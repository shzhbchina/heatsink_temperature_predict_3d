import h5py
import json
import numpy as np
import argparse  # <-- 导入参数解析模块
import sys

'''
python src/h5_to_json_slicer.py "model_param/eval/run1_eval_992.h5" "model_param/eval/run1_eval_992.json" 5 1
'''
# 1. 定义我们要从H5中提取的数据路径
# (这和之前一样，是我们的“目标清单”)
PATHS_TO_EXTRACT = {
    'T_grid_shadow': 'T_grid_shadow',  # 预期 4D (t, z, y, x)
    'T_pred': 'eval/rollout/T_pred',  # 预期 4D (t, z, y, x)
    'q_vol': 'sources/q_vol'  # 预期 3D (z, y, x)
}


# 2. 这是一个新的“切片”函数，它会使用您传入的参数
def slice_data(dataset, key_name, t_idx, z_idx):
    """
    根据传入的t和z索引，从高维数据中提取一个2D (Y-X) 切片。
    """
    shape = dataset.shape
    print(f"  > 正在处理 '{key_name}'，原始形状: {shape}")

    try:
        if key_name in ['T_grid_shadow', 'T_pred']:
            # 这是一个 4D (t, z, y, x) 数组
            # 我们要提取: [t=t_idx, z=z_idx, y=所有, x=所有]
            sliced_data = dataset[t_idx, z_idx, :, :]
            print(f"  > 4D切片(t={t_idx}, z={z_idx})。新形状: {sliced_data.shape}")

        elif key_name == 'q_vol':
            # 这是一个 3D (z, y, x) 数组
            # t_idx 参数在这里用不上，我们只用 z_idx
            # 我们要提取: [z=z_idx, y=所有, x=所有]
            sliced_data = dataset[z_idx, :, :]
            print(f"  > 3D切片(z={z_idx})。新形状: {sliced_data.shape}")

        else:
            # 理论上不会发生，因为我们的清单是固定的
            return None

        # 将NumPy数组转换为Python的“列表”，JSON才能识别
        return sliced_data.tolist()

    except IndexError:
        print(f"  [致命错误] 切片索引超出范围！")
        print(f"  对于 '{key_name}' (形状 {shape})，您请求的 (t={t_idx}, z={z_idx}) 索引不存在。")
        return None
    except Exception as e:
        print(f"  [致命错误] 处理 '{key_name}' 时发生未知错误: {e}")
        return None


# 3. 脚本主程序
def main():
    # --- 3A. 定义我们的4个参数 ---
    parser = argparse.ArgumentParser(
        description="HDF5 Slicer Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python3 h5_slicer_tool.py "input_data.h5" "output_slice.json" 10 5
  (这会提取 t=10, z=5 时的Y-X平面数据)
"""
    )

    parser.add_argument("input_file", help="[参数1] HDF5输入文件的路径 (例如: data.h5)")
    parser.add_argument("output_file", help="[参数2] JSON输出文件的路径 (例如: slice.json)")
    parser.add_argument("t_index", type=int, help="[参数3] 您想要的 T 轴坐标 (时间索引)")
    parser.add_argument("z_index", type=int, help="[参数4] 您想要的 Z 轴坐标 (Z平面索引)")

    # --- 3B. 解析命令行传入的参数 ---
    args = parser.parse_args()

    # 最终用于保存到JSON的字典
    final_json_data = {}

    print(f"--- 正在打开 H5 文件: {args.input_file} ---")

    try:
        with h5py.File(args.input_file, 'r') as hf:

            for key_name, path in PATHS_TO_EXTRACT.items():
                print(f"正在读取路径: {path} ...")

                if path in hf:
                    dataset = hf[path]

                    # --- 3C. 把参数传入切片函数 ---
                    sliced_list = slice_data(dataset, key_name, args.t_index, args.z_index)

                    if sliced_list is not None:
                        final_json_data[key_name] = sliced_list
                        print(f"  [成功] 已处理并添加 '{key_name}'。")
                    else:
                        print(f"  [失败] 未能处理 '{key_name}'。")
                else:
                    print(f"  [警告] 路径 '{path}' 在H5文件中不存在。")

    except FileNotFoundError:
        print(f"[致命错误] H5 文件未找到: {args.input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"[致命错误] 无法读取H5文件: {e}")
        sys.exit(1)

    # --- 3D. 写入到您指定的JSON文件 ---
    if not final_json_data:
        print("--- 未能提取到任何有效数据。JSON文件未创建。 ---")
        sys.exit(0)

    print(f"--- B正在将提取的切片写入 JSON 文件: {args.output_file} ---")

    try:
        with open(args.output_file, 'w') as jf:
            # 我们不使用 indent=4，来让JSON文件体积最小
            json.dump(final_json_data, jf)

        print("\n" + "=" * 30)
        print("★★★ 转换成功！ ★★★")
        print(f"已创建文件: {args.output_file}")
        print(f"切片参数: T={args.t_index}, Z={args.z_index}")
        print("=" * 30 + "\n")

    except Exception as e:
        print(f"[致命错误] 写入 JSON 文件时出错: {e}")
        sys.exit(1)


# 4. 运行主函数
if __name__ == "__main__":
    main()