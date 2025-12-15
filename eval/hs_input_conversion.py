import numpy as np
import pandas as pd
import os
import math
import torch
from heatsink_fno_model import build_fno3d_heatsink_model, load_fno_checkpoint
import h5py

'''
python eval/hs_input_conversion.py
temperature prediction:
    input: heatsink shape csv
    output: predicted temperature h5
'''
try:
    import pydevd_pycharm
    pydevd_pycharm.settrace(host='host.docker.internal', port=5678,
                            stdout_to_server=True, stderr_to_server=True, suspend=False)
except Exception:
    pass


# 和训练代码里的保持一致
T_BASE = 298.0  # K
DT_BASE = 30.0  # K

def build_theta_M_S_from_row(
    row,
    nx_length: int = 20,      # length 方向网格数
    nz_height: int = 40,      # height 方向网格数
    dy_target: float = 0.5e-3,  # width 方向目标步长（约 0.5mm）
    k_solid: float = 205.0,   # 铝的导热系数 [W/m/K]
    dt_base: float = 30.0     # 论文/代码中用的标称温差 ΔT
):
    """
    根据一行 CSV 参数，生成 3 个 3D 体素网格：theta, M, S。

    约定：
    - 三维数组下标顺序为 (z, y, x) = (height, width, length)
    - 网格尺寸：
        Nz = nz_height
        Ny = round(width_m / dy_target)
        Nx = nx_length
    - theta：此处直接采用“以环境温度为基准”的无量纲温度：
        T_base = ambient_C + 273.15 K
        初始时刻假设整个域温度 = T_base，因此 theta 初始为 0。
      （你后面如果想和训练时的归一化完全一致，可以在外层再做一次线性变换）
    - M：固体掩膜，1 表示散热器铝材，0 表示空气。
      采用“体素中心在哪一侧就算哪一相”，等价于“谁占比多归谁”，
      不再精确切分混合体素。
    - S：体积热源项的无量纲形式：
        S_nd = q_vol * L_char^2 / (k_solid * dt_base)
      其中 L_char 取 max(length, width, height)。

    返回：
        theta: (Nz, Ny, Nx) float32
        M:     (Nz, Ny, Nx) float32
        S:     (Nz, Ny, Nx) float32
        meta:  dict，包含网格尺寸、步长、几何参数等辅助信息
    """

    # ========== 1. 读取几何尺寸 ==========
    length_m = float(row["length_m"])          # x 方向总长
    width_m  = float(row["width_m"])           # y 方向总宽
    height_m = float(row["height_m"])          # z 方向总高

    base_thk = float(row["base_thickness_m"])  # 底板厚度
    fin_h    = float(row["fin_height_m"])      # 翅片高度
    fin_thk  = float(row["fin_thickness_m"])   # 单片厚度
    fin_spacing = float(row["fin_spacing_m"])  # 相邻翅片间隙
    fin_count  = int(row["fin_count"])         # 翅片总数

    # 理论上应该满足：height = base_thickness + fin_height
    if abs(base_thk + fin_h - height_m) > 1e-6:
        raise ValueError(
            f"height_m ({height_m}) 应等于 base_thickness_m + fin_height_m "
            f"({base_thk} + {fin_h})"
        )

    # ========== 2. 构建网格尺寸 ==========
    Nx = int(nx_length)             # length 方向网格数
    Nz = int(nz_height)             # height 方向网格数
    Ny = max(1, int(round(width_m / dy_target)))  # width 方向网格数，步长约 0.5mm

    dx = length_m / Nx
    dy = width_m  / Ny
    dz = height_m / Nz

    # 体素中心坐标（物理坐标）
    z_centers = (np.arange(Nz) + 0.5) * dz   # [0, height]
    y_centers = (np.arange(Ny) + 0.5) * dy   # [0, width]
    x_centers = (np.arange(Nx) + 0.5) * dx   # [0, length]

    Zc, Yc, Xc = np.meshgrid(
        z_centers, y_centers, x_centers, indexing="ij"
    )  # 形状 (Nz, Ny, Nx)

    # ========== 3. theta：初始温度场 ==========
    ambient_C = float(row.get("ambient_C", 25.0))  # CSV 里有 ambient_C
    T_base_K = ambient_C + 273.15                  # 以环境温度作为参考
    # 初始时刻假设整个域温度 = T_base -> theta = (T - T_base)/ΔT = 0
    theta = np.zeros((Nz, Ny, Nx), dtype=np.float32)

    # ========== 4. M：固体掩膜 (1=铝, 0=空气) ==========
    M = np.zeros_like(theta, dtype=np.float32)

    # 4.1 底板：全长、全宽、0 <= z <= base_thk
    M[(Zc <= base_thk + 1e-12)] = 1.0

    # 4.2 翅片：沿 width (y) 方向等间距排布
    #  根据数据：width_m ≈ fin_count*fin_thk + (fin_count-1)*fin_spacing
    pitch = fin_thk + fin_spacing  # 相邻翅片“中心到中心”距离

    # 先在 y 方向构造 1D 的翅片区域掩膜
    y_is_fin = np.zeros(Ny, dtype=bool)
    for i in range(fin_count):
        # 第 i 片翅片覆盖区间 [i*pitch, i*pitch + fin_thk]
        y_start = i * pitch
        y_end   = y_start + fin_thk
        y_is_fin |= ((y_centers >= y_start - 1e-12) &
                     (y_centers <= y_end   + 1e-12))

    # 广播成 3D
    y_is_fin_3d = y_is_fin[None, :, None]  # (1, Ny, 1)

    # 翅片在 base_thk 之上，直到 height_m
    fin_region_z = (Zc > base_thk + 1e-12) & (Zc <= height_m + 1e-12)

    # 填充翅片区域
    M[fin_region_z & y_is_fin_3d] = 1.0

    # 至此，M 里：
    # - M==1 的体素：散热器铝材
    # - M==0 的体素：空气
    # 混合体素按中心位置归类，近似满足“谁占比多归谁”的规则。

    # ========== 5. S：热源体素 + 无量纲化 ==========
    # 先在 q_vol 空间累加（单位：W/m^3），最后统一转成无量纲 S_nd
    S_q = np.zeros_like(theta, dtype=np.float64)  # 用 double 做中间计算

    def add_source(prefix_P: str, prefix_geo: str):
        """
        将一个热源（src1 或 src2）累加到 S_q 中。
        参数命名约定：
            prefix_P  = 'P1_W' 或 'P2_W'
            prefix_geo= 'src1' 或 'src2'
        """

        # 功率
        if prefix_P not in row.index:
            return
        P = float(row[prefix_P])  # [W]
        if not np.isfinite(P) or P <= 0.0:
            return

        # 几何参数（可能是归一化 0~1，也可能是绝对值，做自动检测）
        L_raw  = float(row.get(f"{prefix_geo}_length", 0.0))
        W_raw  = float(row.get(f"{prefix_geo}_width", 0.0))
        cL_raw = float(row.get(f"{prefix_geo}_clength", 0.5))
        cW_raw = float(row.get(f"{prefix_geo}_cwidth", 0.5))

        if L_raw <= 0 or W_raw <= 0:
            return

        # 如果 length/width/center 都在 [0,1] 左右，就按“归一化比例”解释
        if (L_raw <= 1.0 and W_raw <= 1.0 and
                cL_raw <= 1.0 + 1e-6 and cW_raw <= 1.0 + 1e-6):
            src_L = L_raw  * length_m
            src_W = W_raw  * width_m
            cL    = cL_raw * length_m
            cW    = cW_raw * width_m
        else:
            # 否则当作“绝对物理长度(米)”使用
            src_L = L_raw
            src_W = W_raw
            cL    = cL_raw
            cW    = cW_raw

        # 矩形热源在平面上的投影范围
        x_min = max(0.0, cL - 0.5 * src_L)
        x_max = min(length_m, cL + 0.5 * src_L)
        y_min = max(0.0, cW - 0.5 * src_W)
        y_max = min(width_m, cW + 0.5 * src_W)

        # 热源只作用在 z=0 这一层（即 k=0 这一层的体素）
        k0 = 0

        xs = (x_centers >= x_min - 1e-12) & (x_centers <= x_max + 1e-12)
        ys = (y_centers >= y_min - 1e-12) & (y_centers <= y_max + 1e-12)
        mask2d = ys[:, None] & xs[None, :]   # (Ny, Nx)

        # 只在固体内生热（在底板里的那部分）
        solid2d = (M[k0] > 0.5)
        mask2d &= solid2d

        n_cells = int(mask2d.sum())
        if n_cells == 0:
            return

        # 单个体素体积
        cell_vol = dx * dy * dz

        # 每个热源体素的体积热源密度 q_vol [W/m^3]：
        # 总功率 P 在 n_cells 个体素上均匀分布
        q_vol = P / (n_cells * cell_vol)

        # 写入到底层 z=0
        S_q[k0, mask2d] += q_vol

    # 最多两个热源：src1, src2
    add_source("P1_W", "src1")
    add_source("P2_W", "src2")

    # 把 q_vol 转成无量纲 S_nd：
    #   S_nd = q_vol * L_char^2 / (k_solid * dt_base)
    # 与你原训练代码一致：S = q_vol * L^2 / (k * ΔT)
    L_char = max(length_m, width_m, height_m)
    coeff = (L_char ** 2) / (k_solid * dt_base)

    S_nd = (S_q * coeff).astype(np.float32)

    # 一些辅助信息打包出去，方便后续检查或可视化
    meta = dict(
        Nx=Nx, Ny=Ny, Nz=Nz,
        dx=dx, dy=dy, dz=dz,
        length_m=length_m, width_m=width_m, height_m=height_m,
        base_thickness_m=base_thk, fin_height_m=fin_h,
        fin_thickness_m=fin_thk, fin_spacing_m=fin_spacing,
        fin_count=fin_count,
        ambient_C=ambient_C,
        T_base_K=T_base_K,
        L_char=L_char,
        k_solid=k_solid,
        dt_base=dt_base,
    )

    return theta, M, S_nd, meta


def build_theta_M_S_from_csv(csv_path: str, row_index: int = 0):
    """
    从 CSV 文件中读取第 row_index 行，构建 theta, M, S 三个 3D 体素网格。

    参数：
        csv_path  : CSV 文件路径
        row_index : 使用第几行（默认 0）

    返回：
        theta, M, S, meta  同上
    """
    df = pd.read_csv(csv_path)
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row_index={row_index} 超出 CSV 行数范围 [0, {len(df)-1}]")

    row = df.iloc[row_index]
    return build_theta_M_S_from_row(row)

def pad_3d_grids_with_zeros(theta, M, S):
    """
    将 3 个 3D 网格 (Nz, Ny, Nx) 在 z/y/x 三个维度两侧各外延一层体素（值为 0）。

    输入：
        theta, M, S : np.ndarray, 形状均为 (Nz, Ny, Nx)

    输出：
        theta_p, M_p, S_p : np.ndarray, 形状均为 (Nz+2, Ny+2, Nx+2)
    """
    pad_width = ((1, 1), (1, 1), (1, 1))  # (前, 后) × 3 维

    theta_p = np.pad(theta, pad_width, mode="constant", constant_values=0.0)
    M_p     = np.pad(M,     pad_width, mode="constant", constant_values=0.0)
    S_p     = np.pad(S,     pad_width, mode="constant", constant_values=0.0)

    return theta_p, M_p, S_p



def build_ctx_from_csv(csv_path, row_index=0):
    df = pd.read_csv(csv_path)
    row = df.iloc[row_index]

    # 几何尺寸：注意这里和训练代码对应关系
    # length -> x, width -> y, height -> z
    length_m = float(row["length_m"])
    width_m  = float(row["width_m"])
    height_m = float(row["height_m"])

    Lx_abs = length_m
    Ly_abs = width_m
    Lz_abs = height_m

    # ===== 重力向量：一定要保持符号一致 =====
    # 如果 CSV 有 g_x, g_y, g_z，就直接用；否则就用训练代码的默认 [0, -9.81, 0]
    if all(k in row.index for k in ["g_x", "g_y", "g_z"]):
        g_vec = np.array([float(row["g_x"]),
                          float(row["g_y"]),
                          float(row["g_z"])], dtype=np.float64)
    else:
        # 注意这里是 -9.81，对应 “负 y 方向向下”，和 HeatsinkH5 默认完全一致
        g_vec = np.array([0.0, -9.81, 0.0], dtype=np.float64)

    g_mag = float(np.linalg.norm(g_vec))
    if g_mag < 1e-12:
        g_hat = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        g_mag = 1e-12
    else:
        g_hat = (g_vec / g_mag).astype(np.float32)

    # ===== 环境温度 =====
    T_amb_C = float(row["ambient_C"])
    T_amb_K = T_amb_C + 273.15
    theta_inf_mean_hat = (T_amb_K - T_BASE) / DT_BASE

    # ===== alpha（热扩散率）=====
    # 优先直接读 alpha；否则由 k, rho, cp 算
    if "alpha" in row.index:
        alpha = float(row["alpha"])
    else:
        k=205.0
        rho=2700.0
        cp=900.0
        alpha = k / (rho * cp)

    ctx_glb = np.array([
        math.log(max(Lz_abs, 1e-12)),
        math.log(max(Ly_abs, 1e-12)),
        math.log(max(Lx_abs, 1e-12)),
        float(g_hat[0]),           # 和训练完全一致：就是 g_hat
        float(g_hat[1]),
        float(g_hat[2]),
        math.log(max(g_mag, 1e-12)),
        float(theta_inf_mean_hat),
        math.log(max(alpha, 1e-12)),
    ], dtype=np.float32)

    return ctx_glb, alpha


def prepare_fno_inputs_from_grids(
    theta_grid: np.ndarray,
    M_grid: np.ndarray,
    S_grid: np.ndarray,
    ctx_glb: np.ndarray,
    dtau: float,
    device: str = "cuda:0"
):
    """
    将 {theta, M, S, ctx_glb, dtau} 打包成直接可送入 FNO3D 模型的输入张量。

    参数
    ----
    theta_grid : np.ndarray, shape = (Z, Y, X)
        无量纲温度 θ = (T - T_BASE) / DT_BASE
    M_grid     : np.ndarray, shape = (Z, Y, X)
        固体 mask，1=散热器，0=空气
    S_grid     : np.ndarray, shape = (Z, Y, X)
        无量纲体积热源 S_nd
    ctx_glb    : np.ndarray, shape = (9,)
        来自 build_ctx_from_csv，与训练代码 HeatsinkH5.ctx_glb 对齐
    dtau       : float
        当前时间步的无量纲时间步长 dtau = dt * alpha / L^2
    device     : str
        目标设备，例如 "cuda:0" 或 "cpu"

    返回
    ----
    x_torch   : torch.Tensor, shape = (1, 3, Z, Y, X)
    ctx_torch : torch.Tensor, shape = (1, 10)
    """
    # 1. 确保 numpy 类型正确
    theta_np = np.asarray(theta_grid, dtype=np.float32)
    M_np     = np.asarray(M_grid,     dtype=np.float32)
    S_np     = np.asarray(S_grid,     dtype=np.float32)
    ctx_glb  = np.asarray(ctx_glb,    dtype=np.float32)

    assert theta_np.shape == M_np.shape == S_np.shape, "theta/M/S 形状必须一致 (Z,Y,X)"
    assert ctx_glb.shape == (9,), "ctx_glb 形状必须是 (9,)"

    # 2. 构造 x: (1, 3, Z, Y, X)
    x_np = np.stack([theta_np, M_np, S_np], axis=0)  # (3, Z, Y, X)
    x_torch = torch.from_numpy(x_np).unsqueeze(0)    # (1, 3, Z, Y, X)

    # 3. 构造 ctx_vec = [ctx_glb, log_dtau_clamped] → (1, 10)
    log_dtau = dtau
    log_dtau = float(np.clip(log_dtau, -20.0, 20.0))

    ctx_vec = np.concatenate(
        [ctx_glb, np.array([log_dtau], dtype=np.float32)],
        axis=0
    )  # (10,)

    ctx_torch = torch.from_numpy(ctx_vec.astype(np.float32)).unsqueeze(0)  # (1, 10)

    # 4. 放到目标 device
    dev = torch.device(device)
    x_torch   = x_torch.to(dev)
    ctx_torch = ctx_torch.to(dev)

    return x_torch, ctx_torch

def compute_dtau_from_csv(csv_path, dt_phys_seconds, row_index=0,
                          k_al=205.0, rho_al=2700.0, cp_al=900.0):
    """
    根据 CSV 中的 length_m / width_m / height_m 和给定的物理时间步长 Δt，
    计算无量纲时间步长 dtau = alpha * dt / L^2。

    参数
    ----
    csv_path : str
        包含几何参数的 csv 文件路径（需要有列：length_m, width_m, height_m）。
    dt_phys_seconds : float
        物理时间步长 Δt，单位：秒。
    row_index : int
        使用 CSV 中第几行（从 0 开始），默认 0。
    k_al : float
        铝的导热系数 k，单位 W/m/K。
    rho_al : float
        铝的密度 rho，单位 kg/m^3。
    cp_al : float
        铝的比热 cp，单位 J/kg/K。

    返回
    ----
    dtau_nd : float
        无量纲时间步长 Δτ = α * Δt / L^2
    """
    # 1. 计算铝的热扩散率 alpha [m^2/s]
    alpha = k_al / (rho_al * cp_al)

    # 2. 用 pandas 读取 CSV
    df = pd.read_csv(csv_path)

    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row_index={row_index} 超出 CSV 行数范围 (0 ~ {len(df) - 1})")

    row = df.iloc[row_index]

    # 从这一行里取 length/width/height（单位：m）
    Lx = float(row["length_m"])   # x 方向长度 [m]
    Ly = float(row["width_m"])    # y 方向长度 [m]
    Lz = float(row["height_m"])   # z 方向长度 [m]

    # 3. L 取三者最大值
    L = max(Lx, Ly, Lz)

    # 4. 计算无量纲时间步长
    dt_phys = float(dt_phys_seconds)
    dtau_nd = alpha * dt_phys / (L * L)

    return dtau_nd

def save_history_to_h5(
    h5_path: str,
    T_history: np.ndarray,
    t_history: np.ndarray,
    theta_history: np.ndarray = None,
    meta: dict = None,
    overwrite: bool = True,
):
    """
    将温度时间序列写入 .h5 文件。

    参数
    ----
    h5_path : str
        输出 h5 文件路径。
    T_history : np.ndarray
        绝对温度时间序列，形状 (Nt, Z, Y, X)，单位 K。
    t_history : np.ndarray
        物理时间序列，形状 (Nt, )，单位 s，对应 T_history 的每一帧。
    theta_history : np.ndarray, 可选
        无量纲温度时间序列，形状 (Nt, Z, Y, X)，可选。
    meta : dict, 可选
        额外元数据，会写到 "meta" group 下面。
        - 若 value 是标量或一维数组，则写成 dataset
        - 若 value 是 str，则写成 attribute
    overwrite : bool
        若目标文件已存在，是否覆盖。
    """
    T_history = np.asarray(T_history, dtype=np.float32)
    t_history = np.asarray(t_history, dtype=np.float32)

    assert T_history.ndim == 4, f"T_history 维度应为 4, got {T_history.ndim}"
    assert t_history.ndim == 1, f"t_history 维度应为 1, got {t_history.ndim}"
    Nt_T = T_history.shape[0]
    Nt_t = t_history.shape[0]
    assert Nt_T == Nt_t, f"时间步数不一致: T_history[0]={Nt_T}, t_history={Nt_t}"

    if theta_history is not None:
        theta_history = np.asarray(theta_history, dtype=np.float32)
        assert theta_history.shape == T_history.shape, \
            f"theta_history 形状 {theta_history.shape} 必须与 T_history {T_history.shape} 一致"

    # 若文件存在且允许覆盖，先删掉
    if os.path.exists(h5_path) and overwrite:
        os.remove(h5_path)

    with h5py.File(h5_path, "w") as f:
        # 主数据集
        dset_T = f.create_dataset(
            "T_history", data=T_history,
            compression="gzip", compression_opts=4
        )
        dset_t = f.create_dataset(
            "time_phys", data=t_history,
            compression="gzip", compression_opts=4
        )

        # 可选：无量纲温度
        if theta_history is not None:
            f.create_dataset(
                "theta_history", data=theta_history,
                compression="gzip", compression_opts=4
            )

        # 可选：meta 信息
        if meta is not None:
            g_meta = f.create_group("meta")
            for key, val in meta.items():
                if isinstance(val, str):
                    # 字符串写成 attribute
                    g_meta.attrs[key] = val
                else:
                    arr = np.asarray(val)
                    g_meta.create_dataset(key, data=arr)

        # 给数据集加一点简单的说明（optional）
        dset_T.attrs["units"] = "K"
        dset_t.attrs["units"] = "s"
        if theta_history is not None:
            f["theta_history"].attrs["meaning"] = "non-dimensional temperature theta = (T - T_BASE)/DT_BASE"


if __name__ == "__main__":
    # ================== 前面这部分你已有 ==================
    cur_dir = os.path.dirname(__file__)
    csv_path = os.path.join(cur_dir, "testcsv.csv")  # 你可以改成自己的路径

    # 物理时间步长序列（单位：秒）
    dtau_t = [100, 100, 200, 400, 400, 400, 800, 1200]

    theta, M, S, meta = build_theta_M_S_from_csv(csv_path, row_index=0)
    theta_p, M_p, S_p = pad_3d_grids_with_zeros(theta, M, S)

    # ctx_glb: 9 维全局特征；alpha 只是顺便返回，可以不用
    ctx_glb, alpha = build_ctx_from_csv(csv_path, row_index=0)

    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = build_fno3d_heatsink_model(
        in_c=3,
        width=24,
        layers=4,
        modes=(12, 24, 4),
        dropout=0.05,
        context_dim=10,
        spec_pad_type="reflect",
        spec_pad=(8, 8, 8),
        ufno_scales=3,
        device=device_name,
    )
    ckpt_path = os.path.join(cur_dir, "ckpt_ep120.pt")
    load_fno_checkpoint(ckpt_path, model, device=device_name)

    model.eval()
    # ======================================================
    # ============ 从这里开始是“多步滚动”部分 ===============
    # 初始温度场：已经 pad 完的一步
    theta_grid_curr = theta_p.copy()  # numpy, shape (Zp, Yp, Xp)

    # 记录所有时间步的theta和T（包含 t=0 初始）
    theta_history = [theta_grid_curr.copy()]  # list of (Zp, Yp, Xp)
    T_history = [theta_grid_curr * DT_BASE + T_BASE]  # 物理温度场
    t_history = [0.0]  # 累计物理时间（秒）

    with torch.no_grad():
        for dt_phys in dtau_t:
            # 1) 计算当前时间步对应的无量纲 dtau_nd
            dtau_nd = compute_dtau_from_csv(
                csv_path,
                dt_phys_seconds=float(dt_phys),
                row_index=0
            )
            log_dtau = math.log(max(dtau_nd, 1e-12))

            # 2) 用当前温度场 + 固定 M_p, S_p, ctx_glb, log_dtau 组装网络输入
            x_torch, ctx_torch = prepare_fno_inputs_from_grids(
                theta_grid=theta_grid_curr,  # 当前这一步的 theta (已 pad)
                M_grid=M_p,
                S_grid=S_p,
                ctx_glb=ctx_glb,
                dtau=log_dtau,  # 这里传的是 log(dtau_nd)，和训练一致
                device=device_name,
            )

            # 3) 单步前向推理
            delta = model(x_torch, ctx_torch)  # (1, 1, Zp, Yp, Xp)
            theta_next = x_torch[:, 0:1, ...] + delta  # θ_{n+1}

            # 4) 拿到 numpy，更新当前温度场
            theta_next_np = theta_next[0, 0].cpu().numpy()  # (Zp, Yp, Xp)
            theta_grid_curr = theta_next_np

            # 5) 记录这一步的结果（theta、T、累计物理时间）
            theta_history.append(theta_grid_curr.copy())
            T_history.append(theta_grid_curr * DT_BASE + T_BASE)
            t_history.append(t_history[-1] + float(dt_phys))

    # 把历史记录堆叠成数组：shape = (N_time+1, Zp, Yp, Xp)
    theta_history_arr = np.stack(theta_history, axis=0)
    T_history_arr = np.stack(T_history, axis=0)
    t_history_arr = np.array(t_history, dtype=np.float32)

    print("总步数(含初始)：", theta_history_arr.shape[0])  # 应该是 len(dtau_t)+1
    print("最终物理时间：", t_history_arr[-1], "s")
    print("theta_history_arr 形状：", theta_history_arr.shape)
    print("T_history_arr 形状：", T_history_arr.shape)

    #write
    out_h5_path = os.path.join(cur_dir, "pred_history.h5")
    meta_info = {
        "dtau_phys_list": dtau_t ,
        "csv_path": csv_path,
        "row_index": 0,
    }
    save_history_to_h5(
        h5_path=out_h5_path,
        T_history=T_history_arr,
        t_history=t_history_arr,
        theta_history=theta_history_arr,  # 如果不想存就改成 None
        meta=meta_info,
        overwrite=True,
    )

    print("EOF")


