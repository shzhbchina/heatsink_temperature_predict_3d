# -*- coding: utf-8 -*-
"""
极简稳妥版：H5 -> MySQL
- 清表重建
- 只生成数值型 TSV（不写 data_key 文本列）
- 服务器端 LOAD DATA INFILE 时用 SET 注入 data_key / 固定 t_idx
- 可选最后重建索引

Convert h5 to mysql

数据源：
  4D: T_grid_shadow            => H5["T_grid_shadow"]         (T,Z,Y,X)
  4D: T_pred                   => H5["eval/rollout/T_pred"]   (T,Z,Y,X)
  3D: q_vol                    => H5["sources/q_vol"]         (Z,Y,X)  (t_idx = -1)

表结构：
  id BIGINT AUTO_INCREMENT PRIMARY KEY
  data_key VARCHAR(50)
  t_idx INT
  z_idx INT
  y_idx INT
  x_idx INT
  value DOUBLE

python heatsink_prj/src/h5_to_mysql.py \
  --h5_file heatsink_prj/dat/run1_eval_992.h5 \
  --db_host localhost \
  --db_user blog_user \
  --db_name my_blog_db \
  --table_name prediction_data \
  --secure-dir /var/lib/mysql-files \
  --reindex minimal

"""

import os
import time
import getpass
import argparse
import numpy as np
import h5py
import mysql.connector

# H5 路径映射
# PATHS = {
#     "T_grid_shadow": "T_grid_shadow",            # 4D
#     "T_pred":        "eval/rollout/T_pred",      # 4D
#     "q_vol":         "sources/q_vol",            # 3D
# }

PATHS = {
    "T_grid_shadow": "T_history",            # 4D
    "T_pred":        "T_history",      # 4D
    "q_vol":         "0",            # 3D
}


def connect_mysql(host, user, password, db):
    print(f"[DB] 连接 MySQL: host={host}, db={db}")
    cnx = mysql.connector.connect(
        host=host, user=user, password=password, database=db, autocommit=True
    )
    return cnx

def drop_and_create_table(cur, table):
    print(f"[TABLE] 丢弃并重建 `{table}`")
    cur.execute(f"DROP TABLE IF EXISTS `{table}`")
    cur.execute(f"""
        CREATE TABLE `{table}` (
            `id` BIGINT NOT NULL AUTO_INCREMENT,
            `data_key` VARCHAR(50) NOT NULL,
            `t_idx` INT NOT NULL,
            `z_idx` INT NOT NULL,
            `y_idx` INT NOT NULL,
            `x_idx` INT NOT NULL,
            `value` DOUBLE NOT NULL,
            PRIMARY KEY (`id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

def create_indexes(cur, table, mode="minimal"):
    plan = [("idx_dtzyx", ["data_key","t_idx","z_idx","y_idx","x_idx"])]
    if mode == "full":
        plan += [
            ("idx_dk_tz", ["data_key","t_idx","z_idx"]),
            ("idx_dk_tx", ["data_key","t_idx","x_idx"]),
        ]
    for name, cols in plan:
        cols_sql = ", ".join(f"`{c}`" for c in cols)
        # 兼容不同 MySQL 的在线DDL能力
        attempts = [
            (f"ALTER TABLE `{table}` ADD INDEX `{name}` ({cols_sql}), ALGORITHM=INPLACE, LOCK=NONE", "INPLACE+NONE"),
            (f"ALTER TABLE `{table}` ADD INDEX `{name}` ({cols_sql}), ALGORITHM=INPLACE", "INPLACE"),
            (f"ALTER TABLE `{table}` ADD INDEX `{name}` ({cols_sql})", "PLAIN"),
        ]
        for sql, tag in attempts:
            try:
                print(f"[INDEX] CREATE `{name}` ({', '.join(cols)}) [{tag}]")
                cur.execute(sql)
                break
            except mysql.connector.Error:
                if tag == "PLAIN":
                    raise
                continue

def write_tsv_4d_numeric(ds, out_path):
    """
    4D (T,Z,Y,X) -> 纯数值 TSV：列为 (t_idx, z_idx, y_idx, x_idx, value)
    """
    T, Z, Y, X = ds.shape
    y_idx, x_idx = np.indices((Y, X))
    with open(out_path, "w", buffering=1024*1024) as fh:
        t0_all = time.time()
        for t in range(T):
            t0 = time.time()
            for z in range(Z):
                plane = ds[t, z, :, :]  # (Y,X) float
                arr = np.column_stack((
                    np.full(Y*X, t, dtype=np.int64),
                    np.full(Y*X, z, dtype=np.int64),
                    y_idx.ravel().astype(np.int64),
                    x_idx.ravel().astype(np.int64),
                    plane.ravel().astype(np.float64),
                ))
                # %d %d %d %d %.10g —— 纯数值，没有任何文本列
                np.savetxt(fh, arr, fmt="%d\t%d\t%d\t%d\t%.10g")
            print(f"    - t={t}  -> {os.path.basename(out_path)} 用时 {time.time()-t0:.2f}s")
        print(f"[TSV] 写完 {os.path.basename(out_path)} ，总用时 {time.time()-t0_all:.2f}s")

def write_tsv_3d_numeric(ds, out_path):
    """
    3D (Z,Y,X) -> 纯数值 TSV：列为 (z_idx, y_idx, x_idx, value)
    t_idx 不在文件里，LOAD 时用 SET t_idx=-1
    """
    Z, Y, X = ds.shape
    y_idx, x_idx = np.indices((Y, X))
    with open(out_path, "w", buffering=1024*1024) as fh:
        t0 = time.time()
        for z in range(Z):
            plane = ds[z, :, :]
            arr = np.column_stack((
                np.full(Y*X, z, dtype=np.int64),
                y_idx.ravel().astype(np.int64),
                x_idx.ravel().astype(np.int64),
                plane.ravel().astype(np.float64),
            ))
            np.savetxt(fh, arr, fmt="%d\t%d\t%d\t%.10g")
        print(f"[TSV] 写完 {os.path.basename(out_path)} ，用时 {time.time()-t0:.2f}s")

def load_infile_4d(cur, table, server_file, data_key):
    # 文件列：(t_idx, z_idx, y_idx, x_idx, value)  ——> SET data_key='...'
    p = server_file.replace("\\", "\\\\")
    sql = (
        f"LOAD DATA INFILE '{p}' INTO TABLE `{table}` "
        f"FIELDS TERMINATED BY '\\t' LINES TERMINATED BY '\\n' "
        f"(t_idx, z_idx, y_idx, x_idx, value) "
        f"SET data_key='{data_key}'"
    )
    t0 = time.time()
    cur.execute(sql)
    print(f"[LOAD] {os.path.basename(server_file)} -> {table} OK （{time.time()-t0:.2f}s）")

def load_infile_3d(cur, table, server_file, data_key):
    # 文件列：(z_idx, y_idx, x_idx, value)  ——> SET data_key='...', t_idx=-1
    p = server_file.replace("\\", "\\\\")
    sql = (
        f"LOAD DATA INFILE '{p}' INTO TABLE `{table}` "
        f"FIELDS TERMINATED BY '\\t' LINES TERMINATED BY '\\n' "
        f"(z_idx, y_idx, x_idx, value) "
        f"SET data_key='{data_key}', t_idx=-1"
    )
    t0 = time.time()
    cur.execute(sql)
    print(f"[LOAD] {os.path.basename(server_file)} -> {table} OK （{time.time()-t0:.2f}s）")


def parse_args():
    ap = argparse.ArgumentParser("H5 -> MySQL")
    # ... 原有参数保持不变 ...
    ap.add_argument("--h5_file", type=str, required=True)
    ap.add_argument("--db_host", type=str, default="localhost")
    ap.add_argument("--db_user", type=str, required=True)
    ap.add_argument("--db_name", type=str, required=True)
    ap.add_argument("--table_name", type=str, required=True)
    ap.add_argument("--secure-dir", type=str, required=True)
    ap.add_argument("--reindex", choices=["none", "minimal", "full"], default="minimal")

    # 【新增】增加密码参数
    ap.add_argument("--password", type=str, default=None, help="直接传入密码，避免交互输入")

    return ap.parse_args()

def main():
    args = parse_args()
    if args.password:
        pwd = args.password
        print(f"[DB] 使用命令行参数传入的密码 (长度: {len(pwd)})")
    else:
        pwd = getpass.getpass(f"请输入 MySQL 用户 '{args.db_user}' 的密码: ")

    cnx = None
    try:
        cnx = connect_mysql(args.db_host, args.db_user, pwd, args.db_name)
        cur = cnx.cursor()

        # 1) 清空并重建表
        drop_and_create_table(cur, args.table_name)

        # 2) 生成数值型 TSV（直接写到 secure-dir），然后服务器端 LOAD
        with h5py.File(args.h5_file, "r") as hf:
            # --- 4D: T_grid_shadow ---
            if PATHS["T_grid_shadow"] in hf:
                ds = hf[PATHS["T_grid_shadow"]]
                tsv = os.path.join(args.secure_dir, f"{args.table_name}_T_grid_shadow.tsv")
                print(f"--- 处理 'T_grid_shadow' （shape={ds.shape}, dtype={ds.dtype}） ---")
                write_tsv_4d_numeric(ds, tsv)
                load_infile_4d(cur, args.table_name, tsv, "T_grid_shadow")
            else:
                print("[WARN] 缺少 T_grid_shadow，跳过")

            # --- 4D: T_pred ---
            if PATHS["T_pred"] in hf:
                ds = hf[PATHS["T_pred"]]
                tsv = os.path.join(args.secure_dir, f"{args.table_name}_T_pred.tsv")
                print(f"--- 处理 'T_pred' （shape={ds.shape}, dtype={ds.dtype}） ---")
                write_tsv_4d_numeric(ds, tsv)
                load_infile_4d(cur, args.table_name, tsv, "T_pred")
            else:
                print("[WARN] 缺少 T_pred，跳过")

            # --- 3D: q_vol ---
            if PATHS["q_vol"] in hf:
                ds = hf[PATHS["q_vol"]]
                tsv = os.path.join(args.secure_dir, f"{args.table_name}_q_vol.tsv")
                print(f"--- 处理 'q_vol' （shape={ds.shape}, dtype={ds.dtype}） ---")
                write_tsv_3d_numeric(ds, tsv)
                load_infile_3d(cur, args.table_name, tsv, "q_vol")
            else:
                print("[WARN] 缺少 q_vol，跳过")

        # 3) 重建索引（可选）
        if args.reindex != "none":
            create_indexes(cur, args.table_name, mode=args.reindex)

        print("[OK] 全部导入完成。")
    except mysql.connector.Error as e:
        msg = f"{e.errno} ({e.sqlstate}): {e.msg}" if hasattr(e, "errno") else str(e)
        print(f"[致命] MySQL 错误：{msg}")
        raise
    except Exception as e:
        print(f"[致命] 异常：{e}")
        raise
    finally:
        if cnx is not None:
            try:
                if cnx.is_connected():
                    cur.close()
                    cnx.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()
