# -*- coding: utf-8 -*-
import os, math, argparse, time as _time
import h5py, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import glob
from pathlib import Path

"""
v8 (fixed) + ENERGY PATH (C)

关键改动：
1) 在数据阶段（MultiPairDataset.__getitem__）构造 ctx = [ctx_glb(9维), log(dtau)(1维)] → 共10维。
2) 训练环节直接使用 cond["ctx"]，不再在loop内拼log(dtau)。
3) 主监督为 Δθ（= y - x[:,0:1]），模型输出 Δθ，y_pred = x[:,0:1] + Δθ。

其余功能保持：FV/strong PDE、Robin BC、能量约束、空气锚点、频域padding等。

#specify h5, 
#if all files --data_glob "dat_test/*.h5" "dat/case_0001_L67_W73_Hb8_F8.h5"
python src/train_fno_pino_h_9.py \
  --data_glob "dat/case_0001_L67_W73_Hb8_F8.h5" \
  --epochs 200 --batch 1 --lr 1e-3 \
  --width 24 --layers 2 --mz 12 --my 12 --mx 12 \
  --use_pde --lam_pde 1.0 --use_bc --lam_bc 1.0 --lam_sup 10000.0 \
  --max_skip 1 \
  --out_dir "$HOME/model_param/heatsink_fno_pino_learnh"

python src/train_fno_pino_h_9_cuda.py \
  --data_glob "dat/case_0001_L67_W73_Hb8_F8.h5"  \
  --epochs 200 --batch 1 --lr 5e-4 \
  --width 24 --layers 4 --mz 12 --my 24 --mx 4 \
  --use_pde --lam_pde 1.0 --use_bc --lam_bc 0.0 --lam_sup 100.0 \
  --use_energy --lam_energy 1.0 \
  --air-anchor 1e-3\
  --spec-pad-type reflect --spec-pad-z 8 --spec-pad-y 8 --spec-pad-x 8 \
  --max_skip 1 --out_dir "model_param/heatsink_fno_pino_learnh" \
  --device cuda:0 --num-workers 0 --pin-memory \
  --preload-gpu


#if continue
python src/train_fno_pino_h_9_cuda.py \
  --data_glob "dat/case_0001_L67_W73_Hb8_F8.h5" \
  --epochs 500  --batch 1 --lr 2e-5 \
  --width 24 --layers 4 --mz 12 --my 24 --mx 4 \
  --use_pde --lam_pde 1.0 --use_bc --lam_bc 0.0 --lam_sup 1.0\
  --use_energy --lam_energy 1.0 \
  --spec-pad-type reflect --spec-pad-z 8 --spec-pad-y 8 --spec-pad-x 8 \
  --max_skip 1 --out_dir "model_param/heatsink_fno_pino_learnh" \
  --ckpt "model_param/heatsink_fno_pino_learnh/ckpt_ep200.pt" \
  --device cuda:0 --num-workers 0 --pin-memory \
  --preload-gpu \
  --resume
  
  
"""

# =================== 常量：新的无量纲基准 ===================
T_BASE = 298.0  # K
DT_BASE = 30.0  # K


import pydevd_pycharm
pydevd_pycharm.settrace(host='host.docker.internal', port=5678,
                        stdout_to_server=True, stderr_to_server=True, suspend=False)


# ------------------------- FD 工具（保留 strong 用） -------------------------
def laplacian_3d(u, dz, dy, dx):
    pad = (1, 1, 1, 1, 1, 1)
    up = F.pad(u, pad, mode='replicate')
    c = up[:, :, 1:-1, 1:-1, 1:-1]
    lap = (up[:, :, 2:, 1:-1, 1:-1] - 2 * c + up[:, :, :-2, 1:-1, 1:-1]) / (dz * dz) \
          + (up[:, :, 1:-1, 2:, 1:-1] - 2 * c + up[:, :, 1:-1, :-2, 1:-1]) / (dy * dy) \
          + (up[:, :, 1:-1, 1:-1, 2:] - 2 * c + up[:, :, 1:-1, 1:-1, :-2]) / (dx * dx)
    return lap


# ======= 固体侧单边梯度（与你现有一致） =======
def one_sided_grad_on_interface_v2(u, Ms, nsurf, dz, dy, dx):
    def neigh_u_z(t):
        tm1 = torch.cat([t[:, :, :1, :, :], t[:, :, :-1, :, :]], dim=2)
        tp1 = torch.cat([t[:, :, 1:, :, :], t[:, :, -1:, :, :]], dim=2)
        return tm1, tp1

    def neigh_u_y(t):
        tm1 = torch.cat([t[:, :, :, :1, :], t[:, :, :, :-1, :]], dim=3)
        tp1 = torch.cat([t[:, :, :, 1:, :], t[:, :, :, -1:, :]], dim=3)
        return tm1, tp1

    def neigh_u_x(t):
        tm1 = torch.cat([t[:, :, :, :, :1], t[:, :, :, :, :-1]], dim=4)
        tp1 = torch.cat([t[:, :, :, :, 1:], t[:, :, :, :, -1:]], dim=4)
        return tm1, tp1

    def neigh_Ms_z(M):
        z0 = torch.zeros_like(M[:, :, :1, :, :])
        Mm1 = torch.cat([z0, M[:, :, :-1, :, :]], dim=2)
        Mp1 = torch.cat([M[:, :, 1:, :, :], z0], dim=2)
        return Mm1, Mp1

    def neigh_Ms_y(M):
        z0 = torch.zeros_like(M[:, :, :, :1, :])
        Mm1 = torch.cat([z0, M[:, :, :, :-1, :]], dim=3)
        Mp1 = torch.cat([M[:, :, :, 1:, :], z0], dim=3)
        return Mm1, Mp1

    def neigh_Ms_x(M):
        z0 = torch.zeros_like(M[:, :, :, :, :1])
        Mm1 = torch.cat([z0, M[:, :, :, :, :-1]], dim=4)
        Mp1 = torch.cat([M[:, :, :, :, 1:], z0], dim=4)
        return Mm1, Mp1

    uzm1, uzp1 = neigh_u_z(u)
    uym1, uyp1 = neigh_u_y(u)
    uxm1, uxp1 = neigh_u_x(u)
    Mzm1, Mzp1 = neigh_Ms_z(Ms)
    Mym1, Myp1 = neigh_Ms_y(Ms)
    Mxm1, Mxp1 = neigh_Ms_x(Ms)

    nz = nsurf[:, 0:1]; ny = nsurf[:, 1:2]; nx = nsurf[:, 2:3]
    Ms_solid = (Ms > 0.5).float()

    gz_bwd = (u - uzm1) / dz; gz_fwd = (uzp1 - u) / dz
    have_minus_z = (Mzm1 > 0.5); have_plus_z = (Mzp1 > 0.5)
    use_bwd_z = have_minus_z & ~have_plus_z; use_fwd_z = have_plus_z & ~have_minus_z
    both_z = have_minus_z & have_plus_z
    use_bwd_z = use_bwd_z | (both_z & (nz >= 0)); use_fwd_z = use_fwd_z | (both_z & (nz < 0))
    gz = torch.zeros_like(u); gz = torch.where(use_bwd_z, gz_bwd, gz); gz = torch.where(use_fwd_z, gz_fwd, gz); gz = gz * Ms_solid

    gy_bwd = (u - uym1) / dy; gy_fwd = (uyp1 - u) / dy
    have_minus_y = (Mym1 > 0.5); have_plus_y = (Myp1 > 0.5)
    use_bwd_y = have_minus_y & ~have_plus_y; use_fwd_y = have_plus_y & ~have_minus_y
    both_y = have_minus_y & have_plus_y
    use_bwd_y = use_bwd_y | (both_y & (ny >= 0)); use_fwd_y = use_fwd_y | (both_y & (ny < 0))
    gy = torch.zeros_like(u); gy = torch.where(use_bwd_y, gy_bwd, gy); gy = torch.where(use_fwd_y, gy_fwd, gy); gy = gy * Ms_solid

    gx_bwd = (u - uxm1) / dx; gx_fwd = (uxp1 - u) / dx
    have_minus_x = (Mxm1 > 0.5); have_plus_x = (Mxp1 > 0.5)
    use_bwd_x = have_minus_x & ~have_plus_x; use_fwd_x = have_plus_x & ~have_minus_x
    both_x = have_minus_x & have_plus_x
    use_bwd_x = use_bwd_x | (both_x & (nx >= 0)); use_fwd_x = use_fwd_x | (both_x & (nx < 0))
    gx = torch.zeros_like(u); gx = torch.where(use_bwd_x, gx_bwd, gx); gx = torch.where(use_fwd_x, gx_fwd, gx); gx = gx * Ms_solid
    return gz, gy, gx


def _unpack_deltas(d):
    if isinstance(d, (tuple, list)):
        return float(d[0]), float(d[1]), float(d[2])
    if torch.is_tensor(d):
        v = d.detach().view(-1)
        return float(v[0].item()), float(v[1].item()), float(v[2].item())
    raise TypeError(f"Unsupported type for deltas: {type(d)}")


# === ENERGY PATH (C) === 辅助：界面等效面积（按 nsurf=[z,y,x]；对流按各向面积绝对值加权）
def _area_weight_from_nsurf(nsurf_xyz, dx, dy, dz, Mi):
    nz = np.abs(nsurf_xyz[0]).astype(np.float32)
    ny = np.abs(nsurf_xyz[1]).astype(np.float32)
    nx = np.abs(nsurf_xyz[2]).astype(np.float32)
    A = nx * (dy * dz) + ny * (dx * dz) + nz * (dx * dy)
    return A * (Mi.astype(np.float32))


# ------------------------- 新增：频域卷积的反射/复制 padding 助手 -------------------------
def _fft_reflect_pad3d(x: torch.Tensor, padz: int, pady: int, padx: int, mode: str):
    if mode == 'none' or (padz == 0 and pady == 0 and padx == 0):
        return x, (0, 0, 0)
    B, C, Z, Y, X = x.shape
    pz = int(max(0, min(padz, max(Z - 1, 0))))
    py = int(max(0, min(pady, max(Y - 1, 0))))
    px = int(max(0, min(padx, max(X - 1, 0))))
    if pz == 0 and py == 0 and px == 0:
        return x, (0, 0, 0)
    if mode not in ('reflect', 'replicate'):
        raise ValueError(f"Unsupported pad mode for spectral conv: {mode}")
    x_pad = F.pad(x, (px, px, py, py, pz, pz), mode=mode)
    return x_pad, (pz, py, px)


# ------------------------- 数据 -------------------------
class HeatsinkH5:
    def __init__(self, path, preload_to_device: str = None):
        self.f = h5py.File(path, 'r')
        self.T = self.f['T_grid_shadow'][:]  # (Nt,Nz,Ny,Nx)
        self.dt = self.f['time_dt'][:]       # (Nt-1,)
        self.Nt = self.T.shape[0]

        # grid
        gp = self.f['grid_padded']
        spacing = gp['spacing'][:]  # [dx,dy,dz]
        dims = gp['dims'][:]        # [Nz,Ny,Nx]

        self.dx, self.dy, self.dz = float(spacing[0]), float(spacing[1]), float(spacing[2])
        self.Nz, self.Ny, self.Nx = int(dims[0]), int(dims[1]), int(dims[2])

        # 绝对尺寸
        self.Lz_abs = self.dz * self.Nz
        self.Ly_abs = self.dy * self.Ny
        self.Lx_abs = self.dx * self.Nx

        # masks & bc
        self.Ms = self.f['mask_solid'][:].astype(np.float32)
        self.Mi = self.f['mask_interface'][:].astype(np.float32)
        bc = self.f['bc/robin']
        self.h_init = bc['h_init'][:].astype(np.float32)
        self.T_inf = bc['T_inf'][:].astype(np.float32)
        self.nsurf = self.f['normal_on_surface'][:].astype(np.float32)  # (3,Z,Y,X)

        # === ENERGY PATH (C) ===
        self.area_weight = _area_weight_from_nsurf(self.nsurf, self.dx, self.dy, self.dz, self.Mi)

        # sources
        if 'sources' in self.f:
            self.q_vol = self.f['sources/q_vol'][:].astype(np.float32)  # W/m^3
            self.q_mask = self.f['sources/q_mask'][:].astype(np.float32) if 'q_mask' in self.f['sources'] else np.ones_like(self.q_vol, np.float32)
        else:
            self.q_vol = np.zeros((self.Nz, self.Ny, self.Nx), np.float32)
            self.q_mask = np.ones_like(self.q_vol, np.float32)

        # const & scales
        cst = self.f['const']
        self.k = float(cst['k_solid'][()])
        self.T_amb = float(cst['T_amb'][()])
        self.rho = float(cst['rho_solid'][()]) if 'rho_solid' in cst else None
        self.cp  = float(cst['cp_solid'][()]) if 'cp_solid'  in cst else None

        # 重力
        if 'g' in cst:
            g_raw = np.array(cst['g'][:], dtype=np.float64).reshape(-1)
            self.g_vec = g_raw[:3] if g_raw.size >= 3 else np.array([0.0, -9.81, 0.0], dtype=np.float64)
        else:
            self.g_vec = np.array([0.0, -9.81, 0.0], dtype=np.float64)
        self.g_mag = float(np.linalg.norm(self.g_vec))
        self.g_hat = self.g_vec / max(self.g_mag, 1e-12)

        # scales
        sc = self.f['scales']
        file_alpha = float(sc['alpha'][()]) if 'alpha' in sc else None
        self.L = float(sc['L'][()])
        self.dTref = float(sc['dT_ref'][()])
        self.qref = float(sc['q_ref'][()]) if 'q_ref' in sc else None

        if file_alpha is not None:
            self.alpha = file_alpha
        else:
            if (self.rho is not None) and (self.cp is not None):
                self.alpha = self.k / (self.rho * self.cp)
            else:
                raise ValueError("[data] alpha 缺失且无法由 k/(rho*cp) 推出，请在 H5 中提供 alpha 或 rho/cp。")

        # hat spacings（沿 L）
        self.dx_hat = self.dx / self.L
        self.dy_hat = self.dy / self.L
        self.dz_hat = self.dz / self.L

        # ----------- 全局 context 参数（9维）-----------
        theta_inf_hat = (self.T_inf - T_BASE) / DT_BASE
        if np.any(self.Mi > 0.5):
            theta_inf_mean_hat = float(theta_inf_hat[self.Mi > 0.5].mean())
        else:
            theta_inf_mean_hat = float(theta_inf_hat.mean())

        self.ctx_glb = np.array([
            math.log(max(self.Lz_abs, 1e-12)),
            math.log(max(self.Ly_abs, 1e-12)),
            math.log(max(self.Lx_abs, 1e-12)),
            float(self.g_hat[0]), float(self.g_hat[1]), float(self.g_hat[2]),
            math.log(max(self.g_mag, 1e-12)),
            theta_inf_mean_hat,
            math.log(max(self.alpha, 1e-12))
        ], dtype=np.float32)

        # Bi = h * L / k 需要
        self.L_over_k = self.L / self.k

        # ====== 源项无量纲化 ======
        if (self.rho is None) or (self.cp is None):
            raise ValueError("[data] 计算源项 S 需要 rho 和 cp，请在 H5 的 const 中提供。")
        coeff = (self.L ** 2) / (self.alpha * self.rho * self.cp * DT_BASE)
        self.S_nd = (self.q_vol * coeff).astype(np.float32)

        # 可选：逐体素/逐时刻 h 真值
        self.h_truth = None
        if 'h_uniform_from_sources_all' in self.f:
            g = self.f['h_uniform_from_sources_all']
            if 'h_uniform_field_Wm2K' in g:
                self.h_truth = g['h_uniform_field_Wm2K'][:].astype(np.float32)
        if (self.h_truth is None) and ('h_from_grad_truth_uniform_all' in self.f):
            g2 = self.f['h_from_grad_truth_uniform_all']
            if 'h_uniform_field_Wm2K' in g2:
                self.h_truth = g2['h_uniform_field_Wm2K'][:].astype(np.float32)

        # === 预加载到 GPU（或指定设备） ===
        self.on_device = None
        if preload_to_device is not None:
            dev = torch.device(preload_to_device)
            self.T_t     = torch.from_numpy(self.T).to(device=dev, dtype=torch.float32)
            self.dt_t    = torch.from_numpy(self.dt.astype(np.float32)).to(device=dev)
            self.Ms_t    = torch.from_numpy(self.Ms).to(device=dev)
            self.Mi_t    = torch.from_numpy(self.Mi).to(device=dev)
            self.Tinf_t  = torch.from_numpy(self.T_inf).to(device=dev)
            self.hini_t  = torch.from_numpy(self.h_init).to(device=dev)
            self.nsurf_t = torch.from_numpy(self.nsurf.astype(np.float32)).to(device=dev)
            self.Snd_t   = torch.from_numpy(self.S_nd).to(device=dev)
            self.area_t  = torch.from_numpy(self.area_weight).to(device=dev)
            self.qvol_t  = torch.from_numpy(self.q_vol).to(device=dev)
            self.qmsk_t  = torch.from_numpy(self.q_mask).to(device=dev)
            self.htruth_t = torch.from_numpy(self.h_truth).to(device=dev) if self.h_truth is not None else None

            self.ctx_glb_t   = torch.tensor(self.ctx_glb, dtype=torch.float32, device=dev)   # 9维
            self.dxh_dyh_dzh_t = torch.tensor([self.dz_hat, self.dy_hat, self.dx_hat],
                                              dtype=torch.float32, device=dev)
            self.L_over_k_t  = torch.tensor([self.L / self.k], dtype=torch.float32, device=dev)

            # 释放 CPU numpy 内存（可选）
            self.T = self.dt = self.Ms = self.Mi = self.T_inf = self.h_init = self.nsurf = self.S_nd = None
            self.h_truth = None
            self.ctx_glb = None
            self.area_weight = None
            self.on_device = dev.type  # 'cuda' 或 'cpu'

    def close(self):
        try: self.f.close()
        except: pass


class MultiPairDataset(Dataset):
    def __init__(self, paths, max_skip=1, preload_to: str = None):
        self.paths = [str(p) for p in paths]
        self.max_skip = int(max(1, max_skip))
        self.preload_to = preload_to
        self.index = []
        self.meta = []
        for i, p in enumerate(self.paths):
            with h5py.File(p, 'r') as f:
                Nt, Nz, Ny, Nx = f['T_grid_shadow'].shape
            self.meta.append((Nt, Nz, Ny, Nx))
            self.index.extend([(i, t0) for t0 in range(Nt - 1)])
        self._cache = {}

    def __len__(self):
        return len(self.index)

    def _get_H(self, i):
        H = self._cache.get(i, None)
        if H is None:
            H = HeatsinkH5(self.paths[i], preload_to_device=self.preload_to)
            self._cache[i] = H
        return H

    def worker_init(self, worker_id):
        self._cache = {}

    def __getitem__(self, idx):
        fi, t0 = self.index[idx]
        H = self._get_H(fi)

        # --- 路径 A：已驻留 GPU ---
        if getattr(H, "on_device", None) == "cuda":
            dev = torch.device("cuda")
            Nt = H.T_t.shape[0]
            k_max = min(self.max_skip, Nt - 1 - t0)
            k = np.random.randint(1, k_max + 1)
            t1 = t0 + k

            T0 = H.T_t[t0]; T1 = H.T_t[t1]
            theta0 = (T0 - T_BASE) / DT_BASE
            theta1 = (T1 - T_BASE) / DT_BASE

            dtau = H.dt_t[t0:t1].sum() * (H.alpha / (H.L * H.L))
            # === 这里构造 ctx = [ctx_glb(9), log(dtau)(1)] ===
            dtau_ctx = torch.log(dtau.view(1) + 1e-12).clamp_(-20.0, 20.0)  # (1,)
            ctx = torch.cat([H.ctx_glb_t, dtau_ctx], dim=0)                  # (10,)

            x = torch.stack([theta0, H.Ms_t, H.Snd_t], dim=0)  # (3,Z,Y,X)
            y = theta1[None, ...]                              # (1,Z,Y,X)

            if H.htruth_t is not None:
                h_t0 = H.htruth_t[t0][None, ...]
                has_h = torch.tensor(1, dtype=torch.uint8, device=dev)
            else:
                h_t0 = torch.zeros((1, H.Nz, H.Ny, H.Nx), dtype=torch.float32, device=dev)
                has_h = torch.tensor(0, dtype=torch.uint8, device=dev)

            cond = {
                "dtau": dtau.view(1).to(torch.float32),
                "Tinf": H.Tinf_t[None, ...],
                "h_init": H.hini_t[None, ...],
                "Mi": H.Mi_t[None, ...],
                "Ms": H.Ms_t[None, ...],
                "nsurf": H.nsurf_t,
                "dxh_dyh_dzh": H.dxh_dyh_dzh_t,
                "L_over_k": H.L_over_k_t,
                "ctx_glb": H.ctx_glb_t,    # 9维（保留）
                "ctx": ctx,                # 10维（新增，训练直接用这个）
                "h_truth_t0": h_t0,
                "has_h_truth": has_h,
                "area": H.area_t[None, ...],  # (1,Z,Y,X)
            }
            return x, y, cond

        # --- 路径 B：CPU 懒加载 ---
        Nt = H.T.shape[0]
        k_max = min(self.max_skip, Nt - 1 - t0)
        k = np.random.randint(1, k_max + 1)
        t1 = t0 + k
        T0 = torch.from_numpy(H.T[t0]); T1 = torch.from_numpy(H.T[t1])
        theta0 = (T0 - T_BASE) / DT_BASE
        theta1 = (T1 - T_BASE) / DT_BASE
        dt = float(H.dt[t0:t1].sum())
        dtau = dt * H.alpha / (H.L * H.L)

        # === CPU 路径同样构造 10维 ctx ===
        ctx_glb_cpu = torch.from_numpy(H.ctx_glb).float()              # (9,)
        dtau_ctx = torch.log(torch.tensor([dtau], dtype=torch.float32) + 1e-12).clamp_(-20.0, 20.0)  # (1,)
        ctx = torch.cat([ctx_glb_cpu, dtau_ctx], dim=0)                # (10,)

        x = torch.stack([theta0, torch.from_numpy(H.Ms), torch.from_numpy(H.S_nd)], dim=0)
        y = theta1[None, ...]
        if H.h_truth is not None:
            h_t0 = torch.from_numpy(H.h_truth[t0])[None, ...]
            has_h = torch.tensor(1, dtype=torch.uint8)
        else:
            h_t0 = torch.zeros((1, H.Nz, H.Ny, H.Nx), dtype=torch.float32)
            has_h = torch.tensor(0, dtype=torch.uint8)
        cond = {
            "dtau": torch.tensor([dtau], dtype=torch.float32),
            "Tinf": torch.from_numpy(H.T_inf[None, ...]),
            "h_init": torch.from_numpy(H.h_init[None, ...]),
            "Mi": torch.from_numpy(H.Mi[None, ...]),
            "Ms": torch.from_numpy(H.Ms[None, ...]),
            "nsurf": torch.from_numpy(H.nsurf),
            "dxh_dyh_dzh": torch.tensor([H.dz_hat, H.dy_hat, H.dx_hat], dtype=torch.float32),
            "L_over_k": torch.tensor([H.L / H.k], dtype=torch.float32),
            "ctx_glb": ctx_glb_cpu,  # 9维（保留）
            "ctx": ctx,              # 10维（新增）
            "h_truth_t0": h_t0,
            "has_h_truth": has_h,
            "area": torch.from_numpy(H.area_weight[None, ...]).float(),  # (1,Z,Y,X)
        }
        return x, y, cond


def discover_h5(pattern_or_dir):
    if os.path.isdir(pattern_or_dir):
        files = [str(p) for p in Path(pattern_or_dir).glob("*.h5")]
    else:
        files = sorted(glob.glob(pattern_or_dir))
    files = [f for f in files if Path(f).suffix == ".h5" and not Path(f).name.endswith(".h5.h5")]
    return sorted(files)


def worker_init_fn(worker_id):
    ds = torch.utils.data.get_worker_info().dataset
    if hasattr(ds, "worker_init"):
        ds.worker_init(worker_id)


# ------------------------- 模型（新增：FFT 前反射/复制 padding） -------------------------
class SpectralConv3d(nn.Module):
    def __init__(self, in_c, out_c, modes_z, modes_y, modes_x,
                 pad_type: str = 'reflect', pad_z: int = 8, pad_y: int = 8, pad_x: int = 8):
        super().__init__()
        self.in_c = in_c; self.out_c = out_c
        self.mz, self.my, self.mx = modes_z, modes_y, modes_x
        self.pad_type = pad_type
        self.pad_z = int(pad_z); self.pad_y = int(pad_y); self.pad_x = int(pad_x)

        scale = 1 / (in_c * out_c)
        self.weight = nn.Parameter(scale * torch.randn(in_c, out_c, self.mz, self.my, self.mx, 2))

    def compl_mul3d(self, a, b):
        op = torch.einsum
        return torch.stack([
            op("bczyx,cozyx->bozyx", a[..., 0], b[..., 0]) - op("bczyx,cozyx->bozyx", a[..., 1], b[..., 1]),
            op("bczyx,cozyx->bozyx", a[..., 0], b[..., 1]) + op("bczyx,cozyx->bozyx", a[..., 1], b[..., 0]),
        ], dim=-1)

    def forward(self, x):
        x_pad, (pz, py, px) = _fft_reflect_pad3d(x, self.pad_z, self.pad_y, self.pad_x, self.pad_type)
        B, C, Zp, Yp, Xp = x_pad.shape
        x_ft = torch.view_as_real(torch.fft.rfftn(x_pad, s=(Zp, Yp, Xp), dim=(-3, -2, -1)))
        out_ft = torch.zeros(B, self.out_c, Zp, Yp, Xp // 2 + 1, 2, device=x_pad.device, dtype=x_pad.dtype)

        mz, my, mx = min(self.mz, Zp), min(self.my, Yp), min(self.mx, Xp // 2 + 1)
        w = self.weight[:, :, :mz, :my, :mx, :]
        out_ft[:, :, :mz, :my, :mx, :] = self.compl_mul3d(x_ft[:, :, :mz, :my, :mx, :], w)

        y_pad = torch.fft.irfftn(torch.view_as_complex(out_ft), s=(Zp, Yp, Xp), dim=(-3, -2, -1))
        if (pz | py | px) > 0:
            y = y_pad[:, :, pz:Zp - pz, py:Yp - py, px:Xp - px]
        else:
            y = y_pad
        return y


class PWBlock(nn.Module):
    def __init__(self, width: int, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Conv3d(width, width * expand, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(width * expand, width, kernel_size=1)
        self.norm = nn.GroupNorm(1, width)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.fc2(self.act(self.fc1(x))); y = self.drop(y)
        return self.norm(x + y)


class FNO3D(nn.Module):
    def __init__(self, in_c=3, width=24, modes=(12, 12, 12), layers=4,
                 add_coords=True, fourier_k=8, use_local=True,
                 gn_groups=1, residual_scale=0.5, dropout=0.0,
                 context_dim: int = 0,
                 spec_pad_type: str = 'reflect',
                 spec_pad=(8, 8, 8)):
        super().__init__()
        self.add_coords = add_coords
        self.fourier_k = fourier_k
        self.use_local = use_local
        self.context_dim = int(context_dim)
        self.spec_pad_type = spec_pad_type
        self.spec_pad = tuple(int(v) for v in spec_pad)

        extra_c = 0
        if add_coords:
            extra_c = 3 + 6 * fourier_k
        self.lift = nn.Conv3d(in_c + extra_c, width, 1)
        mz, my, mx = modes
        pz, py, px = self.spec_pad
        self.specs = nn.ModuleList([
            SpectralConv3d(width, width, mz, my, mx,
                           pad_type=self.spec_pad_type, pad_z=pz, pad_y=py, pad_x=px)
            for _ in range(layers)
        ])
        self.ws = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(layers)])
        self.locals = nn.ModuleList(
            [nn.Conv3d(width, width, 3, padding=1, groups=width) for _ in range(layers)]) if use_local else None
        self.norms = nn.ModuleList([nn.GroupNorm(gn_groups, width) for _ in range(layers)])
        self.gammas = nn.ParameterList([nn.Parameter(torch.tensor(float(residual_scale))) for _ in range(layers)])
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else None
        self.proj = nn.Sequential(nn.Conv3d(width, width, 1), nn.GELU(), nn.Conv3d(width, 1, 1))
        self.ctx_mlp = nn.Linear(self.context_dim, width) if self.context_dim > 0 else None

    @staticmethod
    def _coords(Z, Y, X, device, dtype):
        z = torch.linspace(-1, 1, Z, device=device, dtype=dtype)
        y = torch.linspace(-1, 1, Y, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, X, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        return zz, yy, xx

    def _posenc(self, B, Z, Y, X, device, dtype):
        zz, yy, xx = self._coords(Z, Y, X, device, dtype)
        feats = [zz, yy, xx]
        for k in range(1, self.fourier_k + 1):
            w = k * math.pi
            feats += [torch.sin(w * zz), torch.cos(w * zz)]
            feats += [torch.sin(w * yy), torch.cos(w * yy)]
            feats += [torch.sin(w * xx), torch.cos(w * xx)]
        pe = torch.stack(feats, dim=0).unsqueeze(0).repeat(B, 1, 1, 1, 1)
        return pe

    def forward(self, x, ctx: torch.Tensor = None):
        B, _, Z, Y, X = x.shape
        if self.add_coords:
            pe = self._posenc(B, Z, Y, X, x.device, x.dtype)
            x = torch.cat([x, pe], dim=1)
        h = self.lift(x)
        if (self.ctx_mlp is not None) and (ctx is not None):
            bias = self.ctx_mlp(ctx).view(B, -1, 1, 1, 1)
            h = h + bias
        for i, (sc, w, gn) in enumerate(zip(self.specs, self.ws, self.norms)):
            y = sc(h) + w(h)
            if self.use_local:
                y = y + self.locals[i](h)
            y = F.gelu(y)
            if self.drop is not None:
                y = self.drop(y)
            h = h + self.gammas[i] * y
            h = gn(h)
        return self.proj(h)


# ------------------------- HHead（保持不变） -------------------------
class HHead(nn.Module):
    def __init__(self, in_c=7, width=32, layers=2, out_feat=64, mlp_hidden=64,
                 mask_idx=2, h_min=0.0, h_max=30.0,
                 h_prior=10.0, beta=2.0, use_ctx_film=True, ctx_dim=10):
        super().__init__()
        self.mask_idx = mask_idx
        self.h_min = float(h_min); self.h_max = float(h_max)
        self.beta = float(beta)
        self.use_ctx_film = bool(use_ctx_film)

        blocks = []; c = in_c
        for _ in range(layers):
            blocks += [nn.Conv3d(c, width, kernel_size=1, bias=True), nn.GELU()]
            c = width
        blocks += [nn.Conv3d(c, out_feat, kernel_size=1, bias=True)]
        self.net = nn.Sequential(*blocks)

        self.mlp = nn.Sequential(
            nn.Linear(out_feat, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )

        if self.use_ctx_film:
            self.ctx2affine = nn.Sequential(
                nn.Linear(ctx_dim, 32), nn.GELU(),
                nn.Linear(32, 2)
            )
            nn.init.zeros_(self.ctx2affine[-1].weight)
            nn.init.zeros_(self.ctx2affine[-1].bias)

        with torch.no_grad():
            p = (h_prior - self.h_min) / max(1e-6, (self.h_max - self.h_min))
            p = float(np.clip(p, 1e-6, 1 - 1e-6))
            b0 = math.log(p / (1 - p))  # logit
            nn.init.constant_(self.mlp[-1].bias, b0)
            nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, feats, ctx_vec=None):
        B, _, L, W, H = feats.shape
        z = self.net(feats)  # (B,K,L,W,H)
        mask = feats[:, self.mask_idx:self.mask_idx + 1].float()  # (B,1,L,W,H)
        denom = mask.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        g = (z * mask).sum(dim=(2, 3, 4), keepdim=True) / denom  # (B,K,1,1,1)
        g = g.flatten(1)
        raw = self.mlp(g)  # (B,1)
        if self.use_ctx_film and (ctx_vec is not None):
            ab = self.ctx2affine(ctx_vec)  # (B,2)
            a = ab[:, :1]; b = ab[:, 1:2]
            raw = (1.0 + a) * raw + b
        h_norm = torch.sigmoid(self.beta * raw)
        h_scalar = self.h_min + (self.h_max - self.h_min) * h_norm
        h_scalar = h_scalar.view(B, 1, 1, 1, 1)
        return h_scalar * mask


# ------------------------- FV / PDE / BC -------------------------
def _neighbors_replicate(u, axis):
    if axis == 2:
        um1 = torch.cat([u[:, :, :1, :, :], u[:, :, :-1, :, :]], dim=2)
        up1 = torch.cat([u[:, :, 1:, :, :], u[:, :, -1:, :, :]], dim=2)
    elif axis == 3:
        um1 = torch.cat([u[:, :, :, :1, :], u[:, :, :, :-1, :]], dim=3)
        up1 = torch.cat([u[:, :, :, 1:, :], u[:, :, :, -1:, :]], dim=3)
    elif axis == 4:
        um1 = torch.cat([u[:, :, :, :, :1], u[:, :, :, :, :-1]], dim=4)
        up1 = torch.cat([u[:, :, :, :, 1:], u[:, :, :, :, -1:]], dim=4)
    else:
        raise ValueError("axis must be 2/3/4")
    return um1, up1


def _neighbors_zero(M, axis):
    z0 = torch.zeros_like(M[:, :, :1, :, :])
    if axis == 2:
        Mm1 = torch.cat([z0, M[:, :, :-1, :, :]], dim=2)
        Mp1 = torch.cat([M[:, :, 1:, :, :], z0], dim=2)
    elif axis == 3:
        z0 = torch.zeros_like(M[:, :, :, :1, :])
        Mm1 = torch.cat([z0, M[:, :, :, :-1, :]], dim=3)
        Mp1 = torch.cat([M[:, :, :, 1:, :], z0], dim=3)
    elif axis == 4:
        z0 = torch.zeros_like(M[:, :, :, :, :1])
        Mm1 = torch.cat([z0, M[:, :, :, :, :-1]], dim=4)
        Mp1 = torch.cat([M[:, :, :, :, 1:], z0], dim=4)
    else:
        raise ValueError("axis must be 2/3/4")
    return Mm1, Mp1


def fv_divergence(theta, Ms, Mi, Bi_eff, dz, dy, dx, mode='full', theta_inf=None):
    if theta_inf is None:
        theta_inf = 0.0
    th_zm1, th_zp1 = _neighbors_replicate(theta, 2)
    Ms_zm1, Ms_zp1 = _neighbors_zero(Ms, 2)
    inter_p_z = (Ms > 0.5) & (Ms_zp1 > 0.5)
    bndry_p_z = (Ms > 0.5) & ~(Ms_zp1 > 0.5)
    Fzp = torch.where(inter_p_z, (th_zp1 - theta) / dz, torch.zeros_like(theta))
    Fzp = torch.where(bndry_p_z, -Bi_eff * (theta - theta_inf), Fzp)

    inter_m_z = (Ms > 0.5) & (Ms_zm1 > 0.5)
    bndry_m_z = (Ms > 0.5) & ~(Ms_zm1 > 0.5)
    Fzm = torch.where(inter_m_z, (theta - th_zm1) / dz, torch.zeros_like(theta))
    Fzm = torch.where(bndry_m_z, +Bi_eff * (theta - theta_inf), Fzm)
    div_z = (Fzp - Fzm) / dz

    th_ym1, th_yp1 = _neighbors_replicate(theta, 3)
    Ms_ym1, Ms_yp1 = _neighbors_zero(Ms, 3)
    inter_p_y = (Ms > 0.5) & (Ms_yp1 > 0.5)
    bndry_p_y = (Ms > 0.5) & ~(Ms_yp1 > 0.5)
    Fyp = torch.where(inter_p_y, (th_yp1 - theta) / dy, torch.zeros_like(theta))
    Fyp = torch.where(bndry_p_y, -Bi_eff * (theta - theta_inf), Fyp)

    inter_m_y = (Ms > 0.5) & (Ms_ym1 > 0.5)
    bndry_m_y = (Ms > 0.5) & ~(Ms_ym1 > 0.5)
    Fym = torch.where(inter_m_y, (theta - th_ym1) / dy, torch.zeros_like(theta))
    Fym = torch.where(bndry_m_y, +Bi_eff * (theta - theta_inf), Fym)
    div_y = (Fyp - Fym) / dy

    th_xm1, th_xp1 = _neighbors_replicate(theta, 4)
    Ms_xm1, Ms_xp1 = _neighbors_zero(Ms, 4)
    inter_p_x = (Ms > 0.5) & (Ms_xp1 > 0.5)
    bndry_p_x = (Ms > 0.5) & ~(Ms_xp1 > 0.5)
    Fxp = torch.where(inter_p_x, (th_xp1 - theta) / dx, torch.zeros_like(theta))
    Fxp = torch.where(bndry_p_x, -Bi_eff * (theta - theta_inf), Fxp)

    inter_m_x = (Ms > 0.5) & ~(Ms_xm1 <= 0.5)
    bndry_m_x = (Ms > 0.5) & ~(Ms_xm1 > 0.5)
    Fxm = torch.where(inter_m_x, (theta - th_xm1) / dx, torch.zeros_like(theta))
    Fxm = torch.where(bndry_m_x, +Bi_eff * (theta - theta_inf), Fxm)
    div_x = (Fxp - Fxm) / dx

    total_div = div_z + div_y + div_x
    if mode == 'full':
        output_mask = (Ms > 0.5).float()
    elif mode == 'interior':
        output_mask = ((Ms > 0.5) & ~(Mi > 0.5)).float()
    elif mode == 'boundary':
        output_mask = (Mi > 0.5).float()
    else:
        raise ValueError(f"未知的 fv_divergence 模式: {mode}")
    return total_div * output_mask


def pde_residual_strong(theta_next, theta_curr, dtau, Ms, dxh_dyh_dzh, S):
    dz, dy, dx = _unpack_deltas(dxh_dyh_dzh)
    lap = laplacian_3d(theta_next, dz, dy, dx)
    return (theta_next - theta_curr) - dtau.view(-1, 1, 1, 1, 1) * (lap + S)


def pde_residual_fv(theta_next, theta_curr, dtau, Ms, Mi, Bi_eff, dxh_dyh_dzh, S, theta_inf=None):
    d0 = theta_next.dtype
    dev = theta_next.device
    theta_next = theta_next.double(); theta_curr = theta_curr.double()
    Ms = Ms.double(); Mi = Mi.double(); S = S.double(); Bi_eff = Bi_eff.double()
    if torch.is_tensor(theta_inf):
        theta_inf = theta_inf.double()
    dz, dy, dx = _unpack_deltas(dxh_dyh_dzh)
    dz = torch.as_tensor(dz, dtype=torch.float64, device=dev)
    dy = torch.as_tensor(dy, dtype=torch.float64, device=dev)
    dx = torch.as_tensor(dx, dtype=torch.float64, device=dev)
    Div = fv_divergence(theta_next, Ms, Mi, Bi_eff, dz, dy, dx, mode='interior', theta_inf=theta_inf)
    res = (theta_next - theta_curr) - dtau.view(-1, 1, 1, 1, 1).double() * (Div + S)
    return res.to(d0)


def bc_robin_residual(theta_next, Bi, nsurf, dxh_dyh_dzh, Ms=None, Mi=None, theta_inf=None):
    d0, dev = theta_next.dtype, theta_next.device
    th = theta_next.to(torch.float64)
    Bi = Bi.to(torch.float64)
    ns = nsurf.to(torch.float64)
    Ms64 = Ms.to(torch.float64) if Ms is not None else torch.ones_like(th, dtype=torch.float64)
    dz, dy, dx = _unpack_deltas(dxh_dyh_dzh)
    dz = torch.as_tensor(dz, dtype=torch.float64, device=dev)
    dy = torch.as_tensor(dy, dtype=torch.float64, device=dev)
    dx = torch.as_tensor(dx, dtype=torch.float64, device=dev)
    gz, gy, gx = one_sided_grad_on_interface_v2(th, Ms64, ns, dz, dy, dx)
    nz, ny, nx = ns[:, 0:1], ns[:, 1:2], ns[:, 2:3]
    dth_dn = gz * nz + gy * ny + gx * nx
    if theta_inf is None:
        th_inf = torch.zeros((), dtype=torch.float64, device=dev)
    else:
        th_inf = theta_inf.to(torch.float64) if torch.is_tensor(theta_inf) \
            else torch.as_tensor(theta_inf, dtype=torch.float64, device=dev)
    res = -dth_dn - Bi * (th - th_inf)
    return res.to(d0)


# ------------------------- ckpt I/O -------------------------
def save_ckpt(path, epoch, model, hhead, opt, args_dict):
    ckpt = {"model": model.state_dict(), "args": args_dict, "epoch": int(epoch)}
    if hhead is not None: ckpt["hhead"] = hhead.state_dict()
    if opt is not None:   ckpt["optim"] = opt.state_dict()
    torch.save(ckpt, path)


def load_ckpt(path, model, hhead, opt=None, resume=False, strict=False, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
    if not strict and (missing or unexpected):
        print(f"[warn] model load (non-strict) missing={missing}, unexpected={unexpected}")
    start_epoch = 1
    if ("hhead" in ckpt) and (hhead is not None):
        mh, uh = hhead.load_state_dict(ckpt["hhead"], strict=strict)
        if not strict and (mh or uh):
            print(f"[warn] hhead load (non-strict) missing={mh}, unexpected={uh}")
    if resume and ("optim" in ckpt) and (opt is not None):
        opt.load_state_dict(ckpt["optim"])
        for g in opt.param_groups:
            g["lr"] = args.lr
        start_epoch = int(ckpt.get("epoch", 1))
    return start_epoch


# ------------------------- 训练辅助 -------------------------
def build_env_channels(cond, x):
    th = cond["Tinf"].to(x.device).float()
    if th.dim() == 4:
        th = th.unsqueeze(1)  # -> (1,1,Z,Y,X)
    elif th.dim() == 5:
        pass
    else:
        raise RuntimeError(f"unexpected Tinf shape: {list(th.shape)}")
    B = x.size(0)
    if th.size(0) == 1 and B > 1:
        th = th.repeat(B, 1, 1, 1, 1)
    return th


def _rms(x, mask):
    return torch.sqrt(((x ** 2) * mask).sum() / (mask.sum() + 1e-8))


def _op_norm_factor(dxh_dyh_dzh, dtau):
    dz, dy, dx = _unpack_deltas(dxh_dyh_dzh)
    lam = 2.0 * (1.0 / (dz * dz) + 1.0 / (dy * dy) + 1.0 / (dx * dx))
    lam = torch.as_tensor(lam, dtype=torch.float64, device=dtau.device)
    return lam * dtau.view(-1, 1, 1, 1, 1).double()


# ------------------------- 训练 -------------------------
def train(args):
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
    else:
        device = torch.device("cpu")
        print("[warn] CUDA 不可用，使用 CPU。")

    # 数据
    if args.data_glob and args.data_glob != "default":
        src = args.data_glob
    elif args.data and args.data != "default":
        src = args.data
    else:
        raise ValueError("请提供 --data 或 --data_glob")

    paths = discover_h5(src)
    assert len(paths) > 0, f"未找到任何 H5：{src}"
    print(f"[info] found {len(paths)} files. e.g.: {paths[0]}")

    preload_to = None
    if args.preload_gpu:
        if device.type != "cuda":
            print("[warn] --preload-gpu 被忽略：当前不是 CUDA 设备。")
        else:
            preload_to = device.type
            print("[preload] 全量预加载到 GPU（按 H5 文件驻留），将强制 num_workers=0, pin_memory=False")

    ds = MultiPairDataset(paths, max_skip=args.max_skip, preload_to=preload_to)
    if args.preload_gpu and (device.type == "cuda"):
        eff_workers = 0; eff_pin = False; eff_persist = False
    else:
        eff_workers = args.num_workers; eff_pin = args.pin_memory; eff_persist = (args.num_workers > 0)

    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=eff_workers, worker_init_fn=worker_init_fn,
        pin_memory=eff_pin, persistent_workers=eff_persist
    )

    # 模型
    model = FNO3D(
        in_c=3, width=args.width, modes=(args.mz, args.my, args.mx), layers=args.layers,
        context_dim=10,  # 9 + 1(log dtau)
        spec_pad_type=args.spec_pad_type,
        spec_pad=(args.spec_pad_z, args.spec_pad_y, args.spec_pad_x)
    ).to(device)
    hhead = None
    if args.learn_h:
        hhead = HHead(in_c=7, width=args.h_width, layers=args.h_layers, mask_idx=2,
                      h_min=0.0, h_max=30.0, h_prior=10.0, beta=2.0,
                      use_ctx_film=True, ctx_dim=10).to(device)

    params = list(model.parameters()) + (list(hhead.parameters()) if hhead is not None else [])
    opt = torch.optim.Adam(params, lr=args.lr)

    # 输出
    if args.out_dir:
        out_dir = args.out_dir
    else:
        if args.resume and args.ckpt:
            out_dir = os.path.dirname(os.path.abspath(args.ckpt))
        else:
            exp = args.exp or _time.strftime("exp_%Y%m%d_%H%M%S")
            out_dir = os.path.join("dat/runs", exp)
    os.makedirs(out_dir, exist_ok=True)

    # ckpt
    start_ep = 1
    if args.ckpt:
        start_ep = load_ckpt(args.ckpt, model, hhead, opt=opt if args.resume else None,
                             resume=args.resume, strict=args.strict_load, device=device)
        print(f"[info] loaded ckpt: {args.ckpt}  (start_epoch={start_ep})")

    print(f"[info] data source = {src}, out={out_dir}")
    print(f"[info] scaling(runtime): T_BASE={T_BASE:.1f}K, DT_BASE={DT_BASE:.1f}K; S uses BASE formula L^2/(alpha*rho*cp*DT_BASE)")
    if hhead is not None:
        tot = sum(p.numel() for p in hhead.parameters())
        print(f"[info] HHead params: {tot / 1e3:.1f} K")

    lam_sup = args.lam_sup; lam_pde = args.lam_pde; lam_bc = args.lam_bc
    use_fv = (args.pde_form.lower() == "fv")

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        if hhead is not None: hhead.train()

        m_sup = m_pde_rms = m_bc = m_hsup = 0.0
        m_energy = 0.0; m_Qpred = 0.0; m_Qref = 0.0
        m_step_mse = 0.0; m_step_rmse = 0.0
        m_sup_term = m_pde_term = m_bc_term = m_total_det = 0.0
        m_air_term = 0.0

        n_batches = 0

        for x, y, cond in dl:
            x = x.to(device, non_blocking=True)  # (B,3,...): [θ0, Ms, S]
            y = y.to(device, non_blocking=True)  # (B,1,...)
            Ms = cond["Ms"].to(device, non_blocking=True).float()
            Mi = cond["Mi"].to(device, non_blocking=True).float()
            dtau = cond["dtau"].to(device, non_blocking=True).float()
            S = x[:, 2:3]
            dxh_dyh_dzh = cond["dxh_dyh_dzh"].to(device, non_blocking=True)

            if dxh_dyh_dzh.dim() == 2 and dxh_dyh_dzh.shape[0] > 1:
                if not torch.allclose(dxh_dyh_dzh, dxh_dyh_dzh[0:1].repeat(dxh_dyh_dzh.shape[0], 1)):
                    raise ValueError("Batch 内存在不同的网格间距，请设置 --batch 1 或对数据分组。")

            L_over_k = cond["L_over_k"].to(device, non_blocking=True).view(-1, 1, 1, 1, 1)

            # ====== 直接用数据阶段构造好的 10维 ctx ======
            ctx_vec = cond["ctx"].to(device, non_blocking=True)
            if ctx_vec.dim() == 1:  # 兼容 batch=1 无维度聚合
                ctx_vec = ctx_vec.unsqueeze(0)

            nsurf = cond["nsurf"].to(device, non_blocking=True).float()
            if nsurf.dim() == 4:
                nsurf = nsurf.unsqueeze(0).repeat(x.size(0), 1, 1, 1, 1)

            theta_inf_gridK = build_env_channels(cond, x)  # (B,1,Z,Y,X) [K]
            theta_inf_hat = (theta_inf_gridK - T_BASE) / DT_BASE

            # 预测 Δθ -> y_pred
            if args.rk2:
                delta1 = model(x, ctx_vec)
                y_tilde = x[:, 0:1] + delta1
                x2 = torch.cat([y_tilde, x[:, 1:2], x[:, 2:3]], dim=1)
                delta2 = model(x2, ctx_vec)
                delta = 0.5 * (delta1 + delta2)
            else:
                delta = model(x, ctx_vec)
            y_pred = x[:, 0:1] + delta
            Bsz = x.size(0)

            # ====== 用数据/初值 h 构造 Bi_eff（不输网络）======
            h_truth_t0 = cond["h_truth_t0"].to(device, non_blocking=True).float().repeat(Bsz, 1, 1, 1, 1)
            has_h = cond["has_h_truth"].to(device, non_blocking=True).float().view(-1, 1, 1, 1, 1)
            h_data = h_truth_t0 * has_h + (1.0 - has_h) * cond["h_init"].to(device, non_blocking=True).float().repeat(
                Bsz, 1, 1, 1, 1)
            h_data = h_data * Mi
            Bi_eff = h_data * L_over_k  # Bi = h * L / k

            # 监督（固体）：Δθ
            target_delta = (y - x[:, 0:1])
            sup = ((delta - target_delta) ** 2 * Ms).sum() / (Ms.sum() + 1e-8)
            total = lam_sup * sup

            # 空气锚
            air_mse_term = torch.tensor(0.0, device=device)
            if args.air_anchor > 0:
                air = (1.0 - Ms)
                air_mse = ((y_pred - x[:, 0:1]) ** 2 * air).sum() / (air.sum() + 1e-8)
                air_mse_term = args.air_anchor * air_mse
                total += air_mse_term

            # PDE（FV 或 strong）
            if args.use_pde:
                if use_fv:
                    _ = pde_residual_fv(y, x[:, 0:1], dtau, Ms, Mi, Bi_eff, dxh_dyh_dzh, S, theta_inf=theta_inf_hat)
                    res_p = pde_residual_fv(y_pred, x[:, 0:1], dtau, Ms, Mi, Bi_eff, dxh_dyh_dzh, S,
                                            theta_inf=theta_inf_hat)
                else:
                    res_p = pde_residual_strong(y_pred, x[:, 0:1], dtau, Ms, dxh_dyh_dzh, S)
                mask_int = ((Ms > 0.5) & ~(Mi > 0.5)).float()
                pde = ((res_p ** 2) * mask_int).sum() / (mask_int.sum() + 1e-8)
                total = total + lam_pde * pde
                pde_rms = torch.sqrt(((res_p ** 2) * mask_int).sum() / (mask_int.sum() + 1e-8))
            else:
                pde = torch.tensor(0.0, device=device)
                pde_rms = torch.tensor(0.0, device=device)

            # BC 残差
            if args.use_bc:
                res_b = bc_robin_residual(y_pred, Bi_eff, nsurf, dxh_dyh_dzh, Ms=Ms, Mi=Mi, theta_inf=theta_inf_hat)
                bc = (res_b ** 2) * Mi
                bc = bc.sum() / (Mi.sum() + 1e-8)
                total = total + lam_bc * bc
            else:
                bc = torch.tensor(0.0, device=device)

            # ====== hhead：仅做自身监督（若有真值），不参与 Bi_eff ======
            h_sup_loss = torch.tensor(0.0, device=device)
            if (hhead is not None) and (args.lam_h_sup > 0.0):
                feats = torch.cat([
                    x[:, 0:1],  # θ0
                    x[:, 2:3],  # S
                    Mi,         # mask_idx=2
                    Ms,
                    nsurf[:, 0:1], nsurf[:, 1:2], nsurf[:, 2:3],
                ], dim=1)
                h_pred_field = hhead(feats, ctx_vec=ctx_vec)  # (B,1,Z,Y,X)
                denom = (Mi * has_h).sum() + 1e-8
                if denom > 0:
                    h_sup_loss = ((h_pred_field - h_truth_t0) ** 2 * Mi * has_h).sum() / denom

            # --- PDE/BC 规范化（保持你的实现）
            scale_p_ini = _op_norm_factor(dxh_dyh_dzh, dtau)
            res_p_n = (res_p.double() / (scale_p_ini + 1e-12)).float() if args.use_pde else torch.tensor(0.0, device=device)
            mask_int = ((Ms > 0.5) & ~(Mi > 0.5)).float()
            pde_n = ((res_p_n ** 2) * mask_int).sum() / (mask_int.sum() + 1e-8) if args.use_pde else torch.tensor(0.0, device=device)
            c_pde = 0.5
            w_pde = torch.as_tensor(c_pde, device=device)
            pde_term = lam_pde * w_pde * pde_n if args.use_pde else torch.tensor(0.0, device=device)

            Bi_mean = (Bi_eff * Mi).sum() / (Mi.sum() + 1e-8)
            bc_scale = (1.0 + Bi_mean.detach()).double()
            res_b_n = (res_b.double() / (bc_scale + 1e-12)).float() if args.use_bc else torch.tensor(0.0, device=device)
            bc_n = ((res_b_n ** 2) * Mi).sum() / (Mi.sum() + 1e-8) if args.use_bc else torch.tensor(0.0, device=device)
            c_bc = 0.5
            w_bc = torch.as_tensor(c_bc, device=device)
            bc_term = lam_bc * w_bc * bc_n if args.use_bc else torch.tensor(0.0, device=device)

            # === ENERGY PATH (C) ===
            energy_term = torch.tensor(0.0, device=device)
            if args.use_energy:
                A = cond["area"].to(device, non_blocking=True).float()  # (1,Z,Y,X)
                if A.dim() == 4:  # (1,Z,Y,X) -> (B,1,Z,Y,X)
                    A = A.unsqueeze(1).repeat(Bsz, 1, 1, 1, 1)
                T_pred = y_pred * DT_BASE + T_BASE
                T_true = y * DT_BASE + T_BASE
                Q_pred = (h_data.double() * A.double() * (T_pred.double() - theta_inf_gridK.double())).sum(dim=(1, 2, 3, 4))
                Q_ref  = (h_data.double() * A.double() * (T_true.double() - theta_inf_gridK.double())).sum(dim=(1, 2, 3, 4))
                energy_norm = Q_ref.abs().clamp_min(1e-6)
                energy_loss = ((Q_pred - Q_ref) / energy_norm) ** 2
                energy_term = args.lam_energy * energy_loss.mean()
                total = total + energy_term

            # 总损失
            total = lam_sup * sup + pde_term + bc_term + energy_term + air_mse_term
            opt.zero_grad(); total.backward(); opt.step()

            # ================== 统计 ==================
            m_sup += float(sup.detach().cpu())
            m_pde_rms += float(pde_rms.detach().cpu())
            m_bc += float(bc.detach().cpu())
            m_hsup += float(h_sup_loss.detach().cpu())
            if args.use_energy:
                m_energy += float(energy_term.detach().cpu())
                m_Qpred += float(Q_pred.mean().detach().cpu())
                m_Qref  += float(Q_ref.mean().detach().cpu())

            step_mse = ((y_pred - y) ** 2 * Ms).sum() / (Ms.sum() + 1e-8)
            step_rmse = torch.sqrt(step_mse)
            sup_term = (lam_sup * sup).detach()
            pde_det = (pde_term).detach()
            bc_det = (bc_term).detach()
            total_det = (lam_sup * sup + pde_term + bc_term + energy_term + air_mse_term).detach()

            m_step_mse += float(step_mse.detach().cpu())
            m_step_rmse += float(step_rmse.detach().cpu())
            m_sup_term += float(sup_term.cpu())
            m_pde_term += float(pde_det.cpu())
            m_bc_term += float(bc_det.cpu())
            m_total_det += float(total_det.cpu())
            m_air_term += float(air_mse_term.detach().cpu())

            n_batches += 1

        # ============ 打印 =============
        print(f"[ep {ep:03d}]")
        print(
            f"          loss_terms: (lam_sup*sup)={m_sup_term / n_batches:.4e}  pde_term={m_pde_term / n_batches:.4e}  bc_term={m_bc_term / n_batches:.4e}  "
            f"{'energy_term=' + format((m_energy / n_batches), '.4e') if args.use_energy else ''} air_terms={m_air_term / n_batches:.4e} total={m_total_det / n_batches:.4e}")
        print(
            f"          mean_error: avg_step_mse={m_step_mse / n_batches:.4e}  avg_step_rmse={m_step_rmse / n_batches:.4e}")

        if ep % args.ckpt_every == 0:
            save_ckpt(os.path.join(out_dir, f"ckpt_ep{ep}.pt"), ep, model, hhead, opt, vars(args))

    save_ckpt(os.path.join(out_dir, "ckpt_final.pt"), args.epochs, model, hhead, opt, vars(args))


# ------------------------- 参数 -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="default", help="单文件或目录（目录时只读该目录下 *.h5）")
    ap.add_argument("--data_glob", type=str, default="default", help="通配符模式（如 dat/train/*.h5）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--mz", type=int, default=12); ap.add_argument("--my", type=int, default=12); ap.add_argument("--mx", type=int, default=12)
    ap.add_argument("--lam_sup", type=float, default=1.0)
    ap.add_argument("--use_pde", action="store_true"); ap.add_argument("--lam_pde", type=float, default=1e-2)
    ap.add_argument("--use_bc", action="store_true");  ap.add_argument("--lam_bc", type=float, default=1e-2)
    ap.add_argument("--learn_h", action="store_true")
    ap.add_argument("--h_width", type=int, default=16)
    ap.add_argument("--h_layers", type=int, default=2)
    ap.add_argument("--lam_h_sup", type=float, default=1.0, help="真值 h 对小网络 h 的监督权重 (MSE)")
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--exp", type=str, default="")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--strict_load", action="store_true")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--pde_form", type=str, choices=["fv", "strong"], default="fv",
                    help="PDE 残差形式：有限体积(fv) 或 强式(strong)")
    ap.add_argument("--max_skip", type=int, default=1, help="随机跨度上限K，训练样本为(t, t+k), k∈[1,K]")
    ap.add_argument("--rk2", action="store_true", help="使用 Heun(RK2) 时间推进")
    ap.add_argument("--device", type=str, default="cuda", help="cuda / cuda:0 / cpu")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (Win 建议 0 或 1)")
    ap.add_argument("--pin-memory", action="store_true", help="DataLoader pin_memory (配合 CUDA 建议开)")
    ap.add_argument("--preload-gpu", action="store_true",
                    help="将训练所需数据一次性加载到 GPU 上；强制 num_workers=0、pin_memory=False")

    # === ENERGY PATH (C) ===
    ap.add_argument("--use_energy", action="store_true", help="启用路径 C：能量积分约束（基于 h*A*ΔT 的总功率匹配）")
    ap.add_argument("--lam_energy", type=float, default=1.0, help="能量约束的损失系数")

    # === 新增：频域卷积前的反射/复制 padding 选项 ===
    ap.add_argument("--spec-pad-type", type=str, choices=["none", "reflect", "replicate"],
                    default="reflect", help="频域卷积前的空间 padding 类型")
    ap.add_argument("--spec-pad-z", type=int, default=8, help="FFT 前 z 轴对称 padding 宽度（格点）")
    ap.add_argument("--spec-pad-y", type=int, default=8, help="FFT 前 y 轴对称 padding 宽度（格点）")
    ap.add_argument("--spec-pad-x", type=int, default=8, help="FFT 前 x 轴对称 padding 宽度（格点）")
    # 增加air anchor 锚定空气温度
    ap.add_argument("--air-anchor", type=float, default=0.0, help="空气域锚点权重：让空气温度保持初值或接近T_inf")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
