# -*- coding: utf-8 -*-
"""
v13 (FULLSEQ 多步监督 & 广播修复版)  + 任选两帧一步跨越训练

新增：
- --twoframe：从同一条样本序列任意抽取两帧 ts<te，计算跨区间 dtau，做“一步估计”训练；
  适合配合 RK2 / L-稳定残差（大步长一步跨越）。
- AnyTwoFrameDataset：支撑 --twoframe 模式的数据读取。

新增（本次修改）：
- --val_dir：指定一个包含验证集 .h5 文件的文件夹，每个 epoch 结束后在验证集上计算 MSE/RMSE；
- --dropout：FNO 内部的 3D dropout，用于防止过拟合；
- --weight-decay：Adam 的 weight_decay(L2 正则)，用于防止过拟合。

保留：
- --fullseq（整段展开 + TBPTT）
- 短滚动多步 (--multi-steps)
- uFNO 多尺度频域 + 门控、FV/Strong PDE 残差、Robin 边界、能量项、按形状&网格分桶、断点续训

从头（示例）：
python src/train_fno_pino_h_15_cuda.py \
  --data_glob "dat_test/*.h5" \
  --val_dir "dat_val" \
  --epochs 200 \
  --batch 1 --bucket-by-shape \
  --lr 2e-4 \
  --width 24 --layers 4 --mz 12 --my 24 --mx 4 \
  --use_pde --lam_pde 200.0 \
  --use_bc --lam_bc 0.0 \
  --lam_sup 0.2 \
  --use_energy --lam_energy 4.0 \
  --air-anchor 2e-1 \
  --spec-pad-type reflect --spec-pad-z 8 --spec-pad-y 8 --spec-pad-x 8 \
  --dropout 0.15 \
  --weight-decay 3e-4 \
  --out_dir "model_param/heatsink_fno_pino_h_8ls" \
  --ckpt "model_param/heatsink_fno_pino_h_8ls/ckpt_ep29.pt" \
  --resume \
  --device cuda:0 \
  --fullseq \
  --tbptt 2 \
  --detach-every 0 \
  --ckpt_every 1

从头4层
python src/train_fno_pino_h_15_cuda.py \
  --data_glob "dat_test/*.h5" \
  --val_dir "dat_val" \
  --epochs 200 \
  --batch 1 --bucket-by-shape \
  --lr 2e-4 \
  --width 24 --layers 4 --mz 12 --my 24 --mx 4 \
  --use_pde --lam_pde 200.0 \
  --use_bc --lam_bc 0.0 \
  --lam_sup 0.2 \
  --use_energy --lam_energy 4.0 \
  --air-anchor 2e-1 \
  --spec-pad-type reflect --spec-pad-z 8 --spec-pad-y 8 --spec-pad-x 8 \
  --dropout 0.05 \
  --weight-decay 1e-4 \
  --out_dir "model_param/heatsink_fno_pino_h_4ls" \
  --ckpt "model_param/heatsink_fno_pino_h_4ls/ckpt_ep34.pt" \
  --resume \
  --device cuda:0 \
  --fullseq \
  --tbptt 2 \
  --detach-every 0 \
  --ckpt_every 1
  
"""

import os, math, argparse, time as _time, glob
from pathlib import Path
import h5py, numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# =======================================================
# 常量
# =======================================================
T_BASE = 298.0  # K
DT_BASE = 30.0  # K

# 若你本机无 pydevd_pycharm，可忽略
try:
    import pydevd_pycharm
    pydevd_pycharm.settrace(host='host.docker.internal', port=5678,
                            stdout_to_server=True, stderr_to_server=True, suspend=False)
except Exception:
    pass


# =======================================================
# 工具 & 数值算子
# =======================================================
def laplacian_3d(u, dz, dy, dx):
    pad = (1, 1, 1, 1, 1, 1)
    up = F.pad(u, pad, mode='replicate')
    c = up[:, :, 1:-1, 1:-1, 1:-1]
    lap = (up[:, :, 2:, 1:-1, 1:-1] - 2 * c + up[:, :, :-2, 1:-1, 1:-1]) / (dz * dz) \
          + (up[:, :, 1:-1, 2:, 1:-1] - 2 * c + up[:, :, 1:-1, :-2, 1:-1]) / (dy * dy) \
          + (up[:, :, 1:-1, 1:-1, 2:] - 2 * c + up[:, :, 1:-1, 1:-1, :-2]) / (dx * dx)
    return lap


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

    uzm1, uzp1 = neigh_u_z(u); uym1, uyp1 = neigh_u_y(u); uxm1, uxp1 = neigh_u_x(u)
    Mzm1, Mzp1 = neigh_Ms_z(Ms); Mym1, Myp1 = neigh_Ms_y(Ms); Mxm1, Mxp1 = neigh_Ms_x(Ms)

    nz = nsurf[:, 0:1]; ny = nsurf[:, 1:2]; nx = nsurf[:, 2:3]
    Ms_solid = (Ms > 0.5).float()

    gz_bwd = (u - uzm1) / dz; gz_fwd = (uzp1 - u) / dz
    have_minus_z = (Mzm1 > 0.5); have_plus_z = (Mzp1 > 0.5)
    use_bwd_z = have_minus_z & ~have_plus_z; use_fwd_z = have_plus_z & ~have_minus_z
    both_z = have_minus_z & have_plus_z
    use_bwd_z = use_bwd_z | (both_z & (nz >= 0)); use_fwd_z = use_fwd_z | (both_z & (nz < 0))
    gz = torch.zeros_like(u); gz = torch.where(use_bwd_z, gz_bwd, gz); gz = torch.where(use_fwd_z, gz_fwd, gz)
    gz = gz * Ms_solid

    gy_bwd = (u - uym1) / dy; gy_fwd = (uyp1 - u) / dy
    have_minus_y = (Mym1 > 0.5); have_plus_y = (Myp1 > 0.5)
    use_bwd_y = have_minus_y & ~have_plus_y; use_fwd_y = have_plus_y & ~have_minus_y
    both_y = have_minus_y & have_plus_y
    use_bwd_y = use_bwd_y | (both_y & (ny >= 0)); use_fwd_y = use_fwd_y | (both_y & (ny < 0))
    gy = torch.zeros_like(u); gy = torch.where(use_bwd_y, gy_bwd, gy); gy = torch.where(use_fwd_y, gy_fwd, gy)
    gy = gy * Ms_solid

    gx_bwd = (u - uxm1) / dx; gx_fwd = (uxp1 - u) / dx
    have_minus_x = (Mxm1 > 0.5); have_plus_x = (Mxp1 > 0.5)
    use_bwd_x = have_minus_x & ~have_plus_x; use_fwd_x = have_plus_x & ~have_minus_x
    both_x = have_minus_x & have_plus_x
    use_bwd_x = use_bwd_x | (both_x & (nx >= 0)); use_fwd_x = use_fwd_x | (both_x & (nx < 0))
    gx = torch.zeros_like(u); gx = torch.where(use_bwd_x, gx_bwd, gx); gx = torch.where(use_fwd_x, gx_fwd, gx)
    gx = gx * Ms_solid
    return gz, gy, gx


def _unpack_deltas(d):
    if isinstance(d, (tuple, list)):
        return float(d[0]), float(d[1]), float(d[2])
    if torch.is_tensor(d):
        v = d.detach().view(-1)
        return float(v[0].item()), float(v[1].item()), float(v[2].item())
    raise TypeError(f"Unsupported type for deltas: {type(d)}")


def _area_weight_from_nsurf(nsurf_xyz, dx, dy, dz, Mi):
    nz = np.abs(nsurf_xyz[0]).astype(np.float32)
    ny = np.abs(nsurf_xyz[1]).astype(np.float32)
    nx = np.abs(nsurf_xyz[2]).astype(np.float32)
    A = nx * (dy * dz) + ny * (dx * dz) + nz * (dx * dy)
    return A * (Mi.astype(np.float32))


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


# =======================================================
# 数据读取
# =======================================================
class HeatsinkH5:
    def __init__(self, path, preload_to_device: str = None):
        self.f = h5py.File(path, 'r')
        self.T = self.f['T_grid_shadow'][:]  # (Nt,Nz,Ny,Nx)
        self.dt = self.f['time_dt'][:]       # (Nt-1,)
        self.Nt = self.T.shape[0]

        gp = self.f['grid_padded']
        spacing = gp['spacing'][:]  # [dx,dy,dz]
        dims = gp['dims'][:]        # [Nz,Ny,Nx]
        self.dx, self.dy, self.dz = float(spacing[0]), float(spacing[1]), float(spacing[2])
        self.Nz, self.Ny, self.Nx = int(dims[0]), int(dims[1]), int(dims[2])

        self.Lz_abs = self.dz * self.Nz
        self.Ly_abs = self.dy * self.Ny
        self.Lx_abs = self.dx * self.Nx

        self.Ms = self.f['mask_solid'][:].astype(np.float32)
        self.Mi = self.f['mask_interface'][:].astype(np.float32)
        bc = self.f['bc/robin']
        self.h_init = bc['h_init'][:].astype(np.float32)
        self.T_inf = bc['T_inf'][:].astype(np.float32)
        self.nsurf = self.f['normal_on_surface'][:].astype(np.float32)

        self.area_weight = _area_weight_from_nsurf(self.nsurf, self.dx, self.dy, self.dz, self.Mi)

        if 'sources' in self.f:
            self.q_vol = self.f['sources/q_vol'][:].astype(np.float32)
            self.q_mask = self.f['sources/q_mask'][:].astype(np.float32) if 'q_mask' in self.f['sources'] else np.ones_like(self.q_vol, np.float32)
        else:
            self.q_vol = np.zeros((self.Nz, self.Ny, self.Nx), np.float32)
            self.q_mask = np.ones_like(self.q_vol, np.float32)

        cst = self.f['const']
        self.k = float(cst['k_solid'][()])
        self.T_amb = float(cst['T_amb'][()])
        self.rho = float(cst['rho_solid'][()]) if 'rho_solid' in cst else None
        self.cp  = float(cst['cp_solid'][()])  if 'cp_solid'  in cst else None

        if 'g' in cst:
            g_raw = np.array(cst['g'][:], dtype=np.float64).reshape(-1)
            self.g_vec = g_raw[:3] if g_raw.size >= 3 else np.array([0.0, -9.81, 0.0], dtype=np.float64)
        else:
            self.g_vec = np.array([0.0, -9.81, 0.0], dtype=np.float64)
        self.g_mag = float(np.linalg.norm(self.g_vec))
        self.g_hat = self.g_vec / max(self.g_mag, 1e-12)

        sc = self.f['scales']
        file_alpha = float(sc['alpha'][()]) if 'alpha' in sc else None
        self.L = float(sc['L'][()])
        self.dTref = float(sc['dT_ref'][()]) if 'dT_ref' in sc else DT_BASE
        self.qref = float(sc['q_ref'][()]) if 'q_ref' in sc else None

        if file_alpha is not None:
            self.alpha = file_alpha
        else:
            if (self.rho is not None) and (self.cp is not None):
                self.alpha = self.k / (self.rho * self.cp)
            else:
                raise ValueError("[data] alpha 缺失且无法由 k/(rho*cp) 推出。")

        self.dx_hat = self.dx / self.L
        self.dy_hat = self.dy / self.L
        self.dz_hat = self.dz / self.L

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

        self.L_over_k = self.L / self.k

        if (self.rho is None) or (self.cp is None):
            raise ValueError("[data] 计算源项 S 需要 rho 和 cp")
        coeff = (self.L ** 2) / (self.alpha * self.rho * self.cp * DT_BASE)
        self.S_nd = (self.q_vol * coeff).astype(np.float32)

        self.h_truth = None
        if 'h_uniform_from_sources_all' in self.f:
            g = self.f['h_uniform_from_sources_all']
            if 'h_uniform_field_Wm2K' in g:
                self.h_truth = g['h_uniform_field_Wm2K'][:].astype(np.float32)
        if (self.h_truth is None) and ('h_from_grad_truth_uniform_all' in self.f):
            g2 = self.f['h_from_grad_truth_uniform_all']
            if 'h_uniform_field_Wm2K' in g2:
                self.h_truth = g2['h_uniform_field_Wm2K'][:].astype(np.float32)

        self.on_device = None
        if preload_to_device is not None:
            dev = torch.device(preload_to_device)
            self.T_t = torch.from_numpy(self.T).to(device=dev, dtype=torch.float32)
            self.dt_t = torch.from_numpy(self.dt.astype(np.float32)).to(device=dev)
            self.Ms_t = torch.from_numpy(self.Ms).to(device=dev)
            self.Mi_t = torch.from_numpy(self.Mi).to(device=dev)
            self.Tinf_t = torch.from_numpy(self.T_inf).to(device=dev)
            self.hini_t = torch.from_numpy(self.h_init).to(device=dev)
            self.nsurf_t = torch.from_numpy(self.nsurf.astype(np.float32)).to(device=dev)
            self.Snd_t = torch.from_numpy(self.S_nd).to(device=dev)
            self.area_t = torch.from_numpy(self.area_weight).to(device=dev)
            self.qvol_t = torch.from_numpy(self.q_vol).to(device=dev)
            self.qmsk_t = torch.from_numpy(self.q_mask).to(device=dev)
            self.htruth_t = torch.from_numpy(self.h_truth).to(device=dev) if self.h_truth is not None else None

            self.ctx_glb_t = torch.tensor(self.ctx_glb, dtype=torch.float32, device=dev)
            self.dxh_dyh_dzh_t = torch.tensor([self.dz_hat, self.dy_hat, self.dx_hat],
                                              dtype=torch.float32, device=dev)
            self.L_over_k_t = torch.tensor([self.L / self.k], dtype=torch.float32, device=dev)

            # 释放 CPU numpy（可选）
            self.T = self.dt = self.Ms = self.Mi = self.T_inf = self.h_init = self.nsurf = self.S_nd = None
            self.h_truth = None
            self.ctx_glb = None
            self.area_weight = None
            self.on_device = dev.type

    def close(self):
        try:
            self.f.close()
        except:
            pass


class MultiPairDataset(Dataset):
    """短滚动数据集：每次采样 (t0, t1=t0+k), k∈[1,max_skip]。"""

    def __init__(self, paths, max_skip=1, preload_to: str = None):
        self.paths = [str(p) for p in paths]
        self.max_skip = int(max(1, max_skip))
        self.preload_to = preload_to
        self.index = []
        self.meta = []
        for i, p in enumerate(self.paths):
            with h5py.File(p, 'r') as f:
                Nt, Nz, Ny, Nx = f['T_grid_shadow'].shape
                spacing = f['grid_padded/spacing'][:]  # [dx, dy, dz]
                L = float(f['scales/L'][()])
            dx_hat, dy_hat, dz_hat = float(spacing[0] / L), float(spacing[1] / L), float(spacing[2] / L)
            Q = 1e-6
            def q(v): return int(round(v / Q))
            self.meta.append((Nt, Nz, Ny, Nx, q(dz_hat), q(dy_hat), q(dx_hat)))
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
            dtau_ctx = torch.log(dtau.view(1) + 1e-12).clamp_(-20.0, 20.0)
            ctx = torch.cat([H.ctx_glb_t, dtau_ctx], dim=0)

            x = torch.stack([theta0, H.Ms_t, H.Snd_t], dim=0)
            y = theta1[None, ...]

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
                "ctx_glb": H.ctx_glb_t,
                "ctx": ctx,
                "h_truth_t0": h_t0,
                "has_h_truth": has_h,
                "area": H.area_t[None, ...],
                "shape": torch.tensor([H.Nz, H.Ny, H.Nx], dtype=torch.int32)
            }
            return x, y, cond

        Nt = H.T.shape[0]
        k_max = min(self.max_skip, Nt - 1 - t0)
        k = np.random.randint(1, k_max + 1)
        t1 = t0 + k

        T0 = torch.from_numpy(H.T[t0]); T1 = torch.from_numpy(H.T[t1])
        theta0 = (T0 - T_BASE) / DT_BASE
        theta1 = (T1 - T_BASE) / DT_BASE

        dt = float(H.dt[t0:t1].sum()); dtau = dt * H.alpha / (H.L * H.L)

        ctx_glb_cpu = torch.from_numpy(H.ctx_glb).float()
        dtau_ctx = torch.log(torch.tensor([dtau], dtype=torch.float32) + 1e-12).clamp_(-20.0, 20.0)
        ctx = torch.cat([ctx_glb_cpu, dtau_ctx], dim=0)

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
            "ctx_glb": ctx_glb_cpu,
            "ctx": ctx,
            "h_truth_t0": h_t0,
            "has_h_truth": has_h,
            "area": torch.from_numpy(H.area_weight[None, ...]).float(),
            "shape": torch.tensor([H.Nz, H.Ny, H.Nx], dtype=torch.int32)
        }
        return x, y, cond


class AnyTwoFrameDataset(Dataset):
    """任选两帧：每次从同一条样本随机抽取 ts<te，做“一步跨越”训练。"""

    def __init__(self, paths, preload_to: str = None):
        self.paths = [str(p) for p in paths]
        self.preload_to = preload_to
        self.index = []  # 仅用于加权各样本出现频次
        self.meta = []
        for i, p in enumerate(self.paths):
            with h5py.File(p, 'r') as f:
                Nt, Nz, Ny, Nx = f['T_grid_shadow'].shape
                spacing = f['grid_padded/spacing'][:]  # [dx, dy, dz]
                L = float(f['scales/L'][()])
            dx_hat, dy_hat, dz_hat = float(spacing[0] / L), float(spacing[1] / L), float(spacing[2] / L)
            Q = 1e-6
            def q(v): return int(round(v / Q))
            self.meta.append((Nt, Nz, Ny, Nx, q(dz_hat), q(dy_hat), q(dx_hat)))
            # 为使长序列样本更常被采到，按 (Nt-1) 次写入索引
            self.index.extend([i for _ in range(max(1, Nt - 1))])
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
        fi = self.index[idx]
        H = self._get_H(fi)

        if getattr(H, "on_device", None) == "cuda":
            dev = torch.device("cuda")
            Nt = H.T_t.shape[0]
            # 随机 ts<te
            ts = np.random.randint(0, Nt - 1)
            te = np.random.randint(ts + 1, Nt)

            theta_s = (H.T_t[ts] - T_BASE) / DT_BASE
            theta_e = (H.T_t[te] - T_BASE) / DT_BASE

            dtau = H.dt_t[ts:te].sum() * (H.alpha / (H.L * H.L))
            dtau_ctx = torch.log(dtau.view(1) + 1e-12).clamp_(-20.0, 20.0)
            ctx = torch.cat([H.ctx_glb_t, dtau_ctx], dim=0)

            x = torch.stack([theta_s, H.Ms_t, H.Snd_t], dim=0)
            y = theta_e[None, ...]

            if H.htruth_t is not None:
                h_ts = H.htruth_t[ts][None, ...]
                has_h = torch.tensor(1, dtype=torch.uint8, device=dev)
            else:
                h_ts = torch.zeros((1, H.Nz, H.Ny, H.Nx), dtype=torch.float32, device=dev)
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
                "ctx_glb": H.ctx_glb_t,
                "ctx": ctx,
                "h_truth_t0": h_ts,
                "has_h_truth": has_h,
                "area": H.area_t[None, ...],
                "shape": torch.tensor([H.Nz, H.Ny, H.Nx], dtype=torch.int32)
            }
            return x, y, cond

        # CPU 分支
        Nt = H.T.shape[0]
        ts = np.random.randint(0, Nt - 1)
        te = np.random.randint(ts + 1, Nt)

        theta_s = (torch.from_numpy(H.T[ts]) - T_BASE) / DT_BASE
        theta_e = (torch.from_numpy(H.T[te]) - T_BASE) / DT_BASE

        dt = float(H.dt[ts:te].sum()); dtau = dt * H.alpha / (H.L * H.L)

        ctx_glb_cpu = torch.from_numpy(H.ctx_glb).float()
        dtau_ctx = torch.log(torch.tensor([dtau], dtype=torch.float32) + 1e-12).clamp_(-20.0, 20.0)
        ctx = torch.cat([ctx_glb_cpu, dtau_ctx], dim=0)

        x = torch.stack([theta_s, torch.from_numpy(H.Ms), torch.from_numpy(H.S_nd)], dim=0)
        y = theta_e[None, ...]

        if H.h_truth is not None:
            h_ts = torch.from_numpy(H.h_truth[ts])[None, ...]
            has_h = torch.tensor(1, dtype=torch.uint8)
        else:
            h_ts = torch.zeros((1, H.Nz, H.Ny, H.Nx), dtype=torch.float32)
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
            "ctx_glb": ctx_glb_cpu,
            "ctx": ctx,
            "h_truth_t0": h_ts,
            "has_h_truth": has_h,
            "area": torch.from_numpy(H.area_weight[None, ...]).float(),
            "shape": torch.tensor([H.Nz, H.Ny, H.Nx], dtype=torch.int32)
        }
        return x, y, cond


class FullSeqDataset(Dataset):
    """FULLSEQ：每个样本=整段时间序列 (Nt,Z,Y,X) + 空间/全局条件。"""
    def __init__(self, paths, preload_to: str = None):
        self.paths = [str(p) for p in paths]
        self.preload_to = preload_to
        self.meta = []
        for i, p in enumerate(self.paths):
            with h5py.File(p, 'r') as f:
                Nt, Nz, Ny, Nx = f['T_grid_shadow'].shape
                spacing = f['grid_padded/spacing'][:]  # [dx, dy, dz]
                L = float(f['scales/L'][()])
            dx_hat, dy_hat, dz_hat = float(spacing[0] / L), float(spacing[1] / L), float(spacing[2] / L)
            Q = 1e-6
            def q(v): return int(round(v / Q))
            self.meta.append((Nt, Nz, Ny, Nx, q(dz_hat), q(dy_hat), q(dx_hat)))
        self._cache = {}

    def __len__(self):
        return len(self.paths)

    def _get_H(self, i):
        H = self._cache.get(i, None)
        if H is None:
            H = HeatsinkH5(self.paths[i], preload_to_device=self.preload_to)
            self._cache[i] = H
        return H

    def worker_init(self, worker_id):
        self._cache = {}

    def __getitem__(self, idx):
        H = self._get_H(idx)
        if getattr(H, "on_device", None) == "cuda":
            T = H.T_t
            theta_seq = (T - T_BASE) / DT_BASE
            dtau_seq = H.dt_t * (H.alpha / (H.L * H.L))
            cond = {
                "Tinf": H.Tinf_t,
                "h_init": H.hini_t,
                "Mi": H.Mi_t,
                "Ms": H.Ms_t,
                "nsurf": H.nsurf_t,
                "dxh_dyh_dzh": H.dxh_dyh_dzh_t,
                "L_over_k": H.L_over_k_t,
                "ctx_glb": H.ctx_glb_t,
                "S": H.Snd_t,
                "area": H.area_t,
                "dtau_seq": dtau_seq,
                "shape": torch.tensor([H.Nz, H.Ny, H.Nx], dtype=torch.int32, device=theta_seq.device),
            }
            return theta_seq, cond

        T = torch.from_numpy(H.T)
        theta_seq = (T - T_BASE) / DT_BASE
        dtau_seq = torch.from_numpy(H.dt.astype(np.float32)) * (H.alpha / (H.L * H.L))
        cond = {
            "Tinf": torch.from_numpy(H.T_inf).float(),
            "h_init": torch.from_numpy(H.h_init).float(),
            "Mi": torch.from_numpy(H.Mi).float(),
            "Ms": torch.from_numpy(H.Ms).float(),
            "nsurf": torch.from_numpy(H.nsurf).float(),
            "dxh_dyh_dzh": torch.tensor([H.dz_hat, H.dy_hat, H.dx_hat], dtype=torch.float32),
            "L_over_k": torch.tensor([H.L / H.k], dtype=torch.float32),
            "ctx_glb": torch.from_numpy(H.ctx_glb).float(),
            "S": torch.from_numpy(H.S_nd).float(),
            "area": torch.from_numpy(H.area_weight).float(),
            "dtau_seq": dtau_seq.float(),
            "shape": torch.tensor([H.Nz, H.Ny, H.Nx], dtype=torch.int32),
        }
        return theta_seq, cond


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


# =======================================================
# 频域层 + uFNO
# =======================================================
class SpectralConv3d(nn.Module):
    def __init__(self, in_c, out_c, modes_z, modes_y, modes_x,
                 pad_type: str = 'reflect', pad_z: int = 8, pad_y: int = 8, pad_x: int = 8):
        super().__init__()
        self.in_c = in_c; self.out_c = out_c
        self.mz, self.my, self.mx = int(modes_z), int(modes_y), int(modes_x)
        self.pad_type = pad_type
        self.pad_z = int(pad_z); self.pad_y = int(pad_y); self.pad_x = int(pad_x)
        scale = 1 / max(1, in_c * out_c)
        self.weight = nn.Parameter(
            scale * torch.randn(in_c, out_c, max(1, self.mz), max(1, self.my), max(1, self.mx), 2))
        self.gate = nn.Parameter(torch.zeros(1, out_c, 1, 1, 1))

    @staticmethod
    def compl_mul3d(a, b):
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
        if mz > 0 and my > 0 and mx > 0:
            w = self.weight[:, :, :mz, :my, :mx, :]
            out_ft[:, :, :mz, :my, :mx, :] = self.compl_mul3d(x_ft[:, :, :mz, :my, :mx, :], w)

        y_pad = torch.fft.irfftn(torch.view_as_complex(out_ft), s=(Zp, Yp, Xp), dim=(-3, -2, -1))
        if (pz | py | px) > 0:
            y = y_pad[:, :, pz:Zp - pz, py:Yp - py, px:Xp - px]
        else:
            y = y_pad
        scale = 1.0 + torch.tanh(self.gate)
        y = y * scale
        return y


class MultiScaleSpectralBlock(nn.Module):
    def __init__(self, width, modes, scale_count=3,
                 pad_type='reflect', pad=(8, 8, 8)):
        super().__init__()
        mz, my, mx = modes
        pz, py, px = pad
        self.scale_count = int(max(1, scale_count))
        self.specs = nn.ModuleList()
        for s in range(self.scale_count):
            mz_s = max(1, mz // (2 ** s))
            my_s = max(1, my // (2 ** s))
            mx_s = max(1, mx // (2 ** s))
            self.specs.append(SpectralConv3d(width, width, mz_s, my_s, mx_s,
                                             pad_type=pad_type, pad_z=pz, pad_y=py, pad_x=px))
        self.scale_logits = nn.Parameter(torch.zeros(self.scale_count))

    def forward(self, h):
        if self.scale_count == 1:
            return self.specs[0](h)
        ws = torch.softmax(self.scale_logits, dim=0)
        y = 0.0
        for s, sc in enumerate(self.specs):
            y = y + ws[s] * sc(h)
        return y


class FNO3D(nn.Module):
    def __init__(self, in_c=3, width=24, modes=(12, 12, 12), layers=4,
                 add_coords=True, fourier_k=8, use_local=True,
                 gn_groups=1, residual_scale=0.5, dropout=0.0,
                 context_dim: int = 0,
                 spec_pad_type: str = 'reflect',
                 spec_pad=(8, 8, 8),
                 ufno_scales: int = 3):
        super().__init__()
        self.add_coords = add_coords
        self.fourier_k = fourier_k
        self.use_local = use_local
        self.context_dim = int(context_dim)
        self.spec_pad_type = spec_pad_type
        self.spec_pad = tuple(int(v) for v in spec_pad)
        self.ufno_scales = int(max(1, ufno_scales))

        extra_c = 0
        if add_coords:
            extra_c = 3 + 6 * fourier_k
        self.lift = nn.Conv3d(in_c + extra_c, width, 1)

        mz, my, mx = modes
        pz, py, px = self.spec_pad

        self.specs = nn.ModuleList([
            MultiScaleSpectralBlock(width, (mz, my, mx),
                                    scale_count=self.ufno_scales,
                                    pad_type=self.spec_pad_type, pad=(pz, py, px))
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
        self.layer_films = nn.ModuleList()
        if self.context_dim > 0:
            for _ in range(layers):
                lin = nn.Linear(self.context_dim, 2 * width)
                nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
                self.layer_films.append(lin)

    @staticmethod
    def _coords(Z, Y, X, device, dtype):
        Z = int(Z); Y = int(Y); X = int(X)
        z = torch.linspace(-1.0, 1.0, Z, device=device, dtype=dtype)
        y = torch.linspace(-1.0, 1.0, Y, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, X, device=device, dtype=dtype)
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
            h = h + self.ctx_mlp(ctx).view(B, -1, 1, 1, 1)

        for i, (spec_blk, w, gn) in enumerate(zip(self.specs, self.ws, self.norms)):
            y = spec_blk(h) + w(h)
            if self.use_local:
                y = y + self.locals[i](h)
            y = F.gelu(y)
            if self.drop is not None:
                y = self.drop(y)

            if (ctx is not None) and (len(self.layer_films) > 0):
                ab = self.layer_films[i](ctx)
                gamma, beta = torch.chunk(ab, 2, dim=-1)
                y = y * (1.0 + gamma.view(B, -1, 1, 1, 1)) + beta.view(B, -1, 1, 1, 1)

            h = h + self.gammas[i] * y
            h = gn(h)
        return self.proj(h)


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
            p = float(np.clip(p, 1e-6, 1 - p))
            b0 = math.log(p / (1 - p))
            nn.init.constant_(self.mlp[-1].bias, b0)
            nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, feats, ctx_vec=None):
        B, _, Z, Y, X = feats.shape
        z = self.net(feats)
        mask = feats[:, self.mask_idx:self.mask_idx + 1].float()
        denom = mask.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        g = (z * mask).sum(dim=(2, 3, 4), keepdim=True) / denom
        g = g.flatten(1)
        raw = self.mlp(g)
        if self.use_ctx_film and (ctx_vec is not None):
            ab = self.ctx2affine(ctx_vec)
            a = ab[:, :1]; b = ab[:, 1:2]
            raw = (1.0 + a) * raw + b
        h_norm = torch.sigmoid(self.beta * raw)
        h_scalar = self.h_min + (self.h_max - self.h_min) * h_norm
        h_scalar = h_scalar.view(B, 1, 1, 1, 1)
        return h_scalar * mask


# =======================================================
# PDE / BC / 能量
# =======================================================
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
    d0 = theta_next.dtype; dev = theta_next.device
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
    th = theta_next.to(torch.float64); Bi = Bi.to(torch.float64)
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


# =======================================================
# ckpt I/O（含旧->多尺度 第0分支映射）
# =======================================================
def save_ckpt(path, epoch, model, hhead, opt, args_dict):
    ckpt = {"model": model.state_dict(), "args": args_dict, "epoch": int(epoch)}
    if hhead is not None: ckpt["hhead"] = hhead.state_dict()
    if opt is not None:   ckpt["optim"] = opt.state_dict()
    torch.save(ckpt, path)


def load_ckpt(path, model, hhead, opt=None, resume=False, strict=False, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    ckpt_state = ckpt["model"]
    model_state = model.state_dict()

    remapped = {}
    for k, v in ckpt_state.items():
        if k.startswith("specs.") and (".weight" in k or ".gate" in k) and (".specs." not in k):
            parts = k.split(".")
            if len(parts) == 3 and parts[0] == "specs":
                L = parts[1]; tail = parts[2]
                new_k = f"specs.{L}.specs.0.{tail}"
                remapped[new_k] = v
        else:
            remapped[k] = v

    filtered = {}
    skipped = []
    for k, v in remapped.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    if skipped:
        print(f"[ckpt] skip {len(skipped)} keys due to name/shape mismatch (modes/ufno_scales 变更时属正常).")

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing or unexpected:
        print(f"[warn] model load (non-strict) missing={missing}, unexpected={unexpected}")

    start_epoch = 1
    if ("hhead" in ckpt) and (hhead is not None):
        try:
            mh, uh = hhead.load_state_dict(ckpt["hhead"], strict=False)
            if mh or uh:
                print(f"[warn] hhead load (non-strict) missing={mh}, unexpected={uh}")
        except RuntimeError as e:
            print(f"[warn] hhead load skipped due to shape mismatch: {e}")

    if resume and ("optim" in ckpt) and (opt is not None):
        try:
            opt.load_state_dict(ckpt["optim"])
            start_epoch = int(ckpt.get("epoch", 1))
        except Exception as e:
            print(f"[warn] optimizer state not loaded: {e}")

    return start_epoch


# =======================================================
# Bucketing
# =======================================================
class ShapeBucketBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size: int, drop_last: bool = False,
                 shuffle: bool = True, seed: int = 42):
        self.ds = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        self.shape2idxs = {}
        if hasattr(self.ds, "index"):
            iter_items = list(enumerate(self.ds.index))
            for ds_idx, item in iter_items:
                if isinstance(item, tuple):
                    fi = item[0]
                else:
                    fi = item
                meta = self.ds.meta[fi]
                key = meta[1:] if len(meta) >= 7 else meta[1:]
                self.shape2idxs.setdefault(tuple(key), []).append(ds_idx)
        else:
            for ds_idx, meta in enumerate(self.ds.meta):
                key = meta[1:] if len(meta) >= 7 else meta[1:]
                self.shape2idxs.setdefault(tuple(key), []).append(ds_idx)

        stats = [(k, len(v)) for k, v in self.shape2idxs.items()]
        stats.sort(key=lambda kv: -kv[1])
        print(f"[bucket] total buckets: {len(stats)}")
        for i, (shape, cnt) in enumerate(stats[:10]):
            print(f"  bucket#{i:02d} shape={shape} count={cnt}")
        if len(stats) > 10:
            rest = sum(cnt for _, cnt in stats[10:])
            print(f"  ... {len(stats) - 10} more buckets (total samples in rest = {rest})")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        per_bucket_batches = []
        for shape, idxs in self.shape2idxs.items():
            idxs = idxs.copy()
            if self.shuffle:
                idxs = torch.tensor(idxs)[torch.randperm(len(idxs), generator=g)].tolist()
            n_full = len(idxs) // self.batch_size
            end = n_full * self.batch_size
            full_batches = [idxs[i:i + self.batch_size] for i in range(0, end, self.batch_size)]
            if (not self.drop_last) and (end < len(idxs)):
                full_batches.append(idxs[end:])
            per_bucket_batches.append(full_batches)

        order = list(range(len(per_bucket_batches)))
        if self.shuffle:
            order = torch.tensor(order)[torch.randperm(len(order), generator=g)].tolist()

        cursors = [0 for _ in per_bucket_batches]
        remaining = True
        while remaining:
            remaining = False
            for bi in order:
                b_list = per_bucket_batches[bi]
                ci = cursors[bi]
                if ci < len(b_list):
                    yield b_list[ci]
                    cursors[bi] += 1
                    remaining = True

    def __len__(self):
        total = 0
        for idxs in self.shape2idxs.values():
            n_full = len(idxs) // self.batch_size
            if self.drop_last:
                total += n_full
            else:
                total += n_full + (1 if (len(idxs) % self.batch_size) > 0 else 0)
        return total


# =======================================================
# 训练相关
# =======================================================
def build_env_channels(cond, B, device):
    Tinf = cond["Tinf"].to(device).float()
    if Tinf.dim() == 3:
        Tinf = Tinf.unsqueeze(0).unsqueeze(0)
    elif Tinf.dim() == 4:
        Tinf = Tinf.unsqueeze(0)
    elif Tinf.dim() == 5:
        pass
    else:
        raise RuntimeError(f"unexpected Tinf shape: {list(Tinf.shape)}")
    if Tinf.size(0) == 1 and B > 1:
        Tinf = Tinf.expand(B, -1, -1, -1, -1)
    return Tinf


def _op_norm_factor(dxh_dyh_dzh, dtau):
    dz, dy, dx = _unpack_deltas(dxh_dyh_dzh)
    lam = 2.0 * (1.0 / (dz * dz) + 1.0 / (dy * dy) + 1.0 / (dx * dx))
    lam = torch.as_tensor(lam, dtype=torch.float64, device=dtau.device)
    return lam * dtau.view(-1, 1, 1, 1, 1).double()


# =======================================================
# 验证集评估函数：只看监督误差 (MSE / RMSE)
# =======================================================
def evaluate(model, args, device, val_dl):
    """
    在给定的 DataLoader 上做一次前向推理，只计算监督误差 (MSE / RMSE)，
    不回传梯度，也不算 PDE / BC / 能量项。
    """
    if val_dl is None:
        return None, None

    was_training = model.training
    model.eval()

    total_mse = 0.0
    total_rmse = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_dl:
            if args.fullseq:
                # ---------- FULLSEQ 验证：整段展开，从 t0 推到 t_end ----------
                theta_seq, cond = batch
                theta_seq = theta_seq.to(device, non_blocking=True)   # (B,Nt,Z,Y,X)
                B, Nt, Z, Y, X = theta_seq.shape

                Ms_sp = cond["Ms"].to(device, non_blocking=True).float()
                S_sp  = cond["S"].to(device, non_blocking=True).float()
                ctx_glb = cond["ctx_glb"].to(device, non_blocking=True).float()
                dtau_seq = cond["dtau_seq"].to(device, non_blocking=True).float()

                def to_b1zyx(sp):
                    if sp.dim() == 3:      # (Z,Y,X)
                        sp = sp.unsqueeze(0).unsqueeze(0)
                    elif sp.dim() == 4:    # (B,Z,Y,X)
                        sp = sp.unsqueeze(1)
                    elif sp.dim() == 5:    # (B,1,Z,Y,X)
                        pass
                    else:
                        raise RuntimeError(f"unexpected spatial tensor shape: {list(sp.shape)}")
                    if sp.size(0) == 1 and B > 1:
                        sp = sp.expand(B, -1, -1, -1, -1)
                    return sp

                Ms = to_b1zyx(Ms_sp)   # (B,1,Z,Y,X)
                S  = to_b1zyx(S_sp)    # (B,1,Z,Y,X)

                theta_curr = theta_seq[:, 0:1, ...]  # 从 t0 开始

                for t in range(Nt - 1):
                    if dtau_seq.dim() == 1:
                        dtau_t = dtau_seq[t].view(1).repeat(B).to(device)
                    else:
                        dtau_t = dtau_seq[:, t].to(device)

                    log_dtau = torch.log(dtau_t + 1e-12).clamp_(-20.0, 20.0).view(B, 1)
                    ctx_vec = torch.cat([ctx_glb.view(1, -1).expand(B, -1), log_dtau], dim=1)

                    if args.rk2:
                        delta1 = model(torch.cat([theta_curr, Ms, S], dim=1), ctx_vec)
                        y_tilde = theta_curr + delta1
                        delta2 = model(torch.cat([y_tilde, Ms, S], dim=1), ctx_vec)
                        delta = 0.5 * (delta1 + delta2)
                    else:
                        delta = model(torch.cat([theta_curr, Ms, S], dim=1), ctx_vec)

                    theta_curr = theta_curr + delta

                theta_true_end = theta_seq[:, -1:, ...]
                step_mse = ((theta_curr - theta_true_end) ** 2 * Ms).sum() / (Ms.sum() + 1e-8)
                step_rmse = torch.sqrt(step_mse)

            elif args.twoframe:
                # ---------- TWOFREAME 验证：任选两帧一步跨越 ----------
                x, y, cond = batch
                x = x.to(device, non_blocking=True)  # (B,3,Z,Y,X)
                y = y.to(device, non_blocking=True)  # (B,1,Z,Y,X)

                Ms = cond["Ms"].to(device, non_blocking=True).float()
                ctx_vec = cond["ctx"].to(device, non_blocking=True).float()
                if ctx_vec.dim() == 1:
                    ctx_vec = ctx_vec.unsqueeze(0)

                theta_curr = x[:, 0:1]
                S = x[:, 2:3]

                if args.rk2:
                    delta1 = model(torch.cat([theta_curr, x[:, 1:2], S], dim=1), ctx_vec)
                    y_tilde = theta_curr + delta1
                    delta2 = model(torch.cat([y_tilde, x[:, 1:2], S], dim=1), ctx_vec)
                    delta = 0.5 * (delta1 + delta2)
                else:
                    delta = model(torch.cat([theta_curr, x[:, 1:2], S], dim=1), ctx_vec)

                theta_next = theta_curr + delta
                step_mse = ((theta_next - y) ** 2 * Ms).sum() / (Ms.sum() + 1e-8)
                step_rmse = torch.sqrt(step_mse)

            else:
                # ---------- 短滚动验证：按 multi-steps 滚动 ----------
                x, y, cond = batch
                x = x.to(device, non_blocking=True)  # (B,3,Z,Y,X)
                y = y.to(device, non_blocking=True)  # (B,1,Z,Y,X)
                Ms = cond["Ms"].to(device, non_blocking=True).float()

                ctx_vec = cond["ctx"].to(device, non_blocking=True).float()
                if ctx_vec.dim() == 1:
                    ctx_vec = ctx_vec.unsqueeze(0)

                theta_curr = x[:, 0:1]

                for step in range(args.multi_steps):
                    if args.rk2:
                        delta1 = model(torch.cat([theta_curr, x[:, 1:2], x[:, 2:3]], dim=1), ctx_vec)
                        y_tilde = theta_curr + delta1
                        delta2 = model(torch.cat([y_tilde, x[:, 1:2], x[:, 2:3]], dim=1), ctx_vec)
                        delta = 0.5 * (delta1 + delta2)
                    else:
                        delta = model(torch.cat([theta_curr, x[:, 1:2], x[:, 2:3]], dim=1), ctx_vec)

                    theta_curr = theta_curr + delta

                step_mse = ((theta_curr - y) ** 2 * Ms).sum() / (Ms.sum() + 1e-8)
                step_rmse = torch.sqrt(step_mse)

            total_mse += float(step_mse.detach().cpu())
            total_rmse += float(step_rmse.detach().cpu())
            n_batches += 1

    if was_training:
        model.train()

    if n_batches == 0:
        return None, None
    return total_mse / n_batches, total_rmse / n_batches


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

    if args.data_glob and args.data_glob != "default":
        src = args.data_glob
    elif args.data and args.data != "default":
        src = args.data
    else:
        raise ValueError("请提供 --data 或 --data_glob")

    paths = discover_h5(src)
    assert len(paths) > 0, f"未找到任何 H5：{src}"
    print(f"[info] found {len(paths)} files. e.g.: {paths[0]}")

    preload_to = None    # or "cuda"
    if args.preload_gpu:
        if device.type != "cuda":
            print("[warn] --preload-gpu 被忽略：当前不是 CUDA 设备。")
        else:
            preload_to = device.type
            print("[preload] 全量预加载到 GPU（按 H5 驻留），将强制 num_workers=0, pin_memory=False")

    # 选择训练数据集
    if args.fullseq:
        print("[info] FULLSEQ 模式：每个样本使用整段时间序列，逐步监督 Nt-1 个子步")
        ds = FullSeqDataset(paths, preload_to=preload_to)
    elif args.twoframe:
        print("[info] TWOFREAME 模式：每个样本随机任选两帧 ts<te，做一步跨越监督")
        ds = AnyTwoFrameDataset(paths, preload_to=preload_to)
    else:
        ds = MultiPairDataset(paths, max_skip=args.max_skip, preload_to=preload_to)

    # 训练 DataLoader + 分桶
    use_bucket = args.bucket_by_shape and (args.batch >= 1)
    batch_sampler = None
    if use_bucket and (not args.preload_gpu or args.batch > 1):
        batch_sampler = ShapeBucketBatchSampler(
            ds, batch_size=args.batch, drop_last=args.drop_last,
            shuffle=True, seed=args.seed
        )
        if args.preload_gpu and (device.type == "cuda"):
            eff_workers = 0; eff_pin = False; eff_persist = False
        else:
            eff_workers = args.num_workers; eff_pin = args.pin_memory; eff_persist = (args.num_workers > 0)
        dl = torch.utils.data.DataLoader(
            ds, batch_sampler=batch_sampler,
            num_workers=eff_workers, worker_init_fn=worker_init_fn,
            pin_memory=eff_pin, persistent_workers=eff_persist
        )
    else:
        if args.preload_gpu and (device.type == "cuda"):
            eff_workers = 0; eff_pin = False; eff_persist = False
        else:
            eff_workers = args.num_workers; eff_pin = args.pin_memory; eff_persist = (args.num_workers > 0)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=args.batch, shuffle=True,
            num_workers=eff_workers, worker_init_fn=worker_init_fn,
            pin_memory=eff_pin, persistent_workers=eff_persist
        )

    # ================= 验证集 DataLoader =================
    val_dl = None
    if getattr(args, "val_dir", ""):
        val_paths = discover_h5(args.val_dir)
        if len(val_paths) == 0:
            print(f"[warn] val_dir={args.val_dir} 内未找到任何 .h5 文件，跳过验证集。")
        else:
            print(f"[info] found {len(val_paths)} val files. e.g.: {val_paths[0]}")
            if args.fullseq:
                val_ds = FullSeqDataset(val_paths, preload_to=preload_to)
            elif args.twoframe:
                val_ds = AnyTwoFrameDataset(val_paths, preload_to=preload_to)
            else:
                val_ds = MultiPairDataset(val_paths, max_skip=args.max_skip, preload_to=preload_to)

            if args.preload_gpu and (device.type == "cuda"):
                v_workers = 0; v_pin = False; v_persist = False
            else:
                v_workers = args.num_workers; v_pin = args.pin_memory; v_persist = (args.num_workers > 0)

            val_dl = torch.utils.data.DataLoader(
                val_ds, batch_size=1, shuffle=False,
                num_workers=v_workers, worker_init_fn=worker_init_fn,
                pin_memory=v_pin, persistent_workers=v_persist
            )
    # ====================================================

    model = FNO3D(
        in_c=3, width=args.width, modes=(args.mz, args.my, args.mx), layers=args.layers,
        dropout=args.dropout,                # 新增：dropout 从命令行控制
        context_dim=10,
        spec_pad_type=args.spec_pad_type,
        spec_pad=(args.spec_pad_z, args.spec_pad_y, args.spec_pad_x),
        ufno_scales=args.ufno_scales
    ).to(device)

    hhead = None
    if args.learn_h:
        hhead = HHead(in_c=7, width=args.h_width, layers=args.h_layers, mask_idx=2,
                      h_min=0.0, h_max=30.0, h_prior=10.0, beta=2.0,
                      use_ctx_film=True, ctx_dim=10).to(device)

    params = list(model.parameters()) + (list(hhead.parameters()) if hhead is not None else [])
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)  # 新增：weight_decay

    if args.out_dir:
        out_dir = args.out_dir
    else:
        if args.resume and args.ckpt:
            out_dir = os.path.dirname(os.path.abspath(args.ckpt))
        else:
            exp = args.exp or _time.strftime("exp_%Y%m%d_%H%M%S")
            out_dir = os.path.join("dat/runs", exp)
    os.makedirs(out_dir, exist_ok=True)

    start_ep = 1
    if args.ckpt:
        start_ep = load_ckpt(args.ckpt, model, hhead, opt=opt if args.resume else None,
                             resume=args.resume, strict=args.strict_load, device=device)
        print(f"[info] loaded ckpt: {args.ckpt}  (start_epoch={start_ep})")

    print(f"[info] data source = {src}, out={out_dir}")
    print(f"[info] scaling(runtime): T_BASE={T_BASE:.1f}K, DT_BASE={DT_BASE:.1f}K; "
          f"S uses BASE formula L^2/(alpha*rho*cp*DT_BASE)")

    if hhead is not None:
        tot = sum(p.numel() for p in hhead.parameters())
        print(f"[info] HHead params: {tot / 1e3:.1f} K")

    lam_sup = args.lam_sup; lam_pde = args.lam_pde; lam_bc = args.lam_bc
    use_fv = (args.pde_form.lower() == "fv")

    # =============== 训练循环 ===============
    for ep in range(start_ep, args.epochs + 1):
        if isinstance(batch_sampler, ShapeBucketBatchSampler):
            batch_sampler.set_epoch(ep)

        model.train()
        if hhead is not None: hhead.train()

        m_energy = m_step_mse = m_step_rmse = 0.0
        m_sup_term = m_pde_term = m_bc_term = m_total_det = 0.0
        m_air_term = 0.0
        n_batches = 0

        for batch in dl:
            opt.zero_grad(set_to_none=True)

            if args.fullseq:
                # ---------- FULLSEQ 原路径 ----------
                theta_seq, cond = batch
                theta_seq = theta_seq.to(device, non_blocking=True)   # (B,Nt,Z,Y,X)
                B, Nt, Z, Y, X = theta_seq.shape

                Ms_sp = cond["Ms"].to(device, non_blocking=True).float()
                Mi_sp = cond["Mi"].to(device, non_blocking=True).float()
                S_sp  = cond["S"].to(device, non_blocking=True).float()
                area_sp = cond["area"].to(device, non_blocking=True).float()
                dxh_dyh_dzh = cond["dxh_dyh_dzh"].to(device, non_blocking=True)
                L_over_k = cond["L_over_k"].to(device, non_blocking=True).view(1,1,1,1,1)
                ctx_glb = cond["ctx_glb"].to(device, non_blocking=True).float()
                nsurf = cond["nsurf"].to(device, non_blocking=True).float()
                dtau_seq = cond["dtau_seq"].to(device, non_blocking=True).float()  # (Nt-1,) 或 (B,Nt-1)

                def to_b1zyx(sp):
                    if sp.dim() == 3:   # (Z,Y,X)
                        sp = sp.unsqueeze(0).unsqueeze(0)
                    elif sp.dim() == 4: # (B?,Z,Y,X)
                        sp = sp.unsqueeze(1)
                    elif sp.dim() == 5: # (B,1,Z,Y,X)
                        pass
                    else:
                        raise RuntimeError(f"unexpected spatial tensor shape: {list(sp.shape)}")
                    if sp.size(0) == 1 and B > 1:
                        sp = sp.expand(B, -1, -1, -1, -1)
                    return sp

                Ms = to_b1zyx(Ms_sp); Mi = to_b1zyx(Mi_sp); S = to_b1zyx(S_sp); A = to_b1zyx(area_sp)
                TinfK = build_env_channels(cond, B, device)
                theta_inf_hat = (TinfK - T_BASE) / DT_BASE

                if nsurf.dim() == 4:  # (3,Z,Y,X) -> (B,3,Z,Y,X)
                    nsurf = nsurf.unsqueeze(0).expand(B, -1, -1, -1, -1)

                theta_curr = theta_seq[:, 0:1, ...]  # (B,1,Z,Y,X)

                # ===== TBPTT：按窗口分块累积梯度 =====
                win = int(max(0, args.tbptt))
                if win == 0:
                    win = Nt - 1  # 等同整段一次回传（可能 OOM）

                sup_acc = pde_term_acc = bc_term_acc = energy_term_acc = air_term_acc = 0.0

                for s in range(0, Nt - 1, win):
                    e = min(Nt - 1, s + win)
                    total_chunk = torch.tensor(0.0, device=device)

                    for t in range(s, e):
                        theta_true_next = theta_seq[:, t + 1:t + 2, ...]
                        if dtau_seq.dim() == 1:
                            dtau_t = dtau_seq[t].view(1).repeat(B).to(device)
                        else:
                            dtau_t = dtau_seq[:, t].to(device)

                        log_dtau = torch.log(dtau_t + 1e-12).clamp_(-20.0, 20.0).view(B, 1)
                        ctx_vec = torch.cat([ctx_glb.view(1, -1).expand(B, -1), log_dtau], dim=1)

                        if args.rk2:
                            delta1 = model(torch.cat([theta_curr, Ms, S], dim=1), ctx_vec)
                            y_tilde = theta_curr + delta1
                            delta2 = model(torch.cat([y_tilde, Ms, S], dim=1), ctx_vec)
                            delta = 0.5 * (delta1 + delta2)
                        else:
                            delta = model(torch.cat([theta_curr, Ms, S], dim=1), ctx_vec)

                        theta_next = theta_curr + delta

                        sup = ((theta_next - theta_true_next) ** 2 * Ms).sum() / (Ms.sum() + 1e-8)

                        air_mse_term = torch.tensor(0.0, device=device)
                        if args.air_anchor > 0:
                            air = (1.0 - Ms)
                            air_mse = ((theta_next - theta_curr) ** 2 * air).sum() / (air.sum() + 1e-8)
                            air_mse_term = args.air_anchor * air_mse

                        if args.use_pde:
                            h_data = to_b1zyx(cond["h_init"].to(device).float()) * Mi
                            Bi_eff = h_data * L_over_k
                            if use_fv:
                                res_p = pde_residual_fv(theta_next, theta_curr,
                                                        dtau_t.view(B, 1), Ms, Mi, Bi_eff,
                                                        dxh_dyh_dzh, S, theta_inf=theta_inf_hat)
                            else:
                                res_p = pde_residual_strong(theta_next, theta_curr,
                                                            dtau_t.view(B, 1), Ms, dxh_dyh_dzh, S)
                            mask_int = ((Ms > 0.5) & ~(Mi > 0.5)).float()
                            pde = ((res_p ** 2) * mask_int).sum() / (mask_int.sum() + 1e-8)
                        else:
                            pde = torch.tensor(0.0, device=device)
                            res_p = torch.tensor(0.0, device=device)
                            mask_int = ((Ms > 0.5) & ~(Mi > 0.5)).float()

                        if args.use_bc:
                            h_data = to_b1zyx(cond["h_init"].to(device).float()) * Mi
                            Bi_eff = h_data * L_over_k
                            res_b = bc_robin_residual(theta_next, Bi_eff,
                                                      nsurf, dxh_dyh_dzh, Ms=Ms, Mi=Mi,
                                                      theta_inf=theta_inf_hat)
                            bc = (res_b ** 2 * Mi).sum() / (Mi.sum() + 1e-8)
                        else:
                            bc = torch.tensor(0.0, device=device)
                            Bi_eff = torch.tensor(0.0, device=device)

                        scale_p_ini = _op_norm_factor(dxh_dyh_dzh, dtau_t)
                        res_p_n = (res_p.double() / (scale_p_ini + 1e-12)).float() if args.use_pde \
                                  else torch.tensor(0.0, device=device)
                        pde_n = ((res_p_n ** 2) * mask_int).sum() / (mask_int.sum() + 1e-8) if args.use_pde \
                                else torch.tensor(0.0, device=device)
                        c_pde = 0.5; w_pde = torch.as_tensor(c_pde, device=device)
                        pde_term = lam_pde * w_pde * pde_n if args.use_pde else torch.tensor(0.0, device=device)

                        if args.use_bc:
                            Bi_mean = (Bi_eff * Mi).sum() / (Mi.sum() + 1e-8)
                            bc_scale = (1.0 + Bi_mean.detach()).double()
                            res_b_n = (res_b.double() / (bc_scale + 1e-12)).float()
                            bc_n = ((res_b_n ** 2) * Mi).sum() / (Mi.sum() + 1e-8)
                            c_bc = 0.5; w_bc = torch.as_tensor(c_bc, device=device)
                            bc_term = lam_bc * w_bc * bc_n
                        else:
                            bc_term = torch.tensor(0.0, device=device)

                        energy_term = torch.tensor(0.0, device=device)
                        if args.use_energy:
                            T_pred = theta_next * DT_BASE + T_BASE
                            T_true = theta_true_next * DT_BASE + T_BASE
                            Q_pred = (Bi_eff.double() * A.double() * (T_pred.double() - TinfK.double())).sum(dim=(1,2,3,4))
                            Q_ref  = (Bi_eff.double() * A.double() * (T_true.double() - TinfK.double())).sum(dim=(1,2,3,4))
                            energy_norm = Q_ref.abs().clamp_min(1e-6)
                            energy_loss = ((Q_pred - Q_ref) / energy_norm) ** 2
                            energy_term = args.lam_energy * energy_loss.mean()

                        step_loss = lam_sup * sup + pde_term + bc_term + energy_term + air_mse_term
                        total_chunk = total_chunk + step_loss

                        # 累计统计量（按步平均打印）
                        sup_acc += float(sup.detach().cpu())
                        pde_term_acc += float(pde_term.detach().cpu())
                        bc_term_acc += float(bc_term.detach().cpu())
                        energy_term_acc += float(energy_term.detach().cpu())
                        air_term_acc += float(air_mse_term.detach().cpu())

                        # 下一步
                        if args.detach_every > 0 and ((t + 1) % args.detach_every == 0):
                            theta_curr = theta_next.detach()
                        else:
                            theta_curr = theta_next

                    # 这一窗口回传并释放图
                    total_chunk.backward()
                    del total_chunk
                    # 截断反传链路，防止上一窗口的图继续保留
                    theta_curr = theta_curr.detach()

                # 整段累积完梯度后再更新
                opt.step()

                # 末帧误差（仅用于日志）
                step_mse = ((theta_curr - theta_seq[:, -1:, ...]) ** 2 * Ms).sum() / (Ms.sum() + 1e-8)
                step_rmse = torch.sqrt(step_mse)

                steps_count = (Nt - 1)
                m_step_mse += float(step_mse.detach().cpu())
                m_step_rmse += float(step_rmse.detach().cpu())
                m_sup_term += lam_sup * (sup_acc / max(1, steps_count))
                m_pde_term += pde_term_acc / max(1, steps_count)
                m_bc_term  += bc_term_acc  / max(1, steps_count)
                m_air_term += air_term_acc / max(1, steps_count)
                if args.use_energy:
                    m_energy += energy_term_acc / max(1, steps_count)
                n_batches += 1

            elif args.twoframe:
                # ---------- 新增：任选两帧一步跨越 ----------
                x, y, cond = batch
                x = x.to(device, non_blocking=True)  # (B,3,Z,Y,X)  [theta_s, Ms, S]
                y = y.to(device, non_blocking=True)  # (B,1,Z,Y,X)  theta_e

                Ms = cond["Ms"].to(device, non_blocking=True).float()
                Mi = cond["Mi"].to(device, non_blocking=True).float()
                dtau = cond["dtau"].to(device, non_blocking=True).float()
                dxh_dyh_dzh = cond["dxh_dyh_dzh"].to(device, non_blocking=True)
                L_over_k = cond["L_over_k"].to(device, non_blocking=True).view(-1, 1, 1, 1, 1)
                ctx_vec = cond["ctx"].to(device, non_blocking=True)
                if ctx_vec.dim() == 1: ctx_vec = ctx_vec.unsqueeze(0)
                nsurf = cond["nsurf"].to(device, non_blocking=True).float()
                if nsurf.dim() == 4:
                    nsurf = nsurf.unsqueeze(0).repeat(x.size(0), 1, 1, 1, 1)
                TinfK = build_env_channels(cond, x.size(0), device)
                theta_inf_hat = (TinfK - T_BASE) / DT_BASE

                theta_curr = x[:, 0:1]
                S = x[:, 2:3]

                # 单步（可 RK2）
                if args.rk2:
                    delta1 = model(torch.cat([theta_curr, x[:, 1:2], S], dim=1), ctx_vec)
                    y_tilde = theta_curr + delta1
                    delta2 = model(torch.cat([y_tilde, x[:, 1:2], S], dim=1), ctx_vec)
                    delta = 0.5 * (delta1 + delta2)
                else:
                    delta = model(torch.cat([theta_curr, x[:, 1:2], S], dim=1), ctx_vec)
                theta_next = theta_curr + delta

                sup = ((theta_next - y) ** 2 * Ms).sum() / (Ms.sum() + 1e-8)

                air_mse_term = torch.tensor(0.0, device=device)
                if args.air_anchor > 0:
                    air = (1.0 - Ms)
                    air_mse = ((theta_next - theta_curr) ** 2 * air).sum() / (air.sum() + 1e-8)
                    air_mse_term = args.air_anchor * air_mse

                if args.use_pde:
                    h_data = cond["h_init"].to(device).float().repeat(x.size(0), 1, 1, 1, 1) * Mi
                    Bi_eff = h_data * L_over_k
                    if use_fv:
                        res_p = pde_residual_fv(theta_next, theta_curr, dtau, Ms, Mi,
                                                Bi_eff, dxh_dyh_dzh, S, theta_inf=theta_inf_hat)
                    else:
                        res_p = pde_residual_strong(theta_next, theta_curr, dtau, Ms, dxh_dyh_dzh, S)
                    mask_int = ((Ms > 0.5) & ~(Mi > 0.5)).float()
                    pde = ((res_p ** 2) * mask_int).sum() / (mask_int.sum() + 1e-8)
                else:
                    pde = torch.tensor(0.0, device=device)
                    res_p = torch.tensor(0.0, device=device)
                    mask_int = ((Ms > 0.5) & ~(Mi > 0.5)).float()

                if args.use_bc:
                    h_data = cond["h_init"].to(device).float().repeat(x.size(0), 1, 1, 1, 1) * Mi
                    Bi_eff = h_data * L_over_k
                    res_b = bc_robin_residual(theta_next, Bi_eff, nsurf, dxh_dyh_dzh, Ms=Ms, Mi=Mi,
                                              theta_inf=theta_inf_hat)
                    bc = (res_b ** 2) * Mi
                    bc = bc.sum() / (Mi.sum() + 1e-8)
                else:
                    bc = torch.tensor(0.0, device=device)

                scale_p_ini = _op_norm_factor(dxh_dyh_dzh, dtau)
                res_p_n = (res_p.double() / (scale_p_ini + 1e-12)).float() if args.use_pde else torch.tensor(0.0, device=device)
                pde_n = ((res_p_n ** 2) * mask_int).sum() / (mask_int.sum() + 1e-8) if args.use_pde else torch.tensor(0.0, device=device)
                c_pde = 0.5; w_pde = torch.as_tensor(c_pde, device=device)
                pde_term = lam_pde * w_pde * pde_n if args.use_pde else torch.tensor(0.0, device=device)

                if args.use_bc:
                    Bi_mean = (Bi_eff * Mi).sum() / (Mi.sum() + 1e-8)
                    bc_scale = (1.0 + Bi_mean.detach()).double()
                    res_b_n = (res_b.double() / (bc_scale + 1e-12)).float()
                    bc_n = ((res_b_n ** 2) * Mi).sum() / (Mi.sum() + 1e-8)
                    c_bc = 0.5; w_bc = torch.as_tensor(c_bc, device=device)
                    bc_term = lam_bc * w_bc * bc_n
                else:
                    bc_term = torch.tensor(0.0, device=device)

                energy_term = torch.tensor(0.0, device=device)
                if args.use_energy:
                    A = cond["area"].to(device, non_blocking=True).float()
                    if A.dim() == 4:
                        A = A.unsqueeze(1).repeat(x.size(0), 1, 1, 1, 1)
                    T_pred = theta_next * DT_BASE + T_BASE
                    T_true = y * DT_BASE + T_BASE
                    Q_pred = (Bi_eff.double() * A.double() * (T_pred.double() - TinfK.double())).sum(dim=(1, 2, 3, 4))
                    Q_ref  = (Bi_eff.double() * A.double() * (T_true.double() - TinfK.double())).sum(dim=(1, 2, 3, 4))
                    energy_norm = Q_ref.abs().clamp_min(1e-6)
                    energy_loss = ((Q_pred - Q_ref) / energy_norm) ** 2
                    energy_term = args.lam_energy * energy_loss.mean()

                total = lam_sup * sup + pde_term + bc_term + energy_term + air_mse_term
                total.backward()
                opt.step()

                # 统计与日志
                m_sup_term += float((lam_sup * sup).detach().cpu())
                m_pde_term += float(pde_term.detach().cpu())
                m_bc_term  += float(bc_term.detach().cpu())
                m_air_term += float(air_mse_term.detach().cpu())
                if args.use_energy:
                    m_energy += float(energy_term.detach().cpu())
                m_total_det += float(total.detach().cpu())

                step_mse = ((theta_next - y) ** 2 * Ms).sum() / (Ms.sum() + 1e-8)
                step_rmse = torch.sqrt(step_mse)
                m_step_mse += float(step_mse.detach().cpu())
                m_step_rmse += float(step_rmse.detach().cpu())
                n_batches += 1

            else:
                # ---------- 短滚动原路径 ----------
                x, y, cond = batch
                x = x.to(device, non_blocking=True)  # (B,3,Z,Y,X)
                y = y.to(device, non_blocking=True)  # (B,1,Z,Y,X)
                Ms = cond["Ms"].to(device, non_blocking=True).float()
                Mi = cond["Mi"].to(device, non_blocking=True).float()
                dtau = cond["dtau"].to(device, non_blocking=True).float()
                S = x[:, 2:3]
                dxh_dyh_dzh = cond["dxh_dyh_dzh"].to(device, non_blocking=True)

                if dxh_dyh_dzh.dim() == 2 and dxh_dyh_dzh.shape[0] > 1:
                    if not torch.allclose(dxh_dyh_dzh, dxh_dyh_dzh[0:1].repeat(dxh_dyh_dzh.shape[0], 1)):
                        raise ValueError("Batch 内存在不同的网格间距，请 --batch 1 或开启分桶并确保同形状采样。")

                L_over_k = cond["L_over_k"].to(device, non_blocking=True).view(-1, 1, 1, 1, 1)
                ctx_vec = cond["ctx"].to(device, non_blocking=True)
                if ctx_vec.dim() == 1:
                    ctx_vec = ctx_vec.unsqueeze(0)
                nsurf = cond["nsurf"].to(device, non_blocking=True).float()
                if nsurf.dim() == 4:
                    nsurf = nsurf.unsqueeze(0).repeat(x.size(0), 1, 1, 1, 1)

                TinfK = build_env_channels(cond, x.size(0), device)
                theta_inf_hat = (TinfK - T_BASE) / DT_BASE

                Bsz = x.size(0)
                theta_curr = x[:, 0:1]

                total = torch.tensor(0.0, device=device)
                for step in range(args.multi_steps):
                    if args.rk2:
                        delta1 = model(torch.cat([theta_curr, x[:, 1:2], x[:, 2:3]], dim=1), ctx_vec)
                        y_tilde = theta_curr + delta1
                        delta2 = model(torch.cat([y_tilde, x[:, 1:2], x[:, 2:3]], dim=1), ctx_vec)
                        delta = 0.5 * (delta1 + delta2)
                    else:
                        delta = model(torch.cat([theta_curr, x[:, 1:2], x[:, 2:3]], dim=1), ctx_vec)
                    theta_next = theta_curr + delta

                    target_delta = (y - x[:, 0:1])
                    sup = ((theta_next - x[:, 0:1] - target_delta) ** 2 * Ms).sum() / (Ms.sum() + 1e-8)

                    air_mse_term = torch.tensor(0.0, device=device)
                    if args.air_anchor > 0:
                        air = (1.0 - Ms)
                        air_mse = ((theta_next - theta_curr) ** 2 * air).sum() / (air.sum() + 1e-8)
                        air_mse_term = args.air_anchor * air_mse

                    if args.use_pde:
                        h_data = cond["h_init"].to(device).float().repeat(Bsz, 1, 1, 1, 1) * Mi
                        Bi_eff = h_data * L_over_k
                        if use_fv:
                            res_p = pde_residual_fv(theta_next, theta_curr, dtau, Ms, Mi,
                                                    Bi_eff, dxh_dyh_dzh, x[:, 2:3], theta_inf=theta_inf_hat)
                        else:
                            res_p = pde_residual_strong(theta_next, theta_curr, dtau, Ms, dxh_dyh_dzh, x[:, 2:3])
                        mask_int = ((Ms > 0.5) & ~(Mi > 0.5)).float()
                        pde = ((res_p ** 2) * mask_int).sum() / (mask_int.sum() + 1e-8)
                    else:
                        pde = torch.tensor(0.0, device=device)
                        res_p = torch.tensor(0.0, device=device)
                        mask_int = ((Ms > 0.5) & ~(Mi > 0.5)).float()

                    if args.use_bc:
                        h_data = cond["h_init"].to(device, non_blocking=True).float().repeat(Bsz, 1, 1, 1, 1) * Mi
                        Bi_eff = h_data * L_over_k
                        res_b = bc_robin_residual(theta_next, Bi_eff, nsurf, dxh_dyh_dzh, Ms=Ms, Mi=Mi,
                                                  theta_inf=theta_inf_hat)
                        bc = (res_b ** 2) * Mi
                        bc = bc.sum() / (Mi.sum() + 1e-8)
                    else:
                        bc = torch.tensor(0.0, device=device)

                    scale_p_ini = _op_norm_factor(dxh_dyh_dzh, dtau)
                    res_p_n = (res_p.double() / (scale_p_ini + 1e-12)).float() if args.use_pde else torch.tensor(0.0, device=device)
                    pde_n = ((res_p_n ** 2) * mask_int).sum() / (mask_int.sum() + 1e-8) if args.use_pde else torch.tensor(0.0, device=device)
                    c_pde = 0.5; w_pde = torch.as_tensor(c_pde, device=device)
                    pde_term = lam_pde * w_pde * pde_n if args.use_pde else torch.tensor(0.0, device=device)

                    if args.use_bc:
                        Bi_mean = (Bi_eff * Mi).sum() / (Mi.sum() + 1e-8)
                        bc_scale = (1.0 + Bi_mean.detach()).double()
                        res_b_n = (res_b.double() / (bc_scale + 1e-12)).float()
                        bc_n = ((res_b_n ** 2) * Mi).sum() / (Mi.sum() + 1e-8)
                        c_bc = 0.5; w_bc = torch.as_tensor(c_bc, device=device)
                        bc_term = lam_bc * w_bc * bc_n
                    else:
                        bc_term = torch.tensor(0.0, device=device)

                    energy_term = torch.tensor(0.0, device=device)
                    if args.use_energy:
                        A = cond["area"].to(device, non_blocking=True).float()
                        if A.dim() == 4:
                            A = A.unsqueeze(1).repeat(Bsz, 1, 1, 1, 1)
                        T_pred = theta_next * DT_BASE + T_BASE
                        T_true = y * DT_BASE + T_BASE
                        Q_pred = (Bi_eff.double() * A.double() * (T_pred.double() - TinfK.double())).sum(dim=(1, 2, 3, 4))
                        Q_ref  = (Bi_eff.double() * A.double() * (T_true.double() - TinfK.double())).sum(dim=(1, 2, 3, 4))
                        energy_norm = Q_ref.abs().clamp_min(1e-6)
                        energy_loss = ((Q_pred - Q_ref) / energy_norm) ** 2
                        energy_term = args.lam_energy * energy_loss.mean()

                    step_loss = lam_sup * sup + pde_term + bc_term + energy_term +air_mse_term
                    total = total + step_loss

                    theta_curr = theta_next.detach() if (args.detach_every > 0 and ((step + 1) % args.detach_every == 0)) else theta_next

                total.backward()
                opt.step()

                m_sup_term += float((lam_sup * sup).detach().cpu())
                m_pde_term += float(pde_term.detach().cpu())
                m_bc_term += float(bc_term.detach().cpu())
                m_air_term += float(air_mse_term.detach().cpu())
                if args.use_energy:
                    m_energy += float(energy_term.detach().cpu())
                m_total_det += float(total.detach().cpu())

                step_mse = ((theta_curr - y) ** 2 * Ms).sum() / (Ms.sum() + 1e-8)
                step_rmse = torch.sqrt(step_mse)
                m_step_mse += float(step_mse.detach().cpu())
                m_step_rmse += float(step_rmse.detach().cpu())
                n_batches += 1

        print(f"[ep {ep:03d}]")
        if args.fullseq:
            print(
                f"  loss_terms(avg per-step): (lam_sup*sup)={m_sup_term / max(1,n_batches):.4e}  "
                f"pde_term={m_pde_term / max(1,n_batches):.4e}  bc_term={m_bc_term / max(1,n_batches):.4e}  "
                f"{'energy_term=' + format((m_energy / max(1,n_batches)), '.4e') if args.use_energy else ''}  "
                f"air_terms={m_air_term / max(1,n_batches):.4e}  total={m_total_det / max(1,n_batches):.4e}"
            )
            print(
                f"  mean_error: avg_step_mse={m_step_mse / max(1,n_batches):.4e}  avg_step_rmse={m_step_rmse / max(1,n_batches):.4e}"
            )
        else:
            print(
                f"  loss_terms(avg per-batch): (lam_sup*sup)={m_sup_term / max(1,n_batches):.4e}  "
                f"pde_term={m_pde_term / max(1,n_batches):.4e}  bc_term={m_bc_term / max(1,n_batches):.4e}  "
                f"{'energy_term=' + format((m_energy / max(1,n_batches)), '.4e') if args.use_energy else ''}  "
                f"air_terms={m_air_term / max(1,n_batches):.4e}  total={m_total_det / max(1,n_batches):.4e}"
            )
            print(
                f"  mean_error: avg_step_mse={m_step_mse / max(1,n_batches):.4e}  avg_step_rmse={m_step_rmse / max(1,n_batches):.4e}"
            )

        # ============= 每个 epoch 结束后，跑一遍验证集 =============
        if val_dl is not None:
            val_mse, val_rmse = evaluate(model, args, device, val_dl)
            if val_mse is not None:
                print(f"  [val] mse={val_mse:.4e}  rmse={val_rmse:.4e}")
        # ==========================================================

        if ep % args.ckpt_every == 0:
            save_ckpt(os.path.join(out_dir, f"ckpt_ep{ep}.pt"), ep, model, hhead, opt, vars(args))

    save_ckpt(os.path.join(out_dir, "ckpt_final.pt"), args.epochs, model, hhead, opt, vars(args))


# =======================================================
# 参数
# =======================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="default")
    ap.add_argument("--data_glob", type=str, default="default")

    # ====== 新增：验证集目录 ======
    ap.add_argument("--val_dir", type=str, default="",
                    help="包含验证集 H5 文件的文件夹路径；为空则不做验证评估")
    # =============================

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-3)

    # ====== 新增：防过拟合参数 ======
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="FNO 层的 3D dropout 比例，用于缓解过拟合")
    ap.add_argument("--weight-decay", type=float, default=0.0, dest="weight_decay",
                    help="Adam 权重衰减系数 (L2 正则)，用于缓解过拟合")
    # =============================

    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--mz", type=int, default=12); ap.add_argument("--my", type=int, default=12); ap.add_argument("--mx", type=int, default=12)

    ap.add_argument("--lam_sup", type=float, default=1.0)
    ap.add_argument("--use_pde", action="store_true"); ap.add_argument("--lam_pde", type=float, default=1e-2)
    ap.add_argument("--use_bc", action="store_true");  ap.add_argument("--lam_bc", type=float, default=1e-2)
    ap.add_argument("--learn_h", action="store_true")
    ap.add_argument("--h_width", type=int, default=16)
    ap.add_argument("--h_layers", type=int, default=2)
    ap.add_argument("--lam_h_sup", type=float, default=1.0)
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--exp", type=str, default="")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--strict_load", action="store_true")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--pde_form", type=str, choices=["fv", "strong"], default="fv")
    ap.add_argument("--max_skip", type=int, default=1, help="短滚动：随机跨度上限K（k∈[1,K]）")
    ap.add_argument("--rk2", action="store_true", help="Heun(RK2) 时间推进")
    ap.add_argument("--device", type=str, default="cuda", help="cuda / cuda:0 / cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--preload-gpu", action="store_true")

    ap.add_argument("--use_energy", action="store_true")
    ap.add_argument("--lam_energy", type=float, default=1.0)

    ap.add_argument("--spec-pad-type", type=str, choices=["none", "reflect", "replicate"], default="reflect")
    ap.add_argument("--spec-pad-z", type=int, default=8)
    ap.add_argument("--spec-pad-y", type=int, default=8)
    ap.add_argument("--spec-pad-x", type=int, default=8)

    ap.add_argument("--air-anchor", type=float, default=0.0)

    # 多步/显存控制
    ap.add_argument("--multi-steps", type=int, default=1, dest="multi_steps")
    ap.add_argument("--detach-every", type=int, default=1, dest="detach_every")

    # uFNO
    ap.add_argument("--ufno-scales", type=int, default=3, help="=1 等价 FNO；=3 多尺度(m, m/2, m/4)")

    # Bucketing
    ap.add_argument("--bucket_by_shape", "--bucket-by-shape", dest="bucket_by_shape",
                    action="store_true", default=True,
                    help="按 (Nz,Ny,Nx,量化网格间距) 分桶，保证 batch 内尺寸与网格一致")
    ap.add_argument("--drop-last", action="store_true", help="每桶内不足 batch_size 的小批丢弃")
    ap.add_argument("--seed", type=int, default=42)

    # FULLSEQ：整段监督 + TBPTT
    ap.add_argument("--fullseq", action="store_true", help="整段时间序列展开，从 t=0 逐步滚动到 Nt-1，每步监督")
    ap.add_argument("--tbptt", type=int, default=8, help="FULLSEQ 的截断反传窗口长度；0 表示整段一次回传(可能 OOM)")

    # 新增：任选两帧一步跨越
    ap.add_argument("--twoframe", action="store_true",
                    help="随机任选两帧 ts<te，计算跨区间 dtau 做“一步估计”训练；忽略 multi-steps 循环")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.preload_gpu and args.multi_steps > 1 and (not args.fullseq) and (not args.twoframe):
        print("[note] 多步训练 + --preload-gpu 可能显存吃紧。若显存不足，可关闭 --preload-gpu 或增大 --detach-every。")
    if args.twoframe and args.multi_steps != 1:
        print("[note] --twoframe 模式下一步训练恒为单步（忽略 --multi-steps）。")
    train(args)
