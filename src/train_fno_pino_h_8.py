import os, math, argparse, time as _time
import h5py, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import glob
from pathlib import Path

"""
v8 (fixed) - multi H5 + variable dt + batch-safe + optional h supervision

例：单目录全部 h5（不递归子目录）

#specify h5, 
#if all files --data_glob "dat/*.h5" "dat/case_0001_L67_W73_Hb8_F8.h5"
python src/train_fno_pino_h_8.py \
  --data_glob "dat/case_0001_L67_W73_Hb8_F8.h5" \
  --epochs 200 --batch 1 --lr 1e-3 \
  --width 24 --layers 2 --mz 12 --my 12 --mx 12 \
  --use_pde --lam_pde 1.0 --use_bc --lam_bc 1.0 --lam_sup 10000.0 \
  --max_skip 1 \
  --out_dir "$HOME/model_param/heatsink_fno_pino_learnh"
  
#if continue
python src/train_fno_pino_h_8.py \
  --data_glob "dat/case_0001_L67_W73_Hb8_F8.h5" \
  --epochs 400  --batch 1 --lr 1e-3 \
  --width 24 --layers 2 --mz 12 --my 12 --mx 12 \
  --use_pde --lam_pde 1.0 --use_bc --lam_bc 1.0 --lam_sup 1000.0\
  --max_skip 1 \
  --ckpt "$HOME/model_param/heatsink_fno_pino_learnh/ckpt_ep90.pt" \
  --resume
"""

# =================== 常量：新的无量纲基准 ===================
T_BASE = 298.0   # K
DT_BASE = 30.0   # K

# ---- 可选：PyCharm 远程调试（仅当设置环境变量 PYCHARM_DEBUG=1 才启用，不阻塞）----

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
        # 仅支持 batch 内各样本相同（否则请设置 --batch 1 或分组）
        v = d.detach().view(-1)
        return float(v[0].item()), float(v[1].item()), float(v[2].item())
    raise TypeError(f"Unsupported type for deltas: {type(d)}")

# ------------------------- 数据 -------------------------
class HeatsinkH5:
    def __init__(self, path):
        self.f = h5py.File(path, 'r')
        self.T = self.f['T_grid_shadow'][:]    # (Nt,Nz,Ny,Nx)
        self.dt = self.f['time_dt'][:]         # (Nt-1,)
        self.Nt = self.T.shape[0]

        # grid
        gp = self.f['grid_padded']
        spacing = gp['spacing'][:]   # [dx,dy,dz]
        dims    = gp['dims'][:]      # [Nz,Ny,Nx]

        self.dx, self.dy, self.dz = float(spacing[0]), float(spacing[1]), float(spacing[2])
        self.Nz, self.Ny, self.Nx = int(dims[0]), int(dims[1]), int(dims[2])

        # 绝对尺寸（与 z,y,x 次序一致）
        self.Lz_abs = self.dz * self.Nz
        self.Ly_abs = self.dy * self.Ny
        self.Lx_abs = self.dx * self.Nx

        # masks & bc
        self.Ms = self.f['mask_solid'][:].astype(np.float32)
        self.Mi = self.f['mask_interface'][:].astype(np.float32)
        bc = self.f['bc/robin']
        self.h_init = bc['h_init'][:].astype(np.float32)  # 逐体素/逐边界的 h（若是标量也会广播）
        self.T_inf  = bc['T_inf'][:].astype(np.float32)
        self.nsurf  = self.f['normal_on_surface'][:]

        # sources
        if 'sources' in self.f:
            self.q_vol = self.f['sources/q_vol'][:].astype(np.float32)  # W/m^3
        else:
            self.q_vol = np.zeros((self.Nz, self.Ny, self.Nx), np.float32)

        # const & scales
        cst = self.f['const']
        self.k = float(cst['k_solid'][()])
        self.T_amb = float(cst['T_amb'][()])
        self.rho = float(cst['rho_solid'][()]) if 'rho_solid' in cst else None
        self.cp  = float(cst['cp_solid'][()])  if 'cp_solid'  in cst else None

        # 重力（若无则用 (0,-9.81,0)），按 [gz,gy,gx]
        if 'g' in cst:
            g_raw = np.array(cst['g'][:], dtype=np.float64).reshape(-1)
            self.g_vec = g_raw[:3] if g_raw.size >= 3 else np.array([0.0, -9.81, 0.0], dtype=np.float64)
        else:
            self.g_vec = np.array([0.0, -9.81, 0.0], dtype=np.float64)
        self.g_mag = float(np.linalg.norm(self.g_vec))
        self.g_hat = self.g_vec / max(self.g_mag, 1e-12)

        # scales（注意：不再使用文件里的 q_ref 参与训练）
        sc  = self.f['scales']
        file_alpha = float(sc['alpha'][()]) if 'alpha' in sc else None
        self.L      = float(sc['L'][()])     # 用于 (x/L) 的无量纲化
        self.dTref  = float(sc['dT_ref'][()])  # 仅打印/参考
        self.qref   = float(sc['q_ref'][()]) if 'q_ref' in sc else None  # 仅打印/参考

        # 优先使用文件 alpha，否则用 k/(rho*cp)
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

        # ----------- 全局 context 参数（给网络，不进体素通道）-----------
        theta_inf_hat = (self.T_inf - T_BASE) / DT_BASE
        if np.any(self.Mi > 0.5):
            theta_inf_mean_hat = float(theta_inf_hat[self.Mi > 0.5].mean())
        else:
            theta_inf_mean_hat = float(theta_inf_hat.mean())

        # 9 维：log Lz, log Ly, log Lx, g_hat(z,y,x), log|g|, mean(theta_inf_hat), log(alpha)
        self.ctx_glb = np.array([
            math.log(max(self.Lz_abs, 1e-12)),
            math.log(max(self.Ly_abs, 1e-12)),
            math.log(max(self.Lx_abs, 1e-12)),
            float(self.g_hat[0]), float(self.g_hat[1]), float(self.g_hat[2]),
            math.log(max(self.g_mag, 1e-12)),
            theta_inf_mean_hat,
            math.log(max(self.alpha, 1e-12))
        ], dtype=np.float32)

        # 用 L 统一尺度：后续 Bi_eff = h * (L/k)
        self.L_over_k = self.L / self.k

        # ====== 源项无量纲化（不再用 q_ref）======
        if (self.rho is None) or (self.cp is None):
            raise ValueError("[data] 计算源项 S 需要 rho 和 cp，请在 H5 的 const 中提供。")
        coeff = (self.L ** 2) / (self.alpha * self.rho * self.cp * DT_BASE)
        self.S_nd = (self.q_vol * coeff).astype(np.float32)

        # ====== 可选：读取逐体素/逐时刻 h 真值（若数据里提供）======
        self.h_truth = None
        if 'h_uniform_from_sources_all' in self.f:
            g = self.f['h_uniform_from_sources_all']
            if 'h_uniform_field_Wm2K' in g:
                self.h_truth = g['h_uniform_field_Wm2K'][:].astype(np.float32)
        if (self.h_truth is None) and ('h_from_grad_truth_uniform_all' in self.f):
            g2 = self.f['h_from_grad_truth_uniform_all']
            if 'h_uniform_field_Wm2K' in g2:
                self.h_truth = g2['h_uniform_field_Wm2K'][:].astype(np.float32)

    def close(self):
        try:
            self.f.close()
        except:
            pass

class MultiPairDataset(Dataset):
    """
    跨多个 H5 的样本对 (t0 -> t1=t0+k).
    输出格式与单文件版一致；h_truth 以张量 + 掩码 has_h_truth 返回，避免 collate 报错。
    """
    def __init__(self, paths, max_skip=1):
        self.paths = [str(p) for p in paths]
        self.max_skip = int(max(1, max_skip))
        self.index = []      # (file_idx, t0)
        self.meta  = []      # [(Nt, Nz, Ny, Nx)]
        for i, p in enumerate(self.paths):
            with h5py.File(p, 'r') as f:
                Nt, Nz, Ny, Nx = f['T_grid_shadow'].shape
            self.meta.append((Nt, Nz, Ny, Nx))
            self.index.extend([(i, t0) for t0 in range(Nt-1)])
        self._cache = {}     # worker 本地缓存：file_idx -> HeatsinkH5

    def __len__(self): return len(self.index)

    def _get_H(self, i):
        H = self._cache.get(i, None)
        if H is None:
            H = HeatsinkH5(self.paths[i])
            self._cache[i] = H
        return H

    def worker_init(self, worker_id):  # 供 DataLoader 的 worker_init_fn 调用
        self._cache = {}

    def __getitem__(self, idx):
        fi, t0 = self.index[idx]
        H = self._get_H(fi)

        Nt = H.T.shape[0]
        k_max = min(self.max_skip, Nt - 1 - t0)
        k = np.random.randint(1, k_max + 1)
        t1 = t0 + k

        T0 = torch.from_numpy(H.T[t0]); T1 = torch.from_numpy(H.T[t1])
        theta0 = (T0 - T_BASE)/DT_BASE
        theta1 = (T1 - T_BASE)/DT_BASE

        dt = float(H.dt[t0:t1].sum())
        dtau = dt * H.alpha / (H.L * H.L)

        x = torch.stack([theta0, torch.from_numpy(H.Ms), torch.from_numpy(H.S_nd)], dim=0)  # (3,Z,Y,X)
        y = theta1[None, ...]

        # h 真值与掩码
        if H.h_truth is not None:
            h_t0 = torch.from_numpy(H.h_truth[t0])[None, ...]   # (1,Z,Y,X)
            has_h = torch.tensor(1, dtype=torch.uint8)
        else:
            h_t0 = torch.zeros((1, H.Nz, H.Ny, H.Nx), dtype=torch.float32)
            has_h = torch.tensor(0, dtype=torch.uint8)

        cond = {
            "dtau": torch.tensor([dtau], dtype=torch.float32),                 # (1,)
            "Tinf": torch.from_numpy(H.T_inf[None, ...]),                      # (1,Z,Y,X)
            "h_init": torch.from_numpy(H.h_init[None, ...]),                   # (1,Z,Y,X)
            "Mi": torch.from_numpy(H.Mi[None, ...]),                           # (1,Z,Y,X)
            "Ms": torch.from_numpy(H.Ms[None, ...]),                           # (1,Z,Y,X)
            "nsurf": torch.from_numpy(H.nsurf),                                 # (3,Z,Y,X)
            "dxh_dyh_dzh": torch.tensor([H.dz_hat, H.dy_hat, H.dx_hat], dtype=torch.float32),  # (3,)
            "L_over_k": torch.tensor([H.L/H.k], dtype=torch.float32),          # (1,)
            "ctx_glb": torch.from_numpy(H.ctx_glb),                             # (9,)
            "h_truth_t0": h_t0,                                                # (1,Z,Y,X)
            "has_h_truth": has_h,                                              # ()
        }
        return x, y, cond

def discover_h5(pattern_or_dir):
    """
    - 若输入为目录，则仅匹配此目录下 *.h5（不递归子目录）
    - 若输入为通配符模式，则按模式匹配（常用 "dir/*.h5"）
    - 过滤掉 ".h5.h5" 等中间件
    """
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

# ------------------------- 模型 -------------------------
class SpectralConv3d(nn.Module):
    def __init__(self, in_c, out_c, modes_z, modes_y, modes_x):
        super().__init__()
        self.in_c = in_c; self.out_c = out_c
        self.mz, self.my, self.mx = modes_z, modes_y, modes_x
        scale = 1 / (in_c * out_c)
        self.weight = nn.Parameter(scale * torch.randn(in_c, out_c, self.mz, self.my, self.mx, 2))

    def compl_mul3d(self, a, b):
        op = torch.einsum
        return torch.stack([
            op("bczyx,cozyx->bozyx", a[..., 0], b[..., 0]) - op("bczyx,cozyx->bozyx", a[..., 1], b[..., 1]),
            op("bczyx,cozyx->bozyx", a[..., 0], b[..., 1]) + op("bczyx,cozyx->bozyx", a[..., 1], b[..., 0]),
        ], dim=-1)

    def forward(self, x):
        B, C, Z, Y, X = x.shape
        x_ft = torch.view_as_real(torch.fft.rfftn(x, s=(Z, Y, X), dim=(-3, -2, -1)))
        out_ft = torch.zeros(B, self.out_c, Z, Y, X // 2 + 1, 2, device=x.device, dtype=x.dtype)
        mz, my, mx = min(self.mz, Z), min(self.my, Y), min(self.mx, X // 2 + 1)
        w = self.weight[:, :, :mz, :my, :mx, :]
        out_ft[:, :, :mz, :my, :mx, :] = self.compl_mul3d(x_ft[:, :, :mz, :my, :mx, :], w)
        return torch.fft.irfftn(torch.view_as_complex(out_ft), s=(Z, Y, X), dim=(-3, -2, -1))

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
                 context_dim: int = 0):
        super().__init__()
        self.add_coords = add_coords
        self.fourier_k = fourier_k
        self.use_local = use_local
        self.context_dim = int(context_dim)

        extra_c = 0
        if add_coords:
            extra_c = 3 + 6 * fourier_k
        self.lift = nn.Conv3d(in_c + extra_c, width, 1)
        mz, my, mx = modes
        self.specs = nn.ModuleList([SpectralConv3d(width, width, mz, my, mx) for _ in range(layers)])
        self.ws = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(layers)])
        self.locals = nn.ModuleList([nn.Conv3d(width, width, 3, padding=1, groups=width) for _ in range(layers)]) if use_local else None
        self.norms = nn.ModuleList([nn.GroupNorm(gn_groups, width) for _ in range(layers)])
        self.gammas = nn.ParameterList([nn.Parameter(torch.tensor(float(residual_scale))) for _ in range(layers)])
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else None
        self.proj = nn.Sequential(nn.Conv3d(width, width, 1), nn.GELU(), nn.Conv3d(width, 1, 1))

        # ---- 全局参数注入：Linear(ctx)->(B,width,1,1,1) 作为逐通道偏置 ----
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

# ------------------------- HHead：标量 h × mask（全局参数 FiLM 调制） -------------------------
class HHead(nn.Module):
    def __init__(self, in_c=7, width=32, layers=2, out_feat=64, mlp_hidden=64,
                 mask_idx=2, h_min=0.0, h_max=30.0,
                 h_prior=10.0, beta=2.0, use_ctx_film=True, ctx_dim=10):
        """
        in_c=7 组成：[θ0, S, Mi(作为mask_idx=2), Ms, nz, ny, nx]
        ctx_dim=10 = 原 9 维 + log(dtau)
        """
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
        z = self.net(feats)                                     # (B,K,L,W,H)
        mask = feats[:, self.mask_idx:self.mask_idx+1].float()  # (B,1,L,W,H)
        denom = mask.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        g = (z * mask).sum(dim=(2, 3, 4), keepdim=True) / denom # (B,K,1,1,1)
        g = g.flatten(1)                                        # (B,K)

        raw = self.mlp(g)                                       # (B,1)

        if self.use_ctx_film and (ctx_vec is not None):
            ab = self.ctx2affine(ctx_vec)                       # (B,2)
            a = ab[:, :1]; b = ab[:, 1:2]
            raw = (1.0 + a) * raw + b

        h_norm = torch.sigmoid(self.beta * raw)                 # (B,1)
        h_scalar = self.h_min + (self.h_max - self.h_min) * h_norm
        h_scalar = h_scalar.view(B, 1, 1, 1, 1)
        return h_scalar * mask                                  # (B,1,L,W,H)

# ------------------------- FV 工具 -------------------------
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

# Robin 通量统一为 -Bi*(theta - theta_inf)
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

    inter_m_x = (Ms > 0.5) & (Ms_xm1 > 0.5)
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

# ------------------------- PDE 残差 -------------------------
def pde_residual_strong(theta_next, theta_curr, dtau, Ms, dxh_dyh_dzh, S):
    dz, dy, dx = _unpack_deltas(dxh_dyh_dzh)
    lap = laplacian_3d(theta_next, dz, dy, dx)
    return (theta_next - theta_curr) - dtau.view(-1, 1, 1, 1, 1) * (lap + S)

# def pde_residual_fv(theta_next, theta_curr, dtau, Ms, Mi, Bi_eff, dxh_dyh_dzh, S, theta_inf=None):
#     dz, dy, dx = _unpack_deltas(dxh_dyh_dzh)
#     Div = fv_divergence(theta_next, Ms, Mi, Bi_eff, dz, dy, dx, mode='interior', theta_inf=theta_inf)
#     return (theta_next - theta_curr) - dtau.view(-1, 1, 1, 1, 1) * (Div + S)

def pde_residual_fv(theta_next, theta_curr, dtau, Ms, Mi, Bi_eff, dxh_dyh_dzh, S, theta_inf=None): #double
    # keep original dtype to cast back
    d0 = theta_next.dtype
    dev = theta_next.device

    # to float64 for stable small differences
    theta_next = theta_next.double()
    theta_curr = theta_curr.double()
    Ms   = Ms.double()
    Mi   = Mi.double()
    S    = S.double()
    Bi_eff = Bi_eff.double()
    if torch.is_tensor(theta_inf):
        theta_inf = theta_inf.double()

    dz, dy, dx = _unpack_deltas(dxh_dyh_dzh)
    dz = torch.as_tensor(dz, dtype=torch.float64, device=dev)
    dy = torch.as_tensor(dy, dtype=torch.float64, device=dev)
    dx = torch.as_tensor(dx, dtype=torch.float64, device=dev)

    Div = fv_divergence(theta_next, Ms, Mi, Bi_eff, dz, dy, dx, mode='interior', theta_inf=theta_inf)
    res = (theta_next - theta_curr) - dtau.view(-1,1,1,1,1).double() * (Div + S)
    return res.to(d0)

# ------------------------- Robin 残差（θ-θ_inf） -------------------------

def bc_robin_residual(theta_next, Bi, nsurf, dxh_dyh_dzh, Ms=None, Mi=None, theta_inf=None):
    """内部用 float64 计算（含单边梯度），再还原到原 dtype。"""
    d0, dev = theta_next.dtype, theta_next.device

    th   = theta_next.to(torch.float64)
    Bi   = Bi.to(torch.float64)
    ns   = nsurf.to(torch.float64)
    Ms64 = Ms.to(torch.float64) if Ms is not None else torch.ones_like(th, dtype=torch.float64)
    # Mi 不直接用到（这里只是保持签名一致）
    dz, dy, dx = _unpack_deltas(dxh_dyh_dzh)
    dz = torch.as_tensor(dz, dtype=torch.float64, device=dev)
    dy = torch.as_tensor(dy, dtype=torch.float64, device=dev)
    dx = torch.as_tensor(dx, dtype=torch.float64, device=dev)

    # 固体侧单边梯度（内部用 64 位）
    gz, gy, gx = one_sided_grad_on_interface_v2(th, Ms64, ns, dz, dy, dx)
    nz, ny, nx = ns[:, 0:1], ns[:, 1:2], ns[:, 2:3]
    dth_dn = gz * nz + gy * ny + gx * nx

    # theta_inf 处理与广播
    if theta_inf is None:
        th_inf = torch.zeros((), dtype=torch.float64, device=dev)  # 标量 0
    else:
        th_inf = theta_inf.to(torch.float64) if torch.is_tensor(theta_inf) \
                 else torch.as_tensor(theta_inf, dtype=torch.float64, device=dev)

    res = -dth_dn - Bi * (th - th_inf)  # Robin: -∂θ/∂n - Bi(θ-θ_inf)
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
    # if resume and ("optim" in ckpt):
    #     if opt is not None:
    #         opt.load_state_dict(ckpt["optim"])
    #     start_epoch = int(ckpt.get("epoch", 1))
    if resume and ("optim" in ckpt) and (opt is not None):
        opt.load_state_dict(ckpt["optim"])
        # 关键：把学习率改回命令行的值
        for g in opt.param_groups:
            g["lr"] = args.lr
        start_epoch = int(ckpt.get("epoch", 1))

    return start_epoch

# ------------------------- 训练辅助 -------------------------
def build_env_channels(cond, x):
    """
    返回 (B,1,Z,Y,X) 维度的 theta_inf 网格。
    支持 cond["Tinf"] 是 (1,Z,Y,X) 或 (B,1,Z,Y,X) 等形态。
    """
    th = cond["Tinf"].to(x.device).float()
    if th.dim() == 4:          # (1,Z,Y,X)
        th = th.unsqueeze(1)   # -> (1,1,Z,Y,X)
    elif th.dim() == 5:
        pass                   # 已是 (B,1,Z,Y,X)
    else:
        raise RuntimeError(f"unexpected Tinf shape: {list(th.shape)}")
    B = x.size(0)
    if th.size(0) == 1 and B > 1:
        th = th.repeat(B, 1, 1, 1, 1)
    return th

def _rms(x, mask):
    return torch.sqrt(((x**2) * mask).sum() / (mask.sum() + 1e-8))

def _op_norm_factor(dxh_dyh_dzh, dtau):
    # 允许 (B,3) 或 (3,)；你前面已有 “batch 内网格一致” 的检查
    dz, dy, dx = _unpack_deltas(dxh_dyh_dzh)
    lam = 2.0*(1.0/(dz*dz) + 1.0/(dy*dy) + 1.0/(dx*dx))     # ~ 离散拉普拉斯的谱半径
    lam = torch.as_tensor(lam, dtype=torch.float64, device=dtau.device)
    return lam * dtau.view(-1,1,1,1,1).double()             # 形状可与残差广播

# ------------------------- 训练 -------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 选择数据源（优先 data_glob，其次 data）
    if args.data_glob and args.data_glob != "default":
        src = args.data_glob
    elif args.data and args.data != "default":
        src = args.data
    else:
        raise ValueError("请提供 --data 或 --data_glob")

    paths = discover_h5(src)
    assert len(paths) > 0, f"未找到任何 H5：{src}"
    print(f"[info] found {len(paths)} files. e.g.: {paths[0]}")

    ds = MultiPairDataset(paths, max_skip=args.max_skip)
    # 若不同样本网格或尺度不一致，建议 batch=1；一致可用更大 batch。
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=1,
        worker_init_fn=worker_init_fn, pin_memory=False
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=False, num_workers=1,
        worker_init_fn=worker_init_fn, pin_memory=False
    )

    # FNO 输入通道数仍为 3；context_dim=10（原 9 维 + log(dtau)）
    model = FNO3D(in_c=3, width=args.width, modes=(args.mz, args.my, args.mx), layers=args.layers,
                  context_dim=10).to(device)

    hhead = None
    if args.learn_h:
        # HHead ctx_dim=10（原 9 维 + log(dtau)）
        hhead = HHead(in_c=7, width=args.h_width, layers=args.h_layers, mask_idx=2,
                      h_min=0.0, h_max=30.0, h_prior=10.0, beta=2.0,
                      use_ctx_film=True, ctx_dim=10).to(device)

    params = list(model.parameters()) + (list(hhead.parameters()) if hhead is not None else [])
    opt = torch.optim.Adam(params, lr=args.lr)

    # 输出目录
    if args.out_dir:
        out_dir = args.out_dir
    else:
        if args.resume and args.ckpt:
            out_dir = os.path.dirname(os.path.abspath(args.ckpt))
        else:
            exp = args.exp or _time.strftime("exp_%Y%m%d_%H%M%S")
            out_dir = os.path.join("dat/runs", exp)
    os.makedirs(out_dir, exist_ok=True)

    # 加载 ckpt
    start_ep = 1
    if args.ckpt:
        start_ep = load_ckpt(args.ckpt, model, hhead, opt=opt if args.resume else None,
                             resume=args.resume, strict=args.strict_load, device=device)
        print(f"[info] loaded ckpt: {args.ckpt}  (start_epoch={start_ep})")

    print(f"[info] data source = {src}, out={out_dir}")
    print(f"[info] scaling(runtime): T_BASE={T_BASE:.1f}K, DT_BASE={DT_BASE:.1f}K; S uses BASE formula L^2/(alpha*rho*cp*DT_BASE)")
    if hhead is not None:
        tot = sum(p.numel() for p in hhead.parameters())
        print(f"[info] HHead params: {tot/1e3:.1f} K")

    lam_sup = args.lam_sup; lam_pde = args.lam_pde; lam_bc = args.lam_bc
    use_fv = (args.pde_form.lower() == "fv")

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        if hhead is not None:
            hhead.train()
        m_sup = m_pde_rms = m_bc = m_hsup = 0.0; n_batches = 0

        for x, y, cond in dl:
            x = x.to(device)      # (B,3,...): [θ0, Ms, S]
            y = y.to(device)      # (B,1,...)
            Ms = cond["Ms"].to(device).float()
            Mi = cond["Mi"].to(device).float()
            dtau = cond["dtau"].to(device).float()             # (B,1)
            S = x[:, 2:3]
            dxh_dyh_dzh = cond["dxh_dyh_dzh"].to(device)       # (B,3) 或 (3,)

            # batch 内各样本混合不同网格间距会导致问题，这里做一个软断言
            if dxh_dyh_dzh.dim() == 2 and dxh_dyh_dzh.shape[0] > 1:
                if not torch.allclose(dxh_dyh_dzh, dxh_dyh_dzh[0:1].repeat(dxh_dyh_dzh.shape[0], 1)):
                    raise ValueError("Batch 内存在不同的网格间距，请设置 --batch 1 或对数据分组。")

            # L/k 与 全局 context（含 log(alpha)） + NEW: log(dtau)（逐样本）
            L_over_k = cond["L_over_k"].to(device).view(-1, 1, 1, 1, 1)  # (B,1,1,1,1)
            ctx_glb = cond["ctx_glb"].to(device)                          # (B,9) 由 collate 叠起来
            if ctx_glb.dim() == 1:                                        # 兼容 batch=1
                ctx_glb = ctx_glb.unsqueeze(0)
            dtau_ctx = torch.log(dtau.view(-1, 1) + 1e-12)                # (B,1)
            ctx_ext = torch.cat([ctx_glb, dtau_ctx], dim=1)               # (B,10)

            # nsurf：可能是 (B,3,Z,Y,X) 或 (3,Z,Y,X)
            nsurf = cond["nsurf"].to(device).float()
            if nsurf.dim() == 4:
                nsurf = nsurf.unsqueeze(0).repeat(x.size(0), 1, 1, 1, 1)  # -> (B,3,....)

            # 仅用于物理项的 θ_inf 栅格（不作为网络输入）
            theta_inf = build_env_channels(cond, x)  # (B,1,Z,Y,X)

            # ===== 预测 r -> y_pred（可选 RK2） =====
            if args.rk2:
                r1 = model(x, ctx_ext)
                y_tilde = x[:, 0:1] + dtau.view(-1, 1, 1, 1, 1) * r1
                x2 = torch.cat([y_tilde, x[:, 1:2], x[:, 2:3]], dim=1)
                r2 = model(x2, ctx_ext)
                r = 0.5 * (r1 + r2)
            else:
                r = model(x, ctx_ext)
            y_pred = x[:, 0:1] + dtau.view(-1, 1, 1, 1, 1) * r
            Bsz = x.size(0)

            # ====== 用数据/初值 h 构造 Bi_eff：仅用于 PDE/BC（不输入网络）======
            h_truth_t0 = cond["h_truth_t0"].to(device).float().repeat(Bsz, 1, 1, 1, 1)
            has_h = cond["has_h_truth"].to(device).float().view(-1, 1, 1, 1, 1)  # (B,1,1,1,1)
            h_data = h_truth_t0 * has_h + (1.0 - has_h) * cond["h_init"].to(device).float().repeat(Bsz,1,1,1,1)
            h_data = h_data * Mi
            Bi_eff = h_data * L_over_k   # Bi = h * L / k

            # 监督（固体）
            target_r = (y - x[:, 0:1]) / dtau.view(-1, 1, 1, 1, 1)
            sup = ((r - target_r) ** 2 * Ms).sum() / (Ms.sum() + 1e-8)
            total = lam_sup * sup

            # PDE（FV 或 strong）
            if args.use_pde:
                if use_fv:
                    res_p_sim = pde_residual_fv(y, x[:, 0:1], dtau, Ms, Mi, Bi_eff, dxh_dyh_dzh, S,
                                            theta_inf=theta_inf)
                    res_p = pde_residual_fv(y_pred, x[:, 0:1], dtau, Ms, Mi, Bi_eff, dxh_dyh_dzh, S, theta_inf=theta_inf)
                else:
                    res_p = pde_residual_strong(y_pred, x[:, 0:1], dtau, Ms, dxh_dyh_dzh, S)
                mask_int = ((Ms > 0.5) & ~(Mi > 0.5)).float()
                pde = ((res_p ** 2) * mask_int).sum() / (mask_int.sum() + 1e-8)
                pde_sim = ((res_p_sim ** 2) * mask_int).sum() / (mask_int.sum() + 1e-8)
                total = total + lam_pde * pde
                pde_rms = torch.sqrt(((res_p ** 2) * mask_int).sum() / (mask_int.sum() + 1e-8))
            else:
                pde = torch.tensor(0.0, device=device); pde_rms = torch.tensor(0.0, device=device)

            # BC 残差（θ-θ_inf）
            if args.use_bc:
                res_b = bc_robin_residual(y_pred, Bi_eff, nsurf, dxh_dyh_dzh, Ms=Ms, Mi=Mi, theta_inf=theta_inf)
                res_b_sim = bc_robin_residual(y, Bi_eff, nsurf, dxh_dyh_dzh, Ms=Ms, Mi=Mi, theta_inf=theta_inf)
                bc = (res_b ** 2) * Mi
                bc = bc.sum() / (Mi.sum() + 1e-8)
                total = total + lam_bc * bc
            else:
                bc = torch.tensor(0.0, device=device)


            # ====== hhead：仅做自身监督（若有真值），不参与 Bi_eff ======
            h_sup_loss = torch.tensor(0.0, device=device)
            if (hhead is not None) and (args.lam_h_sup > 0.0):
                feats = torch.cat([
                    x[:, 0:1],           # θ0
                    x[:, 2:3],           # S
                    Mi,                  # mask_idx=2
                    Ms,
                    nsurf[:, 0:1], nsurf[:, 1:2], nsurf[:, 2:3],
                ], dim=1)
                h_pred_field = hhead(feats, ctx_vec=ctx_ext)  # (B,1,Z,Y,X)
                denom = (Mi * has_h).sum() + 1e-8
                if denom > 0:
                    h_sup_loss = ((h_pred_field - h_truth_t0) ** 2 * Mi * has_h).sum() / denom
                # total = total + args.lam_h_sup * h_sup_loss


            # --- 规范化尺度（PDE） ---
            scale_p_ini = _op_norm_factor(dxh_dyh_dzh, dtau)  # double
            # scale_p=torch.sqrt(scale_p_ini)
            scale_p = scale_p_ini
            res_p_n = (res_p.double() / (scale_p + 1e-12)).float()  # 模型
            res_p_true_n = (res_p_sim.double() / (scale_p + 1e-12)).float().detach()  # oracle(真值)
            mask_int = ((Ms > 0.5) & ~(Mi > 0.5)).float()
            pde_n = ((res_p_n ** 2) * mask_int).sum() / (mask_int.sum() + 1e-8)
            res_p_true_n_rms = torch.sqrt(((res_p_true_n ** 2) * mask_int).sum() / (mask_int.sum() + 1e-8)).detach()
            c_pde = 0.5  # 你可以试 0.3~1.0
            # w_pde = (c_pde / (res_p_true_n_rms + 1e-6)).clamp(0.1, 5.0).detach()  # 标量
            w_pde = (c_pde / (res_p_true_n_rms + 1e-6)).detach()
            pde_term = lam_pde * w_pde * pde_n

            # --- 规范化尺度（BC，Robin: -∂n - Bi），用一个简单稳健的尺度：1 + E[Bi] ---
            Bi_mean = (Bi_eff * Mi).sum() / (Mi.sum() + 1e-8)
            bc_scale = (1.0 + Bi_mean.detach()).double()
            res_b_n = (res_b.double() / (bc_scale + 1e-12)).float()
            res_b_n_sim = (res_b_sim.double() / (bc_scale + 1e-12)).float().detach()
            bc_n = ((res_b_n ** 2) * Mi).sum() / (Mi.sum() + 1e-8)
            b_ref_rms = torch.sqrt(((res_b_n_sim ** 2) * Mi).sum() / (Mi.sum() + 1e-8)).detach()
            c_bc = 0.5
            # w_bc = (c_bc / (b_ref_rms + 1e-6)).clamp(0.1, 5.0).detach()
            w_bc = (c_bc / (b_ref_rms + 1e-6)).detach()
            bc_term = lam_bc * w_bc * bc_n

            # total = lam_sup * sup + lam_pde * pde_n + lam_bc * bc_n
            total = lam_sup * sup + pde_term + bc_term

            opt.zero_grad(); total.backward(); opt.step()

            m_sup     += float(sup.detach().cpu())
            m_pde_rms += float(pde_rms.detach().cpu())
            m_bc      += float(bc.detach().cpu())
            m_hsup    += float(h_sup_loss.detach().cpu())
            n_batches += 1

        print(f"[ep {ep:03d}] sup={m_sup/n_batches:.4e}  pde_rms={m_pde_rms/n_batches:.4e}  bc={m_bc/n_batches:.4e}  hSUP={m_hsup/n_batches:.4e}  ({'FV' if use_fv else 'STR'})")

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
    ap.add_argument("--use_bc",  action="store_true"); ap.add_argument("--lam_bc",  type=float, default=1e-2)
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
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
