import os, math, argparse, time as _time
import h5py, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

"""
python /workspace/src/train_fno_pino_h_6.py \
  --data dat/gridized_vtk_padded_with_scales_h.h5 \
  --epochs 200 --batch 1 --lr 5e-3 \
  --width 24 --layers 1 --mz 12 --my 12 --mx 12 \
  --use_pde --lam_pde 10.0 --use_bc --lam_bc 100.0 \
  --learn_h --h_width 16 --h_layers 1 \
  --lam_h_sup 10.0 \
  --out_dir dat/runs/heatsink_fno_pino_learnh
"""

# =================== 常量：新的无量纲基准 ===================
T_BASE = 298.0   # K
DT_BASE = 30.0   # K

# 如在容器里无调试需求，注释掉以下三行
import pydevd_pycharm
pydevd_pycharm.settrace(host='host.docker.internal', port=5678,
                        stdout_to_server=True, stderr_to_server=True, suspend=True)

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

# ------------------------- 数据 -------------------------
class HeatsinkH5:
    def __init__(self, path):
        self.f = h5py.File(path, 'r')
        self.T = self.f['T_grid_shadow'][:]    # (Nt,Nz,Ny,Nx)
        self.dt = self.f['time_dt'][:]         # (Nt-1,)

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
        self.T_amb = float(cst['T_amb'][()])  # 文件中的环境温度（可能与 T_inf 等值）
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
        # 采用新的无量纲基准：theta = (T - T_BASE)/DT_BASE
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
        self.S_nd = (self.q_vol * coeff).astype(np.float32)  # 与新 θ 定义一致

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

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, H: HeatsinkH5):
        self.H = H
        self.Nt = H.T.shape[0]
        self.pairs = [(i, i + 1) for i in range(self.Nt - 1)]
        self.Ms = torch.from_numpy(H.Ms[None, ...])
        self.Mi = torch.from_numpy(H.Mi[None, ...])
        # 源项使用 BASE 无量纲，不再用 q_ref
        self.S  = torch.from_numpy(H.S_nd[None, ...])
        self.Tinf = torch.from_numpy(H.T_inf[None, ...])
        self.h_init = torch.from_numpy(H.h_init[None, ...])
        self.nsurf = torch.from_numpy(H.nsurf)   # (3,Nz,Ny,Nx)
        self.h_truth = torch.from_numpy(H.h_truth) if (H.h_truth is not None) else None

        # 全局参数（不进 voxel 通道）—— 含 alpha
        self.ctx_glb = torch.from_numpy(H.ctx_glb)        # (9,)
        self.L_over_k = torch.tensor([H.L_over_k], dtype=torch.float32)  # (1,)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        t0, t1 = self.pairs[idx]
        T0 = torch.from_numpy(self.H.T[t0]); T1 = torch.from_numpy(self.H.T[t1])

        # 新的无量纲化：theta = (T - T_BASE)/DT_BASE
        theta0 = (T0 - T_BASE) / DT_BASE
        theta1 = (T1 - T_BASE) / DT_BASE

        dt = float(self.H.dt[t0])
        dtau = dt * self.H.alpha / (self.H.L * self.H.L)

        # FNO 输入：仍为 3 通道
        x = torch.stack([theta0, self.Ms[0], self.S[0]], dim=0)  # (3, Z,Y,X)
        y = theta1[None, ...]

        cond = {
            "dtau": torch.tensor([dtau], dtype=torch.float32),
            "Tinf": self.Tinf,
            "h_init": self.h_init,          # 逐体素初值 h（仅物理项）
            "Mi":   self.Mi,
            "Ms":   self.Ms,
            "nsurf": self.nsurf,
            "dxh_dyh_dzh": torch.tensor([self.H.dz_hat, self.H.dy_hat, self.H.dx_hat], dtype=torch.float32),
            "L_over_k": self.L_over_k,      # (1,)
            "ctx_glb": self.ctx_glb         # (9,)
        }
        if self.h_truth is not None:
            cond["h_truth_t0"] = self.h_truth[t0][None, ...]
        return x, y, cond

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
                 h_prior=10.0, beta=2.0, use_ctx_film=True, ctx_dim=9):
        """
        in_c=7 组成：[θ0, S, Mi(作为mask_idx=2), Ms, nz, ny, nx]
        ctx_dim=9（在原 8 维基础上加入 log(alpha)）
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

def pde_residual_fv(theta_next, theta_curr, dtau, Ms, Mi, Bi_eff, dxh_dyh_dzh, S, theta_inf=None):
    dz, dy, dx = _unpack_deltas(dxh_dyh_dzh)
    Div = fv_divergence(theta_next, Ms, Mi, Bi_eff, dz, dy, dx, mode='interior', theta_inf=theta_inf)
    return (theta_next - theta_curr) - dtau.view(-1, 1, 1, 1, 1) * (Div + S)

# ------------------------- Robin 残差（θ-θ_inf） -------------------------
def bc_robin_residual(theta_next, Bi, nsurf, dxh_dyh_dzh, Ms=None, Mi=None, theta_inf=None):
    dz, dy, dx = _unpack_deltas(dxh_dyh_dzh)
    gz, gy, gx = one_sided_grad_on_interface_v2(theta_next, Ms, nsurf, dz, dy, dx)
    nz = nsurf[:, 0:1]; ny = nsurf[:, 1:2]; nx = nsurf[:, 2:3]
    dth_dn = gz * nz + gy * ny + gx * nx
    if theta_inf is None:
        theta_inf = 0.0
    return -dth_dn - Bi * (theta_next - theta_inf)

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
    if resume and ("optim" in ckpt):
        if opt is not None:
            opt.load_state_dict(ckpt["optim"])
        start_epoch = int(ckpt.get("epoch", 1))
    return start_epoch

# ------------------------- 训练 -------------------------
def build_env_channels(cond, H, x):
    # 仅用于物理项；新的无量纲：theta = (T - T_BASE)/DT_BASE
    B = x.size(0)
    theta_inf = (cond["Tinf"].to(x.device).float() - T_BASE) / DT_BASE
    theta_inf = theta_inf.repeat(B, 1, 1, 1, 1)    # (B,1,Z,Y,X)
    return theta_inf

def train(args):
    device = torch.device('cpu')
    H = HeatsinkH5(args.data)
    test_case = 1
    if test_case:
        ds_ori = PairDataset(H)
        ds = Subset(ds_ori, np.arange(5))
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
    else:
        ds = PairDataset(H)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    # FNO 输入通道数仍为 3: [θ0, Ms, S]；用 context_dim=9 注入“参数”（在原 8 维基础上加入 log(alpha)）
    model = FNO3D(in_c=3, width=args.width, modes=(args.mz, args.my, args.mx), layers=args.layers,
                  context_dim=9).to(device)

    hhead = None
    if args.learn_h:
        # HHead 输入不含 theta_inf 通道：in_c=7；ctx_dim=9
        hhead = HHead(in_c=7, width=args.h_width, layers=args.h_layers, mask_idx=2,
                      h_min=0.0, h_max=30.0, h_prior=10.0, beta=2.0,
                      use_ctx_film=True, ctx_dim=9).to(device)

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

    print(f"[info] data={args.data}, out={out_dir}")
    print(f"[info] grid: Nz,Ny,Nx= {H.Nz,H.Ny,H.Nx}; abs(mm): Lz={H.Lz_abs*1e3:.2f}, Ly={H.Ly_abs*1e3:.2f}, Lx={H.Lx_abs*1e3:.2f}")
    print(f"[info] g={H.g_vec} (|g|={H.g_mag:.3f})")
    print(f"[info] scales(file): alpha(file?)={getattr(H,'alpha',float('nan')):.3e}, L_ref={H.L:.3e}, dTref(file)={H.dTref:.3e}, qref(file)={H.qref if H.qref is not None else float('nan'):.3e}  [qref 未用于训练]")
    print(f"[info] scaling(runtime): T_BASE={T_BASE:.1f}K, DT_BASE={DT_BASE:.1f}K; S uses BASE formula L^2/(alpha*rho*cp*DT_BASE)")
    if hhead is not None:
        tot = sum(p.numel() for p in hhead.parameters())
        print(f"[info] HHead params: {tot/1e3:.1f} K")
    if H.h_truth is not None:
        print(f"[info] Found h_truth from H5: shape={H.h_truth.shape} -> BC/PDE 将使用数据 h；hhead 仅做监督 λ={args.lam_h_sup}")

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
            dtau = cond["dtau"].to(device).float()
            nsurf = cond["nsurf"].to(device).float().repeat(x.size(0), 1, 1, 1, 1)
            S = x[:, 2:3]
            dxh_dyh_dzh = cond["dxh_dyh_dzh"].to(device)

            # 参数：L/k 与 全局 context（含 log(alpha)）
            L_over_k = cond["L_over_k"].to(device).view(1, 1, 1, 1, 1)
            ctx_glb = cond["ctx_glb"].to(device).repeat(x.size(0), 1)  # (B,9)

            # 仅用于物理项的 θ_inf 栅格（不作为网络输入）
            theta_inf = build_env_channels(cond, H, x)

            # 预测 r -> y_pred
            r = model(x, ctx_glb)
            y_pred = x[:, 0:1] + dtau.view(-1, 1, 1, 1, 1) * r
            Bsz = x.size(0)

            # ====== 用数据/初值 h 构造 Bi_eff：仅用于 PDE/BC（不输入网络）======
            if "h_truth_t0" in cond and cond["h_truth_t0"] is not None:
                h_data = cond["h_truth_t0"].to(device).float().repeat(Bsz, 1, 1, 1, 1)
            else:
                raise ValueError("missing h_truth_t0")

            h_data = h_data * Mi
            Bi_eff = h_data * L_over_k   # Bi = h * L / k

            # ====== hhead：仅做自身监督（若有真值），不参与 Bi_eff ======
            h_sup_loss = torch.tensor(0.0, device=device)
            if (hhead is not None) and ("h_truth_t0" in cond) and (cond["h_truth_t0"] is not None) and (args.lam_h_sup > 0.0):
                feats = torch.cat([
                    x[:, 0:1],           # θ0
                    x[:, 2:3],           # S
                    Mi,                  # mask_idx=2
                    Ms,
                    nsurf[:, 0:1], nsurf[:, 1:2], nsurf[:, 2:3],
                ], dim=1)               # in_c = 7
                h_pred_field = hhead(feats, ctx_vec=ctx_glb)      # (B,1,Z,Y,X)
                h_truth_t0 = cond["h_truth_t0"].to(device).float().repeat(Bsz, 1, 1, 1, 1)
                h_sup_loss = ((h_pred_field - h_truth_t0) ** 2 * Mi).sum() / (Mi.sum() + 1e-8)

            # 监督（固体）
            target_r = (y - x[:, 0:1]) / dtau.view(-1, 1, 1, 1, 1)
            sup = ((r - target_r) ** 2 * Ms).sum() / (Ms.sum() + 1e-8)
            total = lam_sup * sup

            # PDE（FV 或 strong）
            if args.use_pde:
                if use_fv:
                    res_p = pde_residual_fv(y_pred, x[:, 0:1], dtau, Ms, Mi, Bi_eff, dxh_dyh_dzh, S, theta_inf=theta_inf)
                else:
                    res_p = pde_residual_strong(y_pred, x[:, 0:1], dtau, Ms, dxh_dyh_dzh, S)

                mask_int = ((Ms > 0.5) & ~(Mi > 0.5)).float()
                pde = ((res_p ** 2) * mask_int).sum() / (mask_int.sum() + 1e-8)
                total = total + lam_pde * pde
                pde_rms = torch.sqrt(((res_p ** 2) * mask_int).sum() / (mask_int.sum() + 1e-8))
            else:
                pde = torch.tensor(0.0, device=device); pde_rms = torch.tensor(0.0, device=device)

            # BC 残差（θ-θ_inf）
            if args.use_bc:
                res_b = bc_robin_residual(y_pred, Bi_eff, nsurf, dxh_dyh_dzh, Ms=Ms, Mi=Mi, theta_inf=theta_inf)
                bc = (res_b ** 2) * Mi
                bc = bc.sum() / (Mi.sum() + 1e-8)
                total = total + lam_bc * bc
            else:
                bc = torch.tensor(0.0, device=device)

            # h 监督（不影响 Bi_eff）
            if hhead is not None:
                total = total + args.lam_h_sup * h_sup_loss

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
    H.close()

# ------------------------- 参数 -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="H5 path (e.g., dat/xxx.h5)")
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
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
