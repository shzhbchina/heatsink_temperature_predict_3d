# /workspace/src/train_fno_pino.py
import os, json, math, argparse, time as _time
import h5py, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

'''
python /workspace/src/train_fno_pino_h_2.py \
  --data dat/gridized_vtk_padded_with_scales_h.h5 \
  --epochs 200 --batch 1 --lr 1e-2 \
  --width 24 --layers 1 --mz 12 --my 12 --mx 12 \
  --use_pde --lam_pde 1.0 --use_bc --lam_bc 100.0 \
  --learn_h --h_width 16 --h_layers 2 --lam_h_l2 1e-6 \
  --lam_h_sup 10.0 --use_h_truth_for_bc \
  --out_dir dat/runs/heatsink_fno_pino_learnh
'''

# 如在容器里无调试需求，注释掉以下三行
import pydevd_pycharm
pydevd_pycharm.settrace(host='host.docker.internal', port=5678,
                        stdout_to_server=True, stderr_to_server=True, suspend=True)

# ------------------------- FD 工具（保留 strong 用） -------------------------
def laplacian_3d(u, dz, dy, dx):
    pad = (1,1,1,1,1,1)
    up = F.pad(u, pad, mode='replicate')
    c  = up[:,:,1:-1,1:-1,1:-1]
    lap = (up[:,:,2:,1:-1,1:-1] - 2*c + up[:,:,:-2,1:-1,1:-1])/(dz*dz) \
        + (up[:,:,1:-1,2:,1:-1] - 2*c + up[:,:,1:-1,:-2,1:-1])/(dy*dy) \
        + (up[:,:,1:-1,1:-1,2:] - 2*c + up[:,:,1:-1,1:-1,:-2])/(dx*dx)
    return lap

# ======= 固体侧单边梯度（与你现有一致） =======
def one_sided_grad_on_interface_v2(u, Ms, nsurf, dz, dy, dx):
    def neigh_u_z(t):
        tm1 = torch.cat([t[:, :, :1, :, :],  t[:, :, :-1, :, :]], dim=2)
        tp1 = torch.cat([t[:, :, 1:, :, :],  t[:, :, -1:, :, :]], dim=2)
        return tm1, tp1
    def neigh_u_y(t):
        tm1 = torch.cat([t[:, :, :, :1, :],  t[:, :, :, :-1, :]], dim=3)
        tp1 = torch.cat([t[:, :, :, 1:, :],  t[:, :, :, -1:, :]], dim=3)
        return tm1, tp1
    def neigh_u_x(t):
        tm1 = torch.cat([t[:, :, :, :, :1],  t[:, :, :, :, :-1]], dim=4)
        tp1 = torch.cat([t[:, :, :, :, 1:],  t[:, :, :, :, -1:]], dim=4)
        return tm1, tp1
    def neigh_Ms_z(M):
        z0  = torch.zeros_like(M[:, :, :1, :, :])
        Mm1 = torch.cat([z0,            M[:, :, :-1, :, :]], dim=2)
        Mp1 = torch.cat([M[:, :, 1:, :, :], z0],            dim=2)
        return Mm1, Mp1
    def neigh_Ms_y(M):
        z0  = torch.zeros_like(M[:, :, :, :1, :])
        Mm1 = torch.cat([z0,            M[:, :, :, :-1, :]], dim=3)
        Mp1 = torch.cat([M[:, :, :, 1:, :], z0],            dim=3)
        return Mm1, Mp1
    def neigh_Ms_x(M):
        z0  = torch.zeros_like(M[:, :, :, :, :1])
        Mm1 = torch.cat([z0,            M[:, :, :, :, :-1]], dim=4)
        Mp1 = torch.cat([M[:, :, :, :, 1:], z0],             dim=4)
        return Mm1, Mp1

    uzm1, uzp1 = neigh_u_z(u);  uym1, uyp1 = neigh_u_y(u);  uxm1, uxp1 = neigh_u_x(u)
    Mzm1, Mzp1 = neigh_Ms_z(Ms); Mym1, Myp1 = neigh_Ms_y(Ms); Mxm1, Mxp1 = neigh_Ms_x(Ms)
    nz = nsurf[:, 0:1]; ny = nsurf[:, 1:2]; nx = nsurf[:, 2:3]
    Ms_solid = (Ms > 0.5).float()

    gz_bwd = (u - uzm1) / dz; gz_fwd = (uzp1 - u) / dz
    have_minus_z = (Mzm1 > 0.5); have_plus_z  = (Mzp1 > 0.5)
    use_bwd_z = have_minus_z & ~have_plus_z; use_fwd_z = have_plus_z  & ~have_minus_z
    both_z    = have_minus_z &  have_plus_z
    use_bwd_z = use_bwd_z | (both_z & (nz >= 0)); use_fwd_z = use_fwd_z | (both_z & (nz <  0))
    gz = torch.zeros_like(u); gz = torch.where(use_bwd_z, gz_bwd, gz); gz = torch.where(use_fwd_z, gz_fwd, gz); gz = gz * Ms_solid

    gy_bwd = (u - uym1) / dy; gy_fwd = (uyp1 - u) / dy
    have_minus_y = (Mym1 > 0.5); have_plus_y  = (Myp1 > 0.5)
    use_bwd_y = have_minus_y & ~have_plus_y; use_fwd_y = have_plus_y  & ~have_minus_y
    both_y    = have_minus_y &  have_plus_y
    use_bwd_y = use_bwd_y | (both_y & (ny >= 0)); use_fwd_y = use_fwd_y | (both_y & (ny <  0))
    gy = torch.zeros_like(u); gy = torch.where(use_bwd_y, gy_bwd, gy); gy = torch.where(use_fwd_y, gy_fwd, gy); gy = gy * Ms_solid

    gx_bwd = (u - uxm1) / dx; gx_fwd = (uxp1 - u) / dx
    have_minus_x = (Mxm1 > 0.5); have_plus_x  = (Mxp1 > 0.5)
    use_bwd_x = have_minus_x & ~have_plus_x; use_fwd_x = have_plus_x  & ~have_minus_x
    both_x    = have_minus_x &  have_plus_x
    use_bwd_x = use_bwd_x | (both_x & (nx >= 0)); use_fwd_x = use_fwd_x | (both_x & (nx <  0))
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
        self.time = self.f['time'][:]          # (Nt,)
        self.dt = self.f['time_dt'][:]         # (Nt-1,)
        gp = self.f['grid_padded']
        spacing = gp['spacing'][:]
        dims = gp['dims'][:]
        self.dx, self.dy, self.dz = float(spacing[0]), float(spacing[1]), float(spacing[2])
        self.Nz, self.Ny, self.Nx = int(dims[0]), int(dims[1]), int(dims[2])
        # masks & bc
        self.Ms = self.f['mask_solid'][:].astype(np.float32)
        self.Mi = self.f['mask_interface'][:].astype(np.float32)
        bc = self.f['bc/robin']
        self.h_init = bc['h_init'][:].astype(np.float32)
        self.T_inf  = bc['T_inf'][:].astype(np.float32)
        self.nsurf  = self.f['normal_on_surface'][:]
        # sources
        if 'sources' in self.f:
            self.q_vol = self.f['sources/q_vol'][:].astype(np.float32)
            self.q_mask= self.f['sources/q_mask'][:].astype(np.float32) if 'sources/q_mask' in self.f else np.ones_like(self.q_vol, np.float32)
        else:
            self.q_vol = np.zeros((self.Nz,self.Ny,self.Nx), np.float32)
            self.q_mask= np.zeros_like(self.q_vol)
        # const & scales
        cst = self.f['const']
        self.k = float(cst['k_solid'][()])
        self.T_amb = float(cst['T_amb'][()])
        self.rho = float(cst['rho_solid'][()]) if 'rho_solid' in cst else None
        self.cp  = float(cst['cp_solid'][()])  if 'cp_solid'  in cst else None
        sc  = self.f['scales']
        self.alpha  = float(sc['alpha'][()]) if 'alpha' in sc else None
        self.L      = float(sc['L'][()])
        self.dTref  = float(sc['dT_ref'][()])
        self.qref   = float(sc['q_ref'][()])
        # hat spacings
        self.dx_hat = self.dx/self.L; self.dy_hat = self.dy/self.L; self.dz_hat = self.dz/self.L
        # Bi baseline
        self.Bi = (self.h_init*self.L/self.k).astype(np.float32)

        # ====== 可选：读取真值 h（统一场，基于能量守恒）======
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
        try: self.f.close()
        except: pass

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, H: HeatsinkH5):
        self.H = H
        self.Nt = H.T.shape[0]
        self.pairs = [(i, i+1) for i in range(self.Nt-1)]
        self.Ms = torch.from_numpy(H.Ms[None, ...])
        self.Mi = torch.from_numpy(H.Mi[None, ...])
        self.S  = torch.from_numpy((H.q_vol/H.qref)[None, ...])
        self.Tinf = torch.from_numpy(H.T_inf[None, ...])
        self.Bi = torch.from_numpy(H.Bi[None, ...])
        self.h_init = torch.from_numpy(H.h_init[None, ...])
        self.nsurf = torch.from_numpy(H.nsurf)   # (3,Nz,Ny,Nx)
        self.alpha = H.alpha; self.L = H.L
        self.dxh, self.dyh, self.dzh = H.dz_hat, H.dy_hat, H.dx_hat  # (z,y,x)
        self.h_truth = torch.from_numpy(H.h_truth) if (H.h_truth is not None) else None

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        t0, t1 = self.pairs[idx]
        T0 = torch.from_numpy(self.H.T[t0]); T1 = torch.from_numpy(self.H.T[t1])
        theta0 = (T0 - self.H.T_amb)/self.H.dTref
        theta1 = (T1 - self.H.T_amb)/self.H.dTref
        dt = float(self.H.dt[t0])
        dtau = dt * self.H.alpha / (self.H.L*self.H.L)

        # >>> CHANGED: FNO 输入不再包含 dtau_field
        x = torch.stack([theta0, self.Ms[0], self.S[0]], dim=0)  # (3, Nz,Ny,Nx)  [θ0, Ms, S]
        y = theta1[None, ...]

        cond = {
            "dtau": torch.tensor([dtau], dtype=torch.float32),
            "Tinf": self.Tinf,
            "Bi":   self.Bi,
            "h_init": self.h_init,
            "Mi":   self.Mi,
            "Ms":   self.Ms,
            "nsurf": self.nsurf,
            "dxh_dyh_dzh": torch.tensor([self.H.dz_hat, self.H.dy_hat, self.H.dx_hat], dtype=torch.float32),
            "L_over_k": torch.tensor([self.H.L/self.H.k], dtype=torch.float32)
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
        scale = 1/(in_c*out_c)
        self.weight = nn.Parameter(scale*torch.randn(in_c, out_c, self.mz, self.my, self.mx, 2))
    def compl_mul3d(self, a, b):
        op = torch.einsum
        return torch.stack([
            op("bczyx,cozyx->bozyx", a[...,0], b[...,0]) - op("bczyx,cozyx->bozyx", a[...,1], b[...,1]),
            op("bczyx,cozyx->bozyx", a[...,0], b[...,1]) + op("bczyx,cozyx->bozyx", a[...,1], b[...,0]),
        ], dim=-1)
    def forward(self, x):
        B, C, Z, Y, X = x.shape
        x_ft = torch.view_as_real(torch.fft.rfftn(x, s=(Z,Y,X), dim=(-3,-2,-1)))
        out_ft = torch.zeros(B, self.out_c, Z, Y, X//2+1, 2, device=x.device, dtype=x.dtype)
        mz, my, mx = min(self.mz,Z), min(self.my,Y), min(self.mx,X//2+1)
        w = self.weight[:, :, :mz, :my, :mx, :]
        out_ft[:, :, :mz, :my, :mx, :] = self.compl_mul3d(x_ft[:, :, :mz, :my, :mx, :], w)
        return torch.fft.irfftn(torch.view_as_complex(out_ft), s=(Z,Y,X), dim=(-3,-2,-1))

class PWBlock(nn.Module):
    def __init__(self, width: int, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Conv3d(width, width*expand, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(width*expand, width, kernel_size=1)
        self.norm = nn.GroupNorm(1, width)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x):
        y = self.fc2(self.act(self.fc1(x))); y = self.drop(y)
        return self.norm(x + y)

class FNN3D(nn.Module):
    def __init__(self, in_c=3, width=64, layers=6, add_coords=True, fourier_k=4, dropout=0.0):  # >>> CHANGED in_c=3
        super().__init__()
        self.add_coords = add_coords; self.fourier_k = fourier_k
        extra_c = 0
        if add_coords: extra_c = 3 + 6 * fourier_k
        self.stem = nn.Conv3d(in_c + extra_c, width, kernel_size=1)
        self.blocks = nn.ModuleList([PWBlock(width, expand=2, dropout=dropout) for _ in range(layers)])
        self.head = nn.Conv3d(width, 1, kernel_size=1)
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
        if self.fourier_k > 0:
            for k in range(1, self.fourier_k+1):
                feats += [torch.sin(k*math.pi*zz), torch.cos(k*math.pi*zz)]
                feats += [torch.sin(k*math.pi*yy), torch.cos(k*math.pi*yy)]
                feats += [torch.sin(k*math.pi*xx), torch.cos(k*math.pi*xx)]
        pe = torch.stack(feats, dim=0).unsqueeze(0).repeat(B, 1, 1, 1, 1)
        return pe
    def forward(self, x):
        B, C, Z, Y, X = x.shape
        if self.add_coords:
            pe = self._posenc(B, Z, Y, X, x.device, x.dtype)
            x = torch.cat([x, pe], dim=1)
        h = self.stem(x)
        for blk in self.blocks: h = blk(h)
        return self.head(h)

class FNO3D(nn.Module):
    def __init__(self, in_c=3, width=24, modes=(12,12,12), layers=4,   # >>> CHANGED in_c=3
                 add_coords=True, fourier_k=8, use_local=True,
                 gn_groups=1, residual_scale=0.5, dropout=0.0):
        super().__init__()
        self.add_coords = add_coords
        self.fourier_k = fourier_k
        self.use_local = use_local
        extra_c = 0
        if add_coords:
            extra_c = 3 + 6 * fourier_k
        self.lift = nn.Conv3d(in_c + extra_c, width, 1)
        mz, my, mx = modes
        self.specs  = nn.ModuleList([SpectralConv3d(width, width, mz, my, mx) for _ in range(layers)])
        self.ws     = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(layers)])
        self.locals = nn.ModuleList([nn.Conv3d(width, width, 3, padding=1, groups=width) for _ in range(layers)]) if use_local else None
        self.norms  = nn.ModuleList([nn.GroupNorm(gn_groups, width) for _ in range(layers)])
        self.gammas = nn.ParameterList([nn.Parameter(torch.tensor(float(residual_scale))) for _ in range(layers)])
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else None
        self.proj  = nn.Sequential(nn.Conv3d(width, width, 1), nn.GELU(), nn.Conv3d(width, 1, 1))
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
        for k in range(1, self.fourier_k+1):
            w = k * math.pi
            feats += [torch.sin(w*zz), torch.cos(w*zz)]
            feats += [torch.sin(w*yy), torch.cos(w*yy)]
            feats += [torch.sin(w*xx), torch.cos(w*xx)]
        pe = torch.stack(feats, dim=0).unsqueeze(0).repeat(B, 1, 1, 1, 1)
        return pe
    def forward(self, x):
        B, _, Z, Y, X = x.shape
        if self.add_coords:
            pe = self._posenc(B, Z, Y, X, x.device, x.dtype)
            x  = torch.cat([x, pe], dim=1)
        x = self.lift(x)
        for i, (sc, w, gn) in enumerate(zip(self.specs, self.ws, self.norms)):
            y = sc(x) + w(x)
            if self.use_local:
                y = y + self.locals[i](x)
            y = F.gelu(y)
            if self.drop is not None:
                y = self.drop(y)
            x = x + self.gammas[i] * y
            x = gn(x)
        return self.proj(x)

# ------------------------- HHead：标量 h ∈ [0,30] × mask -------------------------
class HHead(nn.Module):
    """
    单路 1x1x1 卷积 -> K通道体素特征 -> (边界mask加权)全局平均 -> 小MLP -> 标量 h∈[0,30]
    -> 广播 * mask  => 体素换热系数场
    """
    def __init__(self, in_c=8, width=32, layers=2, out_feat=64, mlp_hidden=64,
                 mask_idx=2, h_min=0.0, h_max=30.0):
        super().__init__()
        self.mask_idx = mask_idx
        self.h_min = float(h_min); self.h_max = float(h_max)

        blocks = []
        c = in_c
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

    def forward(self, feats):
        B, C, L, W, H = feats.shape
        z = self.net(feats)                                     # (B,K,L,W,H)
        mask = feats[:, self.mask_idx:self.mask_idx+1].float()  # (B,1,L,W,H)
        denom = mask.sum(dim=(2,3,4), keepdim=True).clamp_min(1e-6)
        g = (z * mask).sum(dim=(2,3,4), keepdim=True) / denom   # (B,K,1,1,1)
        g = g.flatten(1)                                        # (B,K)
        raw = self.mlp(g)                                       # (B,1)
        # h_scalar = self.h_min + (self.h_max - self.h_min) * torch.sigmoid(raw)  # [0,30]
        h_scalar=raw
        h_scalar = h_scalar.view(B,1,1,1,1)
        return h_scalar * mask                                  # (B,1,L,W,H)

# ------------------------- FV 工具 -------------------------
def _neighbors_replicate(u, axis):
    if axis == 2:
        um1 = torch.cat([u[:,:, :1, :, :], u[:,:, :-1, :, :]], dim=2)
        up1 = torch.cat([u[:,:, 1:, :, :], u[:,:, -1:, :, :]], dim=2)
    elif axis == 3:
        um1 = torch.cat([u[:,:,:, :1, :], u[:,:,:,:-1, :]], dim=3)
        up1 = torch.cat([u[:,:,:, 1:, :], u[:,:,:,-1:, :]], dim=3)
    elif axis == 4:
        um1 = torch.cat([u[:,:,:,:, :1], u[:,:,:,:, :-1]], dim=4)
        up1 = torch.cat([u[:,:,:,:, 1:], u[:,:,:,:, -1:]], dim=4)
    else:
        raise ValueError("axis must be 2/3/4")
    return um1, up1

def _neighbors_zero(M, axis):
    z0 = torch.zeros_like(M[:,:,:1,:,:])
    if axis == 2:
        Mm1 = torch.cat([z0,               M[:,:,:-1,:,:]], dim=2)
        Mp1 = torch.cat([M[:,:,1:,:,:],    z0],             dim=2)
    elif axis == 3:
        z0 = torch.zeros_like(M[:,:,:,:1,:])
        Mm1 = torch.cat([z0,               M[:,:,:,:-1,:]], dim=3)
        Mp1 = torch.cat([M[:,:,: ,1:,:],   z0],             dim=3)
    elif axis == 4:
        z0 = torch.zeros_like(M[:,:,:,:,:1])
        Mm1 = torch.cat([z0,               M[:,:,:,:,:-1]], dim=4)
        Mp1 = torch.cat([M[:,:,:,:,1:],    z0],             dim=4)
    else:
        raise ValueError("axis must be 2/3/4")
    return Mm1, Mp1

# >>> CHANGED: Robin 通量统一为 -Bi*(theta - theta_inf)
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
    return (theta_next - theta_curr) - dtau.view(-1,1,1,1,1) * (lap + S)

# >>> CHANGED: 透传 theta_inf 到 FV
def pde_residual_fv(theta_next, theta_curr, dtau, Ms, Mi, Bi_eff, dxh_dyh_dzh, S, theta_inf=None):
    dz, dy, dx = _unpack_deltas(dxh_dyh_dzh)
    Div = fv_divergence(theta_next, Ms, Mi, Bi_eff, dz, dy, dx, mode='interior', theta_inf=theta_inf)
    return (theta_next - theta_curr) - dtau.view(-1,1,1,1,1) * (Div + S)

# ------------------------- Robin 残差（θ-θ_inf） -------------------------
def bc_robin_residual(theta_next, Bi, nsurf, dxh_dyh_dzh, Ms=None, Mi=None, theta_inf=None):
    dz, dy, dx = _unpack_deltas(dxh_dyh_dzh)
    gz, gy, gx = one_sided_grad_on_interface_v2(theta_next, Ms, nsurf, dz, dy, dx)
    nz = nsurf[:,0:1]; ny = nsurf[:,1:2]; nx = nsurf[:,2:3]
    dth_dn = gz*nz + gy*ny + gx*nx
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
    B, _, Z, Y, X = x.shape
    theta_inf = (cond["Tinf"].to(x.device).float() - H.T_amb) / H.dTref
    theta_inf = theta_inf.repeat(B,1,1,1,1)    # (B,1,Z,Y,X)
    return theta_inf

def train(args):
    device = torch.device('cpu')
    H = HeatsinkH5(args.data)
    test_case=1
    if test_case:
        ds_ori = (PairDataset(H))
        ds = Subset(ds_ori, np.arange(5))
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
    else:
        ds = PairDataset(H)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    # >>> CHANGED: FNO 输入通道数 = 3
    model = FNO3D(in_c=3, width=args.width, modes=(args.mz,args.my,args.mx), layers=args.layers).to(device)

    hhead = None
    if args.learn_h:
        # >>> CHANGED: HHead 输入通道 = 8：[θ0, S, Mi, Ms, nz,ny,nx, theta_inf]
        hhead = HHead(in_c=8, width=args.h_width, layers=args.h_layers, mask_idx=2).to(device)

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
    print(f"[info] grid: Nz,Ny,Nx= {H.Nz,H.Ny,H.Nx}; Nt={H.T.shape[0]}; dxh,dyh,dzh= {H.dz_hat,H.dy_hat,H.dx_hat}")
    print(f"[info] scales: alpha={H.alpha:.3e}, L={H.L:.3e}, dTref={H.dTref:.3e}, qref={H.qref:.3e}")
    if hhead is not None:
        tot = sum(p.numel() for p in hhead.parameters())
        print(f"[info] HHead params: {tot/1e3:.1f} K")
    if H.h_truth is not None:
        print(f"[info] Found h_truth from H5: shape={H.h_truth.shape} -> will use for BC = {args.use_h_truth_for_bc}, h-supervision λ={args.lam_h_sup}")

    lam_sup = args.lam_sup; lam_pde = args.lam_pde; lam_bc = args.lam_bc; lam_h_l2 = args.lam_h_l2
    use_fv = (args.pde_form.lower() == "fv")

    for ep in range(start_ep, args.epochs+1):
        model.train();  hhead.train() if hhead is not None else None
        m_sup=m_pde_rms=m_bc=m_reg=m_hsup=0.0; n_batches=0

        for x, y, cond in dl:
            x = x.to(device)      # (B,3,...): [θ0, Ms, S]
            y = y.to(device)      # (B,1,...)
            Ms = cond["Ms"].to(device).float()
            Mi = cond["Mi"].to(device).float()
            dtau = cond["dtau"].to(device).float()
            nsurf = cond["nsurf"].to(device).float().repeat(x.size(0),1,1,1,1)
            Bi_base = cond["Bi"].to(device).float().repeat(x.size(0),1,1,1,1)
            h_init = cond["h_init"].to(device).float().repeat(x.size(0),1,1,1,1)
            S = x[:,2:3]  # 注意：现在 x 通道为 [θ0, Ms, S]
            dxh_dyh_dzh = cond["dxh_dyh_dzh"].to(device)
            L_over_k = cond["L_over_k"].to(device).view(1,1,1,1,1)

            # 环境：theta_inf（无量纲，与 θ 一致）
            theta_inf = build_env_channels(cond, H, x)

            # 预测 r -> y_pred
            r = model(x)
            y_pred = x[:,0:1] + dtau.view(-1,1,1,1,1) * r
            Bsz = x.size(0)

            # ----- 计算 Bi_eff：优先真值 h；否则小网；再否则基线 -----
            reg_h = torch.tensor(0.0, device=device)
            h_sup_loss = torch.tensor(0.0, device=device)

            h_truth_t0 = cond.get("h_truth_t0", None)
            if (h_truth_t0 is not None) and args.use_h_truth_for_bc:
                h_truth_t0 = h_truth_t0.to(device).float().repeat(Bsz,1,1,1,1)
                Bi_eff = (h_truth_t0 * L_over_k)  # Bi = h*L/k
                if (hhead is not None) and (args.lam_h_sup > 0.0):
                    # >>> CHANGED: feats 不包含 dtau_field
                    feats = torch.cat([
                        x[:,0:1],           # θ0
                        x[:,2:3],           # S
                        Mi,                 # 边界mask（mask_idx=2）
                        Ms,                 # 固体mask
                        nsurf[:,0:1], nsurf[:,1:2], nsurf[:,2:3],
                        theta_inf           # θ_inf
                    ], dim=1)              # in_c = 8
                    h_pred_field = hhead(feats)      # (B,1,Z,Y,X)，已在边界上非零
                    h_sup_loss = ((h_pred_field - h_truth_t0)**2 * Mi).sum() / (Mi.sum() + 1e-8)
            else:
                if hhead is not None:
                    feats = torch.cat([
                        x[:,0:1], x[:,2:3], Mi, Ms,
                        nsurf[:,0:1], nsurf[:,1:2], nsurf[:,2:3],
                        theta_inf
                    ], dim=1)
                    h_pred_field = hhead(feats)      # [0,30]*Mi
                    Bi_eff = h_pred_field * L_over_k
                    reg_h = ((h_pred_field**2) * Mi).sum() / (Mi.sum() + 1e-8)
                else:
                    Bi_eff = Bi_base

            # 监督（固体）
            target_r = (y - x[:, 0:1]) / dtau.view(-1, 1, 1, 1, 1)
            sup = ((r - target_r) ** 2 * Ms).sum() / (Ms.sum() + 1e-8)
            total = lam_sup*sup

            # PDE（FV 或 strong）
            if args.use_pde:
                if use_fv:
                    res_p = pde_residual_fv(y_pred, x[:,0:1], dtau, Ms, Mi, Bi_eff, dxh_dyh_dzh, S, theta_inf=theta_inf)  # >>> CHANGED
                else:
                    res_p = pde_residual_strong(y_pred, x[:,0:1], dtau, Ms, dxh_dyh_dzh, S)
                scale = (res_p.abs()*Ms).mean().detach()
                pde = (((res_p/(scale+1e-8))**2) * Ms).sum() / (Ms.sum() + 1e-8)
                total = total + lam_pde*pde
                pde_rms = torch.sqrt(((res_p**2)*Ms).sum() / (Ms.sum()+1e-8))
            else:
                pde = torch.tensor(0.0, device=device); pde_rms = torch.tensor(0.0, device=device)

            # BC 残差（θ-θ_inf）
            if args.use_bc:
                res_b = bc_robin_residual(y_pred, Bi_eff, nsurf, dxh_dyh_dzh, Ms=Ms, Mi=Mi, theta_inf=theta_inf)  # >>> CHANGED
                bc = (res_b**2 * Mi).sum() / (Mi.sum() + 1e-8)
                total = total + lam_bc*bc
            else:
                bc = torch.tensor(0.0, device=device)

            # Δh 正则 + h 真值监督
            if hhead is not None:
                total = total + lam_h_l2 * reg_h + args.lam_h_sup * h_sup_loss

            opt.zero_grad(); total.backward(); opt.step()

            m_sup     += float(sup.detach().cpu())
            m_pde_rms += float(pde_rms.detach().cpu())
            m_bc      += float(bc.detach().cpu())
            m_reg     += float(reg_h.detach().cpu())
            m_hsup    += float(h_sup_loss.detach().cpu())
            n_batches += 1

        print(f"[ep {ep:03d}] sup={m_sup/n_batches:.4e}  pde_rms={m_pde_rms/n_batches:.4e}  bc={m_bc/n_batches:.4e}  hL2={m_reg/n_batches:.4e}  hSUP={m_hsup/n_batches:.4e}  ({'FV' if use_fv else 'STR'})")

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
    ap.add_argument("--lam_h_l2", type=float, default=1e-6)
    ap.add_argument("--lam_h_sup", type=float, default=1.0, help="真值 h 对小网络 h 的监督权重 (MSE)")
    ap.add_argument("--use_h_truth_for_bc", action="store_true", default=True, help="在 BC/FV 中使用 H5 的统一 h")

    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--exp", type=str, default="")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--strict_load", action="store_true")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--pde_form", type=str, choices=["fv","strong"], default="fv",
                    help="PDE 残差形式：有限体积(fv) 或 强式(strong)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
