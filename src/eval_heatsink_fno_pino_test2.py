# /workspace/src/eval_heatsink_fno_pino.py
"""
用法:
# rollout（自回归累计）
python src/eval_heatsink_fno_pino.py \
  --data dat/gridized_vtk_padded_with_scales.h5 \
  --ckpt dat/runs/heatsink_fno_pino_learnh/ckpt_ep170.pt \
  --out  dat/eval_heatsink_rollout.h5 \
  --save_h \
  --save_h_from_grad_all \
  --save_h_from_grad_truth_all \
  --rollout

# teacher forcing（每步都用真值做输入；写出“对齐版”：索引0=真值t0，索引i+1=用真值ti预测的t(i+1)）
python src/eval_heatsink_fno_pino.py \
  --data dat/gridized_vtk_padded_with_scales.h5 \
  --ckpt dat/runs/heatsink_fno_pino_learnh/ckpt_ep180.pt \
  --out  dat/eval_heatsink_rollout.h5 \
  --teacher_forced \
  --save_h_from_grad_all \
  --save_h_from_grad_truth_all \
  --save_h

# 仅能量守恒法的“每帧统一 h”
python src/eval_heatsink_fno_pino_test2.py \
  --data dat/gridized_vtk_padded_with_scales_h.h5 \
  --ckpt dat/runs/heatsink_fno_pino_learnh/ckpt_ep90.pt \
  --out  dat/eval_heatsink_rollout.h5 \
  --teacher_forced \
  --save_h_uniform_energy_balance_all \
  --h_smooth_dt_thresh 0.5 \
  --print_h_stats \
  --save_h_pred_all
"""
import os, json, argparse
import h5py, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 如在容器里无调试需求，注释/保留以下 try
try:
    import pydevd_pycharm
    pydevd_pycharm.settrace(host='host.docker.internal', port=5678,
                            stdout_to_server=True, stderr_to_server=True, suspend=True)
except Exception:
    pass

# ====== 数值工具（与训练一致） ======
def _unpack_deltas(d):
    if isinstance(d, (tuple, list)):
        return float(d[0]), float(d[1]), float(d[2])
    if torch.is_tensor(d):
        v = d.detach().view(-1)
        return float(v[0].item()), float(v[1].item()), float(v[2].item())
    raise TypeError(type(d))

def laplacian_3d(u, dz, dy, dx):
    pad = (1,1,1,1,1,1)  # (xL,xR,yL,yR,zL,zR)
    up = F.pad(u, pad, mode='replicate')
    c  = up[:,:,1:-1,1:-1,1:-1]
    lap = (up[:,:,2:,1:-1,1:-1] - 2*c + up[:,:,:-2,1:-1,1:-1])/(dz*dz) \
        + (up[:,:,1:-1,2:,1:-1] - 2*c + up[:,:,1:-1,:-2,1:-1])/(dy*dy) \
        + (up[:,:,1:-1,1:-1,2:] - 2*c + up[:,:,1:-1,1:-1,:-2])/(dx*dx)
    return lap  # (B,1,Nz,Ny,Nx)

def one_sided_grad_on_interface_v2(u, Ms, nsurf, dz, dy, dx):
    def neigh_z(t):
        tm1 = torch.cat([t[:,:, :1, :, :], t[:,:, :-1, :, :]], dim=2)
        tm2 = torch.cat([t[:,:, :2, :, :], t[:,:, :-2, :, :]], dim=2)
        tp1 = torch.cat([t[:,:, 1:, :, :], t[:,:, -1:, :, :]], dim=2)
        tp2 = torch.cat([t[:,:, 2:, :, :], t[:,:, -2:, :, :]], dim=2)
        return tm1, tm2, tp1, tp2
    def neigh_y(t):
        tm1 = torch.cat([t[:,:,:, :1, :], t[:,:,:,:-1, :]], dim=3)
        tm2 = torch.cat([t[:,:,:, :2, :], t[:,:,:,:-2, :]], dim=3)
        tp1 = torch.cat([t[:,:,:, 1:, :], t[:,:,:,-1:, :]], dim=3)
        tp2 = torch.cat([t[:,:,:, 2:, :], t[:,:,:,-2:, :]], dim=3)
        return tm1, tm2, tp1, tp2
    def neigh_x(t):
        tm1 = torch.cat([t[:,:,:,:, :1], t[:,:,:,:, :-1]], dim=4)
        tm2 = torch.cat([t[:,:,:,:, :2], t[:,:,:,:, :-2]], dim=4)
        tp1 = torch.cat([t[:,:,:,:, 1:], t[:,:,:,:, -1:]], dim=4)
        tp2 = torch.cat([t[:,:,:,:, 2:], t[:,:,:,:, -2:]], dim=4)
        return tm1, tm2, tp1, tp2

    # z 轴
    u_zm1, u_zm2, u_zp1, u_zp2 = neigh_z(u)
    Ms_zm1, Ms_zm2, Ms_zp1, Ms_zp2 = neigh_z(Ms)
    gz_bwd2 = (3*u - 4*u_zm1 + u_zm2)/(2*dz); gz_bwd1 = (u - u_zm1)/dz
    gz_fwd2 = (-3*u + 4*u_zp1 - u_zp2)/(2*dz); gz_fwd1 = (u_zp1 - u)/dz
    have_bwd2_z = (Ms_zm1>0.5) & (Ms_zm2>0.5)
    have_bwd1_z = (Ms_zm1>0.5)
    have_fwd2_z = (Ms_zp1>0.5) & (Ms_zp2>0.5)
    have_fwd1_z = (Ms_zp1>0.5)
    nz = nsurf[:,0:1]; use_bwd_z = (nz>=0)
    gz = torch.zeros_like(u)
    gz = torch.where(use_bwd_z & have_bwd2_z, gz_bwd2, gz)
    gz = torch.where((~use_bwd_z) & have_fwd2_z, gz_fwd2, gz)
    gz = torch.where(use_bwd_z & (~have_bwd2_z) & have_bwd1_z, gz_bwd1, gz)
    gz = torch.where((~use_bwd_z) & (~have_fwd2_z) & (~have_fwd1_z), torch.zeros_like(u), gz)

    # y 轴
    u_ym1, u_ym2, u_yp1, u_yp2 = neigh_y(u)
    Ms_ym1, Ms_ym2, Ms_yp1, Ms_yp2 = neigh_y(Ms)
    gy_bwd2 = (3*u - 4*u_ym1 + u_ym2)/(2*dy); gy_bwd1 = (u - u_ym1)/dy
    gy_fwd2 = (-3*u + 4*u_yp1 - u_yp2)/(2*dy); gy_fwd1 = (u_yp1 - u)/dy
    have_bwd2_y = (Ms_ym1>0.5) & (Ms_ym2>0.5)
    have_bwd1_y = (Ms_ym1>0.5)
    have_fwd2_y = (Ms_yp1>0.5) & (Ms_yp2>0.5)
    have_fwd1_y = (Ms_yp1>0.5)
    ny = nsurf[:,1:2]; use_bwd_y = (ny>=0)
    gy = torch.zeros_like(u)
    gy = torch.where(use_bwd_y & have_bwd2_y, gy_bwd2, gy)
    gy = torch.where((~use_bwd_y) & have_fwd2_y, gy_fwd2, gy)
    gy = torch.where(use_bwd_y & (~have_bwd2_y) & have_bwd1_y, gy_bwd1, gy)
    gy = torch.where((~use_bwd_y) & (~have_fwd2_y) & (~have_fwd1_y), torch.zeros_like(u), gy)

    # x 轴
    u_xm1, u_xm2, u_xp1, u_xp2 = neigh_x(u)
    Ms_xm1, Ms_xm2, Ms_xp1, Ms_xp2 = neigh_x(Ms)
    gx_bwd2 = (3*u - 4*u_xm1 + u_xm2)/(2*dx); gx_bwd1 = (u - u_xm1)/dx
    gx_fwd2 = (-3*u + 4*u_xp1 - u_xp2)/(2*dx); gx_fwd1 = (u_xp1 - u)/dx
    have_bwd2_x = (Ms_xm1>0.5) & (Ms_xm2>0.5)
    have_bwd1_x = (Ms_xm1>0.5)
    have_fwd2_x = (Ms_xp1>0.5) & (Ms_xp2>0.5)
    have_fwd1_x = (Ms_xp1>0.5)
    nx = nsurf[:,2:3]; use_bwd_x = (nx>=0)
    gx = torch.zeros_like(u)
    gx = torch.where(use_bwd_x & have_bwd2_x, gx_bwd2, gx)
    gx = torch.where((~use_bwd_x) & have_fwd2_x, gx_fwd2, gx)
    gx = torch.where(use_bwd_x & (~have_bwd2_x) & have_bwd1_x, gx_bwd1, gx)
    gx = torch.where((~use_bwd_x) & (~have_fwd2_x) & (~have_fwd1_x), torch.zeros_like(u), gx)
    return gz, gy, gx

def pde_residual_time_balanced(theta_next, theta_curr, dtau, dxh_dyh_dzh, S):
    dz, dy, dx = _unpack_deltas(dxh_dyh_dzh)
    lap = laplacian_3d(theta_next, dz, dy, dx)
    return (theta_next - theta_curr) - dtau.view(-1,1,1,1,1) * (lap + S)

def bc_robin_residual_train_style(theta_next, Bi, nsurf, dxh_dyh_dzh, Ms):
    dz, dy, dx = _unpack_deltas(dxh_dyh_dzh)
    gz, gy, gx = one_sided_grad_on_interface_v2(theta_next, Ms, nsurf, dz, dy, dx)
    nz = nsurf[:,0:1]; ny = nsurf[:,1:2]; nx = nsurf[:,2:3]
    dth_dn = gz*nz + gy*ny + gx*nx
    return -dth_dn - Bi*theta_next

def surface_flux_and_h_from_grad(theta, nsurf5, Ms5, Mi5, dz_hat, dy_hat, dx_hat,
                                 k, L, dTref, T_amb, T_inf):
    """
    从无量纲温度场 theta (Nz,Ny,Nx) 计算：
      - 表面法向热流 q_n [W/m^2]
      - 局部换热系数 h [W/(m^2*K)]
    使用固体侧单边梯度（不使用空气侧）。
    """
    device = theta.device
    th5 = theta[None,None,...]  # (1,1,Nz,Ny,Nx)
    gz, gy, gx = one_sided_grad_on_interface_v2(th5, Ms5, nsurf5, dz_hat, dy_hat, dx_hat)
    nz, ny, nx = nsurf5[:,0:1], nsurf5[:,1:2], nsurf5[:,2:3]
    dth_dn_hat = gz*nz + gy*ny + gx*nx                        # (1,1,Nz,Ny,Nx)

    # 物理热流：q_n = -(k * dTref / L) * dθ/dn_hat
    qn = -(k * dTref / L) * dth_dn_hat                        # (1,1,Nz,Ny,Nx)

    # 表面温度与 T_inf
    T_surf = theta * dTref + T_amb                            # (Nz,Ny,Nx)
    T_inf_t = torch.as_tensor(T_inf, dtype=theta.dtype, device=device)

    # h = q_n / (T_s - T_inf) —— 只在界面上计算
    eps = 1e-8
    denom = (T_surf - T_inf_t).clamp_min(eps)[None,None,...]  # (1,1,Nz,Ny,Nx)
    h_loc = (qn / denom)

    # 只保留界面处
    Mi = Mi5 > 0.5
    qn = torch.where(Mi, qn, torch.zeros_like(qn))
    h_loc = torch.where(Mi, h_loc, torch.zeros_like(h_loc))

    return qn[0,0], h_loc[0,0], T_surf  # 去掉批/通道维

def interface_area_weights(nsurf, dx, dy, dz, Mi):
    """
    体素法向面积近似:
      A_cell ≈ |nx|*(dy*dz) + |ny|*(dx*dz) + |nz|*(dx*dy)
    仅在界面掩膜 Mi 上有效。
    """
    nx = torch.as_tensor(nsurf[0], dtype=torch.float32)
    ny = torch.as_tensor(nsurf[1], dtype=torch.float32)
    nz = torch.as_tensor(nsurf[2], dtype=torch.float32)
    A = nx.abs()* (dy*dz) + ny.abs()* (dx*dz) + nz.abs()* (dx*dy)
    A = A * torch.as_tensor(Mi, dtype=torch.float32)
    return A

# ====== 数据 ======
class HeatsinkH5:
    def __init__(self, path):
        self.f = h5py.File(path, 'r')
        self.T = self.f['T_grid_shadow'][:]    # (Nt,Nz,Ny,Nx)
        self.time = self.f['time'][:]          # (Nt,)
        self.dt = self.f['time_dt'][:]         # (Nt-1,)
        gp = self.f['grid_padded']
        spacing = gp['spacing'][:]             # (dx,dy,dz)
        dims = gp['dims'][:]                   # (Nz,Ny,Nx)
        self.dx, self.dy, self.dz = float(spacing[0]), float(spacing[1]), float(spacing[2])
        self.Nz, self.Ny, self.Nx = int(dims[0]), int(dims[1]), int(dims[2])
        # 掩膜 & 边界
        self.Ms = self.f['mask_solid'][:].astype(np.float32)
        self.Mi = self.f['mask_interface'][:].astype(np.float32)
        bc = self.f['bc/robin']
        self.h_init = bc['h_init'][:].astype(np.float32)
        self.T_inf  = bc['T_inf'][:].astype(np.float32)
        self.nsurf  = self.f['normal_on_surface'][:]  # (3,Nz,Ny,Nx)
        # 源（兼容多命名）
        if 'sources' in self.f:
            sg = self.f['sources']
            if 'q_vol' in sg: self.q_vol = sg['q_vol'][:].astype(np.float32)
            elif 'qvol' in sg: self.q_vol = sg['qvol'][:].astype(np.float32)
            else: self.q_vol = np.zeros((self.Nz,self.Ny,self.Nx), np.float32)
            if 'q_mask' in sg: self.q_mask = sg['q_mask'][:].astype(np.float32)
            elif 'qmask' in sg: self.q_mask = sg['qmask'][:].astype(np.float32)
            else: self.q_mask = np.ones_like(self.q_vol, dtype=np.float32)
        else:
            self.q_vol = np.zeros((self.Nz,self.Ny,self.Nx), np.float32)
            self.q_mask = np.ones((self.Nz,self.Ny,self.Nx), np.float32)
        # 常数 & 尺度
        cst = self.f['const']
        self.k = float(cst['k_solid'][()])
        self.T_amb = float(cst['T_amb'][()])
        self.rho = float(cst['rho_solid'][()])
        self.cp  = float(cst['cp_solid'][()])
        sc  = self.f['scales']
        self.alpha  = float(sc['alpha'][()])
        self.L      = float(sc['L'][()])
        self.dTref  = float(sc['dT_ref'][()])
        self.qref   = float(sc['q_ref'][()])
        # 无量纲网格间距（顺序与训练一致：传 (dz,dy,dx)）
        self.dx_hat = self.dx/self.L; self.dy_hat = self.dy/self.L; self.dz_hat = self.dz/self.L
        self.Bi = (self.h_init*self.L/self.k).astype(np.float32)
    def close(self):
        try: self.f.close()
        except: pass

# ====== 模型（与训练一致） ======
class SpectralConv3d(nn.Module):
    def __init__(self, in_c, out_c, mz, my, mx):
        super().__init__()
        scale = 1/(in_c*out_c)
        self.weight = nn.Parameter(scale*torch.randn(in_c, out_c, mz, my, mx, 2))
        self.mz, self.my, self.mx = mz, my, mx
    def compl_mul3d(self, a, w):
        op = torch.einsum
        return torch.stack([
            op("bczyx,cozyx->bozyx", a[...,0], w[...,0]) - op("bczyx,cozyx->bozyx", a[...,1], w[...,1]),
            op("bczyx,cozyx->bozyx", a[...,0], w[...,1]) + op("bczyx,cozyx->bozyx", a[...,1], w[...,0]),
        ], dim=-1)
    def forward(self, x):
        B,C,Z,Y,X = x.shape
        x_ft = torch.view_as_real(torch.fft.rfftn(x, s=(Z,Y,X), dim=(-3,-2,-1)))
        out_ft = torch.zeros(B, self.weight.shape[1], Z, Y, X//2+1, 2, device=x.device, dtype=x.dtype)
        mz,my,mx = min(self.mz,Z), min(self.my,Y), min(self.mx,X//2+1)
        out_ft[:,:,:mz,:my,:mx,:] = self.compl_mul3d(x_ft[:,:,:mz,:my,:mx,:], self.weight[:,:,:mz,:my,:mx,:])
        return torch.fft.irfftn(torch.view_as_complex(out_ft), s=(Z,Y,X), dim=(-3,-2,-1))

class FNO3D(nn.Module):
    def __init__(self, in_c=4, width=24, modes=(12,12,12), layers=4,
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
        self.norms  = nn.ModuleList([nn.GroupNorm(1, width) for _ in range(layers)])
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
    def __init__(self, in_c=4, width=64, layers=6, add_coords=True, fourier_k=4, dropout=0.0):
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

# class HHead(nn.Module):
#     def __init__(self, in_c=8, width=16, layers=2):
#         super().__init__()
#         blocks=[]; c=in_c
#         for _ in range(layers):
#             blocks += [nn.Conv3d(c, width, 1), nn.GELU()]; c=width
#         blocks += [nn.Conv3d(c, 1, 1)]
#         self.net = nn.Sequential(*blocks)
#     def forward(self, feats): return self.net(feats)
#


class HHead(nn.Module):
    """
    简单版：单路 1x1x1 卷积堆栈 -> 高维体素特征 (K通道)
          -> (边界mask加权)全局平均 -> 小MLP -> 标量
          -> 标量 * mask  => 体素换热系数场
    """
    def __init__(self, in_c=9, width=32, layers=2,
                 out_feat=64,          # 体素特征的通道数 K（提高此值=更强表征）
                 mlp_hidden=64,        # MLP 隐藏维度（略增可提升拟合）
                 mask_idx=2, positive=True):
        super().__init__()
        self.mask_idx = mask_idx
        self.positive = positive

        blocks = []
        c = in_c
        for _ in range(layers):
            blocks += [nn.Conv3d(c, width, kernel_size=1, bias=True), nn.GELU()]
            c = width
        # 输出 K 通道体素特征，而不是直接 1 通道
        blocks += [nn.Conv3d(c, out_feat, kernel_size=1, bias=True)]
        self.net = nn.Sequential(*blocks)

        # 小型 MLP，把 K 维全局向量 -> 标量
        self.mlp = nn.Sequential(
            nn.Linear(out_feat, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )
        # self.out_act = nn.Softplus() if positive else nn.Identity()
        self.out_act = nn.GELU() if positive else nn.Identity()

    def forward(self, feats):
        """
        feats: (B, C, L, W, H)，其中 feats[:, mask_idx] 是边界 mask(0/1)
        return: (B, 1, L, W, H) = h_scalar * mask
        """
        B, C, L, W, H = feats.shape

        # 1) 体素特征 (B, K, L, W, H)
        z = self.net(feats)

        # 2) 边界 mask 加权全局平均 -> (B, K)
        mask = feats[:, self.mask_idx:self.mask_idx+1].float()        # (B,1,L,W,H)
        denom = mask.sum(dim=(2,3,4), keepdim=True).clamp_min(1e-6)   # 防除零
        g = (z * mask).sum(dim=(2,3,4), keepdim=True) / denom         # (B,K,1,1,1)
        g = g.flatten(1)                                              # (B,K)

        # 3) 小 MLP -> 标量（非负）
        h_scalar = self.mlp(g)                                        # (B,1)
        h_scalar = self.out_act(h_scalar).view(B, 1, 1, 1, 1)         # (B,1,1,1,1)

        # 4) 标量 * mask -> 体素场
        out = h_scalar * mask                                         # (B,1,L,W,H)
        return out


# ====== 评估 ======
@torch.no_grad()
def evaluate(args):
    device = torch.device('cpu')

    # --- 读数据
    H = HeatsinkH5(args.data)
    Nt,Nz,Ny,Nx = H.T.shape
    Ms4 = torch.from_numpy(H.Ms[None,...]).float()
    Mi4 = torch.from_numpy(H.Mi[None,...]).float()
    S4  = torch.from_numpy((H.q_vol/H.qref)[None,...]).float()
    nsurf = torch.from_numpy(H.nsurf).float()            # (3,Nz,Ny,Nx)
    # 升到 5D
    Ms5, Mi5, S5 = Ms4[:,None,...], Mi4[:,None,...], S4[:,None,...]
    nsurf5 = nsurf[None,...]
    dxh_dyh_dzh = torch.tensor([H.dz_hat,H.dy_hat,H.dx_hat], dtype=torch.float32)
    L_over_k = torch.tensor([H.L/H.k], dtype=torch.float32)

    # --- build model
    ckpt = torch.load(args.ckpt, map_location='cpu')
    a = ckpt["args"]
    model = FNO3D(in_c=4, width=a["width"], modes=(a["mz"],a["my"],a["mx"]), layers=a["layers"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    learn_h = ("hhead" in ckpt)
    if learn_h:
        hhead = HHead(in_c=9, width=a.get("h_width",16), layers=a.get("h_layers",2)).to(device)
        hhead.load_state_dict(ckpt["hhead"]); hhead.eval()
    else:
        hhead = None

    # --- 度量容器
    per_pair = {"mse_theta": [], "mse_K": [], "mae_K": [], "relL2": [], "pde_rms": [], "bc_rms": []}
    if args.rollout and (not args.teacher_forced):
        theta_roll = torch.zeros(Nt, Nz, Ny, Nx, dtype=torch.float32)
    if args.teacher_forced:
        theta_tf = torch.zeros(Nt, Nz, Ny, Nx, dtype=torch.float32)

    # --- 初值
    theta_prev = torch.from_numpy((H.T[0]-H.T_amb)/H.dTref).float()
    if args.rollout and (not args.teacher_forced): theta_roll[0] = theta_prev
    if args.teacher_forced: theta_tf[0] = theta_prev
    theta_last_pred = theta_prev

    # --- for ALL 分支准备容器（pred / truth / EB）
    save_all_pred   = args.save_h_from_grad_all
    save_all_truth  = (args.save_h_from_grad_truth_all or args.save_h_from_grad_truth_uniform_all)
    save_all_h_eb   = args.save_h_uniform_energy_balance_all

    qn_list_pred, h_list_pred, Ts_list_pred, q_cons_list_pred, time_list_pred = [], [], [], [], []
    qn_list_true, h_list_true, Ts_list_true, q_cons_list_true, time_list_true = [], [], [], [], []

    # 能量守恒法的序列缓存
    h_eb_seq, h_eb_field_seq, time_eb = [], [], []
    Q_gen_seq, dUdt_seq, Q_conv_truth_seq, Q_from_uniform_seq = [], [], [], []
    denom_nonpos_count = 0

    # 面积与体积
    A = interface_area_weights(H.nsurf, H.dx, H.dy, H.dz, H.Mi).numpy().astype(np.float32)  # (Z,Y,X)
    Mi_np = H.Mi.astype(np.uint8)
    dV = float(H.dx * H.dy * H.dz)

    # 体热源功率（常量或时变；这里按当前 H5 读取为常量）：
    Q_gen_const = float((H.q_vol * H.q_mask * H.Ms).sum() * dV)

    # 统计与全时序 h（小网络）缓存
    h_stats = []                     # (t_idx, h_min, h_max, h_mean, h_area_weighted)
    h_pred_seq = [] if args.save_h_pred_all and learn_h else None

    for i in range(Nt-1):
        theta_true_t     = torch.from_numpy((H.T[i  ]-H.T_amb)/H.dTref).float()
        theta_next_true  = torch.from_numpy((H.T[i+1]-H.T_amb)/H.dTref).float()
        dtau = torch.tensor([ H.dt[i]*H.alpha/(H.L*H.L) ], dtype=torch.float32)
        dtau_field = torch.full_like(Ms4[0], float(dtau.item()))

        # 输入（teacher forcing 用真值）
        theta_in = theta_true_t if args.teacher_forced else theta_prev

        # 前向：r = dθ/dτ
        x = torch.stack([theta_in, Ms4[0], S4[0], dtau_field], dim=0).unsqueeze(0)
        r = model(x)
        y_pred = theta_in + dtau.view(1,1,1) * r[0,0]

        # 监督指标
        Ms_cpu = Ms4[0]
        diff_theta = (y_pred - theta_next_true)
        mse_theta = (diff_theta**2 * Ms_cpu).sum().item() / (Ms_cpu.sum().item()+1e-8)
        diff_K = diff_theta * H.dTref
        mse_K = (diff_K**2 * Ms_cpu).sum().item() / (Ms_cpu.sum().item()+1e-8)
        mae_K = (diff_K.abs() * Ms_cpu).sum().item() / (Ms_cpu.sum().item()+1e-8)
        denom = (theta_next_true**2 * Ms_cpu).sum().sqrt().item() + 1e-8
        relL2 = ( (diff_theta**2 * Ms_cpu).sum().sqrt().item() ) / denom
        per_pair["mse_theta"].append(mse_theta)
        per_pair["mse_K"].append(mse_K)
        per_pair["mae_K"].append(mae_K)
        per_pair["relL2"].append(relL2)

        # PDE 残差
        res_p = pde_residual_time_balanced(
            y_pred[None,None,...], theta_in[None,None,...], dtau, dxh_dyh_dzh, S5
        )
        pde_rms = torch.sqrt( ((res_p**2)*Ms5).sum() / (Ms5.sum()+1e-8) ).item()
        per_pair["pde_rms"].append(pde_rms)

        # === 边界 Bi（learn_h 用“小网络 h”，否则用 h_init 的 Bi） ===
        if learn_h:
            dtau5 = dtau.view(1,1,1,1,1).expand(1,1,Nz,Ny,Nx)
            # Tamb5=(H.T_amb/ H.dTref)*torch.ones(1,1,Nz,Ny,Nx)
            # feats = torch.cat([
            #     y_pred[None,None,...], theta_in[None,None,...],
            #     S5, Mi5, Ms5,
            #     nsurf5[:,0:1], nsurf5[:,1:2], nsurf5[:,2:3],
            #     dtau5,
            # ], dim=1)
            Tinf_abs_norm = (H.T_amb - 300.0) / 50.0  # (1,Z,Y,X)
            Tinf_abs_norm = torch.tensor(Tinf_abs_norm).to(device).float().repeat(x.size(0), 1, x.size(2), x.size(3),
                                                                                  x.size(4))  # (B,1,...)
            feats = torch.cat([
                theta_in[None,None,...],
                S5, Mi5, Ms5,
                nsurf5[:,0:1], nsurf5[:,1:2], nsurf5[:,2:3],
                dtau5,Tinf_abs_norm
            ], dim=1)
            # 评估阶段与训练一致：h_pred = softplus(delta_h_raw) * Mi
            delta_h_raw = hhead(feats) * Mi5
            # h_pred = F.softplus(delta_h_raw) * Mi5                       # ⬅ 与训练一致
            h_pred = delta_h_raw
            Bi_eff = h_pred * L_over_k.view(1,1,1,1,1)
            # 打印统计 / 保存全序列 h（仅界面非零）
            if args.print_h_stats:
                h_np = h_pred[0,0].detach().cpu().numpy().astype(np.float32)
                mask = (H.Mi > 0)
                vals = h_np[mask]
                if vals.size > 0:
                    h_min  = float(vals.min()); h_max = float(vals.max()); h_mean = float(vals.mean())
                    h_aw   = float((vals * A[mask]).sum() / (A[mask].sum() + 1e-8))
                    h_stats.append((i+1, h_min, h_max, h_mean, h_aw))
            if h_pred_seq is not None:
                h_pred_seq.append(h_pred[0,0].detach().cpu().numpy().astype(np.float32))
        else:
            Bi_eff = torch.from_numpy(H.Bi[None,None,...]).float()

        res_b = bc_robin_residual_train_style(
            y_pred[None,None,...], Bi_eff, nsurf5, dxh_dyh_dzh, Ms5
        )
        bc_rms = torch.sqrt( ((res_b**2)*Mi5).sum() / (Mi5.sum()+1e-8) ).item()
        per_pair["bc_rms"].append(bc_rms)

        # === 预测场：基于梯度的 q/h（pred）===
        need_last_pred = args.save_h_from_grad and (i == Nt-2)
        if save_all_pred or need_last_pred:
            qn_pred, h_pred_grad, Ts_pred = surface_flux_and_h_from_grad(
                theta=y_pred, nsurf5=nsurf5, Ms5=Ms5, Mi5=Mi5,
                dz_hat=H.dz_hat, dy_hat=H.dy_hat, dx_hat=H.dx_hat,
                k=H.k, L=H.L, dTref=H.dTref, T_amb=H.T_amb, T_inf=H.T_inf
            )
            # 用 Bi_eff 做一致性校核
            Bi_here = Bi_eff[0,0]
            T_inf_t = torch.from_numpy(H.T_inf).to(Ts_pred)
            qn_from_Bi = (H.k / H.L) * Bi_here * (Ts_pred - T_inf_t)
            Mi_t = torch.from_numpy(H.Mi).to(qn_pred)
            diff_q = ((qn_pred - qn_from_Bi) * (Mi_t > 0.5)).abs()
            q_cons_pred = float(diff_q.mean().item())

            if save_all_pred:
                qn_list_pred.append(qn_pred.cpu().numpy().astype(np.float32))
                h_list_pred.append(h_pred_grad.cpu().numpy().astype(np.float32))
                Ts_list_pred.append(Ts_pred.cpu().numpy().astype(np.float32))
                q_cons_list_pred.append(q_cons_pred)
                if len(H.time) > i+1: time_list_pred.append(float(H.time[i+1]))
            if need_last_pred:
                qn_last = qn_pred.cpu().numpy().astype(np.float32)
                h_last  = h_pred_grad.cpu().numpy().astype(np.float32)
                Ts_last = Ts_pred.cpu().numpy().astype(np.float32)
                q_consis_last = q_cons_pred

        # === 真值场：基于梯度的 q/h（truth，对应时刻 i+1）===
        need_last_truth = args.save_h_from_grad_truth and (i == Nt-2)
        if save_all_truth or need_last_truth:
            qn_true, h_true, Ts_true = surface_flux_and_h_from_grad(
                theta=theta_next_true, nsurf5=nsurf5, Ms5=Ms5, Mi5=Mi5,
                dz_hat=H.dz_hat, dy_hat=H.dy_hat, dx_hat=H.dx_hat,
                k=H.k, L=H.L, dTref=H.dTref, T_amb=H.T_amb, T_inf=H.T_inf
            )
            # 真值用 H.Bi 做一致性校核（不依赖网络）
            Bi_truth = torch.from_numpy(H.Bi).to(qn_true)
            T_inf_t = torch.from_numpy(H.T_inf).to(Ts_true)
            qn_from_Bi_truth = (H.k / H.L) * Bi_truth * (Ts_true - T_inf_t)
            Mi_t = torch.from_numpy(H.Mi).to(qn_true)
            diff_q_true = ((qn_true - qn_from_Bi_truth) * (Mi_t > 0.5)).abs()
            q_cons_true = float(diff_q_true.mean().item())

            if save_all_truth:
                qn_list_true.append(qn_true.cpu().numpy().astype(np.float32))
                h_list_true.append(h_true.cpu().numpy().astype(np.float32))
                Ts_list_true.append(Ts_true.cpu().numpy().astype(np.float32))
                q_cons_list_true.append(q_cons_true)
                if len(H.time) > i+1: time_list_true.append(float(H.time[i+1]))
            if need_last_truth:
                qn_last_truth = qn_true.cpu().numpy().astype(np.float32)
                h_last_truth  = h_true.cpu().numpy().astype(np.float32)
                Ts_last_truth = Ts_true.cpu().numpy().astype(np.float32)
                q_consis_last_truth = q_cons_true

        # === 能量守恒法：统一 h（以 i+1 时刻为基准） ===
        if save_all_h_eb:
            # 发热（W）
            Q_gen = Q_gen_const  # 若将来 q_vol 时变，可在此按 i 或 i+1 读取
            # 储能变化率（W）：对齐 t_{i+1} 的向后差分
            dTdt = (H.T[i+1] - H.T[i]) / max(H.dt[i], 1e-12)   # K/s
            dUdt = float((H.rho * H.cp * dTdt * H.Ms).sum() * dV)  # W

            Q_conv_truth = Q_gen - dUdt  # W

            # 分母 ∫(ΔT dA)：只在界面且 |ΔT|>=阈值
            Ts_np = H.T[i+1].astype(np.float32)
            dT_surf = Ts_np - H.T_inf.astype(np.float32)
            dT_thresh = float(getattr(args, "h_smooth_dt_thresh", 0.2))
            mask = (Mi_np>0) & (np.abs(dT_surf) >= dT_thresh)
            denom_area = float((dT_surf[mask] * A[mask]).sum())

            if denom_area <= 0:
                h_eb = 0.0
                denom_nonpos_count += 1
                Q_from_uniform = 0.0
            else:
                h_eb = Q_conv_truth / denom_area
                Q_from_uniform = h_eb * denom_area  # 直接用同一分母重构

            # 缓存
            h_eb_seq.append(np.float32(h_eb))
            Q_gen_seq.append(np.float32(Q_gen))
            dUdt_seq.append(np.float32(dUdt))
            Q_conv_truth_seq.append(np.float32(Q_conv_truth))
            Q_from_uniform_seq.append(np.float32(Q_from_uniform))
            if len(H.time) > i+1: time_eb.append(float(H.time[i+1]))

            # 统一 h 场：界面处=常数，非界面=0（避免占体）
            h_field = np.zeros_like(Ts_np, dtype=np.float32)
            h_field[Mi_np>0] = h_eb
            h_eb_field_seq.append(h_field)

        # 保存序列
        if args.rollout and (not args.teacher_forced): theta_roll[i+1] = y_pred.detach()
        if args.teacher_forced: theta_tf[i+1] = y_pred.detach()

        # 更新缓冲
        theta_last_pred = y_pred.detach()
        theta_prev = theta_next_true if args.teacher_forced else y_pred.detach()

    # ---- 汇总
    summary = {k: {
        "mean": float(np.mean(v)), "std": float(np.std(v)),
        "p50": float(np.percentile(v,50)), "p90": float(np.percentile(v,90))
    } for k,v in per_pair.items()}

    # ---- learn_h 的 h 场（最后一帧预测；与训练一致）
    h_artifacts = None
    if learn_h and args.save_h:
        dtau_last = torch.tensor([ H.dt[-1]*H.alpha/(H.L*H.L) ], dtype=torch.float32)
        dtau5 = dtau_last.view(1,1,1,1,1).expand(1,1,Nz,Ny,Nx)
        Tamb5=(H.T_amb/ H.dTref)*torch.ones(1,1,Nz,Ny,Nx)
        # feats = torch.cat([
        #     theta_last_pred[None,None,...], theta_last_pred[None,None,...],
        #     S5, Mi5, Ms5, nsurf5[:,0:1], nsurf5[:,1:2], nsurf5[:,2:3], dtau5,
        # ], dim=1)
        feats = torch.cat([
            theta_in[None, None, ...],
            S5, Mi5, Ms5,
            nsurf5[:, 0:1], nsurf5[:, 1:2], nsurf5[:, 2:3],
            dtau5, Tamb5
        ], dim=1)
        delta_h_raw = (hhead(feats) if hhead is not None else torch.zeros_like(theta_last_pred[None,None,...])) * Mi5
        h_pred_last = F.softplus(delta_h_raw) * Mi5                 # ⬅ 与训练一致
        h_artifacts = {
            "h_eff":               h_pred_last[0,0].cpu().numpy().astype(np.float32),
            "delta_h_raw_masked":  (delta_h_raw[0,0].cpu().numpy().astype(np.float32)),
            "Mi":                  (Mi5[0,0].cpu().numpy().astype(np.float32)),
        }

    # --- 堆叠 ALL（pred / truth）
    if save_all_pred and len(qn_list_pred) > 0:
        qn_all_pred = np.stack(qn_list_pred, axis=0)
        h_all_pred  = np.stack(h_list_pred,  axis=0)
        Ts_all_pred = np.stack(Ts_list_pred, axis=0)
        q_cons_arr_pred = np.asarray(q_cons_list_pred, dtype=np.float32)
        time_arr_pred = np.asarray(time_list_pred, dtype=np.float64) if len(time_list_pred)>0 else H.time[1:]
    if save_all_truth and len(qn_list_true) > 0:
        qn_all_true = np.stack(qn_list_true, axis=0)
        h_all_true  = np.stack(h_list_true,  axis=0)
        Ts_all_true = np.stack(Ts_list_true, axis=0)
        q_cons_arr_true = np.asarray(q_cons_list_true, dtype=np.float32)
        time_arr_true = np.asarray(time_list_true, dtype=np.float64) if len(time_list_true)>0 else H.time[1:]

    # ---- 写新 H5
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with h5py.File(args.out, "w") as fo:
        g_m = fo.create_group("metrics")
        for k, v in per_pair.items(): g_m.create_dataset(k, data=np.asarray(v, np.float64))
        g_sum = fo.create_group("metrics_summary")
        g_sum.attrs["summary_json"] = json.dumps(summary, ensure_ascii=False)

        # Kelvin 还原
        if args.rollout and (not args.teacher_forced):
            fo.create_dataset("theta_rollout", data=theta_roll.numpy().astype(np.float32))
            T_roll = theta_roll.numpy().astype(np.float32) * H.dTref + H.T_amb
            fo.create_dataset("T_rollout", data=T_roll)
        if args.teacher_forced:
            fo.create_dataset("theta_teacher_forced", data=theta_tf.numpy().astype(np.float32))
            T_tf = theta_tf.numpy().astype(np.float32) * H.dTref + H.T_amb
            fo.create_dataset("T_teacher_forced", data=T_tf)

        if h_artifacts is not None:
            g_h = fo.create_group("h_learned")
            for k, arr in h_artifacts.items(): g_h.create_dataset(k, data=arr)

        # 如果需要保存全时序的小网络 h（仅当 learn_h=true）
        if (h_pred_seq is not None) and (len(h_pred_seq) > 0):
            g_allh = fo.create_group("h_learned_all")
            g_allh.create_dataset("h_pred_Wm2K", data=np.stack(h_pred_seq, axis=0), compression="gzip")
            g_allh.create_dataset("time", data=H.time[1:])

        # 预测版：仅最后一帧（基于梯度的 h 与 q）
        if args.save_h_from_grad:
            g_qh = fo.create_group("h_from_grad_last")
            g_qh.create_dataset("q_n_Wm2", data=qn_last)
            g_qh.create_dataset("h_Wm2K",  data=h_last)
            g_qh.create_dataset("T_surf_K", data=Ts_last)
            g_qh.create_dataset("T_inf_K",  data=H.T_inf.astype(np.float32))
            g_qh.attrs["q_consistency_mean_abs"] = q_consis_last
            A_last = interface_area_weights(H.nsurf, H.dx, H.dy, H.dz, H.Mi)
            g_qh.create_dataset("area_weight", data=A_last.numpy().astype(np.float32))
            dT = (Ts_last - H.T_inf.astype(np.float32))
            mask = (H.Mi>0.5) & (np.abs(dT)>1e-6)
            Q_total = float((qn_last[mask] * A_last.numpy()[mask]).sum())
            denom  = float((dT[mask]      * A_last.numpy()[mask]).sum())
            h_eff_bar = Q_total/denom if denom>0 else 0.0
            g_qh.attrs["h_eff_bar_energy_weighted_Wm2K"] = h_eff_bar
            g_qh.attrs["Q_conv_total_W"] = Q_total

        # 预测版：全时序
        if args.save_h_from_grad_all and len(qn_list_pred) > 0:
            g_all = fo.create_group("h_from_grad_all")
            g_all.create_dataset("q_n_Wm2", data=qn_all_pred, compression="gzip")
            g_all.create_dataset("h_Wm2K",  data=h_all_pred,  compression="gzip")
            g_all.create_dataset("T_surf_K", data=Ts_all_pred, compression="gzip")
            g_all.create_dataset("T_inf_K",  data=H.T_inf.astype(np.float32))
            g_all.create_dataset("time",     data=time_arr_pred)
            g_all.create_dataset("q_consistency_mean_abs", data=q_cons_arr_pred)
            A_np = A
            g_all.create_dataset("area_weight", data=A_np)
            dT_all = (Ts_all_pred - H.T_inf.astype(np.float32)[None,...])
            mask = (H.Mi>0.5)[None,...] & (np.abs(dT_all)>1e-6)
            Q_total = ( (qn_all_pred * A_np[None,...]) * mask ).reshape(qn_all_pred.shape[0], -1).sum(axis=1)
            denom   = ( (dT_all     * A_np[None,...]) * mask ).reshape(h_all_pred.shape[0],  -1).sum(axis=1)
            h_eff_bar = np.divide(Q_total, denom, out=np.zeros_like(Q_total), where=denom>0)
            g_all.create_dataset("h_eff_bar_energy_weighted_Wm2K", data=h_eff_bar)
            g_all.create_dataset("Q_conv_total_W", data=Q_total)

        # 真值版：仅最后一帧
        if args.save_h_from_grad_truth and 'qn_last_truth' in locals():
            g_qh_t = fo.create_group("h_from_grad_truth_last")
            g_qh_t.create_dataset("q_n_Wm2", data=qn_last_truth)
            g_qh_t.create_dataset("h_Wm2K",  data=h_last_truth)
            g_qh_t.create_dataset("T_surf_K", data=Ts_last_truth)
            g_qh_t.create_dataset("T_inf_K",  data=H.T_inf.astype(np.float32))
            # 用真值 Bi 的一致性
            A_last = interface_area_weights(H.nsurf, H.dx, H.dy, H.dz, H.Mi)
            dT = (Ts_last_truth - H.T_inf.astype(np.float32))
            mask = (H.Mi>0.5) & (np.abs(dT)>1e-6)
            Q_total = float((qn_last_truth[mask] * A_last.numpy()[mask]).sum())
            denom  = float((dT[mask]           * A_last.numpy()[mask]).sum())
            h_eff_bar = Q_total/denom if denom>0 else 0.0
            g_qh_t.attrs["h_eff_bar_energy_weighted_Wm2K"] = h_eff_bar
            g_qh_t.attrs["Q_conv_total_W"] = Q_total

        # 真值版：全时序
        if args.save_h_from_grad_truth_all and len(qn_list_true) > 0:
            g_all_t = fo.create_group("h_from_grad_truth_all")
            g_all_t.create_dataset("q_n_Wm2", data=qn_all_true, compression="gzip")
            g_all_t.create_dataset("h_Wm2K",  data=h_all_true,  compression="gzip")
            g_all_t.create_dataset("T_surf_K", data=Ts_all_true, compression="gzip")
            g_all_t.create_dataset("T_inf_K",  data=H.T_inf.astype(np.float32))
            g_all_t.create_dataset("time",     data=time_arr_true)
            g_all_t.create_dataset("q_consistency_mean_abs", data=q_cons_arr_true)
            A_np = A
            g_all_t.create_dataset("area_weight", data=A_np)
            dT_all_t = (Ts_all_true - H.T_inf.astype(np.float32)[None,...])
            mask_t = (H.Mi>0.5)[None,...] & (np.abs(dT_all_t)>1e-6)
            Q_total_t = ( (qn_all_true * A_np[None,...]) * mask_t ).reshape(qn_all_true.shape[0], -1).sum(axis=1)
            denom_t   = ( (dT_all_t    * A_np[None,...]) * mask_t ).reshape(h_all_true.shape[0],  -1).sum(axis=1)
            h_eff_bar_t = np.divide(Q_total_t, denom_t, out=np.zeros_like(Q_total_t), where=denom_t>0)
            g_all_t.create_dataset("h_eff_bar_energy_weighted_Wm2K", data=h_eff_bar_t)
            g_all_t.create_dataset("Q_conv_total_W", data=Q_total_t)

        # === 真值版：全时序（全局统一 h，基于梯度）===
        if args.save_h_from_grad_truth_uniform_all and len(qn_list_true) > 0:
            qn_all_true = np.stack(qn_list_true, axis=0)   # (Nt-1,Z,Y,X)
            Ts_all_true = np.stack(Ts_list_true, axis=0)   # (Nt-1,Z,Y,X)
            time_arr_true = np.asarray(time_list_true, dtype=np.float64) if len(time_list_true)>0 else H.time[1:]
            Mi_np2 = H.Mi.astype(np.uint8)
            A_np = A
            dT_all = (Ts_all_true - H.T_inf.astype(np.float32)[None,...])
            dT_thresh = float(getattr(args, "h_smooth_dt_thresh", 0.2))
            mask = (Mi_np2>0)[None,...] & (np.abs(dT_all) >= dT_thresh)

            QA = ( (qn_all_true * A_np[None,...]) * mask ).reshape(qn_all_true.shape[0], -1).sum(axis=1)
            dTA = ( (dT_all     * A_np[None,...]) * mask ).reshape(dT_all.shape[0],     -1).sum(axis=1)
            h_uniform = np.divide(QA, dTA, out=np.zeros_like(QA, dtype=np.float32), where=dTA>0).astype(np.float32)  # (Nt-1,)

            h_field = np.zeros_like(qn_all_true, dtype=np.float32)
            iface = (Mi_np2>0)
            for t in range(h_field.shape[0]):
                h_field[t, iface] = h_uniform[t]

            Q_from_uniform = ( (h_field * dT_all * A_np[None,...]) * mask ).reshape(h_field.shape[0], -1).sum(axis=1)
            Q_err_rel = np.divide(np.abs(Q_from_uniform-QA), np.maximum(np.abs(QA), 1e-8))

            g_uni = fo.create_group("h_from_grad_truth_uniform_all")
            g_uni.create_dataset("h_uniform_scalar_Wm2K", data=h_uniform)
            g_uni.create_dataset("h_uniform_field_Wm2K", data=h_field, compression="gzip")
            g_uni.create_dataset("time", data=time_arr_true)
            g_uni.create_dataset("T_inf_K",  data=H.T_inf.astype(np.float32))
            g_uni.create_dataset("interface_mask", data=Mi_np2.astype(np.uint8))
            g_uni.create_dataset("area_weight", data=A_np)
            g_uni.create_dataset("Q_conv_total_W_truth_from_q", data=QA.astype(np.float32))
            g_uni.create_dataset("Q_conv_total_W_from_uniform_h", data=Q_from_uniform.astype(np.float32))
            g_uni.create_dataset("Q_relative_error", data=Q_err_rel.astype(np.float32))
            g_uni.attrs["note"] = (
                "Uniform h per frame from truth (solid-side gradient only); "
                "h_uniform(t) = sum(q*A)/sum(dT*A) over interface with |dT|>=threshold."
            )
            g_uni.attrs["dT_threshold_K"] = dT_thresh

        # === 能量守恒法：全时序统一 h ===
        if args.save_h_uniform_energy_balance_all and len(h_eb_seq) > 0:
            h_eb_seq_arr = np.asarray(h_eb_seq, dtype=np.float32)                # (Nt-1,)
            h_eb_field_seq_arr = np.stack(h_eb_field_seq, axis=0)                # (Nt-1,Z,Y,X)
            time_eb_arr = np.asarray(time_eb, dtype=np.float64) if len(time_eb)>0 else H.time[1:]
            Q_gen_arr = np.asarray(Q_gen_seq, dtype=np.float32)
            dUdt_arr  = np.asarray(dUdt_seq,  dtype=np.float32)
            Q_conv_truth_arr   = np.asarray(Q_conv_truth_seq, dtype=np.float32)
            Q_from_uniform_arr = np.asarray(Q_from_uniform_seq, dtype=np.float32)
            rel_err = np.divide(np.abs(Q_from_uniform_arr - Q_conv_truth_arr),
                                np.maximum(np.abs(Q_conv_truth_arr), 1e-8))

            g_eb = fo.create_group("h_uniform_energy_balance_all")
            g_eb.create_dataset("h_uniform_scalar_Wm2K", data=h_eb_seq_arr)
            g_eb.create_dataset("h_uniform_field_Wm2K",  data=h_eb_field_seq_arr, compression="gzip")
            g_eb.create_dataset("time", data=time_eb_arr)
            g_eb.create_dataset("area_weight", data=A)
            g_eb.create_dataset("T_inf_K", data=H.T_inf.astype(np.float32))
            g_eb.create_dataset("interface_mask", data=Mi_np.astype(np.uint8))
            g_eb.create_dataset("Q_gen_W", data=Q_gen_arr)
            g_eb.create_dataset("dUdt_W",  data=dUdt_arr)
            g_eb.create_dataset("Q_conv_truth_W", data=Q_conv_truth_arr)
            g_eb.create_dataset("Q_conv_from_uniform_h_W", data=Q_from_uniform_arr)
            g_eb.create_dataset("Q_relative_error", data=rel_err)
            g_eb.attrs["note"] = (
                "统一 h（能量守恒法）：h = (Q_gen - dU/dt) / ∫(ΔT dA)，界面 ΔT 设阈值过滤；"
                "本路径不依赖表面梯度，数值稳健。"
            )
            g_eb.attrs["dT_threshold_K"] = float(getattr(args, "h_smooth_dt_thresh", 0.2))
            g_eb.attrs["denom_nonpos_count"] = int(denom_nonpos_count)

        fo.attrs["src_h5"] = os.path.abspath(args.data)
        fo.attrs["ckpt"]  = os.path.abspath(args.ckpt)
        fo.attrs["note"]  = (
            "Teacher-forced outputs are aligned: index 0 = ground-truth t0; "
            "index i+1 = prediction for original frame i+1. "
            "Kelvin reconstruction uses T_amb."
        )

    print("[ok] eval done. Summary:")
    for k, s in summary.items():
        print(f"  {k:10s}: mean={s['mean']:.4e}, p90={s['p90']:.4e}")

    # 控制台打印小网络 h 统计
    if args.print_h_stats and learn_h and len(h_stats) > 0:
        print("\n[h_pred stats on interface | frame -> (min, mean, max, area-weighted mean)  (W/m^2/K)]")
        for t_idx, hmin, hmax, hmean, haw in h_stats:
            print(f"  t[{t_idx:03d}]: {hmin:.3f}, {hmean:.3f}, {hmax:.3f}, {haw:.3f}")

    H.close()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out",  type=str, required=True)
    ap.add_argument("--rollout", action="store_true")
    ap.add_argument("--save_h", action="store_true")
    ap.add_argument("--teacher_forced", action="store_true",
                    help="逐步用真值帧作为下一步输入；输出对齐版(Nt)：索引0=真值t0，索引i+1=预测t(i+1)")
    ap.add_argument("--save_h_from_grad", action="store_true",
                    help="基于预测场，保存最后一帧的 q_n 与 h")
    ap.add_argument("--save_h_from_grad_all", action="store_true",
                    help="基于预测场，保存全时序的 q_n 与 h（文件较大）")
    ap.add_argument("--save_h_from_grad_truth", action="store_true",
                    help="基于真值场，保存最后一帧的 q_n 与 h")
    ap.add_argument("--save_h_from_grad_truth_all", action="store_true",
                    help="基于真值场，保存全时序的 q_n 与 h（文件较大）")
    ap.add_argument("--save_h_from_grad_truth_all_smooth", action="store_true",
                    help="基于真值(固体侧单边梯度)的 h 做空间Laplacian平滑+时间EMA去噪，并保存全时序结果（如已实现）")
    ap.add_argument("--save_h_from_grad_truth_uniform_all", action="store_true",
                    help="基于真值(固体侧单边梯度)的 q 与 ΔT 做能量等效平均，得到每一帧单一常数 h，并写出全时序统一 h 场与标量序列")
    ap.add_argument("--save_h_uniform_energy_balance_all", action="store_true",
                    help="全时序：用 Q_gen - dU/dt 与 ∫(ΔT dA) 得到统一 h（能量守恒法），稳健且与 L 无关")
    ap.add_argument("--h_smooth_dt_thresh", type=float, default=0.2,
                    help="对 ∆T=Ts-Tinf 的阈值（K）：仅在 |∆T|>=阈值 的界面参与统一 h 分母/平滑等处理")

    # ★ 新增：控制台打印小网络 h 统计
    ap.add_argument("--print_h_stats", action="store_true",
                    help="控制台打印小网络预测的 h（界面上的 min/mean/max/area-weighted mean）")
    # ★ 新增：保存全时序小网络 h（界面非零）
    ap.add_argument("--save_h_pred_all", action="store_true",
                    help="保存小网络预测的 h 全时序到 h_learned_all/h_pred_Wm2K（仅界面非零）")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
