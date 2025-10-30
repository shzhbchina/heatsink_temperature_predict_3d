# -*- coding: utf-8 -*-
"""
eval_fno_h.py  (适配“Δθ监督 + ctx含log(dtau) + 频域padding”的新训练脚本)

用法示例：
"case_0002_L152_W59_Hb18_F6.h5"
python src/eval_heatsink_fno_pino_test6.py \
  --data dat_eval/case_0991_L324_W220_Hb17_F15.h5 \
  --ckpt model_param/heatsink_fno_pino_learnh/ckpt_ep60.pt \
  --out model_param/eval/run1_eval.h5 \
  --mode both --copy_all \
  --rollout_from 0 --rollout_steps -1 \
  --device cpu
"""

import os, math, argparse
import h5py, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pydevd_pycharm
    pydevd_pycharm.settrace(host='host.docker.internal', port=5678,
                            stdout_to_server=True, stderr_to_server=True, suspend=True)
except Exception:
    pass

# =================== 与训练保持一致的无量纲基准 ===================
T_BASE = 298.0   # K
DT_BASE = 30.0   # K

# ------------------------- 基本工具 -------------------------
def _to_numpy(x):
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return np.asarray(x)

def _safe_log(x):
    return math.log(max(float(x), 1e-12))

def _unpack_deltas(d):
    if isinstance(d, (tuple, list, np.ndarray)):
        return float(d[0]), float(d[1]), float(d[2])
    if torch.is_tensor(d):
        v = d.detach().view(-1)
        return float(v[0].item()), float(v[1].item()), float(v[2].item())
    raise TypeError(f"Unsupported deltas type: {type(d)}")

# ------------------------- 数据读取类（与训练一致的关键字段） -------------------------
class HeatsinkH5:
    def __init__(self, path):
        self.path = path
        self.f = h5py.File(path, 'r')

        # 主数据
        self.T = self.f['T_grid_shadow'][:]    # (Nt,Nz,Ny,Nx)
        Nt, Nz, Ny, Nx = self.T.shape
        self.Nt, self.Nz, self.Ny, self.Nx = Nt, Nz, Ny, Nx

        # 时间步长：优先用 time_dt，没有则用 diff(time)
        if 'time_dt' in self.f:
            self.dt = self.f['time_dt'][:]  # (Nt-1,)
        else:
            assert 'time' in self.f, "H5 既无 time_dt 也无 time"
            t = self.f['time'][:]
            self.dt = np.diff(t)

        # 网格参数（padding 后网格）
        gp = self.f['grid_padded']
        spacing = gp['spacing'][:]   # [dx,dy,dz]
        dims    = gp['dims'][:]      # [Nz,Ny,Nx]
        self.dx, self.dy, self.dz = float(spacing[0]), float(spacing[1]), float(spacing[2])
        assert (int(dims[0])==Nz and int(dims[1])==Ny and int(dims[2])==Nx)
        self.Lz_abs = self.dz * self.Nz
        self.Ly_abs = self.dy * self.Ny
        self.Lx_abs = self.dx * self.Nx

        # 掩膜/法向/BC
        self.Ms = self.f['mask_solid'][:].astype(np.float32)
        self.Mi = self.f['mask_interface'][:].astype(np.float32)
        self.nsurf = self.f['normal_on_surface'][:].astype(np.float32)  # (3,Nz,Ny,Nx)
        bc = self.f['bc/robin']
        self.h_init = bc['h_init'][:].astype(np.float32)
        self.T_inf  = bc['T_inf'][:].astype(np.float32)

        # 源项
        if 'sources' in self.f:
            self.q_vol = self.f['sources/q_vol'][:].astype(np.float32)
        else:
            self.q_vol = np.zeros((Nz, Ny, Nx), np.float32)

        # 常数与尺度
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

        sc  = self.f['scales']
        file_alpha = float(sc['alpha'][()]) if 'alpha' in sc else None
        self.L      = float(sc['L'][()])
        self.dTref  = float(sc['dT_ref'][()]) if 'dT_ref' in sc else DT_BASE
        self.qref   = float(sc['q_ref'][()]) if 'q_ref' in sc else None

        if file_alpha is not None:
            self.alpha = file_alpha
        else:
            if (self.rho is not None) and (self.cp is not None):
                self.alpha = self.k / (self.rho * self.cp)
            else:
                raise ValueError("[data] alpha 缺失且无法由 k/(rho*cp) 推出。")

        # 无量纲网格步长（按训练时的顺序）
        self.dx_hat = self.dx / self.L
        self.dy_hat = self.dy / self.L
        self.dz_hat = self.dz / self.L

        # 9 维全局上下文
        g_mag = float(np.linalg.norm(self.g_vec))
        g_hat = self.g_vec / max(g_mag, 1e-12)
        theta_inf_hat = (self.T_inf - T_BASE) / DT_BASE
        if np.any(self.Mi > 0.5): theta_inf_mean_hat = float(theta_inf_hat[self.Mi > 0.5].mean())
        else:                      theta_inf_mean_hat = float(theta_inf_hat.mean())

        self.ctx_glb = np.array([
            _safe_log(self.Lz_abs), _safe_log(self.Ly_abs), _safe_log(self.Lx_abs),
            float(g_hat[0]), float(g_hat[1]), float(g_hat[2]),
            _safe_log(g_mag), theta_inf_mean_hat, _safe_log(self.alpha)
        ], dtype=np.float32)

        # 源项无量纲（与训练一致）
        if (self.rho is None) or (self.cp is None):
            raise ValueError("[data] 计算 S_nd 需要 rho 与 cp。")
        coeff = (self.L ** 2) / (self.alpha * self.rho * self.cp * DT_BASE)
        self.S_nd = (self.q_vol * coeff).astype(np.float32)

        # 可选：逐时刻 h 真值
        self.h_truth = None
        if 'h_uniform_from_sources_all' in self.f and 'h_uniform_field_Wm2K' in self.f['h_uniform_from_sources_all']:
            self.h_truth = self.f['h_uniform_from_sources_all']['h_uniform_field_Wm2K'][:].astype(np.float32)
        elif 'h_from_grad_truth_uniform_all' in self.f and 'h_uniform_field_Wm2K' in self.f['h_from_grad_truth_uniform_all']:
            self.h_truth = self.f['h_from_grad_truth_uniform_all']['h_uniform_field_Wm2K'][:].astype(np.float32)

    def close(self):
        try: self.f.close()
        except: pass

# ------------------------- 频域padding助手（与训练一致） -------------------------
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

# ------------------------- 模型（与训练保持一致：含频域padding） -------------------------
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

class FNO3D(nn.Module):
    def __init__(self, in_c=3, width=24, modes=(12,12,12), layers=4,
                 add_coords=True, fourier_k=8, use_local=True,
                 gn_groups=1, residual_scale=0.5, dropout=0.0,
                 context_dim: int = 0,
                 spec_pad_type: str = 'reflect', spec_pad=(8,8,8)):
        super().__init__()
        self.add_coords = add_coords
        self.fourier_k = fourier_k
        self.use_local = use_local
        self.context_dim = int(context_dim)
        self.spec_pad_type = spec_pad_type
        self.spec_pad = tuple(int(v) for v in spec_pad)

        extra_c = 0
        if add_coords: extra_c = 3 + 6 * fourier_k
        self.lift = nn.Conv3d(in_c + extra_c, width, 1)
        mz, my, mx = modes
        pz, py, px = self.spec_pad
        self.specs = nn.ModuleList([
            SpectralConv3d(width, width, mz, my, mx,
                           pad_type=self.spec_pad_type, pad_z=pz, pad_y=py, pad_x=px)
            for _ in range(layers)
        ])
        self.ws = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(layers)])
        self.locals = nn.ModuleList([nn.Conv3d(width, width, 3, padding=1, groups=width) for _ in range(layers)]) if use_local else None
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
            b0 = math.log(p / (1 - p))
            nn.init.constant_(self.mlp[-1].bias, b0)
            nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, feats, ctx_vec=None):
        B, _, Z, Y, X = feats.shape
        z = self.net(feats)                                     # (B,K,Z,Y,X)
        mask = feats[:, self.mask_idx:self.mask_idx+1].float()  # (B,1,Z,Y,X)
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
        return h_scalar.view(B, 1, 1, 1, 1) * mask              # (B,1,Z,Y,X)

# ------------------------- ckpt I/O -------------------------
def load_ckpt_for_eval(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})
    width  = int(args.get("width", 24))
    layers = int(args.get("layers", 4))
    mz     = int(args.get("mz", 12)); my = int(args.get("my", 12)); mx = int(args.get("mx", 12))
    rk2    = bool(args.get("rk2", False))

    # 频域 padding 参数（与训练一致）
    spec_pad_type = str(args.get("spec_pad_type", "reflect"))
    spz = int(args.get("spec_pad_z", 8)); spy = int(args.get("spec_pad_y", 8)); spx = int(args.get("spec_pad_x", 8))

    # 模型（context_dim=10：9维ctx_glb + 1维log(dtau)）
    model = FNO3D(in_c=3, width=width, modes=(mz,my,mx), layers=layers, context_dim=10,
                  spec_pad_type=spec_pad_type, spec_pad=(spz, spy, spx)).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[warn] model state: missing={missing}, unexpected={unexpected}")

    # hhead（如存在）
    hhead = None
    if "hhead" in ckpt:
        h_width  = int(args.get("h_width", 16))
        h_layers = int(args.get("h_layers", 2))
        hhead = HHead(in_c=7, width=h_width, layers=h_layers, ctx_dim=10).to(device)
        mh, uh = hhead.load_state_dict(ckpt["hhead"], strict=False)
        if mh or uh:
            print(f"[warn] hhead state: missing={mh}, unexpected={uh}")

    return model.eval(), hhead.eval() if hhead is not None else None, rk2, args

# ------------------------- 评估核心 -------------------------
@torch.inference_mode()
def step_one(model, theta_t, Ms, S_nd, ctx_glb, dtau_scalar, rk2=False, device='cpu'):
    """
    单步推进 θ（注意：模型输出的是 Δθ，不再乘 dtau）
    theta_{t+1} = theta_t + Δθ
    """
    B, _, Z, Y, X = theta_t.shape
    x = torch.cat([theta_t, Ms, S_nd], dim=1)  # (B,3,Z,Y,X)
    dtau = torch.tensor([dtau_scalar], dtype=torch.float32, device=device).view(B,1)
    ctx = torch.cat([ctx_glb, torch.log(dtau + 1e-12)], dim=1)  # (B,10)

    if rk2:
        delta1 = model(x, ctx)
        theta_tilde = theta_t + delta1
        x2 = torch.cat([theta_tilde, Ms, S_nd], dim=1)
        delta2 = model(x2, ctx)
        delta  = 0.5 * (delta1 + delta2)
    else:
        delta = model(x, ctx)

    theta_next = theta_t + delta
    return theta_next

@torch.inference_mode()
def predict_h_field(hhead, theta_t, Ms, Mi, nsurf, S_nd, ctx_glb, dtau_scalar, device='cpu'):
    """
    计算一步对应的 h 预测场（若 hhead 存在）
    """
    if hhead is None: return None
    B = theta_t.size(0)
    feats = torch.cat([
        theta_t,         # (B,1,...)
        S_nd,            # (B,1,...)
        Mi,              # (B,1,...)
        Ms,              # (B,1,...)
        nsurf[:,0:1], nsurf[:,1:2], nsurf[:,2:3]
    ], dim=1)           # (B,7,Z,Y,X)
    dtau = torch.tensor([dtau_scalar], dtype=torch.float32, device=device).view(B,1)
    ctx = torch.cat([ctx_glb, torch.log(dtau + 1e-12)], dim=1)  # (B,10)
    return hhead(feats, ctx_vec=ctx)  # (B,1,Z,Y,X)

def _alloc_like(Nt, Nz, Ny, Nx, with_theta=False, with_h=False):
    out = {"T_pred": np.zeros((Nt, Nz, Ny, Nx), np.float32)}
    if with_theta:
        out["theta_pred"] = np.zeros((Nt, Nz, Ny, Nx), np.float32)
    if with_h:
        out["h_pred"] = np.zeros((Nt, Nz, Ny, Nx), np.float32)
        out["h_scalar"] = np.zeros((Nt,), np.float32)
    return out

def _metrics_T(T_pred, T_true, Ms):
    m = (Ms > 0.5)
    diff = (T_pred - T_true)[m]
    mae = float(np.mean(np.abs(diff))) if diff.size>0 else 0.0
    rmse = float(np.sqrt(np.mean(diff*diff))) if diff.size>0 else 0.0
    return mae, rmse

def _metrics_h(h_pred, h_true, Mi):
    m = (Mi > 0.5)
    diff = (h_pred - h_true)[m]
    if diff.size == 0: return 0.0, 0.0
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff*diff)))
    return mae, rmse

@torch.inference_mode()
def evaluate_modes(H: HeatsinkH5, model: FNO3D, hhead: HHead, rk2: bool,
                   device='cpu', do_teacher=True, do_rollout=True,
                   rollout_from=0, rollout_steps=-1):
    """
    返回两个 dict（可能含空）：teacher, rollout
    dict 包含：T_pred, theta_pred(可选), h_pred(可选), h_scalar(可选), mae_T, rmse_T, (可选) mae_h, rmse_h
    """
    Nt, Nz, Ny, Nx = H.Nt, H.Nz, H.Ny, H.Nx
    Ms = torch.from_numpy(H.Ms[None, None, ...]).to(device)      # (1,1,Z,Y,X)
    Mi = torch.from_numpy(H.Mi[None, None, ...]).to(device)
    S_nd = torch.from_numpy(H.S_nd[None, None, ...]).to(device)
    nsurf = torch.from_numpy(H.nsurf[None, ...]).to(device)      # (1,3,Z,Y,X)
    ctx_glb = torch.from_numpy(H.ctx_glb[None, :]).to(device)    # (1,9)

    # 预计算 dtau 序列（仅用于 ctx 的 log(dtau)）
    dt = H.dt.astype(np.float64)
    dtau_arr = (dt * (H.alpha / (H.L * H.L))).astype(np.float64)  # (Nt-1,)

    results_teacher = {}
    results_rollout = {}

    # ---------- Teacher Forcing ----------
    if do_teacher:
        out = _alloc_like(Nt, Nz, Ny, Nx, with_theta=True, with_h=(hhead is not None))
        mae_T = np.zeros((Nt,), np.float32); rmse_T = np.zeros((Nt,), np.float32)
        mae_h = np.zeros((Nt,), np.float32); rmse_h = np.zeros((Nt,), np.float32)

        theta = (H.T - T_BASE) / DT_BASE
        out["theta_pred"][0] = theta[0].astype(np.float32)
        out["T_pred"][0] = H.T[0].astype(np.float32)

        for t in range(Nt-1):
            theta_t = torch.from_numpy(theta[t:t+1, ...]).to(device).unsqueeze(1)   # (1,1,Z,Y,X)
            dtau = float(dtau_arr[t])
            theta_next = step_one(model, theta_t, Ms, S_nd, ctx_glb, dtau, rk2=rk2, device=device)
            theta_next_np = _to_numpy(theta_next[0,0])
            T_next_np = theta_next_np * DT_BASE + T_BASE

            out["theta_pred"][t+1] = theta_next_np.astype(np.float32)
            out["T_pred"][t+1] = T_next_np.astype(np.float32)

            # h 预测
            if hhead is not None:
                h_field = predict_h_field(hhead, theta_t, Ms, Mi, nsurf, S_nd, ctx_glb, dtau, device=device)
                h_np = _to_numpy(h_field[0,0]).astype(np.float32)
                out["h_pred"][t] = h_np
                m = (H.Mi > 0.5)
                out["h_scalar"][t] = float(h_np[m].mean()) if np.any(m) else 0.0

            # 误差（对 T）
            mae, rmse = _metrics_T(out["T_pred"][t+1], H.T[t+1], H.Ms)
            mae_T[t+1], rmse_T[t+1] = mae, rmse

            # h 误差（如有真值）
            if (hhead is not None) and (H.h_truth is not None):
                mae_h[t], rmse_h[t] = _metrics_h(out["h_pred"][t], H.h_truth[t], H.Mi)

        results_teacher = {
            **out,
            "mae_T": mae_T, "rmse_T": rmse_T
        }
        if hhead is not None:
            results_teacher["mae_h"] = mae_h
            results_teacher["rmse_h"] = rmse_h

    # ---------- Rollout ----------
    if do_rollout:
        if rollout_steps < 0:
            rollout_steps = (Nt-1) - rollout_from
        rollout_steps = int(max(0, min(rollout_steps, Nt-1 - rollout_from)))
        start = int(rollout_from)
        out = _alloc_like(Nt, Nz, Ny, Nx, with_theta=True, with_h=(hhead is not None))
        mae_T = np.zeros((Nt,), np.float32); rmse_T = np.zeros((Nt,), np.float32)
        mae_h = np.zeros((Nt,), np.float32); rmse_h = np.zeros((Nt,), np.float32)

        theta_true = (H.T - T_BASE) / DT_BASE
        # 初始化：真值到 start，预测从 start 开始
        out["theta_pred"][:start+1] = theta_true[:start+1].astype(np.float32)
        out["T_pred"][:start+1] = H.T[:start+1].astype(np.float32)

        theta_curr = torch.from_numpy(theta_true[start:start+1]).to(device).unsqueeze(1)  # (1,1,...)
        for k in range(rollout_steps):
            t = start + k
            dtau = float(dtau_arr[t])
            # 自回滚：用上一步预测作为输入
            theta_next = step_one(model, theta_curr, Ms, S_nd, ctx_glb, dtau, rk2=rk2, device=device)
            theta_next_np = _to_numpy(theta_next[0,0])
            T_next_np = theta_next_np * DT_BASE + T_BASE
            out["theta_pred"][t+1] = theta_next_np.astype(np.float32)
            out["T_pred"][t+1] = T_next_np.astype(np.float32)

            # h 预测（用当前输入 theta_curr）
            if hhead is not None:
                h_field = predict_h_field(hhead, theta_curr, Ms, Mi, nsurf, S_nd, ctx_glb, dtau, device=device)
                h_np = _to_numpy(h_field[0,0]).astype(np.float32)
                out["h_pred"][t] = h_np
                m = (H.Mi > 0.5)
                out["h_scalar"][t] = float(h_np[m].mean()) if np.any(m) else 0.0

            # 误差
            mae, rmse = _metrics_T(out["T_pred"][t+1], H.T[t+1], H.Ms)
            mae_T[t+1], rmse_T[t+1] = mae, rmse

            # 下一步
            theta_curr = theta_next

            # h 真值对比（如有）
            if (hhead is not None) and (H.h_truth is not None):
                mae_h[t], rmse_h[t] = _metrics_h(out["h_pred"][t], H.h_truth[t], H.Mi)

        results_rollout = {
            **out,
            "mae_T": mae_T, "rmse_T": rmse_T,
            "rollout_from": start, "rollout_steps": rollout_steps
        }
        if hhead is not None:
            results_rollout["mae_h"] = mae_h
            results_rollout["rmse_h"] = rmse_h

    return results_teacher, results_rollout

# ------------------------- 写出 H5 -------------------------
def _copy_all(src_h5: h5py.File, dst_h5: h5py.File):
    for name in src_h5.keys():
        if name not in dst_h5:
            src_h5.copy(name, dst_h5, name=name)

def _write_group(g, name, arr, comp=4):
    if arr is None: return
    if name in g: del g[name]
    g.create_dataset(name, data=arr, compression="gzip", compression_opts=comp, shuffle=True)

def save_eval_h5(in_path, out_path, teacher_res, rollout_res, copy_all=False, meta_note=""):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with h5py.File(in_path, "r") as fin, h5py.File(out_path, "w") as fo:
        fo.attrs["src_h5"] = os.path.abspath(in_path)
        fo.attrs["note"] = ("FNO+HHead evaluation results. " + meta_note).strip()

        if copy_all:
            _copy_all(fin, fo)

        ge = fo.create_group("eval")
        # Teacher forcing
        if teacher_res:
            gt = ge.create_group("teacher_forcing")
            for k in ("T_pred", "theta_pred", "h_pred", "h_scalar", "mae_T", "rmse_T", "mae_h", "rmse_h"):
                if k in teacher_res and teacher_res[k] is not None:
                    _write_group(gt, k, teacher_res[k])
        # Rollout
        if rollout_res:
            gr = ge.create_group("rollout")
            for k in ("T_pred", "theta_pred", "h_pred", "h_scalar", "mae_T", "rmse_T", "mae_h", "rmse_h"):
                if k in rollout_res and rollout_res[k] is not None:
                    _write_group(gr, k, rollout_res[k])
            gr.attrs["rollout_from"] = int(rollout_res.get("rollout_from", 0))
            gr.attrs["rollout_steps"] = int(rollout_res.get("rollout_steps", 0))
    print(f"[OK] eval H5 written: {out_path}")

# ------------------------- CLI -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="输入 H5 路径")
    ap.add_argument("--ckpt", type=str, required=True, help="训练 ckpt (.pt)")
    ap.add_argument("--out",  type=str, required=True, help="输出评估 H5 路径")
    ap.add_argument("--mode", type=str, choices=["both","teacher","rollout"], default="both")
    ap.add_argument("--rollout_from", type=int, default=0)
    ap.add_argument("--rollout_steps", type=int, default=-1, help="-1 表示滚到序列末尾")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--copy_all", action="store_true", help="是否复制原 H5 的全部内容到输出，便于对照")
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")

    # 数据与模型
    H = HeatsinkH5(args.data)
    model, hhead, rk2, train_args = load_ckpt_for_eval(args.ckpt, device=device)
    model.eval()
    if hhead is not None: hhead.eval()

    do_teacher = (args.mode in ("both","teacher"))
    do_rollout = (args.mode in ("both","rollout"))

    teacher_res, rollout_res = evaluate_modes(
        H, model, hhead, rk2, device=device,
        do_teacher=do_teacher, do_rollout=do_rollout,
        rollout_from=args.rollout_from, rollout_steps=args.rollout_steps
    )

    meta_note = f"mode={args.mode}; rk2={rk2}; device={device}; ckpt={os.path.basename(args.ckpt)}"
    save_eval_h5(H.path, args.out, teacher_res, rollout_res, copy_all=args.copy_all, meta_note=meta_note)
    H.close()

if __name__ == "__main__":
    main()
