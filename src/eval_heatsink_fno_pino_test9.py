# -*- coding: utf-8 -*-
"""
eval_heatsink_fno_pino_h_14_cuda.py

- 与 train_fno_pino_h_14_cuda.py 中的 FNO3D / MultiScaleSpectralBlock / HHead 结构完全对齐
- ckpt 加载支持 remap（specs.L.weight -> specs.L.specs.0.weight），避免多尺度 uFNO 结构变更导致 silent miss
- step_one / predict_h_field 中 log(dtau) 做 clamp[-20, 20]，和训练保持一致
- 支持：
    * Teacher forcing 逐步评估
    * Rollout（从指定帧开始逐步预测后续）
    * TwoFrame 跨时间两帧一步预测
    * 绝对/相对误差统计（仅 solid、可排除边界外圈、跳过第 0 帧）

使用示例（单文件）：
case_0992_L352_W88_Hb8_F10.h5 case_0994_L169_W179_Hb20_F12.h5
case_0005_L41_W55_Hb10_F5.h5
python src/eval_heatsink_fno_pino_test9.py \
  --data dat_eval/case_0994_L169_W179_Hb20_F12.h5 \
  --ckpt model_param/heatsink_fno_pino_learnh_4ls/ckpt_ep69.pt \
  --out  model_param/eval/run1_eval_0994_4.h5 \
  --mode rollout --rollout_from 0 --rollout_steps -1 \
  --device cuda:0 \
  --copy_all \
  --calc-err --exclude-border 1 --save-err-fields

使用示例（批量目录）：

python src/eval_heatsink_fno_pino_test9.py \
  --data_glob "dat_eval/*.h5" \
  --out_dir  "model_param/eval_batch/run1" \
  --ckpt model_param/heatsink_fno_pino_learnh_8ls/ckpt_ep30(rk2).pt \
  --mode rollout --rollout_from 0 --rollout_steps -1 \
  --device cuda:0 \
  --copy_all \
  --calc-err --exclude-border 1 --save-err-fields

````````````````````````````````

# 整个文件夹批量评估：
python src/eval_heatsink_fno_pino_test9.py \
  --data dat_test \
  --ckpt model_param/heatsink_fno_pino_learnh_4ls/ckpt_ep120.pt \
  --out_dir model_param/eval_batch/run1 \
  --mode rollout --rollout_from 0 --rollout_steps -1 \
  --device cuda:0 \
  --copy_all \
  --calc-err --exclude-border 1 --save-err-fields

"""

import os, math, argparse, random
import h5py, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pydevd_pycharm
    pydevd_pycharm.settrace(host='host.docker.internal', port=5678,
                            stdout_to_server=True, stderr_to_server=True, suspend=False)
except Exception:
    pass

# =================== 无量纲基准（与训练一致） ===================
T_BASE = 298.0   # K
DT_BASE = 30.0   # K

# ------------------------- 小工具 -------------------------
def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
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

def _fft_reflect_pad3d(x: torch.Tensor, padz: int, pady: int, padx: int, mode: str):
    """
    与 train_fno_pino_h_14_cuda.py 中实现保持一致。
    """
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

# ------------------------- 数据读取类（与训练同一物理定义） -------------------------
class HeatsinkH5:
    """
    推理用的 H5 读取类，与训练脚本中的 HeatsinkH5 在物理含义上对齐：
    - T_grid_shadow -> T
    - time_dt / time -> dt
    - grid_padded/spacing, dims -> dx,dy,dz, Nz,Ny,Nx
    - 掩膜 Ms, Mi，法向 nsurf，Robin 边界 h_init, T_inf
    - 常数 k, rho, cp, g, scales 中 L, alpha, q_ref, dT_ref
    - 源项无量纲化 S_nd
    - 9 维 ctx_glb
    """
    def __init__(self, path):
        self.path = path
        self.f = h5py.File(path, 'r')

        # 主数据：温度序列
        self.T = self.f['T_grid_shadow'][:]    # (Nt,Nz,Ny,Nx)
        Nt, Nz, Ny, Nx = self.T.shape
        self.Nt, self.Nz, self.Ny, self.Nx = Nt, Nz, Ny, Nx

        # 时间步长
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
        assert (int(dims[0]) == Nz and int(dims[1]) == Ny and int(dims[2]) == Nx)
        self.Lz_abs = self.dz * self.Nz
        self.Ly_abs = self.dy * self.Ny
        self.Lx_abs = self.dx * self.Nx

        # 掩膜 / 法向 / 边界条件
        self.Ms = self.f['mask_solid'][:].astype(np.float32)
        self.Mi = self.f['mask_interface'][:].astype(np.float32)
        self.nsurf = self.f['normal_on_surface'][:].astype(np.float32)  # (3,Nz,Ny,Nx)
        bc = self.f['bc/robin']
        self.h_init = bc['h_init'][:].astype(np.float32)
        self.T_inf  = bc['T_inf'][:].astype(np.float32)

        # 源项 q_vol
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
        # 这里的 dT_ref 就是我们用来做“温升标尺”的量，≈ T_ref - T_inf 或 T_ref - T_amb
        self.dTref  = float(sc['dT_ref'][()]) if 'dT_ref' in sc else DT_BASE
        self.qref   = float(sc['q_ref'][()]) if 'q_ref' in sc else None

        if file_alpha is not None:
            self.alpha = file_alpha
        else:
            if (self.rho is not None) and (self.cp is not None):
                self.alpha = self.k / (self.rho * self.cp)
            else:
                raise ValueError("[data] alpha 缺失且无法由 k/(rho*cp) 推出。")

        # 无量纲网格步长
        self.dx_hat = self.dx / self.L
        self.dy_hat = self.dy / self.L
        self.dz_hat = self.dz / self.L

        # 9 维全局上下文，与训练一致
        g_mag = float(np.linalg.norm(self.g_vec))
        g_hat = self.g_vec / max(g_mag, 1e-12)
        theta_inf_hat = (self.T_inf - T_BASE) / DT_BASE
        if np.any(self.Mi > 0.5):
            theta_inf_mean_hat = float(theta_inf_hat[self.Mi > 0.5].mean())
        else:
            theta_inf_mean_hat = float(theta_inf_hat.mean())

        self.ctx_glb = np.array([
            _safe_log(self.Lz_abs), _safe_log(self.Ly_abs), _safe_log(self.Lx_abs),
            float(g_hat[0]), float(g_hat[1]), float(g_hat[2]),
            _safe_log(g_mag), theta_inf_mean_hat, _safe_log(self.alpha)
        ], dtype=np.float32)

        # 源项无量纲
        if (self.rho is None) or (self.cp is None):
            raise ValueError("[data] 计算 S_nd 需要 rho 与 cp。")
        coeff = (self.L ** 2) / (self.alpha * self.rho * self.cp * DT_BASE)
        self.S_nd = (self.q_vol * coeff).astype(np.float32)

        # 可选：逐时刻 h 真值（如果有做过反算）
        self.h_truth = None
        if 'h_uniform_from_sources_all' in self.f and 'h_uniform_field_Wm2K' in self.f['h_uniform_from_sources_all']:
            self.h_truth = self.f['h_uniform_from_sources_all']['h_uniform_field_Wm2K'][:].astype(np.float32)
        elif 'h_from_grad_truth_uniform_all' in self.f and 'h_uniform_field_Wm2K' in self.f['h_from_grad_truth_uniform_all']:
            self.h_truth = self.f['h_from_grad_truth_uniform_all']['h_uniform_field_Wm2K'][:].astype(np.float32)

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

# ------------------------- 频域层 & uFNO（结构与训练一模一样） -------------------------
class SpectralConv3d(nn.Module):
    def __init__(self, in_c, out_c, modes_z, modes_y, modes_x,
                 pad_type: str = 'reflect', pad_z: int = 8, pad_y: int = 8, pad_x: int = 8):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.mz, self.my, self.mx = int(modes_z), int(modes_y), int(modes_x)
        self.pad_type = pad_type
        self.pad_z = int(pad_z)
        self.pad_y = int(pad_y)
        self.pad_x = int(pad_x)
        scale = 1 / max(1, in_c * out_c)
        self.weight = nn.Parameter(
            scale * torch.randn(in_c, out_c, max(1, self.mz), max(1, self.my), max(1, self.mx), 2)
        )
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
        out_ft = torch.zeros(
            B, self.out_c, Zp, Yp, Xp // 2 + 1, 2,
            device=x_pad.device, dtype=x_pad.dtype
        )
        mz, my, mx = min(self.mz, Zp), min(self.my, Yp), min(self.mx, Xp // 2 + 1)
        if mz > 0 and my > 0 and mx > 0:
            w = self.weight[:, :, :mz, :my, :mx, :]
            out_ft[:, :, :mz, :my, :mx, :] = self.compl_mul3d(
                x_ft[:, :, :mz, :my, :mx, :], w
            )

        y_pad = torch.fft.irfftn(
            torch.view_as_complex(out_ft),
            s=(Zp, Yp, Xp),
            dim=(-3, -2, -1)
        )
        if (pz | py | px) > 0:
            y = y_pad[:, :, pz:Zp - pz, py:Yp - py, px:Xp - px]
        else:
            y = y_pad
        scale = 1.0 + torch.tanh(self.gate)
        return y * scale

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
            self.specs.append(
                SpectralConv3d(width, width, mz_s, my_s, mx_s,
                               pad_type=pad_type, pad_z=pz, pad_y=py, pad_x=px)
            )
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
            MultiScaleSpectralBlock(
                width, (mz, my, mx),
                scale_count=self.ufno_scales,
                pad_type=self.spec_pad_type,
                pad=(pz, py, px)
            )
            for _ in range(layers)
        ])
        self.ws = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(layers)])
        self.locals = nn.ModuleList(
            [nn.Conv3d(width, width, 3, padding=1, groups=width) for _ in range(layers)]
        ) if use_local else None
        self.norms = nn.ModuleList([nn.GroupNorm(gn_groups, width) for _ in range(layers)])
        self.gammas = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(residual_scale))) for _ in range(layers)]
        )
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else None
        self.proj = nn.Sequential(
            nn.Conv3d(width, width, 1),
            nn.GELU(),
            nn.Conv3d(width, 1, 1)
        )

        self.ctx_mlp = nn.Linear(self.context_dim, width) if self.context_dim > 0 else None
        self.layer_films = nn.ModuleList()
        if self.context_dim > 0:
            for _ in range(layers):
                lin = nn.Linear(self.context_dim, 2 * width)
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
                self.layer_films.append(lin)

    @staticmethod
    def _coords(Z, Y, X, device, dtype):
        z = torch.linspace(-1.0, 1.0, int(Z), device=device, dtype=dtype)
        y = torch.linspace(-1.0, 1.0, int(Y), device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, int(X), device=device, dtype=dtype)
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
    """
    若训练时加了 --learn_h，这里可以加载并预测空间分布 h_field；
    如果训练中没启用 learn_h，ckpt 里也不会有 hhead，本脚本会自动跳过。
    """
    def __init__(self, in_c=7, width=32, layers=2, out_feat=64, mlp_hidden=64,
                 mask_idx=2, h_min=0.0, h_max=30.0,
                 h_prior=10.0, beta=2.0, use_ctx_film=True, ctx_dim=10):
        super().__init__()
        self.mask_idx = mask_idx
        self.h_min = float(h_min)
        self.h_max = float(h_max)
        self.beta = float(beta)
        self.use_ctx_film = bool(use_ctx_film)

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
        if self.use_ctx_film:
            self.ctx2affine = nn.Sequential(
                nn.Linear(ctx_dim, 32), nn.GELU(),
                nn.Linear(32, 2)
            )
            nn.init.zeros_(self.ctx2affine[-1].weight)
            nn.init.zeros_(self.ctx2affine[-1].bias)

        # 初始化 bias 让输出 h 接近 h_prior
        with torch.no_grad():
            p = (h_prior - self.h_min) / max(1e-6, (self.h_max - self.h_min))
            p = float(np.clip(p, 1e-6, 1 - 1e-6))
            b0 = math.log(p / (1 - p))  # logit
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
            a = ab[:, :1]
            b = ab[:, 1:2]
            raw = (1.0 + a) * raw + b
        h_norm = torch.sigmoid(self.beta * raw)
        h_scalar = self.h_min + (self.h_max - self.h_min) * h_norm
        return h_scalar.view(B, 1, 1, 1, 1) * mask

# ------------------------- ckpt I/O（严格对齐 + remap） -------------------------
def load_ckpt_for_eval(ckpt_path, device='cpu'):
    """
    从训练生成的 ckpt 加载 FNO3D + HHead，并根据 args 还原网络结构：
    - width, layers, mz,my,mx, spec_pad_type, spec_pad_z/y/x, ufno_scales
    - 兼容多尺度 uFNO 的权重命名变更（旧 ckpt 只有 specs.L.weight）
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})

    width  = int(args.get("width", 24))
    layers = int(args.get("layers", 4))
    mz     = int(args.get("mz", 12))
    my     = int(args.get("my", 12))
    mx     = int(args.get("mx", 12))

    spec_pad_type = str(args.get("spec_pad_type", "reflect"))
    spz = int(args.get("spec_pad_z", 8))
    spy = int(args.get("spec_pad_y", 8))
    spx = int(args.get("spec_pad_x", 8))
    ufno_scales   = int(args.get("ufno_scales", 3))

    model = FNO3D(
        in_c=3, width=width, modes=(mz, my, mx), layers=layers,
        context_dim=10,
        spec_pad_type=spec_pad_type, spec_pad=(spz, spy, spx),
        ufno_scales=ufno_scales
    ).to(device)

    ckpt_state = ckpt["model"]
    model_state = model.state_dict()

    # 兼容旧权重命名：specs.L.weight -> specs.L.specs.0.weight
    remapped = {}
    for k, v in ckpt_state.items():
        if k.startswith("specs.") and (".weight" in k or ".gate" in k) and (".specs." not in k):
            parts = k.split(".")
            if len(parts) == 3 and parts[0] == "specs":
                L = parts[1]
                tail = parts[2]
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

    hhead = None
    if "hhead" in ckpt:
        h_width  = int(args.get("h_width", 16))
        h_layers = int(args.get("h_layers", 2))
        hhead = HHead(in_c=7, width=h_width, layers=h_layers, ctx_dim=10).to(device)
        mh, uh = hhead.load_state_dict(ckpt["hhead"], strict=False)
        if mh or uh:
            print(f"[warn] hhead load (non-strict) missing={mh}, unexpected={uh}")

    rk2_from_ckpt = bool(args.get("rk2", False))
    return model.eval(), (hhead.eval() if hhead is not None else None), rk2_from_ckpt, args

# ------------------------- 推进/预测（log(dtau) clamp） -------------------------
@torch.inference_mode()
def step_one(model, theta_t, Ms, S_nd, ctx_glb, dtau_scalar, rk2=False, device='cpu'):
    """
    单步时间推进：
    输入：
      - theta_t: (B,1,Z,Y,X) 当前无量纲温升
      - Ms:      (B,1,Z,Y,X) solid mask
      - S_nd:    (B,1,Z,Y,X) 无量纲源项
      - ctx_glb: (B,9)       全局上下文
      - dtau_scalar: 标量 dtau = dt * alpha / L^2
    输出：
      - theta_next: (B,1,Z,Y,X)
    """
    B, _, Z, Y, X = theta_t.shape
    x = torch.cat([theta_t, Ms, S_nd], dim=1)
    dtau = torch.tensor([dtau_scalar], dtype=torch.float32, device=device).view(B, 1)
    log_dtau = torch.log(dtau + 1e-12).clamp_(-20.0, 20.0)
    ctx = torch.cat([ctx_glb, log_dtau], dim=1)  # (B,10)

    if rk2:
        delta1 = model(x, ctx)
        theta_tilde = theta_t + delta1
        delta2 = model(torch.cat([theta_tilde, Ms, S_nd], dim=1), ctx)
        delta  = 0.5 * (delta1 + delta2)
    else:
        delta = model(x, ctx)

    theta_next = theta_t + delta
    return theta_next

@torch.inference_mode()
def predict_h_field(hhead, theta_t, Ms, Mi, nsurf, S_nd, ctx_glb, dtau_scalar, device='cpu'):
    """
    可选：预测 h 场（需要 ckpt 里有 hhead）。
    """
    if hhead is None:
        return None
    B = theta_t.size(0)
    feats = torch.cat([
        theta_t,          # 1
        S_nd,             # 1
        Mi,               # 1
        Ms,               # 1
        nsurf[:, 0:1],    # 1
        nsurf[:, 1:2],    # 1
        nsurf[:, 2:3]     # 1
    ], dim=1)             # 共 7 通道

    dtau = torch.tensor([dtau_scalar], dtype=torch.float32, device=device).view(B, 1)
    log_dtau = torch.log(dtau + 1e-12).clamp_(-20.0, 20.0)
    ctx = torch.cat([ctx_glb, log_dtau], dim=1)
    return hhead(feats, ctx_vec=ctx)

# ------------------------- 误差统计（仅 solid、排边界、跳过第0帧） -------------------------
def _make_core_solid_mask(Ms: np.ndarray, border: int):
    """
    Ms: (Nz,Ny,Nx) float32;
    返回 bool mask，True = 有效（solid 且非边界外圈）
    """
    solid = (Ms > 0.5)
    if border <= 0:
        return solid
    z0, z1 = border, Ms.shape[0] - border
    y0, y1 = border, Ms.shape[1] - border
    x0, x1 = border, Ms.shape[2] - border
    if z0 >= z1 or y0 >= y1 or x0 >= x1:  # 极端小尺寸保护
        return np.zeros_like(solid, dtype=bool)
    core = np.zeros_like(solid, dtype=bool)
    core[z0:z1, y0:y1, x0:x1] = True
    return solid & core

def _timewise_err_stats(T_pred: np.ndarray,
                        T_true: np.ndarray,
                        Ms: np.ndarray,
                        border: int,
                        save_fields: bool,
                        dT_scale: float):
    """
    输入: T_pred/T_true: (Nt,Nz,Ny,Nx), Ms:(Nz,Ny,Nx)

    只统计:
      - t >= 1
      - solid & core 区域（可排除边界外圈）

    相对误差定义为:
      |T_pred - T_true| / max(dT_scale, eps)

    这里 dT_scale 推荐取物理上的温升标尺:
      dT_scale ≈ T_ref - T_inf  或  T_ref - T_amb
    对应数据里的 scales/dT_ref。
    """
    Nt, Nz, Ny, Nx = T_true.shape
    mask = _make_core_solid_mask(Ms, border)  # (Nz,Ny,Nx) bool
    eps = 1e-6

    denom = max(float(dT_scale), eps)  # 固定温升标尺

    keys = [
        "abs_mean", "abs_median", "abs_std", "abs_var",
        "abs_min", "abs_p25", "abs_p75", "abs_max", "abs_count",
        "rel_mean", "rel_median", "rel_std", "rel_var",
        "rel_min", "rel_p25", "rel_p75", "rel_max", "rel_count",
    ]
    stats = {k: np.full((Nt,), np.float32(np.nan)) for k in keys}
    abs_err_vol = np.zeros((Nt, Nz, Ny, Nx), np.float32) if save_fields else None
    rel_err_vol = np.zeros((Nt, Nz, Ny, Nx), np.float32) if save_fields else None

    for t in range(1, Nt):
        diff = T_pred[t] - T_true[t]
        abs_full = np.abs(diff)

        abs_e = abs_full[mask]
        rel_e = (abs_full / denom)[mask]

        if abs_e.size == 0:
            continue

        # 绝对误差统计
        stats["abs_mean"][t]   = float(np.mean(abs_e))
        stats["abs_median"][t] = float(np.median(abs_e))
        stats["abs_std"][t]    = float(np.std(abs_e, ddof=0))
        stats["abs_var"][t]    = float(np.var(abs_e, ddof=0))
        stats["abs_min"][t]    = float(np.min(abs_e))
        stats["abs_p25"][t]    = float(np.percentile(abs_e, 25))
        stats["abs_p75"][t]    = float(np.percentile(abs_e, 75))
        stats["abs_max"][t]    = float(np.max(abs_e))
        stats["abs_count"][t]  = float(abs_e.size)

        # 相对误差统计（相对温升标尺）
        stats["rel_mean"][t]   = float(np.mean(rel_e))
        stats["rel_median"][t] = float(np.median(rel_e))
        stats["rel_std"][t]    = float(np.std(rel_e, ddof=0))
        stats["rel_var"][t]    = float(np.var(rel_e, ddof=0))
        stats["rel_min"][t]    = float(np.min(rel_e))
        stats["rel_p25"][t]    = float(np.percentile(rel_e, 25))
        stats["rel_p75"][t]    = float(np.percentile(rel_e, 75))
        stats["rel_max"][t]    = float(np.max(rel_e))
        stats["rel_count"][t]  = float(rel_e.size)

        if save_fields:
            # 只在有效区域保留误差值
            m = mask.astype(np.float32)
            abs_err_vol[t] = abs_full.astype(np.float32) * m
            rel_err_vol[t] = (abs_full / denom).astype(np.float32) * m

    out = {"stats": stats}
    if save_fields:
        out["abs_err"] = abs_err_vol
        out["rel_err"] = rel_err_vol
    return out

# ------------------------- 评估（Teacher / Rollout） -------------------------
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
    if diff.size == 0:
        return 0.0, 0.0
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return mae, rmse

def _metrics_h(h_pred, h_true, Mi):
    m = (Mi > 0.5)
    diff = (h_pred - h_true)[m]
    if diff.size == 0:
        return 0.0, 0.0
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return mae, rmse

@torch.inference_mode()
def evaluate_modes(H: HeatsinkH5, model: FNO3D, hhead: HHead, rk2: bool,
                   device='cpu', do_teacher=True, do_rollout=True,
                   rollout_from=0, rollout_steps=-1,
                   calc_err=False, exclude_border=1, save_err_fields=False):
    """
    综合评估：
      - Teacher forcing：每一步都喂真值 t 帧，预测 t+1 帧
      - Rollout：从 rollout_from 开始，使用模型预测结果作为下一步输入，逐步滚动 rollout_steps 步
    """
    Nt, Nz, Ny, Nx = H.Nt, H.Nz, H.Ny, H.Nx
    Ms = torch.from_numpy(H.Ms[None, None, ...]).to(device)
    Mi = torch.from_numpy(H.Mi[None, None, ...]).to(device)
    S_nd = torch.from_numpy(H.S_nd[None, None, ...]).to(device)
    nsurf = torch.from_numpy(H.nsurf[None, ...]).to(device)      # (1,3,Z,Y,X)
    ctx_glb = torch.from_numpy(H.ctx_glb[None, :]).to(device)    # (1,9)

    dt = H.dt.astype(np.float64)
    dtau_arr = (dt * (H.alpha / (H.L * H.L))).astype(np.float64)

    results_teacher, results_rollout = {}, {}

    # ---------- Teacher Forcing ----------
    if do_teacher:
        out = _alloc_like(Nt, Nz, Ny, Nx, with_theta=True, with_h=(hhead is not None))
        mae_T = np.zeros((Nt,), np.float32)
        rmse_T = np.zeros((Nt,), np.float32)
        mae_h = np.zeros((Nt,), np.float32)
        rmse_h = np.zeros((Nt,), np.float32)

        theta = (H.T - T_BASE) / DT_BASE
        out["theta_pred"][0] = theta[0].astype(np.float32)
        out["T_pred"][0] = H.T[0].astype(np.float32)

        for t in range(Nt - 1):
            theta_t = torch.from_numpy(theta[t:t + 1, ...]).to(device).unsqueeze(1)   # (1,1,Z,Y,X)
            dtau = float(dtau_arr[t])
            theta_next = step_one(model, theta_t, Ms, S_nd, ctx_glb, dtau,
                                  rk2=rk2, device=device)
            theta_next_np = theta_next[0, 0].detach().cpu().numpy().astype(np.float32)
            T_next_np = (theta_next_np * DT_BASE + T_BASE).astype(np.float32)

            out["theta_pred"][t + 1] = theta_next_np
            out["T_pred"][t + 1] = T_next_np

            if hhead is not None:
                h_field = predict_h_field(hhead, theta_t, Ms, Mi, nsurf, S_nd, ctx_glb,
                                          dtau, device=device)
                h_np = h_field[0, 0].detach().cpu().numpy().astype(np.float32)
                out["h_pred"][t] = h_np
                m = (H.Mi > 0.5)
                out["h_scalar"][t] = float(h_np[m].mean()) if np.any(m) else 0.0

            mae, rmse = _metrics_T(out["T_pred"][t + 1], H.T[t + 1], H.Ms)
            mae_T[t + 1], rmse_T[t + 1] = mae, rmse

            if (hhead is not None) and (H.h_truth is not None):
                mae_h[t], rmse_h[t] = _metrics_h(out["h_pred"][t], H.h_truth[t], H.Mi)

        results_teacher = {**out, "mae_T": mae_T, "rmse_T": rmse_T}
        if hhead is not None:
            results_teacher["mae_h"] = mae_h
            results_teacher["rmse_h"] = rmse_h

        if calc_err:
            err = _timewise_err_stats(
                results_teacher["T_pred"], H.T, H.Ms,
                border=exclude_border,
                save_fields=save_err_fields,
                dT_scale=H.dTref
            )
            results_teacher["err"] = err

    # ---------- Rollout ----------
    if do_rollout:
        if rollout_steps is None or int(rollout_steps) < 0:
            rollout_steps_eff = (Nt - 1) - int(rollout_from)
        else:
            rollout_steps_eff = int(max(0, min(int(rollout_steps),
                                               (Nt - 1) - int(rollout_from))))
        start = int(rollout_from)

        print(f"[rollout] Nt={Nt}, rollout_from={start}, rollout_steps_eff={rollout_steps_eff}")

        out = _alloc_like(Nt, Nz, Ny, Nx, with_theta=True, with_h=(hhead is not None))
        mae_T = np.zeros((Nt,), np.float32)
        rmse_T = np.zeros((Nt,), np.float32)
        mae_h = np.zeros((Nt,), np.float32)
        rmse_h = np.zeros((Nt,), np.float32)

        theta_true = (H.T - T_BASE) / DT_BASE
        out["theta_pred"][:start + 1] = theta_true[:start + 1].astype(np.float32)
        out["T_pred"][:start + 1] = H.T[:start + 1].astype(np.float32)

        theta_curr = torch.from_numpy(theta_true[start:start + 1]).to(device).unsqueeze(1)

        for k in range(rollout_steps_eff):
            t = start + k
            dtau = float(dtau_arr[t])

            theta_next = step_one(model, theta_curr, Ms, S_nd, ctx_glb, dtau,
                                  rk2=rk2, device=device)
            theta_next_np = theta_next[0, 0].detach().cpu().numpy().astype(np.float32)
            T_next_np = (theta_next_np * DT_BASE + T_BASE).astype(np.float32)

            out["theta_pred"][t + 1] = theta_next_np
            out["T_pred"][t + 1] = T_next_np

            if hhead is not None:
                h_field = predict_h_field(hhead, theta_curr, Ms, Mi, nsurf, S_nd,
                                          ctx_glb, dtau, device=device)
                h_np = h_field[0, 0].detach().cpu().numpy().astype(np.float32)
                out["h_pred"][t] = h_np
                m = (H.Mi > 0.5)
                out["h_scalar"][t] = float(h_np[m].mean()) if np.any(m) else 0.0

            mae, rmse = _metrics_T(out["T_pred"][t + 1], H.T[t + 1], H.Ms)
            mae_T[t + 1], rmse_T[t + 1] = mae, rmse

            theta_curr = theta_next

            if (hhead is not None) and (H.h_truth is not None):
                mae_h[t], rmse_h[t] = _metrics_h(out["h_pred"][t], H.h_truth[t], H.Mi)

        results_rollout = {
            **out,
            "mae_T": mae_T, "rmse_T": rmse_T,
            "rollout_from": start, "rollout_steps": rollout_steps_eff
        }
        if hhead is not None:
            results_rollout["mae_h"] = mae_h
            results_rollout["rmse_h"] = rmse_h

        if calc_err:
            err = _timewise_err_stats(
                results_rollout["T_pred"], H.T, H.Ms,
                border=exclude_border,
                save_fields=save_err_fields,
                dT_scale=H.dTref
            )
            results_rollout["err"] = err

    return results_teacher, results_rollout

# ------------------------- TwoFrame（跨时间两帧一步预测） -------------------------
def _parse_pairs_string(pairs_str, Nt):
    pairs = []
    if not pairs_str:
        return pairs
    items = [s.strip() for s in pairs_str.split(",") if s.strip()]
    for it in items:
        if "-" not in it:
            raise ValueError(f"bad pair token: {it} (expect 'ts-te')")
        a, b = it.split("-", 1)
        ts = int(a.strip())
        te = int(b.strip())
        if not (0 <= ts < te <= Nt - 1):
            raise ValueError(f"pair out of range: ({ts},{te}), Nt={Nt}")
        pairs.append((ts, te))
    return pairs

@torch.inference_mode()
def evaluate_twoframe(H: HeatsinkH5, model: FNO3D, rk2: bool,
                      device='cpu',
                      count: int = 0,
                      pairs_str: str = "",
                      min_gap: int = 1,
                      max_gap: int = -1,
                      seed: int = 42):
    """
    TwoFrame 评估：任选两帧 ts<te，一步跨越预测 te 时刻：
      - 若给定 pairs_str（如 "0-5, 0-10, 3-20"），按给定对评估
      - 否则随机抽取 count 对 (ts,te)，间隔在 [min_gap, max_gap] 内
    """
    Nt, Nz, Ny, Nx = H.Nt, H.Nz, H.Ny, H.Nx
    Ms = torch.from_numpy(H.Ms[None, None, ...]).to(device)
    S_nd = torch.from_numpy(H.S_nd[None, None, ...]).to(device)
    ctx_glb = torch.from_numpy(H.ctx_glb[None, :]).to(device)
    dtau_arr = (H.dt.astype(np.float64) * (H.alpha / (H.L * H.L))).astype(np.float64)

    pairs = _parse_pairs_string(pairs_str, Nt)
    mode = "manual"
    if not pairs:
        mode = "random"
        rng = random.Random(seed)
        if max_gap is None or max_gap < 0:
            max_gap_eff = (Nt - 1)
        else:
            max_gap_eff = int(max_gap)
        min_gap = max(1, int(min_gap))
        for _ in range(max(0, int(count))):
            ts = rng.randint(0, Nt - 2)
            gap_hi = min(max_gap_eff, (Nt - 1) - ts)
            if gap_hi < min_gap:
                ts = rng.randint(0, Nt - 1 - min_gap)
                gap_hi = min(max_gap_eff, (Nt - 1) - ts)
            te = ts + rng.randint(min_gap, gap_hi)
            pairs.append((ts, te))

    K = len(pairs)
    if K == 0:
        return {}

    out = {
        "ts": np.zeros((K,), np.int32),
        "te": np.zeros((K,), np.int32),
        "dtau": np.zeros((K,), np.float32),
        "T_in": np.zeros((K, Nz, Ny, Nx), np.float32),
        "T_pred": np.zeros((K, Nz, Ny, Nx), np.float32),
        "T_true": np.zeros((K, Nz, Ny, Nx), np.float32),
        "mae_T": np.zeros((K,), np.float32),
        "rmse_T": np.zeros((K,), np.float32),
    }

    theta_all = (H.T - T_BASE) / DT_BASE

    for i, (ts, te) in enumerate(pairs):
        dtau = float(dtau_arr[ts:te].sum())
        theta_t = torch.from_numpy(theta_all[ts:ts + 1]).to(device).unsqueeze(1)  # (1,1,Z,Y,X)
        theta_pred = step_one(model, theta_t, Ms, S_nd, ctx_glb, dtau,
                              rk2=rk2, device=device)

        theta_pred_np = theta_pred[0, 0].detach().cpu().numpy().astype(np.float32)
        T_pred_np = (theta_pred_np * DT_BASE + T_BASE).astype(np.float32)

        out["ts"][i] = ts
        out["te"][i] = te
        out["dtau"][i] = dtau
        out["T_in"][i] = H.T[ts].astype(np.float32)
        out["T_pred"][i] = T_pred_np
        out["T_true"][i] = H.T[te].astype(np.float32)

        mae, rmse = _metrics_T(out["T_pred"][i], out["T_true"][i], H.Ms)
        out["mae_T"][i] = mae
        out["rmse_T"][i] = rmse

    out["_meta_mode"] = mode
    return out

# ------------------------- H5 写出 -------------------------
def _copy_all(src_h5: h5py.File, dst_h5: h5py.File):
    for name in src_h5.keys():
        if name not in dst_h5:
            src_h5.copy(name, dst_h5, name=name)

def _write_group(g, name, arr, comp=4):
    if arr is None:
        return
    if name in g:
        del g[name]
    g.create_dataset(name, data=arr,
                     compression="gzip", compression_opts=comp,
                     shuffle=True)

def _write_err_group(g, err_obj, comp=4):
    """err_obj: {'stats':dict, 'abs_err':(opt), 'rel_err':(opt)}"""
    if err_obj is None:
        return
    ge = g.create_group("err")
    # stats (1D 按时间)
    gs = ge.create_group("stats")
    for k, v in err_obj["stats"].items():
        _write_group(gs, k, v, comp=comp)
    # 可选 3D 场
    if "abs_err" in err_obj and err_obj["abs_err"] is not None:
        _write_group(ge, "abs_err", err_obj["abs_err"], comp=comp)
    if "rel_err" in err_obj and err_obj["rel_err"] is not None:
        _write_group(ge, "rel_err", err_obj["rel_err"], comp=comp)

def save_eval_h5(in_path, out_path, teacher_res, rollout_res, twoframe_res,
                 copy_all=False, meta_note=""):
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
            for k in ("T_pred", "theta_pred", "h_pred", "h_scalar",
                      "mae_T", "rmse_T", "mae_h", "rmse_h"):
                if k in teacher_res and teacher_res[k] is not None:
                    _write_group(gt, k, teacher_res[k])
            if "err" in teacher_res and teacher_res["err"] is not None:
                _write_err_group(gt, teacher_res["err"])
        # Rollout
        if rollout_res:
            gr = ge.create_group("rollout")
            for k in ("T_pred", "theta_pred", "h_pred", "h_scalar",
                      "mae_T", "rmse_T", "mae_h", "rmse_h"):
                if k in rollout_res and rollout_res[k] is not None:
                    _write_group(gr, k, rollout_res[k])
            gr.attrs["rollout_from"] = int(rollout_res.get("rollout_from", 0))
            gr.attrs["rollout_steps"] = int(rollout_res.get("rollout_steps", 0))
            if "err" in rollout_res and rollout_res["err"] is not None:
                _write_err_group(gr, rollout_res["err"])

        # TwoFrame
        if twoframe_res:
            gf = ge.create_group("twoframe")
            for k in ("ts", "te", "dtau", "T_in", "T_pred", "T_true",
                      "mae_T", "rmse_T"):
                if k in twoframe_res and twoframe_res[k] is not None:
                    _write_group(gf, k, twoframe_res[k])
            gf.attrs["mode"] = str(twoframe_res.get("_meta_mode", ""))
    print(f"[OK] eval H5 written: {out_path}")

# ------------------------- CLI & 批量调度 -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True,
                    help="输入 H5 路径，或包含多个 H5 的目录")
    ap.add_argument("--ckpt", type=str, required=True, help="训练 ckpt (.pt)")

    # 单文件 / 批量输出
    ap.add_argument("--out",  type=str, default="",
                    help="单文件模式下的输出评估 H5 路径")
    ap.add_argument("--out_dir", type=str, default="",
                    help="目录模式下的输出根目录；若未指定，将在 data 目录下创建 eval 子目录")
    ap.add_argument("--suffix", type=str, default="_eval",
                    help="批量模式输出文件名后缀，例如 foo.h5 -> foo_eval.h5")

    ap.add_argument("--recursive", action="store_true",
                    help="当 data 是目录时，是否递归遍历子目录")

    ap.add_argument("--mode", type=str, choices=["both", "teacher", "rollout"], default="both")
    ap.add_argument("--rollout_from", type=int, default=0)
    ap.add_argument("--rollout_steps", type=int, default=-1,
                    help="-1 表示滚到序列末尾")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--copy_all", action="store_true",
                    help="复制原 H5 的全部内容，便于后处理/对照")
    ap.add_argument("--rk2", action="store_true",
                    help="强制启用 RK2（覆盖 ckpt 中的 rk2 设置）")

    # TwoFrame
    ap.add_argument("--twoframe-count", type=int, default=0,
                    help="随机抽取的两帧对数量；=0 表示不做 TwoFrame 随机评估")
    ap.add_argument("--twoframe-pairs", type=str, default="",
                    help="手工指定若干对 ts-te，用逗号分隔；指定后忽略 twoframe-count")
    ap.add_argument("--twoframe-min-gap", type=int, default=1,
                    help="随机抽取时的最小间隔步数（>=1）")
    ap.add_argument("--twoframe-max-gap", type=int, default=-1,
                    help="随机抽取时的最大间隔步数；<0 表示不限")
    ap.add_argument("--twoframe-seed", type=int, default=42,
                    help="随机抽取的种子")

    # 误差开关
    ap.add_argument("--calc-err", action="store_true",
                    help="计算绝对/相对误差及统计（仅 solid，排除外圈，跳过第0帧）")
    ap.add_argument("--exclude-border", type=int, default=1,
                    help="排除的外圈单元厚度（默认1）")
    ap.add_argument("--save-err-fields", action="store_true",
                    help="同时保存逐时间的 3D 绝对/相对误差场（文件可能较大）")
    return ap.parse_args()

def run_single_case(args,
                    model, hhead, rk2,
                    device,
                    data_path: str,
                    out_path: str):
    H = HeatsinkH5(data_path)

    do_teacher = (args.mode in ("both", "teacher"))
    do_rollout = (args.mode in ("both", "rollout"))

    teacher_res, rollout_res = evaluate_modes(
        H, model, hhead, rk2, device=device,
        do_teacher=do_teacher, do_rollout=do_rollout,
        rollout_from=args.rollout_from, rollout_steps=args.rollout_steps,
        calc_err=args.calc_err, exclude_border=args.exclude_border,
        save_err_fields=args.save_err_fields
    )

    twoframe_res = {}
    if (args.twoframe_pairs and args.twoframe_pairs.strip()) or (args.twoframe_count > 0):
        print("[twoframe] generating cross-time predictions ...")
        twoframe_res = evaluate_twoframe(
            H, model, rk2, device=device,
            count=args.twoframe_count,
            pairs_str=args.twoframe_pairs,
            min_gap=args.twoframe_min_gap,
            max_gap=args.twoframe_max_gap,
            seed=args.twoframe_seed
        )

    meta_note = (
        f"mode={args.mode}; rk2={rk2}; device={device}; "
        f"ckpt={os.path.basename(args.ckpt)}; twoframe_count={args.twoframe_count}; "
        f"twoframe_pairs={'manual' if args.twoframe_pairs else 'random'}; "
        f"calc_err={args.calc_err}; exclude_border={args.exclude_border}; "
        f"save_err_fields={args.save_err_fields}"
    )
    save_eval_h5(H.path, out_path, teacher_res, rollout_res, twoframe_res,
                 copy_all=args.copy_all, meta_note=meta_note)
    H.close()

def main():
    args = parse_args()

    # 设备
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 模型只加载一次，多个 case 复用
    model, hhead, rk2_from_ckpt, train_args = load_ckpt_for_eval(args.ckpt, device=device)
    rk2 = bool(args.rk2) or bool(rk2_from_ckpt)
    model.eval()
    if hhead is not None:
        hhead.eval()

    data_path = args.data
    if os.path.isdir(data_path):
        # 批量模式
        if args.out_dir:
            out_root = args.out_dir
        else:
            out_root = os.path.join(data_path, "eval")
        print(f"[info] 批量模式：data_dir={data_path}, out_dir={out_root}")

        h5_list = []
        if args.recursive:
            for root, dirs, files in os.walk(data_path):
                for fn in files:
                    if not fn.endswith(".h5"):
                        continue
                    stem, ext = os.path.splitext(fn)
                    # 跳过已经是 *_suffix.h5 的评估文件
                    if stem.endswith(args.suffix):
                        continue
                    in_f = os.path.join(root, fn)
                    h5_list.append(in_f)
        else:
            for fn in sorted(os.listdir(data_path)):
                if not fn.endswith(".h5"):
                    continue
                stem, ext = os.path.splitext(fn)
                if stem.endswith(args.suffix):
                    continue
                in_f = os.path.join(data_path, fn)
                if os.path.isfile(in_f):
                    h5_list.append(in_f)

        if not h5_list:
            print(f"[warn] 在目录 {data_path} 中未找到需要评估的 .h5 文件。")
            return

        print(f"[info] 发现 {len(h5_list)} 个 H5 文件，将逐个评估。")

        for i, in_path in enumerate(sorted(h5_list)):
            rel = os.path.relpath(in_path, data_path)
            stem, ext = os.path.splitext(rel)
            out_rel = stem + args.suffix + ".h5"
            out_path = os.path.join(out_root, out_rel)
            print(f"[{i+1}/{len(h5_list)}] eval: {in_path} -> {out_path}")
            run_single_case(args, model, hhead, rk2, device, in_path, out_path)

    else:
        # 单文件模式
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"data 路径不是文件也不是目录: {data_path}")

        if args.out:
            out_path = args.out
        else:
            if args.out_dir:
                os.makedirs(args.out_dir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(data_path))[0]
                out_path = os.path.join(args.out_dir, stem + args.suffix + ".h5")
            else:
                base_dir = os.path.dirname(os.path.abspath(data_path))
                stem = os.path.splitext(os.path.basename(data_path))[0]
                out_path = os.path.join(base_dir, stem + args.suffix + ".h5")

        print(f"[info] 单文件模式: {data_path} -> {out_path}")
        run_single_case(args, model, hhead, rk2, device, data_path, out_path)

if __name__ == "__main__":
    main()
