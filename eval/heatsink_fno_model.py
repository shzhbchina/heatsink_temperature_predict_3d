# heatsink_fno_model.py
# 提供 Heatsink FNO3D 模型的定义 + 从 ckpt 加载权重的辅助函数

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------
# 工具：频域 padding（和训练脚本保持一致）
# -------------------------------------------------------
def _fft_reflect_pad3d(x: torch.Tensor, padz: int, pady: int, padx: int, mode: str):
    """
    对输入张量 x 做对称/复制 padding，用于频域卷积。
    x: (B, C, Z, Y, X)
    返回: x_pad, (pz, py, px)
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


# -------------------------------------------------------
# 频域卷积层 + 多尺度 uFNO block
# -------------------------------------------------------
class SpectralConv3d(nn.Module):
    """
    单尺度 3D 频域卷积：FNO 核心算子。
    """

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

        scale = 1.0 / max(1, in_c * out_c)
        self.weight = nn.Parameter(
            scale * torch.randn(in_c, out_c,
                                max(1, self.mz),
                                max(1, self.my),
                                max(1, self.mx), 2)
        )
        # 门控参数
        self.gate = nn.Parameter(torch.zeros(1, out_c, 1, 1, 1))

    @staticmethod
    def compl_mul3d(a, b):
        """
        复数乘法：a, b 形状 (..., 2)，最后一个维度表示 (real, imag)
        """
        op = torch.einsum
        # (b, c, z, y, x, 2)
        real = op("bczyx,cozyx->bozyx", a[..., 0], b[..., 0]) - op("bczyx,cozyx->bozyx", a[..., 1], b[..., 1])
        imag = op("bczyx,cozyx->bozyx", a[..., 0], b[..., 1]) + op("bczyx,cozyx->bozyx", a[..., 1], b[..., 0])
        return torch.stack([real, imag], dim=-1)

    def forward(self, x):
        # x: (B, C, Z, Y, X)
        x_pad, (pz, py, px) = _fft_reflect_pad3d(x, self.pad_z, self.pad_y, self.pad_x, self.pad_type)
        B, C, Zp, Yp, Xp = x_pad.shape

        # rfftn -> (B, C, Zp, Yp, Xp//2+1, 2)
        x_ft = torch.view_as_real(torch.fft.rfftn(x_pad, s=(Zp, Yp, Xp), dim=(-3, -2, -1)))

        out_ft = torch.zeros(
            B, self.out_c, Zp, Yp, Xp // 2 + 1, 2,
            device=x_pad.device, dtype=x_pad.dtype
        )

        mz = min(self.mz, Zp)
        my = min(self.my, Yp)
        mx = min(self.mx, Xp // 2 + 1)

        if mz > 0 and my > 0 and mx > 0:
            w = self.weight[:, :, :mz, :my, :mx, :]
            out_ft[:, :, :mz, :my, :mx, :] = self.compl_mul3d(
                x_ft[:, :, :mz, :my, :mx, :], w
            )

        # ifft 回到空间域
        y_pad = torch.fft.irfftn(
            torch.view_as_complex(out_ft), s=(Zp, Yp, Xp), dim=(-3, -2, -1)
        )

        if (pz | py | px) > 0:
            y = y_pad[:, :, pz:Zp - pz, py:Yp - py, px:Xp - px]
        else:
            y = y_pad

        # 通道门控
        scale = 1.0 + torch.tanh(self.gate)
        y = y * scale
        return y


class MultiScaleSpectralBlock(nn.Module):
    """
    多尺度频域块：在同一层里用不同频带的 SpectralConv，然后做可学习加权和。
    """

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

        # 每个尺度的权重 logits
        self.scale_logits = nn.Parameter(torch.zeros(self.scale_count))

    def forward(self, h):
        if self.scale_count == 1:
            return self.specs[0](h)

        ws = torch.softmax(self.scale_logits, dim=0)
        y = 0.0
        for s, sc in enumerate(self.specs):
            y = y + ws[s] * sc(h)
        return y


# -------------------------------------------------------
# 主体：3D FNO + context FiLM
# -------------------------------------------------------
class FNO3D(nn.Module):
    """
    Heatsink 用的 3D FNO 模型。
    输入:
        x: (B, in_c=3, Z, Y, X)   通常为 [theta, Ms, S]
        ctx: (B, context_dim=10)  ctx_glb[9] + log_dtau[1]
    输出:
        (B, 1, Z, Y, X)           预测的增量 delta_theta
    """

    def __init__(self,
                 in_c=3,
                 width=24,
                 modes=(12, 12, 12),
                 layers=4,
                 add_coords=True,
                 fourier_k=8,
                 use_local=True,
                 gn_groups=1,
                 residual_scale=0.5,
                 dropout=0.0,
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

        # 输入通道加上位置编码通道数
        extra_c = 0
        if add_coords:
            # [z,y,x] + 每个方向的 sin/cosFourier
            extra_c = 3 + 6 * fourier_k

        self.lift = nn.Conv3d(in_c + extra_c, width, kernel_size=1)

        mz, my, mx = modes
        pz, py, px = self.spec_pad

        # uFNO 多尺度频域 block 列表
        self.specs = nn.ModuleList([
            MultiScaleSpectralBlock(
                width, (mz, my, mx),
                scale_count=self.ufno_scales,
                pad_type=self.spec_pad_type,
                pad=(pz, py, px)
            )
            for _ in range(layers)
        ])

        # 频域结果的 1x1 conv
        self.ws = nn.ModuleList([
            nn.Conv3d(width, width, kernel_size=1) for _ in range(layers)
        ])

        # 空间局部卷积 (depthwise 3x3x3)
        self.locals = nn.ModuleList(
            [nn.Conv3d(width, width, kernel_size=3, padding=1, groups=width)
             for _ in range(layers)]
        ) if use_local else None

        # GroupNorm + 残差缩放
        self.norms = nn.ModuleList([
            nn.GroupNorm(gn_groups, width) for _ in range(layers)
        ])
        self.gammas = nn.ParameterList([
            nn.Parameter(torch.tensor(float(residual_scale))) for _ in range(layers)
        ])

        # Dropout
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else None

        # 输出投影到 1 个通道 (delta_theta)
        self.proj = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width, 1, kernel_size=1)
        )

        # 全局 context -> bias
        self.ctx_mlp = nn.Linear(self.context_dim, width) if self.context_dim > 0 else None

        # 每层的 FiLM
        self.layer_films = nn.ModuleList()
        if self.context_dim > 0:
            for _ in range(layers):
                lin = nn.Linear(self.context_dim, 2 * width)
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
                self.layer_films.append(lin)

    # ---------- 位置编码 ----------
    @staticmethod
    def _coords(Z, Y, X, device, dtype):
        Z = int(Z)
        Y = int(Y)
        X = int(X)
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

    # ---------- 前向 ----------
    def forward(self, x, ctx: torch.Tensor = None):
        """
        x   : (B, in_c, Z, Y, X)，一般为 [theta, Ms, S]
        ctx : (B, context_dim=10)，一般为 [ctx_glb(9), log_dtau(1)]
        """
        B, _, Z, Y, X = x.shape

        # 位置编码
        if self.add_coords:
            pe = self._posenc(B, Z, Y, X, x.device, x.dtype)
            x = torch.cat([x, pe], dim=1)

        # lift -> 宽通道
        h = self.lift(x)

        # 全局 context 先加一层 bias
        if (self.ctx_mlp is not None) and (ctx is not None):
            h = h + self.ctx_mlp(ctx).view(B, -1, 1, 1, 1)

        # 逐层 FNO block
        for i, (spec_blk, w, gn) in enumerate(zip(self.specs, self.ws, self.norms)):
            y = spec_blk(h) + w(h)

            if self.use_local:
                y = y + self.locals[i](h)

            y = F.gelu(y)
            if self.drop is not None:
                y = self.drop(y)

            # FiLM
            if (ctx is not None) and (len(self.layer_films) > 0):
                ab = self.layer_films[i](ctx)
                gamma, beta = torch.chunk(ab, 2, dim=-1)
                y = y * (1.0 + gamma.view(B, -1, 1, 1, 1)) + beta.view(B, -1, 1, 1, 1)

            h = h + self.gammas[i] * y
            h = gn(h)

        out = self.proj(h)  # (B, 1, Z, Y, X)
        return out


# -------------------------------------------------------
# 工厂函数：构建和训练时一致的 FNO3D
# -------------------------------------------------------
def build_fno3d_heatsink_model(
    in_c=3,
    width=24,
    layers=4,
    modes=(12, 24, 4),
    dropout=0.05,
    context_dim=10,
    spec_pad_type="reflect",
    spec_pad=(8, 8, 8),
    ufno_scales=3,
    add_coords=True,
    fourier_k=8,
    use_local=True,
    gn_groups=1,
    residual_scale=0.5,
    device: str = "cuda:0",
):
    """
    构建与训练脚本一致的 FNO3D 模型（不含 HHead）。
    参数应与训练脚本里的 --width / --layers / --mz/--my/--mx 等一致。
    """
    model = FNO3D(
        in_c=in_c,
        width=width,
        modes=modes,
        layers=layers,
        add_coords=add_coords,
        fourier_k=fourier_k,
        use_local=use_local,
        gn_groups=gn_groups,
        residual_scale=residual_scale,
        dropout=dropout,
        context_dim=context_dim,
        spec_pad_type=spec_pad_type,
        spec_pad=spec_pad,
        ufno_scales=ufno_scales,
    )
    model.to(device)
    return model


# -------------------------------------------------------
# 从训练 ckpt 加载权重
# -------------------------------------------------------
def load_fno_checkpoint(ckpt_path: str, model: nn.Module, device: str = "cuda:0"):
    """
    从训练脚本保存的 ckpt 中加载 model 权重。
    ckpt 通常由 save_ckpt(...) 保存，里面包含:
        {"model": state_dict, "args": ..., "epoch": int, "hhead":..., "optim":...}

    本函数只关心 "model" 和 "epoch"，忽略 optimizer / hhead。
    会自动处理旧版本与多尺度 uFNO 之间的 spec 参数映射。
    返回:
        epoch (int 或 None)
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" in ckpt:
        ckpt_state = ckpt["model"]
    else:
        # 兜底：万一直接存的就是 state_dict
        ckpt_state = ckpt

    model_state = model.state_dict()

    # 兼容：旧版 "specs.X.weight" -> 新版 "specs.X.specs.0.weight"
    remapped = {}
    for k, v in ckpt_state.items():
        if k.startswith("specs.") and (".weight" in k or ".gate" in k) and (".specs." not in k):
            # 例如 "specs.0.weight" -> "specs.0.specs.0.weight"
            parts = k.split(".")
            if len(parts) == 3 and parts[0] == "specs":
                L = parts[1]
                tail = parts[2]
                new_k = f"specs.{L}.specs.0.{tail}"
                remapped[new_k] = v
            else:
                remapped[k] = v
        else:
            remapped[k] = v

    # 过滤掉 shape 不匹配的 key
    filtered = {}
    skipped = []
    for k, v in remapped.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    if skipped:
        print(f"[load_fno_checkpoint] skip {len(skipped)} keys due to name/shape mismatch (modes/ufno_scales 变更时属正常).")

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[load_fno_checkpoint] missing keys: {missing}")
    if unexpected:
        print(f"[load_fno_checkpoint] unexpected keys: {unexpected}")

    epoch = ckpt.get("epoch", None) if isinstance(ckpt, dict) else None
    if epoch is not None:
        print(f"[load_fno_checkpoint] loaded epoch={epoch} from {ckpt_path}")
    else:
        print(f"[load_fno_checkpoint] loaded weights from {ckpt_path} (epoch unknown)")

    return epoch


__all__ = [
    "FNO3D",
    "build_fno3d_heatsink_model",
    "load_fno_checkpoint",
]
