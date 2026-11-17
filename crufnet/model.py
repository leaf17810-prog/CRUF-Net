from __future__ import annotations
import math
import torch
from .layers import ConvBNAct, MLP, AttentionBlock, PyramidDecoder, ContextEncoder
from torch import nn
from torch.nn import functional as F

def unit(x):
    return F.normalize(x.float(), p=2, dim=1, eps=1e-06).to(x.dtype)

class LayerNorm2d(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

def rope2d(x, h, w):
    d = x.shape[-1]
    freq = 10000.0 ** (-torch.arange(d // 4, device=x.device, dtype=torch.float32) / (d // 4))
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing='ij')
    angle = torch.cat((yy.reshape(-1, 1) * freq, xx.reshape(-1, 1) * freq), dim=-1)
    pairs = x.float().reshape(*x.shape[:-1], d // 2, 2)
    a, b = pairs.unbind(-1)
    return torch.stack((a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), dim=-1).flatten(-2).to(x.dtype)

class NonCausalStateMixer(nn.Module):

    def __init__(self, dim, d_state=64, expand=2, head_dim=64):
        super().__init__()
        self.inner = dim * expand
        self.state = d_state
        self.heads = self.inner // head_dim
        self.head_dim = head_dim
        self.in_proj = nn.Linear(dim, 2 * self.inner + 2 * d_state + self.heads, bias=False)
        self.local = nn.Conv2d(self.inner + 2 * d_state, self.inner + 2 * d_state, 3, padding=1, groups=self.inner + 2 * d_state)
        dt = torch.exp(torch.empty(self.heads).uniform_(math.log(0.001), math.log(0.1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
        self.D = nn.Parameter(torch.ones(self.heads))
        self.norm = nn.LayerNorm(self.inner)
        self.out_proj = nn.Linear(self.inner, dim, bias=False)

    def forward(self, tokens, h, w):
        batch, n, _ = tokens.shape
        z, xbc, dt = self.in_proj(tokens).split((self.inner, self.inner + 2 * self.state, self.heads), dim=-1)
        xbc = F.silu(self.local(xbc.transpose(1, 2).reshape(batch, -1, h, w))).flatten(2).transpose(1, 2)
        v, k, q = xbc.split((self.inner, self.state, self.state), dim=-1)
        dtype = v.dtype
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            v = v.float().reshape(batch, n, self.heads, self.head_dim).transpose(1, 2)
            dt = F.softplus(dt.float() + self.dt_bias.float()).transpose(1, 2).unsqueeze(-1)
            negative_da = -dt * self.A_log.float().exp()[None, :, None, None]
            weight = negative_da.softmax(dim=-2) * n
            q_plain = F.elu(q.float())[:, None] + 1
            k_plain = (F.elu(k.float())[:, None] + 1) * weight
            q_rot = rope2d(q_plain, h, w) / self.inner
            k_rot = rope2d(k_plain, h, w)
            denom = (q_plain / self.inner @ k_plain.mean(-2, keepdim=True).transpose(-2, -1)).clamp_min(1e-06)
            state = k_rot.transpose(-2, -1) / math.sqrt(n) @ (v / math.sqrt(n))
            out = q_rot @ state / denom + v * self.D.float()[None, :, None, None]
            out = out.transpose(1, 2).reshape(batch, n, self.inner)
        out = self.norm(out.to(dtype)) * F.silu(z)
        return self.out_proj(out)

class NonCausalStateBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = NonCausalStateMixer(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=4)

    def forward(self, x, h, w):
        x = x + self.mixer(self.norm1(x), h, w)
        return x + self.mlp(self.norm2(x))

class ContextStage(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.state_blocks = nn.ModuleList([NonCausalStateBlock(dim) for _ in range(2)])
        self.attn_blocks = nn.ModuleList([AttentionBlock(dim, num_heads=10) for _ in range(2)])

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        for block in self.state_blocks:
            t = block(t, h, w)
        for block in self.attn_blocks:
            t = block(t)
        return t.transpose(1, 2).reshape(b, c, h, w)

def raw_descriptors(raw):
    r = raw.float().clamp(0, 1)
    i0, i45, i90, i135 = r.chunk(4, 1)
    mean = r.mean(1, keepdim=True)
    denom = (2 * mean).clamp_min(0.02)
    q = ((i0 - i90) / denom).clamp(-2, 2)
    u = ((i45 - i135) / denom).clamp(-2, 2)
    rho = torch.sqrt(q.square() + u.square() + 1e-08).clamp(0, 2)
    closure = ((i0 + i90 - i45 - i135).abs() / (4 * mean).clamp_min(0.04)).clamp(0, 2)
    local_mean = F.avg_pool2d(mean, 3, stride=1, padding=1, count_include_pad=False)
    local_sq = F.avg_pool2d(mean.square(), 3, stride=1, padding=1, count_include_pad=False)
    contrast = ((local_sq - local_mean.square()).clamp_min(0) + 1e-08).sqrt() / local_mean.clamp_min(0.02)
    saturated = (r >= 0.995).float().mean(1, keepdim=True)
    dark = (r <= 0.005).float().mean(1, keepdim=True)
    return torch.cat((mean, q, u, rho, closure, contrast.clamp_max(2), saturated, dark), dim=1).to(raw.dtype)

def window_partition(x, size=8):
    b, c, h, w = x.shape
    return x.reshape(b, c, h // size, size, w // size, size).permute(0, 2, 4, 3, 5, 1).reshape(-1, size * size, c)

def window_reverse(x, batch, h, w, size=8):
    return x.reshape(batch, h // size, w // size, size, size, -1).permute(0, 5, 1, 3, 2, 4).reshape(batch, -1, h, w)

class ConditionalMixingBlock(nn.Module):

    def __init__(self, channels, shift):
        super().__init__()
        self.channels, self.shift, self.window = (channels, shift, 8)
        self.branch_channels = channels // 2
        self.heads = 3
        self.head_dim = self.branch_channels // self.heads
        self.norm = LayerNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.qkv_local = nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3)
        self.acceptance = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 2, 1))
        nn.init.zeros_(self.acceptance[-1].weight)
        nn.init.zeros_(self.acceptance[-1].bias)
        self.channel_to_spatial = nn.Conv2d(self.branch_channels, self.branch_channels, 1)
        self.spatial_to_channel = nn.Conv2d(self.branch_channels, 1, 1)
        self.spatial_temperature = nn.Parameter(torch.full((self.heads, 1, 1), math.log(10.0)))
        self.channel_temperature = nn.Parameter(torch.full((self.heads, 1, 1), math.log(10.0)))
        self.geometry_logit = nn.Parameter(torch.full((self.heads,), -3.0))
        self.rel_bias = nn.Parameter(torch.zeros(225, self.heads))
        y, x = torch.meshgrid(torch.arange(8), torch.arange(8), indexing='ij')
        coords = torch.stack((y, x)).flatten(1)
        rel = coords[:, :, None] - coords[:, None, :]
        self.register_buffer('relative_index', (rel[0] + 7) * 15 + rel[1] + 7, persistent=False)
        self.merge = nn.Conv2d(channels, channels, 1)
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.norm2 = LayerNorm2d(channels)
        self.ffn = nn.Sequential(nn.Conv2d(channels, channels * 2, 1), nn.GELU(), nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2), nn.GELU(), nn.Conv2d(channels * 2, channels, 1))
        self.ffn_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def geometry_bias(self, normal_windows):
        n = F.normalize(normal_windows.float(), dim=-1, eps=1e-06)
        similarity = (n @ n.transpose(-2, -1)).clamp(-1, 1)
        return self.geometry_logit.sigmoid()[None, :, None, None] * (similarity[:, None] - 1)

    def channel_attention(self, q, k, v):
        b, _, h, w = q.shape
        q, k, v = [z.reshape(b, self.heads, self.head_dim, h * w) for z in (q, k, v)]
        a = F.normalize(q.float(), dim=-1) @ F.normalize(k.float(), dim=-1).transpose(-2, -1)
        a = a * self.channel_temperature.clamp(max=math.log(100)).exp()
        out = a.softmax(-1).to(v.dtype) @ v
        return out.reshape(b, self.branch_channels, h, w)

    def window_attention(self, q, k, v, normal):
        b, _, h, w = q.shape
        ph, pw = (-h % 8, -w % 8)
        hp, wp = (h + ph, w + pw)
        valid = q.new_ones((1, 1, h, w))
        arrays = [F.pad(z, (0, pw, 0, ph)) for z in (q, k, v, normal, valid)]
        if self.shift:
            arrays = [torch.roll(z, (-self.shift, -self.shift), (-2, -1)) for z in arrays]
        qw, kw, vw, nw, validw = [window_partition(z) for z in arrays]
        qw, kw, vw = [z.reshape(-1, 64, self.heads, self.head_dim).transpose(1, 2) for z in (qw, kw, vw)]
        logits = F.normalize(qw.float(), dim=-1) @ F.normalize(kw.float(), dim=-1).transpose(-2, -1)
        logits = logits * self.spatial_temperature.clamp(max=math.log(100)).exp()
        logits = logits + self.rel_bias[self.relative_index].permute(2, 0, 1)[None] + self.geometry_bias(nw)
        n_windows = hp // 8 * (wp // 8)
        blocked = (~validw.squeeze(-1).bool())[:, None, :].expand(-1, 64, -1)
        if self.shift:
            region = q.new_zeros((1, 1, hp, wp))
            slices_h = (slice(0, -8), slice(-8, -self.shift), slice(-self.shift, None))
            slices_w = (slice(0, -8), slice(-8, -self.shift), slice(-self.shift, None))
            for i, sh in enumerate(slices_h):
                for j, sw in enumerate(slices_w):
                    region[:, :, sh, sw] = 3 * i + j
            rw = window_partition(region).squeeze(-1)
            blocked = blocked | (rw[:, :, None] != rw[:, None, :])
        blocked = blocked.repeat(b, 1, 1)
        logits = logits.masked_fill(blocked[:, None], -10000.0)
        attention = logits.softmax(-1).to(vw.dtype)
        y = (attention @ vw).transpose(1, 2).reshape(b * n_windows, 64, self.branch_channels)
        y = window_reverse(y, b, hp, wp)
        if self.shift:
            y = torch.roll(y, (self.shift, self.shift), (-2, -1))
        return y[:, :, :h, :w]

    def forward(self, x, descriptors, coarse_normal):
        if coarse_normal.shape[1] != 3 or coarse_normal.shape[-2:] != x.shape[-2:]:
            raise ValueError('Geometry condition must be a 3-channel predicted normal at the feature resolution')
        qs, qc, ks, kc, vs, vc = self.qkv_local(self.qkv(self.norm(x))).chunk(6, 1)
        accept_s, accept_c = (0.25 + 0.75 * self.acceptance(descriptors).sigmoid()).chunk(2, 1)
        channel_base = self.channel_attention(qc, kc, vc)
        vs = vs * (1 + 0.5 * self.channel_to_spatial(channel_base).tanh()) * accept_s
        spatial = self.window_attention(qs, ks, vs, coarse_normal.detach())
        channel = channel_base * (1 + 0.5 * self.spatial_to_channel(spatial).tanh()) * accept_c
        x = x + self.residual_scale * self.merge(torch.cat((spatial, channel), dim=1))
        return x + self.ffn_scale * self.ffn(self.norm2(x))

class ConditionalReconstructionUnit(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList([ConditionalMixingBlock(channels, 0), ConditionalMixingBlock(channels, 4)])

    def forward(self, x, descriptors, coarse_normal):
        descriptors = F.interpolate(descriptors, size=x.shape[-2:], mode='area')
        coarse_normal = unit(F.interpolate(coarse_normal, size=x.shape[-2:], mode='bilinear', align_corners=False))
        for block in self.blocks:
            x = block(x, descriptors, coarse_normal)
        return x

def blend_normals(base, candidate, acceptance):
    mixed = (1 - acceptance) * base + acceptance * candidate
    return torch.where(mixed.square().sum(1, keepdim=True) > 1e-12, unit(mixed), base)

class CorrectionUtility(nn.Module):

    def __init__(self, feature_channels=24):
        super().__init__()
        self.feature_norm = LayerNorm2d(feature_channels)
        self.predictor = nn.Sequential(nn.Conv2d(feature_channels + 8 + 9, 32, 1), nn.GELU(), nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.GELU(), nn.Conv2d(32, 16, 1), nn.GELU(), nn.Conv2d(16, 1, 1))
        self.training_strength = 1.0

    def forward(self, features, descriptors, base, candidate):
        geometry = torch.cat((base, candidate, candidate - base), 1).detach()
        cues = torch.cat((self.feature_norm(features), descriptors, geometry), 1)
        logits = self.predictor(cues)
        predicted = logits.sigmoid()
        strength = self.training_strength if self.training else 1.0
        acceptance = 1 - strength + strength * predicted
        return (blend_normals(base, candidate, acceptance), acceptance, logits)

class ConditionalDecoder(PyramidDecoder):

    def __init__(self):
        super().__init__(in_dims=(80, 160, 320, 640), fpn_dim=192)
        self.quarter_condition = ConditionalReconstructionUnit(96)
        self.half_condition = ConditionalReconstructionUnit(48)
        self.quarter_head = nn.Conv2d(96, 3, 1)
        self.half_head = nn.Conv2d(48, 3, 1)
        self.utility = CorrectionUtility(24)

    def forward(self, feats, out_size, descriptors):
        c1, c2, c3, c4 = feats
        p4 = self.smooth4(self.lat4(c4))
        p3 = self.smooth3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False))
        p1 = self.smooth1(self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode='bilinear', align_corners=False))
        d3 = self.up43(p4, p3)
        d2 = self.up32(d3, p2)
        d1 = self.up21(d2, p1)
        fq = self.final_up[0](d1)
        coarse = unit(self.quarter_head(fq))
        fq = self.quarter_condition(fq, descriptors, coarse)
        nq = unit(self.quarter_head(fq))
        fh = self.final_up[2](self.final_up[1](fq))
        fh = self.half_condition(fh, descriptors, nq)
        nh = unit(self.half_head(fh))
        full = self.final_up[4](self.final_up[3](fh))
        if full.shape[-2:] != out_size:
            full = F.interpolate(full, size=out_size, mode='bilinear', align_corners=False)
        candidate = unit(self.pred(full))
        base = unit(F.interpolate(nh, size=out_size, mode='bilinear', align_corners=False))
        descriptors = F.interpolate(descriptors, size=out_size, mode='area')
        normal, acceptance, logits = self.utility(full, descriptors, base, candidate)
        return {'normal': normal, 'aux_normals': {'quarter': nq, 'half': nh}, 'refinement': {'base': base, 'candidate': candidate, 'acceptance': acceptance, 'logits': logits}}

class CRUFNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = ContextEncoder(ContextStage)
        self.decoder = ConditionalDecoder()
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for unit_module in (self.decoder.quarter_condition, self.decoder.half_condition):
            for block in unit_module.blocks:
                nn.init.zeros_(block.acceptance[-1].weight)
                nn.init.zeros_(block.acceptance[-1].bias)
        nn.init.zeros_(self.decoder.utility.predictor[-1].weight)
        nn.init.zeros_(self.decoder.utility.predictor[-1].bias)

    def set_refinement_strength(self, strength):
        strength = float(strength)
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('Refinement strength must be finite and in [0,1]')
        self.decoder.utility.training_strength = strength

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 12:
            raise ValueError('Expected Bx12xHxW: raw4, P[1:4], diffuse, specular, view3')
        h, w = x.shape[-2:]
        if h < 32 or w < 32:
            raise ValueError('Both image dimensions must be at least 32 pixels')
        ph, pw = (-h % 32, -w % 32)
        padded = F.pad(x, (0, pw, 0, ph), mode='replicate') if ph or pw else x
        desc = raw_descriptors(padded[:, :4])
        result = self.decoder(self.backbone(padded), padded.shape[-2:], desc)
        result['normal'] = result['normal'][:, :, :h, :w]
        result['refinement'] = {k: v[:, :, :h, :w] for k, v in result['refinement'].items()}
        result['aux_normals']['quarter'] = result['aux_normals']['quarter'][:, :, :math.ceil(h / 4), :math.ceil(w / 4)]
        result['aux_normals']['half'] = result['aux_normals']['half'][:, :, :math.ceil(h / 2), :math.ceil(w / 2)]
        return result

    def diagnostics(self):
        blocks = list(self.decoder.quarter_condition.blocks) + list(self.decoder.half_condition.blocks)
        return {'parameters': sum((p.numel() for p in self.parameters())), 'geometry_strengths': [b.geometry_logit.detach().sigmoid().cpu().tolist() for b in blocks], 'residual_scale_mean': [float(b.residual_scale.detach().mean().cpu()) for b in blocks], 'observation_acceptance_bounds': [0.25, 1.0], 'geometry_condition': 'predicted_only_stop_gradient', 'global_pixel_attention_scales': ['1/16', '1/32'], 'window_attention_scales': ['1/4', '1/2'], 'utility_parameters': sum((p.numel() for p in self.decoder.utility.parameters())), 'utility_training_strength': self.decoder.utility.training_strength, 'utility_eval_strength': 1.0}

    def model_spec(self):
        spec = model_spec()
        spec['parameters'] = sum((p.numel() for p in self.parameters()))
        return spec

def build_model():
    return CRUFNet()

def model_spec():
    return {'name': 'CRUF-Net', 'input_channels': 12, 'input_order': ['I0', 'I45', 'I90', 'I135', 'P1', 'P2', 'P3', 'IrawD', 'IrawS', 'u', 'v', '1'], 'encoder_dimensions': [80, 160, 320, 640], 'pyramid_channels': 192, 'window_size': 8, 'window_shifts': [0, 4], 'normal_output_channels': 3, 'custom_cuda_required': False}
