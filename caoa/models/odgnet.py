"""CAPCM completion network: ODGNet's `UpTrans`, ported.

Ported from `ODGNet/models/UpTrans.py` and the parts of `ODGNet/models/pointnet.py` it uses (original code by
Pingping Cai). Module and attribute names are unchanged, so ODGNet checkpoints load with `strict=True`; the CUDA ops
are replaced by `caoa.models.pointops` / `caoa.models.chamfer`.

Input: (B, N, 3) points, optionally with per-point features (B, N, F) such as the context flag (`in_channels` = 3 + F).
Output: (Pc, P3, P1, P2, cons1, cons2); P3 is the final completion (num_pred points).
"""

import math

import torch
from torch import einsum, nn

from caoa.models.chamfer import ChamferDistanceL1, ChamferDistanceL2
from caoa.models.pointops import furthest_point_sample, gather_operation, grouping_operation


# --- building blocks (ODGNet/models/pointnet.py) ---------------------------------------------

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True, activation_fn=torch.relu):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)  # created even when unused, as in the original
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)
        if self.activation_fn is not None:
            out = self.activation_fn(out)
        return out


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super().__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        return self.conv_2(torch.relu(self.conv_1(x))) + shortcut


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_knn(nsample, xyz, new_xyz, include_self=True):
    """k-NN of new_xyz in xyz. The original fully sorts each row; topk returns the same
    neighbours in the same order (only exactly tied, i.e. duplicated, points may swap)."""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)
    _, indices = torch.topk(sqrdists, nsample + pad, dim=-1, largest=False, sorted=True)
    return indices[:, :, pad:].int()


def sample_and_group_knn(xyz, points, npoint, k, use_xyz=True, idx=None):
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()
    new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint))
    if idx is None:
        idx = query_knn(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz, idx)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)
    if points is not None:
        grouped_points = grouping_operation(points, idx)
        new_points = torch.cat([grouped_xyz, grouped_points], 1) if use_xyz else grouped_points
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    b, _, nsample = xyz.shape
    device = xyz.device
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float, device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample, device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        new_points = (torch.cat([xyz, points], 1) if use_xyz else points).unsqueeze(2)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


class PointNet_SA_Module_KNN(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, if_bn=True, group_all=False, use_xyz=True, if_idx=False):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3
        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp[:-1]:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv.append(Conv2d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points, idx=None):
        if self.group_all:
            new_xyz, new_points, idx, _ = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, _ = sample_and_group_knn(xyz, points, self.npoint, self.nsample, self.use_xyz, idx=idx)
        new_points = torch.max(self.mlp_conv(new_points), 3)[0]
        return (new_xyz, new_points, idx) if self.if_idx else (new_xyz, new_points)


class Transformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super().__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)
        self.pos_mlp = nn.Sequential(nn.Conv2d(3, pos_hidden_dim, 1), nn.BatchNorm2d(pos_hidden_dim), nn.ReLU(), nn.Conv2d(pos_hidden_dim, dim, 1))
        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1),
        )
        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        identity = x
        x = self.linear_start(x)
        b, dim, n = x.shape
        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)
        key = grouping_operation(key, idx_knn)
        qk_rel = query.reshape((b, -1, n, 1)) - key
        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)
        pos_embedding = self.pos_mlp(pos_rel)
        attention = torch.softmax(self.attn_mlp(qk_rel + pos_embedding), -1)
        value = value.reshape((b, -1, n, 1)) + pos_embedding
        agg = einsum("b c i j, b c i j -> b c i", attention, value)
        return self.linear_end(agg) + identity


def get_nearest_index(target, source, k=1, return_dis=False):
    inner = torch.bmm(target.transpose(1, 2), source)
    s_norm_2 = torch.sum(source ** 2, dim=1)
    t_norm_2 = torch.sum(target ** 2, dim=1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
    nearest_dis, nearest_index = torch.topk(d_norm_2, k=k, dim=-1, largest=False)
    return (nearest_index, nearest_dis) if return_dis else nearest_index


def indexing_neighbor(x, index):
    batch_size, num_points, k = index.size()
    id_0 = torch.arange(batch_size).view(-1, 1, 1)
    x = x.transpose(2, 1).contiguous()
    return x[id_0, index].permute(0, 3, 1, 2).contiguous()


class UpTransformer(nn.Module):
    def __init__(self, in_channel, out_channel, dim, n_knn=20, up_factor=2, use_upfeat=True,
                 pos_hidden_dim=64, attn_hidden_multiplier=4, scale_layer=nn.Softmax, attn_channel=True):
        super().__init__()
        self.n_knn = n_knn
        self.up_factor = up_factor
        self.use_upfeat = use_upfeat
        attn_out_channel = dim if attn_channel else 1
        self.mlp_v = MLP_Res(in_dim=in_channel * 2, hidden_dim=in_channel, out_dim=in_channel)
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        if use_upfeat:
            self.conv_upfeat = nn.Conv1d(in_channel, dim, 1)
        self.scale = scale_layer(dim=-1) if scale_layer is not None else nn.Identity()
        self.pos_mlp = nn.Sequential(nn.Conv2d(3, pos_hidden_dim, 1), nn.BatchNorm2d(pos_hidden_dim), nn.ReLU(), nn.Conv2d(pos_hidden_dim, dim, 1))
        attn_mlp = [nn.Conv2d(dim, dim * attn_hidden_multiplier, 1), nn.BatchNorm2d(dim * attn_hidden_multiplier), nn.ReLU()]
        if up_factor:
            attn_mlp.append(nn.ConvTranspose2d(dim * attn_hidden_multiplier, attn_out_channel, (up_factor, 1), (up_factor, 1)))
        else:
            attn_mlp.append(nn.Conv2d(dim * attn_hidden_multiplier, attn_out_channel, 1))
        self.attn_mlp = nn.Sequential(*attn_mlp)
        self.upsample1 = nn.Upsample(scale_factor=(up_factor, 1)) if up_factor else nn.Identity()
        self.upsample2 = nn.Upsample(scale_factor=up_factor) if up_factor else nn.Identity()
        self.conv_end = nn.Conv1d(dim, out_channel, 1)
        self.residual_layer = nn.Conv1d(in_channel, out_channel, 1) if in_channel != out_channel else nn.Identity()

    def forward(self, pos, key, query, upfeat):
        value = self.mlp_v(torch.cat([key, query], 1))
        identity = value
        key = self.conv_key(key)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape
        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        key = grouping_operation(key, idx_knn)
        qk_rel = query.reshape((b, -1, n, 1)) - key
        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)
        pos_embedding = self.pos_mlp(pos_rel)
        if self.use_upfeat:
            upfeat = self.conv_upfeat(upfeat)
            upfeat_rel = upfeat.reshape((b, -1, n, 1)) - grouping_operation(upfeat, idx_knn)
        else:
            upfeat_rel = torch.zeros_like(qk_rel)
        attention = self.scale(self.attn_mlp(qk_rel + pos_embedding + upfeat_rel))
        value = grouping_operation(value, idx_knn) + pos_embedding + upfeat_rel
        value = self.upsample1(value)
        agg = torch.einsum("b c i j, b c i j -> b c i", attention, value)
        y = self.conv_end(agg)
        identity = self.upsample2(self.residual_layer(identity))
        return y + identity


class UpLayer(nn.Module):
    def __init__(self, dim, seed_dim, up_factor=2, i=0, radius=1, n_knn=20, interpolate="three", attn_channel=True):
        super().__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.n_knn = n_knn
        self.interpolate = interpolate
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + seed_dim, layer_dims=[256, dim])
        self.uptrans1 = UpTransformer(dim, dim, dim=64, n_knn=self.n_knn, use_upfeat=True, up_factor=None)
        self.uptrans2 = UpTransformer(dim, dim, dim=64, n_knn=self.n_knn, use_upfeat=True, attn_channel=attn_channel, up_factor=self.up_factor)
        self.upsample = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=dim * 2, hidden_dim=dim, out_dim=dim)
        self.mlp_delta = MLP_CONV(in_channel=dim, layer_dims=[64, 3])

    def forward(self, pcd_prev, seed, seed_feat, K_prev=None):
        if self.interpolate == "nearest":
            idx = get_nearest_index(pcd_prev, seed)
            feat_upsample = indexing_neighbor(seed_feat, idx).squeeze(3)
        elif self.interpolate == "three":
            idx, dis = get_nearest_index(pcd_prev, seed, k=3, return_dis=True)
            dist_recip = 1.0 / (dis + 1e-8)
            weight = dist_recip / torch.sum(dist_recip, dim=2, keepdim=True)
            feat_upsample = torch.sum(indexing_neighbor(seed_feat, idx) * weight.unsqueeze(1), dim=-1)
        else:
            raise ValueError(f"Unknown Interpolation: {self.interpolate}")
        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1, torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))), feat_upsample], 1)
        Q = self.mlp_2(feat_1)
        H = self.uptrans1(pcd_prev, K_prev if K_prev is not None else Q, Q, upfeat=feat_upsample)
        feat_child = self.uptrans2(pcd_prev, K_prev if K_prev is not None else H, H, upfeat=feat_upsample)
        H_up = self.upsample(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))
        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius ** self.i
        return self.upsample(pcd_prev) + delta, K_curr


# --- UpTrans (ODGNet/models/UpTrans.py) --------------------------------------------------------

class UNet(nn.Module):
    def __init__(self, in_channels=3, dim_feat=384, num_seeds=1024, seed_fea=128, num_dicts=256):
        super().__init__()
        self.num_seed = num_seeds
        self.sa_module_1 = PointNet_SA_Module_KNN(num_seeds // 2, 16, in_channels, [32, seed_fea], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(seed_fea, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(num_seeds // 8, 16, seed_fea, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = Transformer(256, dim=64)
        self.sa_module_4 = PointNet_SA_Module_KNN(None, None, 256, [384, dim_feat], group_all=True, if_bn=False)
        self.ps_0 = nn.ConvTranspose1d(dim_feat, 256, num_seeds // 16, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 256, hidden_dim=256, out_dim=256)
        self.ps_1 = nn.ConvTranspose1d(256, 256, 2, 2, bias=True)
        self.ps_2 = nn.ConvTranspose1d(256, 128, 2, 2, bias=True)
        self.refine_1 = FeaRefine(dim=256, hidd_dim=64, num_dicts=num_dicts)
        self.refine_2 = FeaRefine(dim=128, hidd_dim=64, num_dicts=num_dicts // 2)
        self.mlp_3 = MLP_Res(in_dim=seed_fea, hidden_dim=32, out_dim=3)

    def forward(self, point_cloud, feats=None):
        l0_xyz = point_cloud
        l0_fea = feats if feats is not None else point_cloud
        l1_xyz, l1_fea, _ = self.sa_module_1(l0_xyz, l0_fea)
        l1_fea = self.transformer_1(l1_fea, l1_xyz)
        l2_xyz, l2_fea, _ = self.sa_module_2(l1_xyz, l1_fea)
        l2_fea = self.transformer_2(l2_fea, l2_xyz)
        _, l4_fea = self.sa_module_4(l2_xyz, l2_fea)
        u0_fea = self.ps_0(l4_fea)
        u0_fea = self.mlp_1(torch.cat([u0_fea, l4_fea.repeat((1, 1, u0_fea.size(2)))], 1))
        u1_fea = self.ps_1(u0_fea)
        u1_fea, cons1 = self.refine_1(u1_fea)
        u1_fea = torch.cat([l2_fea, u1_fea], dim=2)
        u2_fea = self.ps_2(u1_fea)
        u2_fea, cons2 = self.refine_2(u2_fea)
        u2_xyz = self.mlp_3(u2_fea)
        u2_xyz = torch.cat([l1_xyz, u2_xyz], dim=2)
        u2_fea = torch.cat([l1_fea, u2_fea], dim=2)
        return l4_fea, u2_xyz, u2_fea, cons1, cons2


class FeaRefine(nn.Module):
    def __init__(self, dim=512, hidd_dim=64, num_dicts=128):
        super().__init__()
        self.num_dicts = num_dicts
        self.dict_embed = nn.Embedding(dim, num_dicts)
        self.query = nn.Conv1d(dim, hidd_dim, 1)
        self.key = nn.Conv1d(dim, hidd_dim, 1)

    def forward(self, feat):
        B, C, N = feat.size()
        dict_fea = self.dict_embed.weight
        feature_memory = torch.nn.functional.normalize(dict_fea, dim=0)
        cons = feature_memory.transpose(0, 1) @ feature_memory
        dict_fea = dict_fea.unsqueeze(0).repeat(B, 1, 1)
        q = self.query(feat)
        value = dict_fea
        k = self.key(value)
        d_k = k.size(1)
        scores = torch.softmax(torch.matmul(q.transpose(-2, -1), k) / math.sqrt(d_k), dim=-1)
        output = torch.matmul(scores, value.transpose(-2, -1))
        return (output.transpose(-2, -1) + feat) / 2, cons


class PostProcess(nn.Module):
    def __init__(self, embed_dim=128, upscale=(1, 4, 4)):
        super().__init__()
        self.up_layers = nn.ModuleList(
            UpLayer(dim=embed_dim, seed_dim=embed_dim, up_factor=factor, i=i, n_knn=20, radius=1, interpolate="three", attn_channel=True)
            for i, factor in enumerate(upscale)
        )

    def forward(self, seed, fea):
        pred_pcds = [seed.permute(0, 2, 1).contiguous()]
        K_prev = None
        pcd = seed
        for layer in self.up_layers:
            pcd, K_prev = layer(pcd, seed, fea, K_prev)
            pred_pcds.append(pcd.permute(0, 2, 1).contiguous())
        return pred_pcds


class UpTrans(nn.Module):
    def __init__(self, num_pred=8192, in_channels=3, dim_feat=512, num_seeds=1024, seed_fea=128, num_dicts=512, upscales=(1, 2, 4), l2_loss=0, **_):
        super().__init__()
        self.num_pred = num_pred
        self.in_channels = in_channels
        self.use_L2 = l2_loss
        self.feat_extractor = UNet(in_channels=in_channels, dim_feat=dim_feat, num_seeds=num_seeds, seed_fea=seed_fea, num_dicts=num_dicts)
        self.decoder = PostProcess(embed_dim=seed_fea, upscale=upscales)
        self.loss_func = ChamferDistanceL1() if self.use_L2 == 0 else ChamferDistanceL2()

    @classmethod
    def from_config(cls, model_cfg: dict):
        """From the `model:` section of a CAPCM config."""
        return cls(**{k: v for k, v in model_cfg.items() if k != "NAME"})

    def get_loss(self, ret, gt):
        Pc, P3, P1, P2, cons1, cons2 = ret
        cd0, cd1, cd2, cd3 = (self.loss_func(P, gt) for P in (Pc, P1, P2, P3))
        return (cd0 + cd1 + cd2 + cd3) * 1e3, cd0, cd3

    def get_constrain(self, ret):
        *_, cons1, cons2 = ret
        eye1 = torch.eye(cons1.size(0), device=cons1.device)
        eye2 = torch.eye(cons2.size(0), device=cons2.device)
        return (1e3 * torch.nn.functional.mse_loss(cons1, eye1, reduction="mean"),
                1e3 * torch.nn.functional.mse_loss(cons2, eye2, reduction="mean"))

    def forward(self, partial_point_cloud, feats=None):
        in_pcd = partial_point_cloud.permute(0, 2, 1).contiguous()
        in_feats = None
        if feats is not None:
            in_feats = torch.cat([in_pcd, feats.permute(0, 2, 1).contiguous()], dim=1)
        _, xyz, fea, cons1, cons2 = self.feat_extractor(in_pcd, in_feats)
        pcd, p1, p2, p3 = self.decoder(xyz, fea)
        return pcd, p3, p1, p2, cons1, cons2


def load_checkpoint(model: nn.Module, path, strict: bool = True) -> dict:
    """Load a CAPCM `ckpt-*.pth` (ODGNet's format; a "module." prefix is stripped).

    Returns the checkpoint's bookkeeping: epoch, metrics, best_metrics.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    weights = ckpt.get("model") if ckpt.get("model") is not None else ckpt.get("base_model")
    if weights is None:
        raise RuntimeError(f"no model weights in {path}")
    model.load_state_dict({k.replace("module.", ""): v for k, v in weights.items()}, strict=strict)
    return {k: ckpt.get(k) for k in ("epoch", "metrics", "best_metrics")}
