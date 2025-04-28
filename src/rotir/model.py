"""
Created on Wed Feb 15 15:07:16 2023

@author: rw17789

Edits then performed later for this repo by Richard Lane (mh19137)

This is a consolidated and modified version of the original RoTIR model code,
I've put it all in one file (for ease of use - it's not like I'll be changing much here)
and added some extra documentation, type hints, etc.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from e2cnn import gspaces, nn as e2nn


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(
                ~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float("-inf")
            )

        # Compute the attention and the weighted average
        softmax_temp = 1.0 / queries.size(3) ** 0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class LoFTREncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, attention="linear"):
        super(LoFTREncoderLayer, self).__init__()

        # self.dim = d_model // nhead
        # self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == "linear" else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, E, C]
            source (torch.Tensor): [N, S, E, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        # bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(
            query
        )  # .view(bs, -1, self.nhead, self.dim)  # [N, L, E, C]
        key = self.k_proj(key)  # .view(bs, -1, self.nhead, self.dim)  # [N, S, E, C]
        value = self.v_proj(value)  # .view(bs, -1, self.nhead, self.dim)
        message = self.attention(
            query, key, value, q_mask=x_mask, kv_mask=source_mask
        )  # [N, L, E, C]
        message = self.merge(
            message
        )  # .view(bs, -1, self.nhead*self.dim))  # [N, L, E, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=-1))
        message = self.norm2(message)

        return x + message


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256)):
        super().__init__()
        assert d_model % 4 == 0
        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float()
            * (-torch.log(torch.Tensor([1e4]) / (d_model // 2)))
        )
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer(
            "pe", pe.unsqueeze(0).unsqueeze(0), persistent=False
        )  # [1, 1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        assert x.dim() == 5
        assert self.pe.size(2) == x.size(2)
        for i in range(2):
            assert self.pe.size(i - 2) >= x.size(i - 2)

        return x + self.pe[:, :, :, : x.size(3), : x.size(4)]


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, d_model, nhead, layer_names, attention_type):
        super(LocalFeatureTransformer, self).__init__()

        # self.config = config
        self.d_model = d_model
        self.nhead = nhead
        self.layer_names = layer_names
        # encoder_layer = LoFTREncoderLayer(d_model, nhead, attention_type)
        # self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self.layers = nn.ModuleList(
            [
                LoFTREncoderLayer(d_model, nhead, attention_type)
                for _ in self.layer_names
            ]
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, E, C]
            feat1 (torch.Tensor): [N, S, E, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(-1) and self.d_model == feat1.size(
            -1
        ), "the feature number of src and transformer must be equal"
        assert feat0.size(-1) == feat1.size(
            -1
        ), "number of layers of features must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == "self":
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == "cross":
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1


class Block(torch.nn.Module):
    def __init__(self, in_type, out_type, down=False, double=False):
        super().__init__()
        self.frame = e2nn.SequentialModule(
            e2nn.R2Conv(
                in_type,
                out_type,
                kernel_size=4 if down else 3,
                stride=2 if down else 1,
                padding=1,
            ),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True),
            e2nn.R2Conv(
                out_type,
                out_type,
                kernel_size=4 if down and double else 3,
                stride=2 if down and double else 1,
                padding=1,
            ),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True),
        )
        if in_type.size != out_type.size:
            self.link = e2nn.SequentialModule(
                e2nn.R2Conv(in_type, out_type, kernel_size=1),
                e2nn.InnerBatchNorm(out_type),
            )
        else:
            self.link = None
        if down:
            self.down = e2nn.PointwiseAvgPool(
                out_type, 4 if double else 2, 4 if double else 2
            )
        else:
            self.down = None

    def forward(self, x):
        x1 = self.frame(x)
        x0 = self.link(x) if self.link is not None else x
        if self.down is not None:
            x0 = self.down(x0)
        return x1 + x0


class Feature_Extraction(torch.nn.Module):

    def __init__(
        self,
        in_channel,
        hidden_channel=64,
        n_rotation=4,
        bone_kernel=[False, False],
        num_layers=3,
        up_progress=True,
    ):
        super().__init__()

        self.r2_act = gspaces.Rot2dOnR2(N=n_rotation)
        self.in_type = e2nn.FieldType(
            self.r2_act, in_channel * [self.r2_act.trivial_repr]
        )
        hidden_channel = hidden_channel // n_rotation
        self.hd = hidden_channel
        out_type = e2nn.FieldType(
            self.r2_act, hidden_channel * [self.r2_act.regular_repr]
        )

        self.layer0 = Block(self.in_type, out_type, True, bone_kernel[0])

        self.down_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            in_type = out_type
            out_type = e2nn.FieldType(
                self.r2_act, hidden_channel * (2**i) * [self.r2_act.regular_repr]
            )
            self.down_layers.append(
                Block(in_type, out_type, True if i != 0 else bone_kernel[1])
            )

        self.bottleneck = e2nn.R2Conv(out_type, out_type, 1)

        d3_vector_out_type = e2nn.FieldType(
            self.r2_act,
            hidden_channel * (2 ** (num_layers - 1)) * [self.r2_act.irrep(1)],
        )
        self.layer3_out = e2nn.R2Conv(out_type, d3_vector_out_type, kernel_size=1)

        self.up_progress = up_progress
        if self.up_progress:
            self.up_layers = torch.nn.ModuleList()
            for i in range(2):  # num_layers - 1 or 2
                in_type = out_type
                out_type = e2nn.FieldType(
                    self.r2_act,
                    hidden_channel
                    * (2 ** (num_layers - 2 - i))
                    * [self.r2_act.regular_repr],
                )
                self.up_layers.append(
                    e2nn.SequentialModule(
                        e2nn.R2Upsampling(in_type, 2), e2nn.R2Conv(in_type, out_type, 1)
                    )
                )
                self.up_layers.append(Block(out_type, out_type))

            d1_regular_out_type = e2nn.FieldType(
                self.r2_act, hidden_channel * [self.r2_act.regular_repr]
            )
            self.layer1_out = e2nn.R2Conv(out_type, d1_regular_out_type, 1)
            self.gpool = e2nn.GroupPooling(d1_regular_out_type)

    def forward(self, input):
        x = e2nn.GeometricTensor(input, self.in_type)

        x = self.layer0(x)

        down_list = []
        for _layer in self.down_layers:
            x = _layer(x)
            down_list.append(x)
            # print(x.shape)

        x = self.bottleneck(x)

        d3_out = self.layer3_out(x)

        feature_out3 = rearrange(d3_out.tensor, "b (c d) h w -> b c d h w", d=self.hd)

        if self.up_progress:
            for _up, _layer, _x in zip(
                self.up_layers[0::2], self.up_layers[1::2], down_list[-2::-1]
            ):
                x = _up(x)
                x = x + _x
                x = _layer(x)
                # print(x.shape)

            d1_out = self.gpool(self.layer1_out(x))

            feature_out1 = rearrange(
                d1_out.tensor, "b c (d1 h) (d2 w) -> b (d1 d2) c h w", d1=4, d2=4
            )  # 2 * (nume_layers -1) or 4

        feature = (
            torch.cat([feature_out3, feature_out1], dim=1)
            if self.up_progress
            else feature_out3
        )

        return feature


class ImageRegistration(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_channel = (
            config["Backbone"]["hidden_channel"] // config["Backbone"]["n_rotation"]
        )

        self.backbone = Feature_Extraction(**config["Backbone"])

        self.pos_encoding = PositionEncodingSine(
            d_model=hidden_channel,
            # **config["max_shape"]
            max_shape=(256, 256),
        )

        self.feature_transformer = LocalFeatureTransformer(
            d_model=hidden_channel, **config["Transformer"]
        )

        self.matching_algorithm = config["Matching_algorithm"]["Type"]
        if self.matching_algorithm == "sinkhorn":
            alpha = config["Matching_algorithm"]["alpha"]
            self.bin_score = nn.Parameter(
                torch.tensor(float(alpha), requires_grad=True)
            )
            self.skh_iter = config["Matching_algorithm"]["iters"]

        layer_depth = config["Transformer"]["nhead"]

        self.map_projection = nn.Sequential(
            nn.Linear(layer_depth, 64),
            nn.Dropout(0.25),
            nn.Linear(64, 64),
            nn.Dropout(0.25),
            nn.Linear(64, 6),
        )

    def forward(self, data_dict):
        image_init = data_dict["Template_image"]
        image_term = data_dict["Target_image"]

        feat_init = self.backbone(image_init)
        feat_term = self.backbone(image_term)

        feat_init = rearrange(
            self.pos_encoding(feat_init), "n c1 c2 h w -> n (h w) c1 c2"
        )
        feat_term = rearrange(
            self.pos_encoding(feat_term), "n c1 c2 h w -> n (h w) c1 c2"
        )

        if "Template_square_mask" in data_dict and "Target_square_mask" in data_dict:
            mask_init = rearrange(data_dict["Template_square_mask"], "n h w -> n (h w)")
            mask_term = rearrange(data_dict["Target_square_mask"], "n h w -> n (h w)")
        else:
            mask_init = None
            mask_term = None

        feat_init, feat_term = self.feature_transformer(
            feat_init, feat_term, mask_init, mask_term
        )

        Ch = feat_init.shape[-1]

        feat_init = feat_init.div(Ch**0.5)
        feat_term = feat_term.div(Ch**0.5)

        matching_map = torch.einsum("nlec, nsec -> nlse", feat_init, feat_term)

        matching_map = self.map_projection(matching_map)

        score_map = matching_map[..., 0]

        if "Template_square_mask" in data_dict and "Target_square_mask" in data_dict:
            score_map.masked_fill_(
                ~(torch.einsum("ij, ik -> ijk", mask_init, mask_term)).bool(), -1e9
            )

        if self.matching_algorithm == "sinkhorn":
            score_map = self.log_optimal_transport(
                score_map, self.bin_score, self.skh_iter
            )  # shape: (Bs L+1 S+1)
        else:
            matching_map.mul_(10)
            score_map = F.softmax(score_map, 1) * F.softmax(
                score_map, 2
            )  # shape: (Bs L S)

        angle_map = F.normalize(matching_map[..., 1:3], dim=-1)
        scale_map = matching_map[..., 3:4]  # if matching_map.size(-1) == 6 else None
        trans_map = matching_map[..., -2:]

        return {
            "score_map": score_map,
            "angle_map": angle_map,
            "scale_map": scale_map,
            "trans_map": trans_map,
        }

    def log_optimal_transport(self, scores, alpha, iters=3):
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m * one).to(scores), (n * one).to(scores)

        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha = alpha.expand(b, 1, 1)

        couplings = torch.cat(
            [torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1
        )

        norm = -(ms + ns).log()
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
        Z = Z - norm  # multiply probabilities by M+N
        return Z.exp()

    def log_sinkhorn_iterations(self, Z, log_mu, log_nu, iters):
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(iters):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        return Z + u.unsqueeze(2) + v.unsqueeze(1)
