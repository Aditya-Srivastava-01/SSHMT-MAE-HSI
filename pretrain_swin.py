import torch
import torch.nn as nn

import torch.utils.checkpoint as checkpoint
from utils import get_sinusoid_encoding_table
from einops import rearrange
from timm.models.layers import DropPath
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from group_window_attention import WindowAttention, GroupingModule, get_coordinates


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads, dim_head, dropout, withoutlinear=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        if not withoutlinear:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )
        self.withoutlinear = withoutlinear

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        q = q * self.scale
        dots = (q @ k.transpose(-2, -1))


        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)


        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if self.withoutlinear:
            return out
        out = self.to_out(out)
        return out


# class WindowAttention(nn.Module):
#
#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
#
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x, mask=None):
#
#         B_, N, C = x.shape
#
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))
#
#         if mask is not None:
#             nW = mask.shape[0]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#
#
#         else:
#             attn = self.softmax(attn)
#
#         attn = self.attn_drop(attn)
#
#         x = torch.einsum('bcij,bcjk->bcik', attn, v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x


# def window_partition(x, window_size):
#     B, N, C = x.shape
#     x = x.view(B, N // window_size, window_size, C)
#     windows = x.view(-1, window_size, C)
#     return windows
#
#
# def window_reverse(windows, window_size, N):
#     B = int(windows.shape[0] / (N / window_size))
#     x = windows.view(B, N // window_size, window_size, -1)
#     x = x.view(B, N, -1)
#     return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if self.input_resolution <= self.window_size:
            self.shift_size = 0
            self.window_size = self.input_resolution
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # if self.shift_size > 0:
        #     N = self.input_resolution
        #     img_mask = torch.zeros((1, N, 1))
        #     n_slices = (slice(0, -self.window_size),
        #                 slice(-self.window_size, -self.shift_size),
        #                 slice(-self.shift_size, None))
        #     cnt = 0
        #
        #     for n in n_slices:
        #         img_mask[:, n, :] = cnt
        #         cnt += 1
        #
        #     mask_windows = window_partition(img_mask, self.window_size)
        #     mask_windows = mask_windows.view(-1, self.window_size)
        #     attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(
        #         2)
        #     attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        #
        # else:
        #     attn_mask = None
        #
        # self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, attn_mask, rel_pos_idx):

        # N = self.input_resolution
        # B, L, C = x.shape
        # assert L == N, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)

        # W-MSA/SW-MSA
        x = self.attn(x, mask=attn_mask, pos_idx=rel_pos_idx)  # B*nW, N_vis, C

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # if self.shift_size > 0:
        #     shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
        # else:
        #     shifted_x = x
        #
        # x_windows = window_partition(shifted_x,
        #                              self.window_size)
        # attn_windows = self.attn(x_windows, mask=self.attn_mask)
        #
        # shifted_x = window_reverse(attn_windows, self.window_size, N)
        # if self.shift_size > 0:
        #     x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        # else:
        #     x = shifted_x
        # x = x.view(B, N, C)
        # x = shortcut + self.drop_path(x)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, group, output_dim,
                 mlp_ratio=.125, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,

                 normAll=False, PosEmb=None, ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.norm_layer = norm_layer
        self.window_size = window_size
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        else:
            self.shift_size = window_size // 2

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,)
            for i in range(depth)])

        # self.resNorm = nn.LayerNorm([self.input_resolution, self.dim, 2])
        # self.resLinear = nn.Linear(self.dim * 2, self.dim)
        # self.resLinearPreX = nn.Linear(self.dim, self.dim)
        # self.resLinearX = nn.Linear(self.dim, self.dim)


        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, group=group,
                                         outputdim=output_dim, normAll=normAll, PosEmb=PosEmb)
        else:
            self.downsample = None

    def forward(self, x, coords, patch_mask):
        # prepare the attention mask and relative position bias
        group_block = GroupingModule(self.window_size, 0)
        mask, pos_idx = group_block.prepare(coords, num_tokens=x.shape[1])
        if self.window_size < min(self.input_resolution):
            group_block_shift = GroupingModule(self.window_size, self.shift_size)
            mask_shift, pos_idx_shift = group_block_shift.prepare(coords, num_tokens=x.shape[1])
        else:
            # do not shift
            group_block_shift = group_block
            mask_shift, pos_idx_shift = mask, pos_idx

        # forward with grouping/masking
        for i, blk in enumerate(self.blocks):
            gblk = group_block if i % 2 == 0 else group_block_shift
            attn_mask = mask if i % 2 == 0 else mask_shift
            rel_pos_idx = pos_idx if i % 2 == 0 else pos_idx_shift
            x = gblk.group(x)
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask, rel_pos_idx)
            else:
                x = blk(x, attn_mask, rel_pos_idx)
            x = gblk.merge(x)

        # pre_x = x
        #
        # for blk in self.blocks:
        #     if self.use_checkpoint:
        #         x = checkpoint.checkpoint(blk, x)
        #     else:
        #         x = blk(x)
        #
        #
        # x = torch.cat((pre_x.unsqueeze(-1), x.unsqueeze(-1)), dim=-1)
        # x = self.resNorm(x)
        # pre_x, x = x[:, :, :, 0], x[:, :, :, 1]
        #
        # pre_x = self.resLinearPreX(pre_x)
        # x = self.resLinearX(x)
        # x = torch.cat((pre_x, x), dim=-1)
        # x = self.resLinear(x)

        if self.downsample is not None:


            x, coords, patch_mask = self.downsample(x, coords, patch_mask)
        return x, coords, patch_mask

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class TokenMerging(nn.Module):

    def __init__(self, input_resolution, dim, group, outputdim, norm_layer=nn.LayerNorm, normAll=False, PosEmb="Post"):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(group * dim, outputdim * dim, bias=False)
        self.norm = norm_layer(group * dim)
        # if normAll:
        #     self.norm = norm_layer([input_resolution // group, group * dim])
        # else:
        #     self.norm = norm_layer(group * dim)
        #
        # self.PosEmb = PosEmb
        # if PosEmb == "Post" or PosEmb == "Pre":
        #     self.posPreEmb = nn.Parameter(torch.randn((input_resolution // group, group * dim)))
        #     self.posPostEmb = nn.Parameter(torch.randn((input_resolution // group, dim * outputdim)))
        # self.group = group

    def forward(self, x, coords_prev, mask_prev):
        """
        x: B, N, C
        """
        N = self.input_resolution
        B, L, C = x.shape
        # assert L == N, "input feature has wrong size"
        assert N % self.group == 0, f"x size ({N}) are not even."

        mask = mask_prev.reshape(H // 2, 2, W // 2, 2).permute(0, 2, 1, 3).reshape(-1)
        coords = get_coordinates(H, W, device=x.device).reshape(2, -1).permute(1, 0)
        coords = coords.reshape(H // 2, 2, W // 2, 2, 2).permute(0, 2, 1, 3, 4).reshape(-1, 2)
        coords_vis_local = coords[mask].reshape(-1, 2)
        coords_vis_local = coords_vis_local[:, 0] * H + coords_vis_local[:, 1]
        idx_shuffle = torch.argsort(torch.argsort(coords_vis_local))

        x = torch.index_select(x, 1, index=idx_shuffle)
        x = x.reshape(B, L // 4, 4, C)
        # row-first order to column-first order
        # make it compatible with Swin (https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L342)
        x = torch.cat([x[:, :, 0], x[:, :, 2], x[:, :, 1], x[:, :, 3]], dim=-1)

        # merging by a linear layer
        x = self.norm(x)
        x = self.reduction(x)

        mask_new = mask_prev.view(1, H // 2, 2, W // 2, 2).sum(dim=(2, 4))
        assert torch.unique(mask_new).shape[0] == 2
        mask_new = (mask_new > 0).reshape(1, -1)
        coords_new = get_coordinates(H // 2, W // 2, x.device).reshape(1, 2, -1)
        coords_new = coords_new.transpose(2, 1)[mask_new].reshape(1, -1, 2)

        # xlist = []
        # for i in range(self.group):
        #     xlist.append(x[:, i::self.group, :])
        #
        # x = torch.cat(xlist, -1)
        # x = x.view(B, -1, self.group * C)
        #
        # x = self.norm(x)
        # if self.PosEmb == "Pre":
        #     x += self.posPreEmb
        # x = self.reduction(x)
        # if self.PosEmb == "Post":
        #     x += self.posPostEmb
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class VisionTransformerEncoder(nn.Module):

    def __init__(self, image_size, near_band, num_patches, num_classes, band, dim, heads,
                 pool='cls', dropout=0., emb_dropout=0.,  mode='ViT',
                 group=2, outputdim=1,
                 depths=[2, 2, 6, 2],
                 window_size=10,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, use_checkpoint=False, mlp_ratio=.125,
                 mask_ratio=0.75,
                 init_scaler=0.
                 ):
        super().__init__()
        num_heads = [heads * (outputdim ** i) for i in range(len(depths))]

        patch_dim = near_band*image_size**2

        self.use_cls = True
        self.band = band
        self.image_size = image_size
        self.mode = mode
        self.dim = dim
        # self.patch_dim = patch_dim
        self.near_band = near_band
        self.num_patches = num_patches
        self.num_classes = num_classes

        if self.use_cls:
            patch_cls = 1
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        else:
            patch_cls = 0

        self.embedding_by_msa_pos = nn.Parameter(torch.randn(1, image_size ** 2, near_band))
        self.embedding_by_msa = Attention(dim=near_band, dim_head=self.dim // 4, heads=4,
                                          dropout=dropout)

        self.patch_to_embedding = nn.Linear(patch_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.num_layers = len(depths)
        self.num_features = self.dim * (outputdim ** (self.num_layers - 1))
        self.norm = norm_layer(self.num_features)
        self.out_num_patches = (num_patches // (group ** (self.num_layers - 1)))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, num_classes)
        )  if num_classes > 0 else nn.Identity()

        assert num_patches % (
                group ** (self.num_layers - 1)) == 0, "tokenmerging，group不是num_patches的整数倍"
        self.layers = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        resNorm = nn.LayerNorm([self.dim, 2])
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(self.dim * outputdim ** i_layer),
                               input_resolution=num_patches // (group ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               group=group,
                               output_dim=outputdim,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=TokenMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               )
            self.layers.append(layer)

    def forward_features(self, x, masking_pos):
        x = self.patch_to_embedding(x)
        x = self.dropout(x)

        b, _, c = x.shape

        x_vis = x[~masking_pos].reshape(b, -1, c)

        for i, layer in enumerate(self.layers):
            x_vis = layer(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, masking_pos, mask=None):
        x = self.forward_features(x, masking_pos)
        if self.num_classes > 0:
            x = x.mean(axis=1) if self.pool == 'mean' else self.to_latent(torch.flatten(self.avgpool(x.transpose(1, 2)), 1))
        else:
            x = self.to_latent(torch.flatten(self.avgpool(x.transpose(1, 2)), 1))
        return self.mlp_head(x)

#-------------------------------------------------------------------------------
class VisionTransformerDecoder(nn.Module):

    def __init__(self,
                 image_size=1,
                 near_band=1,
                 num_patches=200,
                 num_classes=147,
                 dim=128,
                 depth=5,
                 dim_head = 16,
                 heads=4,
                 mlp_dim=8,
                 pool='cls',
                 dropout=0.1,
                 emb_dropout=0.1,
                 mode='ViT'
                 ):
        super().__init__()

        self.num_classes = num_classes
        assert num_classes == near_band * image_size ** 2
        self.dropout = nn.Dropout(emb_dropout)
        # self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, dim, mode)
        self.norm =  nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num, mask=None):

        # x = self.transformer(x, mask)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))
        else:
            x = self.head(self.norm(x))
        return x


#-------------------------------------------------------------------------------
class PretrainVisionTransformer(nn.Module):

    def __init__(self,
                 image_size=1,
                 near_band=1,
                 num_patches=200,
                 encoder_num_classes=0,
                 band=200,
                 encoder_dim=147,
                 # encoder_depth=5,
                 encoder_heads=4,
                 encoder_dim_head=16,
                 encoder_mode='ViT',
                 encoder_pool='cls',
                 decoder_num_classes=147,
                 decoder_dim=128,
                 decoder_depth=3,
                 decoder_heads=3,
                 decoder_dim_head=12,
                 decoder_mode='ViT',
                 decoder_pool='cls',
                 mlp_dim=8,
                 dropout=0.1,
                 emb_dropout=0.1,
                 window_size=25,
                 mask_ratio=0.75
                 ):

        super().__init__()
        self.encoder = VisionTransformerEncoder(
            image_size=image_size,
            near_band=near_band,
            num_patches=num_patches,
            num_classes=encoder_num_classes,
            band=band,
            dim=encoder_dim,
            # depth=encoder_depth,
            heads=encoder_heads,
            dropout=dropout,
            emb_dropout=emb_dropout,
            window_size=window_size,
            mask_ratio=mask_ratio)

        self.decoder = VisionTransformerDecoder(
            image_size=image_size,
            near_band=near_band,
            num_patches=num_patches,
            num_classes=decoder_num_classes,
            dim=decoder_dim,
            depth=decoder_depth,
            dim_head = decoder_dim_head,
            heads=decoder_heads,
            mlp_dim=mlp_dim,
            pool=decoder_pool,
            dropout=dropout,
            emb_dropout=emb_dropout,
            mode=decoder_mode)


        self.learn_pos = False
        if self.learn_pos == True:
            self.pos_emb = nn.Parameter(torch.randn(1, num_patches, decoder_dim))
        else:
            self.pos_emb = get_sinusoid_encoding_table(num_patches, decoder_dim)

        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

    def forward(self, x, masking_pos, mask=None):
        b, _, _ = x.shape
        x_vis = self.encoder(x, masking_pos, mask)
        x_visd = self.encoder_to_decoder(x_vis)

        _, _, cv = x_visd.shape

        pos_embed = self.pos_emb.expand(b, -1, -1).type_as(x).to(x.device).detach().clone()
        pos_embed_vis = pos_embed[~masking_pos].reshape(b,-1, cv)
        pos_embed_mask = pos_embed[masking_pos].reshape(b,-1, cv)

        x_full = torch.cat([x_visd + pos_embed_vis, self.mask_token + pos_embed_mask],dim=1)
        x = self.decoder(x_full, pos_embed_mask.shape[1])
        return x

class SwinT(nn.Module):
    def __init__(self, image_size, near_band, num_patches, patch_dim, num_classes, band, dim, heads,
                 pool='cls', dropout=0., emb_dropout=0.,  mode='ViT',
                 group=2, outputdim=1,
                 depths=[2, 2, 6, 2],
                 window_size=10,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, use_checkpoint=False, mlp_ratio=.125,

                 ):

        super().__init__()

        num_heads = [heads * (outputdim ** i) for i in range(len(depths))]

        self.band = band
        self.image_size = image_size
        self.mode = mode
        self.dim = dim
        self.patch_dim = patch_dim
        self.near_band = near_band
        self.num_patches = num_patches

        self.embedding_by_msa_pos = nn.Parameter(torch.randn(1, image_size ** 2, near_band))
        self.embedding_by_msa = Attention(dim=near_band, dim_head=self.dim // 4, heads=4,
                                          dropout=dropout)

        self.patch_to_embedding = nn.Linear(patch_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.num_layers = len(depths)
        self.num_features = self.dim * (outputdim ** (self.num_layers - 1))
        self.norm = norm_layer(self.num_features)
        self.out_num_patches = (num_patches // (group ** (self.num_layers - 1)))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, num_classes)
        )

        assert num_patches % (
                group ** (self.num_layers - 1)) == 0, "tokenmerging，group不是num_patches的整数倍"
        self.layers = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        resNorm = nn.LayerNorm([self.dim, 2])
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(self.dim * outputdim ** i_layer),
                               input_resolution=num_patches // (group ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               group=group,
                               output_dim=outputdim,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=TokenMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               )
            self.layers.append(layer)

    def forward(self, x, mask=None):

        x = self.patch_to_embedding(x)
        x = self.dropout(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

        x = self.norm(x)
        x = self.to_latent(torch.flatten(self.avgpool(x.transpose(1, 2)), 1))
        return self.mlp_head(x)


if __name__ == '__main__':
    # model = SwinT(image_size=7, near_band=7, num_patches=200,
    #           patch_dim=3 * 7 ** 2, num_classes=16, band=200, dim=64,
    #           heads=4, dropout=0.1, emb_dropout=0.1, window_size=25,)
    model = VisionTransformerEncoder(image_size=7, near_band=7, num_patches=200,
                  num_classes=16, band=200, dim=64,
                  heads=4, dropout=0.1, emb_dropout=0.1, window_size=25, )


    def randbool(size, p=0.5):
        return torch.rand(*size) < p

    input_mask = randbool((64, 200), 0.75)
    # model.eval()
    # print(model)
    input = torch.randn(64, 200, 147)
    y = model(input, input_mask)
    print(y.size())