from math import sqrt
from re import L
from regex import B
import torch
import torch.nn as nn

from transformers import SwinModel
import torch
from torch import nn
from RAE.src.stage2.models.lightningDiT import PatchEmbed, Mlp, NormAttention

from timm.models.vision_transformer import PatchEmbed, Mlp
from RAE.src.stage2.models.model_utils import VisionRotaryEmbeddingFast, RMSNorm, SwiGLUFFN, GaussianFourierEmbedding, LabelEmbedder, NormAttention, get_2d_sincos_pos_embed
import torch.nn.functional as F
from typing import *


# PHO
from einops import rearrange
from RAE.src.stage2.models.model_utils import NormAttentionMVRM
from RAE.src.stage2.models.rope import RotaryPositionEmbedding2D, PositionGetter
from torch.utils.checkpoint import checkpoint


def DDTModulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        shift: Tensor of shape (B, L, D)
        scale: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * (1 + scale) + shift, 
        with shift and scale repeated to match L_x if necessary.
    """
    
    if shift.shape[0] != x.shape[0]:
        repeat_v = x.shape[0] // shift.shape[0]  # = num_view
        shift = shift.repeat_interleave(repeat_v, dim=0)
        scale = scale.repeat_interleave(repeat_v, dim=0)    

    B, Lx, D = x.shape
    _, L, _ = shift.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        # if shift has shape (1,d) -> repeat_interleave(n, dim=0) -> (n,d)
        shift = shift.repeat_interleave(repeat, dim=1)
        scale = scale.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * (1 + scale) + shift


def DDTGate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        gate: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * gate, 
        with gate repeated to match L_x if necessary.
    """

    if gate.shape[0] != x.shape[0]:
        repeat_v = x.shape[0] // gate.shape[0]  # = num_view
        gate = gate.repeat_interleave(repeat_v, dim=0)
    
    B, Lx, D = x.shape
    _, L, _ = gate.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        # print(f"gate shape: {gate.shape}, x shape: {x.shape}")
        gate = gate.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * gate


class LightningDDTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        rope=None,
        **block_kwargs
    ):
        super().__init__()

        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        # Initialize attention layer
        self.attn = NormAttentionMVRM(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            rope=rope,
            **block_kwargs
        )

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )

        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, pos=None, layer_idx=None, attn_type=None, analysis=None):
        
        if len(c.shape) < len(x.shape): # t
            c = c.unsqueeze(1)  # (B, 1, C)
        if self.wo_shift:   # f
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:   # t
            '''
                c: concated (time and label) used for condition 
                for adaptive layer normalization, the shift, scale, and gate are computed from the condition c 
                    attention - shift,scale,gate
                    mlp - shift,scale, gate 
                ANOTHER UNDERSTANDING
                    layernorm -> for (n,d) tokens, it normalizes each tokens independently, 
                    first, for one vector token (1,d) normalize it as (v - v.mean() / v.std()), then apply scale SC and shift SH
                    this is done for all tokens, the same scale SC and shift SH is applied. 
                    However, for adaptiveLN, while (v-v.mean())/v.std() is the same, every tokens gets a different SC and SH value.
                    this value is learnable. Also the scale and shift equation is computed as x*(1+scale) + shift, instead of x*scale + shift
            '''
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)        
        x = x + DDTGate(
            self.attn(
                DDTModulate(self.norm1(x), shift_msa, scale_msa), pos=pos, layer_idx=layer_idx, attn_type=attn_type, analysis=analysis
                ), gate_msa
            )
        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)
        return x


class DDTFinalLayer(nn.Module):
    """
    The final layer of DDT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False, output_dim=None):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        
        if output_dim is None:
            output_dim = patch_size * patch_size * out_channels
        
        
        self.linear = nn.Linear(
            hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale)  # no gate
        x = self.linear(x)
        return x


class GARD(nn.Module):
    def __init__(
            self,
            input_size: int = 1,
            patch_size: Union[list, int] = 1,
            in_channels: int = 768,
            hidden_size=[1152, 2048],
            depth=[28, 2],
            num_heads: Union[list[int], int] = [16, 16],
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            wo_shift=False,
            use_pos_embed: bool = True,
            use_global_residual = False,
            ddt_final_layer_output_dim = None
    ):
        super().__init__()
        
        
        # PHO 
        self.num_cls_tkn = 1
        self.vit_patch_size=14  # for giant
        
        self.use_global_residual = use_global_residual
        
        
        self.in_channels = in_channels      
        self.out_channels = in_channels     

        self.encoder_hidden_size = hidden_size[0]   
        self.decoder_hidden_size = hidden_size[1]   
        
        self.num_heads = [num_heads, num_heads] if isinstance(num_heads, int) else list(num_heads)  # (16,16)
        
        self.num_encoder_blocks = depth[0]  # 28
        self.num_decoder_blocks = depth[1]  # 2
        self.num_blocks = depth[0] + depth[1]   # 30
        
        self.use_rope = use_rope    # t
        # analyze patch size
        if isinstance(patch_size, int) or isinstance(patch_size, float):
            patch_size = [patch_size, patch_size]  # [1, 1] patch size for s , x embed 
        assert len(patch_size) == 2, f"patch size should be a list of two numbers, but got {patch_size}"
        self.patch_size = patch_size
        self.s_patch_size = patch_size[0]
        self.x_patch_size = patch_size[1]
        
        self.s_channel_per_token = in_channels * self.s_patch_size * self.s_patch_size   # 768
        # s_input_size = input_size           # 32
        s_patch_size = self.s_patch_size    # 1
        
        # x_input_size = input_size           # 32
        x_patch_size = self.x_patch_size    # 1
        self.x_channel_per_token = in_channels * self.x_patch_size * self.x_patch_size   # 768

        self.s_projector = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size) if self.encoder_hidden_size != self.decoder_hidden_size else nn.Identity()
        self.s_embedder = PatchEmbed(
            # img_size=s_input_size, 
            img_size=None,
            patch_size=s_patch_size, 
            in_chans=self.s_channel_per_token, 
            embed_dim=self.encoder_hidden_size, 
            bias=True,
            strict_img_size=False,          
            dynamic_img_pad=True,            
            )
        
        self.x_embedder = PatchEmbed(
            # img_size=x_input_size, 
            img_size=None,
            patch_size=x_patch_size, 
            in_chans=self.x_channel_per_token, 
            embed_dim=self.decoder_hidden_size, 
            bias=True,
            strict_img_size=False,          
            dynamic_img_pad=True,    
            )
        
        self.t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size)

        self.final_layer = DDTFinalLayer(self.decoder_hidden_size, 1, self.x_channel_per_token, use_rmsnorm=use_rmsnorm, output_dim=ddt_final_layer_output_dim)
        
        
        if use_pos_embed:   # t
            num_patches = self.s_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_cls_tkn, self.encoder_hidden_size), requires_grad=False)
            self.x_pos_embed = None
            
        self.use_pos_embed = use_pos_embed
        enc_num_heads = self.num_heads[0]
        dec_num_heads = self.num_heads[1]
        

        # PHO rope 
        rope_freq=100
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
        self.position_getter = PositionGetter()
        self.blocks = nn.ModuleList([
            LightningDDTBlock(self.encoder_hidden_size if i < self.num_encoder_blocks else self.decoder_hidden_size,
                              enc_num_heads if i < self.num_encoder_blocks else dec_num_heads,
                              mlp_ratio=mlp_ratio,
                              use_qknorm=use_qknorm,
                              use_rmsnorm=use_rmsnorm,
                              use_swiglu=use_swiglu,
                              wo_shift=wo_shift,
                              rope=self.rope,
                              ) for i in range(self.num_blocks)
        ])
        

        self.s_cls_embedder = nn.Linear(self.in_channels, self.encoder_hidden_size)
        self.x_cls_embedder = nn.Linear(self.in_channels, self.decoder_hidden_size)
        
        
        
        self.initialize_weights()


    def initialize_weights(self, xavier_uniform_init: bool = False):
        if xavier_uniform_init:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)


        # Initialize cls embedders (Linear layers)
        nn.init.xavier_uniform_(self.s_cls_embedder.weight)
        nn.init.constant_(self.s_cls_embedder.bias, 0)

        nn.init.xavier_uniform_(self.x_cls_embedder.weight)
        nn.init.constant_(self.x_cls_embedder.bias, 0)
            
            
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)


        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.x_channel_per_token        
        p = self.x_embedder.patch_size[0]
        h, w = self.s_embedder.grid_size
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs


    def _prepare_rope(self, B, S, num_pH, num_pW, device):
        pos = None
        pos_nodiff = None
        if self.rope is not None:
            pos = self.position_getter(B * S, num_pH, num_pW, device=device)   # b*v n 2 
            pos = rearrange(pos, "(b s) n c -> b s n c", b=B)   # b v n 2 
            pos_nodiff = torch.zeros_like(pos).to(pos.dtype)
            pos = pos + 1   # to not account cls_tkn
            pos_special = torch.zeros(B * S, 1, 2).to(device).to(pos.dtype)
            pos_special = rearrange(pos_special, "(b s) n c -> b s n c", b=B)
            pos = torch.cat([pos_special, pos], dim=2)
            pos_nodiff = pos_nodiff + 1     # to not account cls_tkn
            pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
        return pos, pos_nodiff


    def process_attention_encoder(self, x, c, block, attn_type="global", pos=None, layer_idx=None, analysis=None):
        
        b, s, n, _ = x.shape
        if attn_type == "local":
            x = rearrange(x, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
        elif attn_type == "global":
            x = rearrange(x, "b s n c -> b (s n) c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")

        # gradient checkpointing here
        def block_forward(x, c, pos, layer_idx, attn_type, analysis):
            return block(x, c, pos=pos, layer_idx=layer_idx, attn_type=attn_type, analysis=analysis)

        if self.training:
            needs_attn_map = (analysis is not None and
                              analysis.vis_attn_map and
                              layer_idx in analysis.mvrm_attn_map.extract_idx)
            if needs_attn_map:
                # use_reentrant=False: forward runs with autograd (attn gets grad_fn),
                # but q/k/v are freed and recomputed during backward → saves memory
                x = checkpoint(block_forward, x, c, pos, layer_idx, attn_type, analysis, use_reentrant=False)
            else:
                x = checkpoint(block_forward, x, c, pos, layer_idx, attn_type, analysis)
        else:
            x = block(x, c, pos=pos, layer_idx=layer_idx, attn_type=attn_type, analysis=analysis)

        if attn_type == "local":
            x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
        elif attn_type == "global":
            x = rearrange(x, "b (s n) c -> b s n c", b=b, s=s)
        return x


    def process_attention_decoder(self, x, c, block, attn_type="global", pos=None, layer_idx=None, analysis=None):
        
        b, s, n, _ = x.shape
        if attn_type == "local":
            x = rearrange(x, "b s n c -> (b s) n c")
            c = rearrange(c, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
        elif attn_type == "global":
            x = rearrange(x, "b s n c -> b (s n) c")
            c = rearrange(c, "b s n c -> b (s n) c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")


        # x = block(x, c, pos=pos)
        # gradient checkpointing here
        def block_forward(x, c, pos, layer_idx, attn_type, analysis):
            return block(x, c, pos=pos, layer_idx=layer_idx, attn_type=attn_type, analysis=analysis)

        if self.training:
            needs_attn_map = (analysis is not None and
                              analysis.vis_attn_map and
                              layer_idx in analysis.mvrm_attn_map.extract_idx)
            if needs_attn_map:
                x = checkpoint(block_forward, x, c, pos, layer_idx, attn_type, analysis, use_reentrant=False)
            else:
                x = checkpoint(block_forward, x, c, pos, layer_idx, attn_type, analysis)
        else:
            x = block(x, c, pos=pos, layer_idx=layer_idx, attn_type=attn_type, analysis=analysis)


        if attn_type == "local":
            x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
        elif attn_type == "global":
            x = rearrange(x, "b (s n) c -> b s n c", b=b, s=s)
        return x


    def forward(self, x, t, model_img_size, analysis=None):
        
        
        if self.use_global_residual:
            global_residual = x         # b v 973 1536
            
        
        # print('PRINT!!!', model_img_size)
        model_H, model_W = model_img_size
        pH = pW = 14 
        num_pH = model_H//pH    
        num_pW = model_W//pW    
        
        
        cls_tkn = x[:,:,0:1]        # b v 1 d
        patch_tkns = x[:,:,1:]      # b v n d
        
        
        b, v, n, d = patch_tkns.shape
        x = rearrange(patch_tkns, 'b v (num_pH num_pW) d -> (b v) d num_pH num_pW', b=b, v=v, num_pH=num_pH, num_pW=num_pW)   # b v 3072 27 36
        

        # time condition
        t = self.t_embedder(t)          # b*v d        
        c = nn.functional.silu(t)       # b*v d
        

        # encoder - cls embedding 
        s_cls_tkn = self.s_cls_embedder(cls_tkn.reshape(b*v, 1, -1))
        
        
        # encoder - patch embedding 
        s = self.s_embedder(x)  # b*v 1536 27 36 -> b*v 972 1536
        
        

        
        # concat encoder cls tkn w/ patch_tkns 
        s = torch.cat([s_cls_tkn, s], dim=1)      # b*v n+1 d
        
        

        
        # reshape to (b v n d)
        s = rearrange(s, '(b v) n d -> b v n d', b=b, v=v)
            

        # PHO prepare rope
        pos, pos_nodiff = self._prepare_rope(b, v, num_pH, num_pW, s.device)  # b v n+1 2
        g_pos = pos_nodiff
        l_pos = pos
        
        
        # ddt encoder forward apss     
        for i in range(self.num_encoder_blocks):    # len: 28
            # s = self.blocks[i](s, c, feat_rope=self.enc_feat_rope)
            if i % 2 == 0:
                # print(f'encoder - {i} local attn')
                s = self.process_attention_encoder(s, c, self.blocks[i], 'local', pos=l_pos, layer_idx=i, analysis=analysis)    # b v n+1 d
            else:
                # print(f'encoder - {i} global attn')
                s = self.process_attention_encoder(s, c, self.blocks[i], 'global', pos=g_pos, layer_idx=i, analysis=analysis)    # b v n+1 d


        
        # reshape to (b*v n d)
        s = rearrange(s, 'b v n d -> (b v) n d')    # 16 973 1536


        # expand t across views
        t = t.repeat_interleave(s.shape[0]//t.shape[0], dim=0)   # (b*v, d)
        # expand across tokens
        t = t.unsqueeze(1).expand(-1, s.shape[1], -1)   # 16 973 1536
        s = nn.functional.silu(t + s)                   # 16 973 1536
            
        
        # expand encoder output dimension to decoder dimension
        s = self.s_projector(s) # b*v n d1 -> b*v n d2
        
        
        # decoder - cls embedding
        x_cls_tkn = self.x_cls_embedder(cls_tkn.reshape(b*v, 1, -1))
        
        
        # decoder - patch embedding
        x = self.x_embedder(x)  # b*v d h w -> b*v n d
        
        
        # concat decoder cls_tkn w/ patch_tkns
        x = torch.cat([x_cls_tkn, x], dim=1)  # b*v n+1 d
             
        
        # reshape to (b v n d)
        x = rearrange(x, '(b v) n d -> b v n d', b=b, v=v)  # 4 4 973 3072
        s = rearrange(s, '(b v) n d -> b v n d', b=b, v=v)  # 4 4 973 3072
        


        # ddt decoder forward pass
        for i in range(self.num_encoder_blocks, self.num_blocks):   # len: 2
            # x = self.blocks[i](x, s, feat_rope=self.dec_feat_rope)
            if i % 2 == 0:
                # print(f'decoder - {i} local attn')
                x = self.process_attention_decoder(x, s, self.blocks[i], 'local', pos=l_pos, layer_idx=i, analysis=analysis)
            else:
                # print(f'decoder - {i} global attn')
                x = self.process_attention_decoder(x, s, self.blocks[i], 'global', pos=g_pos, layer_idx=i, analysis=analysis)
            
        
        # reshape to (b*v n d)
        x = rearrange(x, 'b v n d -> (b v) n d')
        s = rearrange(s, 'b v n d -> (b v) n d')
        
        
        x = self.final_layer(x, s)  # last adaLN and restore to original dim: 2048 -> 768


        x = rearrange(x, '(b v) n d -> b v n d', b=b, v=v)
        
        if self.use_global_residual:
            x = x + global_residual
        
        return x


    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=(0, 1)):
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:,
                              :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        t = t[: len(t) // 2] # get t for the conditional half
        half_eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)
             ).view(-1, *[1] * (len(cond_eps.shape) - 1)),
            uncond_eps + cfg_scale * (cond_eps - uncond_eps), cond_eps
        )
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward_with_autoguidance(self, x, t, y, cfg_scale, additional_model_forward, cfg_interval=(0, 1)):
        """
        Forward pass of LightningDiT, but also contain the forward pass for the additional model
        """
        model_out = self.forward(x, t, y)
        ag_model_out = additional_model_forward(x, t, y)
        eps = model_out[:, :self.in_channels]
        ag_eps = ag_model_out[:, :self.in_channels]

        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)
             ).view(-1, *[1] * (len(eps.shape) - 1)),
            ag_eps + cfg_scale * (eps - ag_eps), eps
        )

        return eps



