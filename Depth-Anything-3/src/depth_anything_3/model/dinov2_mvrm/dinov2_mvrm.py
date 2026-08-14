# # Copyright (c) Meta Platforms, Inc. and affiliates.
# #
# # This source code is licensed under the Apache License, Version 2.0
# # found in the LICENSE file in the root directory of this source tree.

# # References:
# #   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
# #   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py


# from typing import List
# import torch.nn as nn

# from depth_anything_3.model.dinov2_mvrm.vision_transformer_mvrm import (
#     vit_base,
#     vit_giant2,
#     vit_large,
#     vit_small,
# )



# class DinoV2MVRM(nn.Module):
#     def __init__(
#         self,
#         name: str,
#         out_layers: List[int],
#         alt_start: int = -1,
#         qknorm_start: int = -1,
#         rope_start: int = -1,
#         cat_token: bool = True,
#         mvrm_cfg=None,
#         **kwargs,
#     ):
#         super().__init__()

#         breakpoint()
#         # RAE MVRM
#         # mvrm_cfg = kwargs['mvrm']['stage_2']
#         # transport_cfg = kwargs['mvrm']['transport']
        
#         self.mvrm_cfg = mvrm_cfg 
        
#         breakpoint()
        
#         # DinoV2 
#         assert name in {"vits", "vitb", "vitl", "vitg"}
#         self.name = name
#         self.out_layers = out_layers
#         self.alt_start = alt_start
#         self.qknorm_start = qknorm_start
#         self.rope_start = rope_start
#         self.cat_token = cat_token
#         encoder_map = {
#             "vits": vit_small,
#             "vitb": vit_base,
#             "vitl": vit_large,
#             "vitg": vit_giant2,
#         }
#         encoder_fn = encoder_map[self.name]
#         ffn_layer = "swiglufused" if self.name == "vitg" else "mlp"
#         self.pretrained = encoder_fn(
#             img_size=518,
#             patch_size=14,
#             ffn_layer=ffn_layer,
#             alt_start=alt_start,
#             qknorm_start=qknorm_start,
#             rope_start=rope_start,
#             cat_token=cat_token,
#             mvrm_cfg = self.mvrm_cfg
#         )


#     def forward(self, x, **kwargs):
#         # breakpoint()
#         # self.pretrained -> vision_transformer.py DinoVisionTransformer (original dinov2 repo)
#         return self.pretrained.get_intermediate_layers(
#             x,
#             self.out_layers,
#             **kwargs,
#         )
