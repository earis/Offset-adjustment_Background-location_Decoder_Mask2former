# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, ConvModule
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmengine.model import (BaseModule, ModuleList, caffe2_xavier_init,constant_init,
                            normal_init, xavier_init)
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from ..task_modules.prior_generators import MlvlPointGenerator
from .positional_encoding import SinePositionalEncoding
from .transformer import Mask2FormerTransformerEncoder


@MODELS.register_module()
class MSDeformAttnPixelDecoder(BaseModule):
    """Pixel decoder with multi-scale deformable attention.

    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        strides (list[int] | tuple[int]): Output strides of feature from
            backbone.
        feat_channels (int): Number of channels for feature.
        out_channels (int): Number of channels for output.
        num_outs (int): Number of output scales.
        norm_cfg (:obj:`ConfigDict` or dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (:obj:`ConfigDict` or dict): Config for activation.
            Defaults to dict(type='ReLU').
        encoder (:obj:`ConfigDict` or dict): Config for transformer
            encoder. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer encoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: Union[List[int],
                                    Tuple[int]] = [256, 512, 1024, 2048],
                 strides: Union[List[int], Tuple[int]] = [4, 8, 16, 32],
                 feat_channels: int = 256,
                 out_channels: int = 256,
                 num_outs: int = 3,
                 norm_cfg: ConfigType = dict(type='GN', num_groups=32),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 encoder: ConfigType = None,
                 positional_encoding: ConfigType = dict(
                     num_feats=128, normalize=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.strides = strides
        self.num_input_levels = len(in_channels)
        self.num_encoder_levels = \
            encoder.layer_cfg.self_attn_cfg.num_levels
        assert self.num_encoder_levels >= 1, \
            'num_levels in attn_cfgs must be at least one'
        input_conv_list = []
        input_conv_list_me = []
        # from top to down (low to high resolution)
        for i in range(self.num_input_levels - 1,
                       self.num_input_levels - self.num_encoder_levels - 2,
                       -1):
            input_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True)
            
            input_conv_list.append(input_conv)
        # channels_me = 2048
        # input_conv_me = ConvModule(
        #         channels_me,
        #         feat_channels,
        #         kernel_size=1,
        #         norm_cfg=norm_cfg,
        #         act_cfg=None,
        #         bias=True)
        # input_conv_list_me.append(input_conv_me)

        # self.density_layer = nn.Conv2d(feat_channels,feat_channels, 1, 1)
        # self.reg_layer2 = nn.Sequential(
        #     nn.Conv2d(feat_channels,feat_channels, kernel_size=1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(feat_channels,feat_channels, kernel_size=1, padding=0),
        #     nn.ReLU(inplace=True),
        #     )
        # self.density_layer = ConvModule(
        #         feat_channels,
        #         feat_channels,
        #         kernel_size=1,
        #         norm_cfg=norm_cfg,
        #         bias=True)
        # self.reg_layer2 = nn.Sequential(
        #     ConvModule(
        #         feat_channels,
        #         feat_channels,
        #         kernel_size=1,
        #         norm_cfg=norm_cfg,
        #         bias=True) 
        #     )
        
        self.input_convs = ModuleList(input_conv_list)
        # self.input_convs_me = ModuleList(input_conv_list_me)

        # self.threshold = nn.Linear(256, 256)
        self.encoder = Mask2FormerTransformerEncoder(**encoder)
        self.postional_encoding = SinePositionalEncoding(**positional_encoding)
        # high resolution to low resolution
        self.level_encoding = nn.Embedding(self.num_encoder_levels + 1,
                                           feat_channels)

        # fpn-like structure
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None
        # from top to down (low to high resolution)
        # fpn for the rest features that didn't pass in encoder
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
                       -1):
            lateral_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=None)
            output_conv = ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.mask_feature = Conv2d(
            feat_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.num_outs = num_outs
        self.point_generator = MlvlPointGenerator(strides)

        # self.reg_layer=nn.Sequential(
        #     nn.Conv2d(dim_feedforward, hidden_dim, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(hidden_dim, hidden_dim2, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     )
        # self.density_layer = nn.Conv2d(128, 1, 1)
        # self.reg_layer2=nn.Sequential(
        #     nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(hidden_dim2, hidden_dim, kernel_size=1, padding=0),
        #     nn.ReLU(inplace=True),
        #     )
        

    def init_weights(self) -> None:
        """Initialize weights."""
        for i in range(0, self.num_encoder_levels + 1):
            xavier_init(
                self.input_convs[i].conv,
                gain=1,
                bias=0,
                distribution='uniform')

        for i in range(0, self.num_input_levels - self.num_encoder_levels):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)
            # constant_init(self.threshold, 0.)

        caffe2_xavier_init(self.mask_feature, bias=0)

        normal_init(self.level_encoding, mean=0, std=1)
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        # init_weights defined in MultiScaleDeformableAttention
        for m in self.encoder.layers.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

    def forward(self, feats: List[Tensor], output2class = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c, h, w).

        Returns:
            tuple: A tuple containing the following:

                - mask_feature (Tensor): shape (batch_size, c, h, w).
                - multi_scale_features (list[Tensor]): Multi scale \
                        features, each in shape (batch_size, c, h, w).
        """
        # generate padding mask for each level, for each image
        if len(feats) >= 5:
            output2class=feats[4]
        feats = feats[:4]
        batch_size = feats[0].shape[0]
        encoder_input_list = []
        encoder_input_list_me = []
        padding_mask_list = []
        padding_mask_list_me = []
        level_positional_encoding_list = []
        level_positional_encoding_list_me = []
        spatial_shapes = []
        spatial_shapes_me = []
        reference_points_list = []
        reference_points_list_me = []
        memory_me_list = []
        for i in range(self.num_encoder_levels + 1):
            level_idx = self.num_input_levels - i - 1
            feat = feats[level_idx]
            feat_projected = self.input_convs[i](feat)
            if output2class is not None:
                feat_projected = feat_projected + F.interpolate(
                    output2class,
                    size=feat_projected.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
            
            feat_hw = torch._shape_as_tensor(feat)[2:].to(feat.device)

            # no padding
            padding_mask_resized = feat.new_zeros(
                (batch_size, ) + feat.shape[-2:], dtype=torch.bool)
            pos_embed = self.postional_encoding(padding_mask_resized)
            level_embed = self.level_encoding.weight[i]
            level_pos_embed = level_embed.view(1, -1, 1, 1) + pos_embed
            # (h_i * w_i, 2)
            reference_points = self.point_generator.single_level_grid_priors(
                feat.shape[-2:], level_idx, device=feat.device)
            # normalize
            feat_wh = feat_hw.unsqueeze(0).flip(dims=[0, 1])
            factor = feat_wh * self.strides[level_idx]
            reference_points = reference_points / factor

            # shape (batch_size, c, h_i, w_i) -> (h_i * w_i, batch_size, c)
            feat_projected = feat_projected.flatten(2).permute(0, 2, 1)
            

            level_pos_embed = level_pos_embed.flatten(2).permute(0, 2, 1)
            padding_mask_resized = padding_mask_resized.flatten(1)
            if i != 0:
                encoder_input_list.append(feat_projected)
                # encoder_input_list_me.append(feat_projected_me)
                padding_mask_list.append(padding_mask_resized)
                level_positional_encoding_list.append(level_pos_embed)
                spatial_shapes.append(feat_hw)
                reference_points_list.append(reference_points)
            if i ==0:
                encoder_input_list_me.append(feat_projected)
                padding_mask_list_me.append(padding_mask_resized)
                level_positional_encoding_list_me.append(level_pos_embed)
                spatial_shapes_me.append(feat_hw)
                reference_points_list_me.append(reference_points)
        padding_masks_me = torch.cat(padding_mask_list_me, dim=1)
        # shape (total_num_queries, batch_size, c)
        encoder_inputs_me = torch.cat(encoder_input_list_me, dim=1)
        # encoder_inputs_size = encoder_inputs.shape
        # original_tensor = torch.zeros(encoder_inputs_size)
        # original_tensor = feat_projected_me
        # feat_projected_me_shape2 = feat_projected_me.shape[1]
        # original_tensor[:,:feat_projected_me_shape2,:] = feat_projected_me
        level_positional_encodings_me = torch.cat(
            level_positional_encoding_list_me, dim=1)
        # shape (num_encoder_levels, 2), from low
        # resolution to high resolution
        spatial_shapes_me = torch.cat(spatial_shapes_me).view(-1, 2)
        # shape (0, h_0*w_0, h_0*w_0+h_1*w_1, ...)
        level_start_index_me = torch.cat((spatial_shapes_me.new_zeros(
            (1, )), spatial_shapes_me.prod(1).cumsum(0)[:-1]))
        reference_points_me = torch.cat(reference_points_list_me, dim=0)
        reference_points_me = reference_points_me[None, :, None].repeat(
            batch_size, 1, 2, 1)
        valid_radios_me = reference_points_me.new_ones(
            (batch_size, 2, 2))
        # threshold_me =  
        # shape (num_total_queries, batch_size, c)

        # memory_me = self.encoder(
        #     query=encoder_inputs_me,
        #     query_pos=level_positional_encodings_me,
        #     key_padding_mask=padding_masks_me,
        #     spatial_shapes=spatial_shapes_me,
        #     reference_points=reference_points_me,
        #     level_start_index=level_start_index_me,
        #     valid_ratios=valid_radios_me,something = None)
        # memory_me_list.append(memory_me)
        memory_me_list.append(encoder_inputs_me)
        memory_me_list.append(level_positional_encodings_me)
        memory_me_list.append(padding_masks_me)
        memory_me_list.append(spatial_shapes_me)
        memory_me_list.append(reference_points_me)
        memory_me_list.append(level_start_index_me)
        memory_me_list.append(valid_radios_me)
        memory_me_list.append(None)
        # (batch_size, c, num_total_queries)
        # memory_me = memory_me.permute(0, 2, 1)
        
        # feat_projected_me = self.input_convs_me[0](feats[0])
        # shape (batch_size, total_num_queries),
        # total_num_queries=sum([., h_i * w_i,.])
        padding_masks = torch.cat(padding_mask_list, dim=1)
        # shape (total_num_queries, batch_size, c)
        encoder_inputs = torch.cat(encoder_input_list, dim=1)
        # encoder_inputs_size = encoder_inputs.shape
        # original_tensor = torch.zeros(encoder_inputs_size)
        # original_tensor = feat_projected_me
        # feat_projected_me_shape2 = feat_projected_me.shape[1]
        # original_tensor[:,:feat_projected_me_shape2,:] = feat_projected_me
        level_positional_encodings = torch.cat(
            level_positional_encoding_list, dim=1)
        # shape (num_encoder_levels, 2), from low
        # resolution to high resolution
        num_queries_per_level = [e[0] * e[1] for e in spatial_shapes]
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        # shape (0, h_0*w_0, h_0*w_0+h_1*w_1, ...)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = torch.cat(reference_points_list, dim=0)
        reference_points = reference_points[None, :, None].repeat(
            batch_size, 1, self.num_encoder_levels, 1)
        valid_radios = reference_points.new_ones(
            (batch_size, self.num_encoder_levels, 2))
        # shape (num_total_queries, batch_size, c)
        memory = self.encoder(
            query=encoder_inputs,
            query_pos=level_positional_encodings,
            key_padding_mask=padding_masks,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_radios,something = memory_me_list)
        # (batch_size, c, num_total_queries)
        memory = memory.permute(0, 2, 1)

        

        # from low resolution to high resolution
        outs = torch.split(memory, num_queries_per_level, dim=-1)
        outs = [
            x.reshape(batch_size, -1, spatial_shapes[i][0],
                      spatial_shapes[i][1]) for i, x in enumerate(outs)
        ]

        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
                       -1):
            x = feats[i]
            cur_feat = self.lateral_convs[i](x)
            y = cur_feat + F.interpolate(
                outs[-1],
                size=cur_feat.shape[-2:],
                mode='bilinear',
                align_corners=False)
            y = self.output_convs[i](y)
            outs.append(y)
        multi_scale_features = outs[:self.num_outs]
        
        mask_feature = self.mask_feature(outs[-1])
        return mask_feature, multi_scale_features