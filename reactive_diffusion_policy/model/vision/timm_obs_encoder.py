import copy

import timm
import peft
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
from functools import partial

from reactive_diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from reactive_diffusion_policy.common.pytorch_util import replace_submodules
from reactive_diffusion_policy.model.vision.choice_randomizer import RandomChoice
# from timm.layers.attention_pool import AttentionPoolLatent
try:
    from timm.layers.attention_pool import AttentionPoolLatent   # 혹시 포크/구버전 대응
except ImportError:
    # 공식 timm 경로 (권장)
    from timm.layers.attention_pool2d import AttentionPool2d as AttentionPoolLatent


# logger = logging.getLogger(__name__)
from loguru import logger


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class TimmObsEncoder(ModuleAttrMixin):
    def __init__(self,
                 shape_meta: dict,
                 obs_horizon: int,
                 model_name: str,
                 pretrained: bool,
                 frozen: bool,
                 global_pool: str,
                 transforms: list,
                 # replace BatchNorm with GroupNorm
                 use_group_norm: bool = False,
                 # use single rgb model for all rgb inputs
                 share_rgb_model: bool = False,
                 # renormalize rgb input with imagenet normalization
                 # assuming input in [0,1]
                 imagenet_norm: bool = False,
                 three_augment: bool = False,
                 feature_aggregation: str = 'spatial_embedding',
                 downsample_ratio: int = 32,
                 position_encording: str = 'learnable',
                 use_lora: bool = False,
                 lora_rank: int = 8,
                 drop_path_rate: float = 0.0,
                 fused_model_name: str = '',
                 ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_fused_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()
        key_eval_transform_map = nn.ModuleDict()

        original_image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert original_image_shape is None or original_image_shape == shape[1:]
                original_image_shape = shape[1:]

        # fix image shape to 224x224 to align with pretrained models
        image_shape = [224, 224]


        assert global_pool == ''
        if 'resnet' in model_name:
            model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool=global_pool,  # '' means no pooling
                num_classes=0,  # remove classification layer
            )
        else:
            model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool=global_pool,  # '' means no pooling
                num_classes=0,  # remove classification layer
                img_size=image_shape[0],  # 224
                drop_path_rate=drop_path_rate,  # stochastic depth
            )

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False

        if use_lora:
            assert pretrained and not frozen
            lora_config = peft.LoraConfig(
                r=lora_rank,
                lora_alpha=8,
                lora_dropout=0.0,
                target_modules=["qkv"],
            )
            model = peft.get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        fused_model = None
        if fused_model_name != '':
            assert feature_aggregation == 'map'
            fused_model = timm.create_model(
                model_name=fused_model_name,
                pretrained=True,
                global_pool=global_pool,
                num_classes=0,
                img_size=image_shape[0],
                drop_path_rate=0.0,
            )
            for param in fused_model.parameters():
                param.requires_grad = False

        feature_dim = None
        num_heads = None
        if model_name.startswith('resnet'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('convnext'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('vit'):
            feature_dim = model.num_features
            # 08/07 3dtacdex3d 안함 no<
            # feature_dim = model.num_features + 64
            logger.info(f"feature_dim: {feature_dim}")
            num_heads = model.blocks[0].attn.num_heads
            if fused_model_name != '':
                feature_dim = feature_dim + fused_model.num_features

        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8),
                    num_channels=x.num_features)
            )
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                             # compatible with input of any size
                             torchvision.transforms.Resize(size=image_shape[0], antialias=True),
                             torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                             torchvision.transforms.Resize(size=image_shape[0], antialias=True)
                         ] + transforms[1:]
            if imagenet_norm:
                transforms = transforms + [
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        eval_transforms = None
        if transforms is not None:
            eval_transforms = [
                                  # compatible with input of any size
                                  torchvision.transforms.Resize(size=image_shape[0], antialias=True),
                                  # compatible with non-square input
                                  torchvision.transforms.RandomCrop(size=int(image_shape[0])),
                                  torchvision.transforms.Resize(size=image_shape[0], antialias=True)
                              ]
            if imagenet_norm:
                eval_transforms = eval_transforms + [
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        eval_transform = nn.Identity() if transforms is None else torch.nn.Sequential(*eval_transforms)

        if three_augment:
            # Following DeiT III: https://arxiv.org/abs/2204.07118
            primary_tfl = [
                # compatible with input of any size
                torchvision.transforms.Resize(size=image_shape[0], antialias=True),
                torchvision.transforms.RandomCrop(image_shape[0], padding=4, padding_mode='reflect'),
            ]
            secondary_tfl = [
                RandomChoice([torchvision.transforms.Grayscale(num_output_channels=3),
                              torchvision.transforms.RandomSolarize(threshold=0.5, p=1.0),
                              torchvision.transforms.GaussianBlur(kernel_size=5)]),
                torchvision.transforms.ColorJitter(0.3, 0.3, 0.3)
            ]
            final_tfl = [
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            transform = torch.nn.Sequential(*primary_tfl, *secondary_tfl, *final_tfl)
            assert eval_transform is not None and eval_transform != nn.Identity()

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model
                key_fused_model_map[key] = fused_model

                this_transform = transform
                key_transform_map[key] = this_transform
                key_eval_transform_map[key] = eval_transform
            elif type == 'low_dim':
                if not attr.get('ignore_by_policy', False):
                    low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        feature_map_shape = [x // downsample_ratio for x in image_shape]

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print('rgb keys:         ', rgb_keys)
        print('low_dim_keys keys:', low_dim_keys)

        self.model_name = model_name
        self.shape_meta = shape_meta
        self.obs_horizon = obs_horizon
        self.key_model_map = key_model_map
        self.key_fused_model_map = key_fused_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.key_eval_transform_map = key_eval_transform_map
        self.feature_aggregation = feature_aggregation
        self.fused_model_name = fused_model_name
        if model_name.startswith('vit'):
            # assert self.feature_aggregation is None # vit uses the CLS token
            if self.feature_aggregation == 'cls_token':
                pass
            elif self.feature_aggregation == 'map':
                # Multihead Attention Pooling, following https://arxiv.org/abs/1810.00825
                self.attn_pool = AttentionPoolLatent(
                    in_features=feature_dim,
                    num_heads=num_heads,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )
            # else:
            #     raise NotImplementedError(f"Unsupported feature_aggregation: {self.feature_aggregation}")
        logger.info(f"self.feature_aggregation: {self.feature_aggregation}")
        if self.feature_aggregation == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif self.feature_aggregation == 'spatial_embedding':
            self.spatial_embedding = torch.nn.Parameter(
                torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif self.feature_aggregation == 'transformer':
            if position_encording == 'learnable':
                self.position_embedding = torch.nn.Parameter(
                    torch.randn(feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim))
            elif position_encording == 'sinusoidal':
                num_features = feature_map_shape[0] * feature_map_shape[1] + 1
                self.position_embedding = torch.zeros(num_features, feature_dim)
                position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, feature_dim, 2).float() * (-math.log(2 * num_features) / feature_dim))
                self.position_embedding[:, 0::2] = torch.sin(position * div_term)
                self.position_embedding[:, 1::2] = torch.cos(position * div_term)
            self.aggregation_transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4),
                num_layers=4)
        elif self.feature_aggregation == 'attention_pool_2d':
            logger.info(f"In the attention_pool_2d")
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def aggregate_feature(self, feature, fused_feature=None):
        if self.model_name.startswith('vit'):
            if self.feature_aggregation == 'cls_token':
                return feature[:, 0, :]
            elif self.feature_aggregation == 'map':
                feature = feature[:, 1:, :]
                if fused_feature is not None:
                    num_tokens = feature.shape[1]
                    fused_feature = fused_feature[:, -num_tokens:, :]
                    feature = torch.cat([feature, fused_feature], dim=2)
                feature = self.attn_pool(feature)
                return feature

        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2)  # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2)  # B, 7*7, 512

        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1])
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1])
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1)
        elif self.feature_aggregation == 'transformer':
            zero_feature = torch.zeros(feature.shape[0], 1, feature.shape[-1], device=feature.device)
            if self.position_embedding.device != feature.device:
                self.position_embedding = self.position_embedding.to(feature.device)
            feature_with_pos_embedding = torch.concat([zero_feature, feature], dim=1) + self.position_embedding
            feature_output = self.aggregation_transformer(feature_with_pos_embedding)
            return feature_output[:, 0]
        else:
            assert self.feature_aggregation is None
            return feature

    def forward(self, obs_dict):
        features = list()
        # compatible with diffusion unet image policy
        batch_size = next(iter(obs_dict.values())).shape[0] // self.obs_horizon

        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            # compatible with diffusion unet image policy
            BT = img.shape[0]
            assert BT == batch_size * self.obs_horizon
            B = BT // self.obs_horizon
            assert img.shape[1:] == self.key_shape_map[key]
            if self.training:
                img = self.key_transform_map[key](img)
            else:
                img = self.key_eval_transform_map[key](img)
            raw_feature = self.key_model_map[key](img)
            fused_feature = None
            if self.fused_model_name != '':
                fused_feature = self.key_fused_model_map[key](img)

            feature = self.aggregate_feature(raw_feature, fused_feature)
            # compatible with diffusion unet image policy
            assert len(feature.shape) == 2 and feature.shape[0] == BT
            features.append(feature.reshape(B, -1))
            # logger.debug(f"{key} features: {[f.shape for f in features]}")

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            # compatible with diffusion unet image policy
            BT = data.shape[0]
            assert BT == batch_size * self.obs_horizon
            B = BT // self.obs_horizon
            # 08/07 3dtacdex3d
            # assert data.shape[1:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))
            # if key.endswith('tcp_wrench'):
            #     # For wrench, we use the default normalizer
            #     logger.debug(f"{key} BT shape: data.shape: {data.shape}")
            #     features[-1] = features[-1] * 0.01
            # logger.debug(f"{key} features: {[f.shape for f in features]}, data shape: {data.shape}")

        # concatenate all features
        result = torch.cat(features, dim=-1)

        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            # compatible with diffusion unet image policy
            this_obs = torch.zeros(
                (1 * self.obs_horizon, ) + shape,
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
            logger.debug(f"example_obs_dict: {key} shape: {this_obs.shape}")
        logger.debug(f"example_obs_dict: {example_obs_dict.keys()}")
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        # compatible with diffusion unet image policy
        assert example_output.shape[0] == 1
        # compatible with diffusion unet image policy
        return [example_output.shape[1] // self.obs_horizon]

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}


if __name__ == '__main__':
    timm_obs_encoder = TimmObsEncoder(
        shape_meta=None,
        model_name='resnet18.a1_in1k',
        pretrained=False,
        global_pool='',
        transforms=None
    )
