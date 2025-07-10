from typing import Dict, Callable, Tuple
import numpy as np
from reactive_diffusion_policy.common.cv2_util import get_image_transform
from loguru import logger

def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        is_extended_obs: bool = False,
        use_constant_rgb: bool = False, # 07/07 
        constant_rgb_value: float = 0.5  # 07/07
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    if is_extended_obs:
        obs_shape_meta = shape_meta['extended_obs']
    else:
        obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            # obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
            obs_dict_np[key] = np.moveaxis(out_imgs, -1, 1) # 07/07
            logger.info(f"get_real_obs_dict_rgb_size: {obs_dict_np[key].shape}")
            if use_constant_rgb:
                obs_dict_np[key][...] = constant_rgb_value
        elif type == 'low_dim':
            if "wrt" in key:
                continue
            this_data_in = env_obs[key]
            # print(f"this_data_in in get_real_obs_dict: {this_data_in}")
            if 'pose' in key and shape == (2,): # not use in mcy
                # take X,Y coordinates
                this_data_in = this_data_in[...,[0,1]]
                # print(f"this_data_in in get_real_obs_dict and pose : {this_data_in}")
            obs_dict_np[key] = this_data_in
    return obs_dict_np


def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res
