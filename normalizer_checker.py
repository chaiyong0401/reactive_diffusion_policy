import pickle
import torch

# 경로를 정확히 지정
normalizer_path = "/home/dyros-recruit/mcy_ws/reactive_diffusion_policy_umi/data/outputs/2025.05.14/17.23.03_train_latent_diffusion_unet_image_real_peel_image_gelsight_emb_ldp_24fps_0514172301/normalizer.pkl"

with open(normalizer_path, 'rb') as f:
    normalizer = pickle.load(f)

# 예: 모든 키 출력
print("Normalizer keys:", list(normalizer.params_dict.keys()))

# 예: 각 항목의 offset과 scale 값 확인
for key in normalizer.params_dict:
    print(f"\n[ {key} ]")
    offset = normalizer.params_dict[key]['offset']
    scale = normalizer.params_dict[key]['scale']
    print("offset:", offset)
    print("scale:", scale)
