# DP w. GelSight Emb. (Peeling)
#python eval_real_robot_flexiv.py \
#      --config-name train_diffusion_unet_real_image_workspace \
#      task=real_peel_image_gelsight_emb_absolute_12fps \
#      +task.env_runner.output_dir=/path/for/saving/videos \
#      +ckpt_path=/path/to/dp/checkpoint

# RDP w. Force (Peeling)
# python eval_real_robot_flexiv.py \
#       --config-name train_latent_diffusion_unet_real_image_workspace \
#       task=real_peel_image_wrench_ldp_24fps \
#       +task.env_runner.output_dir=/path/for/saving/videos \
#       at=at_peel \
#       +ckpt_path=/path/to/ldp/checkpoint \
#       at_load_dir=/path/to/at/checkpoint


#!/bin/bash
GPU_ID=0
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "PYTHONPATH is $PYTHONPATH"
python eval_real_robot_flexiv.py \
      --config-name train_latent_diffusion_unet_real_image_workspace \
      task=real_umi_image_xela_wrench_ldp_24fps \
      +task.env_runner.output_dir=/home/embodied-ai/mcy/reactive_diffusion_policy_umi/saving_video \
      at=at_peel \
      +ckpt_path="/home/embodied-ai/mcy/reactive_diffusion_policy_umi/data/outputs/2025.07.25/23.43.27_train_latent_diffusion_unet_image_real_umi_image_xela_wrench_ldp_24fps_0725234325/checkpoints/latest.ckpt" \
      at_load_dir="/home/embodied-ai/mcy/reactive_diffusion_policy_umi/data/outputs/2025.07.25/21.23.35_train_vae_real_umi_image_xela_wrench_at_24fps_0725212334/checkpoints/latest.ckpt"
      # +ckpt_path="/home/embodied-ai/mcy/reactive_diffusion_policy_umi/data/outputs/2025.07.05/19.52.59_train_latent_diffusion_unet_image_real_umi_image_xela_wrench_ldp_24fps_0705195257/checkpoints/latest.ckpt" \
      # at_load_dir="/home/embodied-ai/mcy/reactive_diffusion_policy_umi/data/outputs/2025.07.05/18.46.09_train_vae_real_umi_image_xela_wrench_at_24fps_0705184608/checkpoints/latest.ckpt"