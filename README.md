<h3 align="center">
    Reactive Diffuison Policy:
</h3>
<h4 align="center">
    Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation
</h4>
<h4 align="center">
    In Submission
</h4>

<p align="center">
    <a href="https://hanxue.me">Han Xue</a><sup>1*</sup>,
    Jieji Ren<sup>1*</sup>,
    <a href="https://wendichen.me">Wendi Chen</a><sup>1*</sup>,
    <br>
    <a href="https://www.gu-zhang.com">Gu Zhang</a><sup>234‡</sup>,
    Yuan Fang<sup>1†</sup>,
    <a href="https://softrobotics.sjtu.edu.cn">Guoying Gu</a>,
    <a href="http://hxu.rocks">Huazhe xu</a><sup>234‡</sup>,
    <a href="https://www.mvig.org">Cewu Lu</a><sup>1‡</sup>
    <br>
    <sup>1</sup>Shanghai Jiao Tong University
    <sup>2</sup>Tsinghua University, IIIS
    <sup>3</sup>Shanghai Qi Zhi Institute
    <sup>4</sup>Shanghai AI Lab
    <br>
    <sup>*</sup>Equal contribution
    <sup>†</sup>Equal contribution
    <sup>‡</sup>Equal advising
    <br>
</p>

<div align="center">
<a href='https://arxiv.org/abs/2503.02881'><img alt='arXiv' src='https://img.shields.io/badge/arXiv-2503.02881-red.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://reactive-diffusion-policy.github.io'><img alt='project website' src='https://img.shields.io/website-up-down-green-red/http/cv.lbesson.qc.to.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<img alt='powered by Pytorch' src='https://img.shields.io/badge/PyTorch-❤️-F8C6B5?logo=pytorch&logoColor=white'> &nbsp;&nbsp;&nbsp;&nbsp;
</div>

<p align="center">
<img src="assets/teaser.png" alt="teaser" style="width:90%;" />
</p>

## TODO
- [x] Release the code of TactAR and [Quest3 APP](https://github.com/xiaoxiaoxh/TactAR_APP).
- [ ] Release the code of RDP. (ETA: April 6th)
- [ ] Release the data. (ETA: April 9th)
- [ ] Release the pretrained models. (ETA: April 9th)
- [ ] Add guide for customized tactile/force sensors.
- [ ] Add guide for collecting the tactile dataset.
- [ ] Support single robot arm.
- [ ] Support more robots (e.g. Franka).
- [ ] Support 3D visual input in TactAR.
- [ ] Add docker image for easy setup.

## ⚙️ Environment Setup
### Hardware
#### Device List
- Meta Quest 3 VR headset.
- Workstation with Ubuntu 22.04 for compatibility with ROS2 Humble.
    > A workstation with a GPU (e.g., NVIDIA RTX 3090) is required.
      If GelSight Mini is used, a high-performance CPU (e.g., Core i9-13900K) is required to
      ensure 24 FPS tactile sensing.
- 2 robot arms with (optional) joint torque sensors.
    > We use [Flexiv Rizon 4](https://www.flexiv.com/products/rizon) with the [GRAV](https://www.flexiv.com/products/grav) gripper by default and 
      will support single robot and other robots soon.
- 1-3 [RealSense](https://www.intelrealsense.com) cameras.
    > We use D435 for wrist camera and D415 for external cameras.
      Follow the [official document](https://dev.intelrealsense.com/docs/compiling-librealsense-for-linux-ubuntu-guide) to install librealsense2. 
- (Optional) 1-2 [GelSight Mini](https://www.gelsight.com/gelsightmini) tactile sensors with [tracking marker gel](https://www.gelsight.com/product/tracking-marker-replacement-gel).
    > We use 1 sensor for each robot arm. Download the [CAD model](https://drive.google.com/drive/folders/13tS5cMgPOnqIQvKm3XiM-n6DmyEc4qy2?usp=share_link) and 3D print the mount to attach the sensor to the GRAV gripper.

### Software
#### Quest 3 Setup
Build and install the TactAR APP on the Quest 3 according to
the [README in our Unity Repo](https://github.com/xiaoxiaoxh/TactAR_APP).

#### Workstation Setup
1. Follow the [official document](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
to install ROS2 Humble.
2. Since ROS2 has some compatibility issues with Conda,
   we recommend using a virtual environment with `venv`.
   ```bash
   python3 -m venv rdp_venv
   source rdp_venv/bin/activate
   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
   pip install -r requirements.txt
   ```

## 📦 Data Collection
### TactAR Setup
1. For the workstation,
the environment and the task have to be configured first and
then start several services for teleoperating robots, publishing sensor data and record the data.
    1. Environment and Task configuration.
        - **Calibration.**
          Example calibration files are proviced in [data/calibration](data/calibration).
          Each `A_to_B_transform.json` contains the transformation from coordinate system A to coordinate system B. 
            > We use calibration files only for bimanual manipulation.
        - **Environment Configuration.**
          Edit [reactive_diffusion_policy/config/task/real_robot_env.yaml](reactive_diffusion_policy/config/task/real_robot_env.yaml)
          to configure the environment settings including `host_ip`, `robot_ip`, `vr_server_ip` and `calibration_path`.
        - **Task Configuration.**
          Create task configuration file which assigns the camera and sensor to use.
          You can take [reactive_diffusion_policy/config/task/peeling_two_realsense_one_gelsight_one_mctac_24fps.yaml](reactive_diffusion_policy/config/task/peeling_two_realsense_one_gelsight_one_mctac_24fps.yaml)
          as an example.
   2. Start services.
      ```bash
      # start teleoperation server
      python teleop.py task=[task_config_file_name]
      # start camera node launcher
      python camera_node_launcher.py task=[task_config_file_name]
      # start data recorder
      python record_data.py --save_to_disk --save_file_dir [task_data_dir] --save_file_name [record_seq_file_name]
      ```
2. For Quest 3,
follow the [user guide in our Unity Repo](https://github.com/xiaoxiaoxh/TactAR_APP/blob/master/Docs/User_Guide.md)
to run the TactAR APP.

### (Important) Data Collection Tips
Please refer to [docs/data_collection_tips.md](docs/data_collection_tips.md).

### Data Postprocessing
Change the config in [post_process_data.py](post_process_data.py)
(including `TAG`, `ACTION_DIM`, `TEMPORAL_DOWNSAMPLE_RATIO` and `SENSOR_MODE`)
and execute the command
```bash
python post_process_data.py
```
We use [Zarr](https://zarr.dev/) to store the data.
After postprocessing, you may see the following structure:
```
 ├── action (25710, 4) float32
 ├── external_img (25710, 240, 320, 3) uint8
 ├── left_gripper1_img (25710, 240, 320, 3) uint8
 ├── left_gripper1_initial_marker (25710, 63, 2) float32
 ├── left_gripper1_marker_offset (25710, 63, 2) float32
 ├── left_gripper1_marker_offset_emb (25710, 15) float32
 ├── left_gripper2_img (25710, 240, 320, 3) uint8
 ├── left_gripper2_initial_marker (25710, 25, 2) float32
 ├── left_gripper2_marker_offset (25710, 25, 2) float32
 ├── left_gripper2_marker_offset_emb (25710, 15) float32
 ├── left_robot_gripper_force (25710, 1) float32
 ├── left_robot_gripper_width (25710, 1) float32
 ├── left_robot_tcp_pose (25710, 9) float32
 ├── left_robot_tcp_vel (25710, 6) float32
 ├── left_robot_tcp_wrench (25710, 6) float32
 ├── left_wrist_img (25710, 240, 320, 3) uint8
 ├── right_robot_gripper_force (25710, 1) float32
 ├── right_robot_gripper_width (25710, 1) float32
 ├── right_robot_tcp_pose (25710, 9) float32
 ├── right_robot_tcp_vel (25710, 6) float32
 ├── right_robot_tcp_wrench (25710, 6) float32
 ├── target (25710, 4) float32
 └── timestamp (25710,) float32
```

## Q&A
Please refer to [docs/Q&A.md](docs/Q&A.md).

## 🙏 Acknowledgement
Our work is built upon [Diffusion Policy](https://github.com/real-stanford/diffusion_policy). Thanks for their great work!

## 🔗 Citation
If you find our work useful, please consider citing:
```
@article{xue2025reactive,
  title     = {Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation},
  author    = {Xue, Han and Ren, Jieji and Chen, Wendi and Zhang, Gu and Fang, Yuan and Gu, Guoying and Xu, Huazhe and Lu, Cewu},
  journal   = {arXiv preprint arXiv:2503.02881},
  year      = {2025}
}
```
