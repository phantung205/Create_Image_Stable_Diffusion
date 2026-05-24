import os
import torch
import controlnet_hinter

""" phần chung """
# root project
root_project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# config
output_folder = "static/outputs"
upload_folder = "static/uploads"

# tham số chung
guidance_scale = 0.75
height = 512
width= 512
seed = 42



""" phần text to image"""
# list model
model_list = ["nota-ai/bk-sdm-small",
              "CompVis/stable-diffusion-v1-4",
              "runwayml/stable-diffusion-v1-5",
              "prompthero/openjourney",
              "hakurei/waifu-diffusion",
              "stabilityai/stable-diffusion-2-1",
              "dreamlike-art/dreamlike-photoreal-2.0",
              ]
default_model = model_list[2]

# config tham số
num_inference_steps = 100



""" phần image and prompt to image"""

# Cấu hình ControlNet Mapping
controlnet_mapping = {
    "canny_edge": {
        "model_id": "lllyasviel/sd-controlnet-canny",
        "hinter": controlnet_hinter.hint_canny
    },
    "pose": {
        "model_id": "lllyasviel/sd-controlnet-openpose",
        "hinter": controlnet_hinter.hint_openpose
    },
}

# định nghĩa model và load controlnet
controlnet_type = "canny_edge"
base_model_id = "digiplay/Juggernaut_final"

# cấu hình thông số sinh ảnh
my_neg_prompt = "lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
controlnet_steps = 20
controlnet_strength = 1.0
num_image = 2
