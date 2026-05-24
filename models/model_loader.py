from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from configs import settings
import torch
from utils import device_utils


# device
device = device_utils.get_device()
dtype = device_utils.get_dtype(device)


# load model stable diffusion vao gpu
def load_text2img_pipeline(model_name = settings.default_model):
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        use_safetensors=True
    )
    pipeline.to(device)

    return pipeline


# Load Stable Diffusion + ControlNet pipeline.
def load_controlnet_pipeline():
    # Load ControlNet model
    controlnet = ControlNetModel.from_pretrained(
        settings.controlnet_mapping[
            settings.controlnet_type
        ]["model_id"],
        torch_dtype=dtype
    )
    controlnet.to(device)

    # Create Stable Diffusion pipeline with ControlNet
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        settings.base_model_id,
        controlnet=controlnet,
        torch_dtype=dtype
    )
    pipeline.to(device)

    return pipeline
