import torch
from diffusers.utils import load_image
from configs import settings


def generate_controlnet(prompt ,image_path,pipeline,
                        num_image=settings.num_image,
                        my_neg_prompt=settings.my_neg_prompt,
                        guidance_scale=settings.guidance_scale,
                        num_inference_steps=settings.controlnet_steps,
                        width=settings.width,
                        height=settings.height,
                        controlnet_conditioning_scale=settings.controlnet_strength,
                    ):
    # giúp kết quả ko bị thay đổi lung tung khi chạy lại
    generator = torch.manual_seed(settings.seed)

    # load ảnh input
    image  = load_image(image_path)

    # tạo ảnh hint image
    hint_image = settings.controlnet_mapping[settings.controlnet_type]["hinter"](image)

    # generate ảnh
    result = pipeline(
        prompt=prompt,
        image=hint_image,
        # những thứ ko muốn tạo
        negative_prompt=my_neg_prompt,
        # bám theo prompt mạnh hay yếu
        guidance_scale=guidance_scale,
        # số bước diffusion
        num_inference_steps=num_inference_steps,
        # AI nghe ảnh input mạnh tới đâu
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=generator,
        width=width,
        height=height,
        num_images_per_prompt=num_image
    )

    images = result.images

    return images

