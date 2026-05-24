import torch
from configs import settings


def generate_text2img(prompt,pipeline, num_image=settings.num_image,
                        guidance_scale=settings.guidance_scale,
                        num_inference_steps=settings.num_inference_steps,
                        height=settings.height,
                        width=settings.width):

    # giúp kết quả ko bị thay đổi lung tung khi chạy lại
    generator = torch.manual_seed(settings.seed)

    # bắt đầu sinh ảnh
    result = pipeline(
        prompt=prompt,
        # bám theo prompt mạnh hay yếu
        guidance_scale=guidance_scale,
        # số bước diffusion
        num_inference_steps=num_inference_steps,
        generator=generator,
        num_images_per_request = num_image,  # nếu muônd sinh bao nhiêu ảnh để số bấy nhiêu
        height=height,
        width=width
    )

    images = result.images

    return images