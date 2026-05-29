import os
import uuid
from configs import settings
from models import text2img_model,controlnet_model


# lưu ảnh đầu ra model dự đoán
def save_output_image(image):
    # tạo thư mục nếu chưa tồn tại
    os.makedirs(
        settings.output_folder,
        exist_ok=True
    )

    # tạo tên file ngẫu nhiên
    filename = "{}.png".format(uuid.uuid4())

    # đường dẫn file tuyệt đối
    save_path = os.path.join(settings.output_folder,filename)
    image.save(save_path)

    image_url = "/static/outputs/{}".format(filename)
    return image_url

# lưu ảnh người dụng nhập vào
def save_upload_image(file):
    # tạo thư mục nếu chưa tồn tại
    os.makedirs(
        settings.upload_folder,
        exist_ok=True
    )

    # lấy đuôi ảnh người dùng up
    ext = file.filename.split(".")[-1]

    # ten file
    filename = "{}.{}".format(uuid.uuid4(),ext)

    # đường dẫn file tuyệt đối
    save_path = os.path.join(settings.upload_folder, filename)

    file.save(save_path)
    return save_path

def process_text2img(prompt,pipeline,num_image=settings.num_image,negative_prompt=None,
                     guidance_scale=settings.guidance_scale,
                     num_inference_steps=settings.num_inference_steps,
                     width=settings.width,height=settings.height):

    # gọi đến model text2img sinh ảnh
    images = text2img_model.generate_text2img(prompt=prompt,pipeline=pipeline,num_image=num_image,negative_prompt=negative_prompt,
                                              guidance_scale=guidance_scale,num_inference_steps=num_inference_steps,
                                              width=width,height=height)
    output_paths = []
    # save từng ảnh
    for image in images:
        path = save_output_image(
            image
        )
        output_paths.append(path)
    return output_paths

def process_controlnet(prompt,upload_file,pipeline,num_image=settings.num_image,
                       negative_prompt=None,guidance_scale=settings.guidance_scale,
                       num_inference_steps=settings.controlnet_steps,width=settings.width,
                       height=settings.height,controlnet_conditioning_scale=settings.controlnet_strength):
    # save upload image
    image_path = save_upload_image(
        upload_file
    )

    images = controlnet_model.generate_controlnet(prompt=prompt, image_path=image_path,pipeline=pipeline, num_image=num_image,
                                                  negative_prompt=negative_prompt,guidance_scale=guidance_scale,num_inference_steps=num_inference_steps,
                                                  width=width,height=height,controlnet_conditioning_scale=controlnet_conditioning_scale)

    output_paths = []

    # save từng ảnh
    for image in images:
        path = save_output_image(image)

        output_paths.append(path)

    return output_paths