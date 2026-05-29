from flask import Flask,render_template,request
from services.generation_service import process_text2img,process_controlnet
from models.model_loader import load_text2img_pipeline,load_controlnet_pipeline


# tạo app flask
app = Flask(__name__)

# load model ,pipline
print("Loading Text2Img Pipeline...")
text2img_pipeline = load_text2img_pipeline()

print("Loading ControlNet Pipeline...")
controlnet_pipeline = load_controlnet_pipeline()

# trạng chủ
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template(
            "index.html"
        )

    else:
        # lấy data từ promt
        prompt = request.form["prompt"]
        # get này ko bắt buộc có
        negative_prompt = request.form.get("negative_prompt")
        num_image = int(request.form["num_image"])
        guidance_scale = float(request.form["guidance_scale"])
        num_inference_steps = int(request.form["num_inference_steps"])
        width = int(request.form["width"])
        height = int(request.form["height"])
        controlnet_conditioning_scale = float(request.form["controlnet_conditioning_scale"])

        # lấy ảnh upload
        image_file = request.files["image"]

        # text 2 iamge
        if image_file.filename == "":
            output_paths = process_text2img(
                prompt=prompt,
                pipeline=text2img_pipeline,
                num_image=num_image,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height
            )

        else:
            output_paths = process_controlnet(
                prompt=prompt,
                upload_file=image_file,
                pipeline=controlnet_pipeline,
                num_image=num_image,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                controlnet_conditioning_scale=controlnet_conditioning_scale
            )

            # render html
        return render_template(
            "index.html",
            image_paths=output_paths
        )

if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=8888
    )