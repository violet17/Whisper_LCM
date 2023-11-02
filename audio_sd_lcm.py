
# conda create -n sd python=3.10
# conda activate sd
# conda install ffmpeg

# pip install openvino-2023.2.0-12778-cp310-cp310-win_amd64.whl
# pip install torch==2.1.0 torchvision
# pip install diffusers transformers omegaconf accelerate
# pip install numpy opencv-python pillow
# pip install gradio==3.41.1 
# pip install sentencepiece
# pip install bigdl-llm==2.4.0b20231009 py-cpuinfo
# pip install SpeechRecognition

## 运行代码：
# python audiosd_v0.1.py

import os
import time
import torch
import numpy as np
import gradio as gr
from PIL import Image
import cv2
import openvino.frontend.pytorch.torchdynamo.backend
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
# from diffusers import StableDiffusionImg2ImgPipeline
# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
# from diffusers import StableDiffusionInpaintPipeline

from lcm.lcm_scheduler import LCMScheduler
from lcm.lcm_pipeline import LatentConsistencyModelPipeline
from lcm.lcm_i2i_pipeline import LatentConsistencyModelImg2ImgPipeline, LCMSchedulerWithTimestamp

import speech_recognition as sr
# from bigdl.llm.transformers import AutoModelForSpeechSeq2Seq
# from transformers import WhisperProcessor
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer

import random
# from concurrent.futures import ThreadPoolExecutor
# import uuid


os.environ["PYTORCH_TRACING_MODE"] = "TORCHFX"
#Adjust the ‘GPU’ parameter to ‘GPU.0’, ‘GPU.1’ etc. if there are multiple GPUs on the system
os.environ["OPENVINO_TORCH_BACKEND_DEVICE"] = "GPU" 
# enable caching
os.environ["OPENVINO_TORCH_MODEL_CACHING"] = "1"

MAX_SEED = np.iinfo(np.int32).max

templates = {
    "None": "{prompt}. best quality, masterpiece, 8k, RAW photo, an extremely delicate and beautiful, extremely detailed, ultra-detailed, highres", 
    "Cinematic": "cinematic still for {prompt} . best quality, masterpiece, 8k, emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy", 
    "Anime": "anime artwork for {prompt} . masterpiece, anime style, key visual, vibrant, studio anime,  highly detailed", 
    "3D": "professional 3d model for {prompt} . masterpiece, octane render, highly detailed, volumetric, dramatic lighting", 
    "Steam": "steampunk style for {prompt} . masterpiece, antique, mechanical, brass and copper tones, gears, intricate, detailed", 
    "Pop": "Pop Art style {prompt} . masterpiece, bright colors, bold outlines, popular culture themes, ironic or kitsch", 
    "Pixel": "pixel-art for {prompt} . masterpiece, low-res, blocky, pixel art style, 8-bit graphics"
}


# def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
#     if randomize_seed:
#         seed = random.randint(0, MAX_SEED)
#     return seed

# def save_image(img, profile: gr.OAuthProfile | None, metadata: dict, root_path='./'):
#     unique_name = str(uuid.uuid4()) + '.png'
#     unique_name = os.path.join(root_path, unique_name)
#     img.save(unique_name)
#     # gr_user_history.save_image(label=metadata["prompt"], image=img, profile=profile, metadata=metadata)
#     return unique_name

# def save_images(image_array, profile: gr.OAuthProfile | None, metadata: dict):
#     paths = []
#     root_path = './images/'
#     os.makedirs(root_path, exist_ok=True)
#     with ThreadPoolExecutor() as executor:
#         paths = list(executor.map(save_image, image_array, [profile]*len(image_array), [metadata]*len(image_array), [root_path]*len(image_array)))
#     return paths



def load_whisper_model(model_path):
    # whisper_model_path = model_path+"whisper-medium-int4"
    whisper_model_path = model_path+"whisper-medium"
    print("loading whisper---------")
    t0 = time.time()
    # processor = WhisperProcessor.from_pretrained(whisper_model_path)
    # whisper =  AutoModelForSpeechSeq2Seq.load_low_bit(whisper_model_path, trust_remote_code=True, optimize_model=False)
    # whisper =  AutoModelForSpeechSeq2Seq.load_low_bit(whisper_model_path, trust_remote_code=True, optimize_model=False, tie_word_embeddings=False)
    processor = WhisperProcessor.from_pretrained(whisper_model_path)
    whisper =  WhisperForConditionalGeneration.from_pretrained(whisper_model_path)
    whisper.config.forced_decoder_ids = None
    t1 = time.time()
    print("loading whisper----------Done, cost time(s): ", t1-t0)
    return processor, whisper

def load_marian_model(model_path):
    marian_model_path = model_path+"opus-mt-zh-en"
    print("loading Marian---------")
    t0 = time.time()
    marian_tokenizer = MarianTokenizer.from_pretrained(marian_model_path)
    marian_model = MarianMTModel.from_pretrained(marian_model_path)
    t1 = time.time()
    print("loading Marian----------Done, cost time(s): ", t1-t0)
    return marian_tokenizer, marian_model

def load_sd_model(model_path, mode=0):
    # global sd_model
    sd_model_path = model_path + "stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors"
    original_config_file = model_path + "stable-diffusion-v1-5/v1-inference.yaml"
    print("loading SD---------")
    t0 = time.time()
    # if mode == 0: #txt2img
    sd_model = StableDiffusionPipeline.from_single_file(sd_model_path, original_config_file=original_config_file, load_safety_checker=False)   
    if mode == 1: #img2img
        sd_model = StableDiffusionImg2ImgPipeline(**sd_model.components)
    elif mode == 2: # controlnet
        controlnet_path = model_path + "lllyasviel--sd-controlnet-canny"
        controlnet = ControlNetModel.from_pretrained(controlnet_path)
        sd_model = StableDiffusionControlNetPipeline(**sd_model.components, controlnet=controlnet)
        sd_model.controlnet = torch.compile(sd_model.controlnet, backend="openvino")
    elif mode == 3: #inpaiting
        sd_model = StableDiffusionInpaintPipeline(**sd_model.components)
    

    sd_model.scheduler = DPMSolverMultistepScheduler.from_config(sd_model.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True) #DPM++ 2M Karras


    sd_model.unet = torch.compile(sd_model.unet, backend="openvino")
    # sd_model.vae.encode = torch.compile(sd_model.vae.encode, backend="openvino")
    sd_model.vae.decode = torch.compile(sd_model.vae.decode, backend="openvino")
    t1 = time.time()
    print("loading SD----------Done, cost time(s): ", t1-t0)

    print("warmup----------")
    if mode == 0:
        sd_model(prompt="", num_inference_steps=1, guidance_scale=7, output_type="pil", width=512, height=512).images[0]
    elif mode == 1:
        # init_image = Image.fromarray(np.uint8(np.random.rand(512, 512, 3)*255))
        init_image =  Image.fromarray(cv2.imread("input_image_vermeer.png"))
        sd_model(prompt="", image=init_image, num_inference_steps=2, guidance_scale=7, output_type="pil").images[0]
    elif mode == 2:
        # init_image = Image.fromarray(np.uint8(np.random.rand(512, 512, 3)*255))
        init_image =  Image.fromarray(cv2.imread("input_image_vermeer.png"))
        sd_model(prompt="", image=init_image, guess_mode=True, num_inference_steps=1, guidance_scale=7, output_type="pil", width=512, height=512).images[0]
    elif mode == 3:
        # init_image = Image.fromarray(np.uint8(np.random.rand(512, 512, 3)*255))
        init_image =  Image.fromarray(cv2.imread("input_image_vermeer.png"))
        sd_model(prompt="", image=init_image, mask_image=init_image, num_inference_steps=1, guidance_scale=7, output_type="pil", width=512, height=512).images[0]
    print("warmup----------Done")

    return sd_model

def load_lcm_model(model_path, mode=0):
    lcm_scheduler_path = model_path + "LCM_Dreamshaper_v7/scheduler/scheduler_config.json"
    lcm_model_path = model_path + "LCM_Dreamshaper_v7" 
    print("loading LCM---------")
    t0 = time.time()
    if mode == 0:
        lcm_scheduler = LCMScheduler.from_pretrained(lcm_scheduler_path)
        lcm_model = LatentConsistencyModelPipeline.from_pretrained(lcm_model_path, scheduler=lcm_scheduler, safety_checker=None, torch_dtype=torch.float32)

        lcm_model.text_encoder = torch.compile(lcm_model.text_encoder, backend="openvino")
        lcm_model.unet = torch.compile(lcm_model.unet, backend="openvino")
        lcm_model.vae.decode = torch.compile(lcm_model.vae.decode, backend="openvino")
        t1 = time.time()
        print("loading SD----------Done, cost time(s): ", t1-t0)

        print("warmup----------")    
        # lcm_model(prompt="", width=512, height=512, guidance_scale=8, num_inference_steps=1, num_images_per_prompt=1, output_type="pil")
        lcm_model(prompt="", width=512, height=512, guidance_scale=8, num_inference_steps=1, num_images_per_prompt=1, lcm_origin_steps=50, output_type="pil")
        print("warmup----------Done")

    elif mode == 2:
        lcm_scheduler = LCMSchedulerWithTimestamp.from_pretrained(lcm_scheduler_path)
        lcm_model = LatentConsistencyModelImg2ImgPipeline.from_pretrained(lcm_model_path, scheduler=lcm_scheduler, safety_checker=None, torch_dtype=torch.float32)

        lcm_model.text_encoder = torch.compile(lcm_model.text_encoder, backend="openvino")
        lcm_model.unet = torch.compile(lcm_model.unet, backend="openvino")
        lcm_model.vae.encode = torch.compile(lcm_model.vae.encode, backend="openvino")
        lcm_model.vae.decode = torch.compile(lcm_model.vae.decode, backend="openvino")
        t1 = time.time()
        print("loading SD----------Done, cost time(s): ", t1-t0)

        print("warmup----------")
        # image =  Image.fromarray(cv2.imread("input_image_vermeer.png"))
        image = Image.fromarray(np.uint8(np.random.rand(512, 512, 3)*255))
        # lcm_model(prompt="", image=image, strength=0.5, width=512, height=512, guidance_scale=8, num_inference_steps=1, num_images_per_prompt=1, output_type="pil")
        lcm_model(prompt="", image=image, strength=0.5, width=512, height=512, guidance_scale=8, num_inference_steps=1, num_images_per_prompt=1, lcm_origin_steps=50, output_type="pil")

        print("warmup----------Done")


    return lcm_model


def resample(audio, src_sample_rate, dst_sample_rate):
    """
    Resample audio to specific sample rate

    Parameters:
      audio: input audio signal
      src_sample_rate: source audio sample rate
      dst_sample_rate: destination audio sample rate
    Returns:
      resampled_audio: input audio signal resampled with dst_sample_rate
    """
    if src_sample_rate == dst_sample_rate:
        return audio
    duration = audio.shape[0] / src_sample_rate
    resampled_data = np.zeros(shape=(int(duration * dst_sample_rate)), dtype=np.float32)
    x_old = np.linspace(0, duration, audio.shape[0], dtype=np.float32)
    x_new = np.linspace(0, duration, resampled_data.shape[0], dtype=np.float32)
    resampled_audio = np.interp(x_new, x_old, audio)
    return resampled_audio.astype(np.float32)

def get_input_features(processor, audio_file, device):
    with sr.AudioFile(audio_file) as source:
        audio = sr.Recognizer().record(source)  # read the entire audio file
    frame_data = np.frombuffer(audio.frame_data, np.int16).flatten().astype(np.float32) / 32768.0
    # audio.ndim == 2:
    #    audio = audio.mean(axis=1)
    if audio.sample_rate != 16000:
        print("=====rasample audio to 16000Hz")
        frame_data = resample(frame_data, audio.sample_rate, 16000)
    input_features = processor(frame_data, sampling_rate=16000, return_tensors="pt").input_features
    # if device == "xpu":
    #     input_features = input_features.half().contiguous().to(device)
    #else:
    #    input_features = input_features.contiguous().to(device)
    input_features = input_features.contiguous().to(device)
    return input_features

def get_prompt(processor, whisper, audio, device="cpu"):
    with torch.inference_mode():
        input_features = get_input_features(processor, audio, device)#0.09s
        predicted_ids = whisper.generate(input_features)
        output_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0] 
    return output_str

def translate(prompt):
    if prompt == "":
        return ""
    
    texts = prompt.split('\n')
    output = ""

    for text in texts:
        encoded_text = marian_tokenizer.encode(text, return_tensors='pt')
        translation = marian_model.generate(encoded_text)
        decoded_translation = marian_tokenizer.decode(translation[0], skip_special_tokens=True)
        if output == "":
            output = decoded_translation
        else:
            output = output + '\n' + decoded_translation
    return output

def predict(mode, audio_input, prompt=None, param_template="None", height=512, width=512, steps=20, image_input=None, strength=0.5):
    global model_path, lcm_model, whisper, processor, marian_tokenizer, marian_model, last_width, last_height, last_mode
    # mode = 0
    print("\n--------start generate ", mode)
    mode = int(mode)
    # print("\n--------start mode: ", mode, mode==1, type(mode), mode != last_mode)
    # prompt_template = "cinematic still {prompt_input} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy"
    # negative_prompt = "anime, people, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, NSFW, low res, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    cfg_scale = 7
    # print("last: ", last_height, last_width, mode, last_mode, width != last_width, height != last_height , mode != last_mode, type(mode), type(last_mode))


    if width != last_width or height != last_height or mode != last_mode:
        print("------reload model")
        lcm_model = load_lcm_model(model_path, mode)
        last_width = width
        last_height = height
        last_mode = mode
        # sd_model.unet = torch.compile(sd_model.unet, backend="openvino")
        # # sd_model.vae.encode = torch.compile(sd_model.vae.encode, backend="openvino")
        # sd_model.vae.decode = torch.compile(sd_model.vae.decode, backend="openvino")
        # if mode == 2:
        #     sd_model.controlnet = torch.compile(sd_model.controlnet, backend="openvino")



    sr_latency = None
    if audio_input:
        t0 = time.time()
        prompt_zh = get_prompt(processor, whisper, audio_input)
        t1 = time.time()
        sr_latency = (t1 - t0) 
        print("SR time cost(s): ", sr_latency)
        sr_latency = str(round(sr_latency, 2))
        print("Chinese prompt: ", prompt_zh)
        # prompt_zh = prompt #test!!!!!!!!!!
        prompt_en = translate(prompt_zh)
        print("English prompt: ", prompt_en)
        # prompt = prompt_template.format(prompt_input=promptTranslated)
        print("param_template: ", param_template)
        prompt = templates[param_template].format(prompt=prompt_en)
        # print("start------mode, image_input.shape", mode, image_input)
        yield prompt_zh, None, sr_latency, None

    print("prompt after Rich: ", prompt)
    seed = random.randint(0, MAX_SEED) 
    torch.manual_seed(seed)
    if mode == 0: #txt2img   
        print("-----txt2img")   
        # print("0------mode, image_input.shape", mode,  image_input)

        t0 = time.time()
        output = lcm_model(
                        prompt=prompt,
                        num_inference_steps=steps,
                        guidance_scale=cfg_scale,
                        output_type="pil",
                        width=width,
                        height=height,
                        lcm_origin_steps=50,
                ).images[0]
        t1 = time.time()
        output.save("results/result.png")
        sd_latency = (t1 - t0)
        print("SD time cost(s): ", sd_latency)
        sd_latency = str(round(sd_latency, 2)) 
        yield prompt_zh, output, sr_latency, sd_latency
    elif mode == 1: #img2img
        # print("1------mode, image_input.shape", mode, image_input)
        print("-----img2img")
        # sd_model = load_sd_model(model_path, mode)
        # prompt = "ghibli style"
        # prompt = ""
        # negative_prompt = ""
        # steps = 20
        # cfg_scale = 7

        # init_image =  Image.fromarray(cv2.imread("input_image_vermeer.png"))
        # denoising_strength =
        # print("-----image_input: ", image_input)
        # image_input.save("temp.png")
        if image_input is None:
            print("Please input image!!!")
            return
        init_image = image_input#['image']
        # print("init_image: ", init_image.shape)


        t0 = time.time()
        output = lcm_model(
                        prompt=prompt,                                      
                        image=init_image, #
                        strength=strength,
                        num_inference_steps=steps,
                        guidance_scale=cfg_scale,
                        lcm_origin_steps=50,
                        output_type="pil",
                        # width=512,
                        # height=512,
                ).images[0]
        t1 = time.time()
        output.save("results/result.png")
        sd_latency = (t1 - t0)
        print("SD time cost(s): ", sd_latency)
        sd_latency = str(round(sd_latency, 2)) 
        yield prompt_zh, output, sr_latency, sd_latency
    # elif mode == 2 : #controlnet
    #     print("-----controlnet")
    #     # sd_model = load_sd_model(model_path, mode)
    #     # image = cv2.imread("input_image_vermeer.png")
    #     # print("-----image_input: ", image_input.shape)
    #     image = np.array(image_input)#['image']
    #     low_threshold = 100
    #     high_threshold = 200

    #     image = cv2.Canny(image, low_threshold, high_threshold)
    #     image = image[:, :, None]
    #     image = np.concatenate([image, image, image], axis=2)
    #     canny_image = Image.fromarray(image)

    #     t0 = time.time()
    #     output = sd_model(
    #                     prompt="",
    #                     negative_prompt=negative_prompt,
    #                     image=canny_image,
    #                     guess_mode=True,
    #                     num_inference_steps=steps,
    #                     guidance_scale=cfg_scale,
    #                     output_type="pil",
    #                     width=512,
    #                     height=512,
    #             ).images[0]
    #     t1 = time.time()
    #     output.save("results/result.png")
    #     sd_latency = (t1 - t0)
    #     print("SD time cost(s): ", sd_latency)
    #     sd_latency = str(round(sd_latency, 2)) 
    #     yield prompt_zh, output, sr_latency, sd_latency
    # elif mode == 3: #inpainting
    #     print("-----inpainting")
    #     # sd_model = load_sd_model(model_path, mode)
    #     # print("-----image_input: ", image_input.shape)
    #     # init_image = Image.open("overture-creations-5sI6fQgYIuo.png").convert("RGB").resize((512,512))#cv2.imread("overture-creations-5sI6fQgYIuo.png")
    #     # mask_image = Image.open("overture-creations-5sI6fQgYIuo_mask.png").convert("RGB").resize((512,512))#cv2.imread("overture-creations-5sI6fQgYIuo_mask.png")
    #     # denoising_strength =
    #     init_image = image_input['image']
    #     mask_image = image_input['mask']


    #     t0 = time.time()
    #     output = sd_model(
    #                     prompt=prompt,                   
    #                     negative_prompt=negative_prompt,
    #                     image=init_image,
    #                     mask_image=mask_image, 
    #                     num_inference_steps=steps,
    #                     guidance_scale=cfg_scale,
    #                     output_type="pil",
    #                     width=512,
    #                     height=512,
    #             ).images[0]
    #     t1 = time.time()
    #     output.save("results/result.png")
    #     sd_latency = (t1 - t0)
    #     print("SD time cost(s): ", sd_latency)
    #     sd_latency = str(round(sd_latency, 2))
    #     yield prompt_zh, output, sr_latency, sd_latency



def reset_user_input():
    return gr.update(value='')

def reset_state():
    return [], [], "", ""



if __name__ == '__main__':
    global model_path, lcm_model, whisper, processor, marian_tokenizer, marian_model, last_width, last_height, last_mode
    model_path = "./models/"

    if not os.path.exists("./results"):
        os.makedirs("./results")
    
    last_mode = 0
    last_width = 512
    last_height = 512
    

    # sd_model = load_sd_model(model_path)
    lcm_model = load_lcm_model(model_path)
    processor, whisper = load_whisper_model(model_path)
    marian_tokenizer, marian_model = load_marian_model(model_path)

    



    """Override Chatbot.postprocess"""
    # gr.Chatbot.postprocess = postprocess



    outlen=512
    device_list = ["iGPU", "CPU"]
    template_choices = ["None", "Cinematic", "Anime", "Pixel", "Pop"]
    # Main UI Framework display:flex;flex-wrap:wrap;
    with gr.Blocks(theme=gr.themes.Base.load("theme3.json"), css=".gradio-container {display:flex;flex-wrap: wrap;} footer {visibility: hidden}") as demo: ## 可以在huging face下载模板
    #with gr.Blocks(css=css) as demo: ## 可以在huging face下载模板
        gr.HTML("""<h1 align="center">英特尔AI生图应用</h1>""")
        with gr.Tab("文生图"):
            mode = gr.Textbox(value=0, label="模式", visible=False)
            with gr.Row():
                with gr.Column(scale=2):
                    # sr_device = gr.Dropdown(device_list,value="CPU",label="选择推理设备", interactive=True, visible=False)
                    sr_device = gr.Textbox(value="CPU",label="推理设备", visible=True)
                    sr_model_name = gr.Textbox(label="语音识别模型", value="whisper-medium")
                    with gr.Row():
                        # sd_device = gr.Dropdown(device_list,value="iGPU",label="选择推理设备", interactive=True, visible=False)       
                        sd_device = gr.Textbox(value="iGPU",label="推理设备", visible=True)         
                        sd_model_name = gr.Textbox(label="图像生成模型", value="LCM_Dreamshaper_v7")
                        param_template = gr.Dropdown(template_choices, label='风格', value=template_choices[0], interactive=True)
                        height = gr.Slider(256, 768, value=512, step=32, label="高度", interactive=True)
                        width = gr.Slider(256, 768, value=512, step=32, label="宽度", interactive=True)
                        steps = gr.Slider(1, 8, value=4, step=1, label="生成步数", interactive=True)
                        # seed = gr.Slider(0, 2048, value=outlen, step=1.0, label="生成种子", interactive=True)
                    with gr.Row():                   
                        sr_latency = gr.Textbox(label="语音识别耗时（s）", visible=True)
                        sd_latency = gr.Textbox(label="图像生成耗时（s）", visible=True)
                with gr.Column(scale=8):
                    prompt_img = gr.Textbox(label = "提示词", show_label=True, visible=True)
                    # promptTranslated = gr.Textbox(label = "English", show_label=True, visible=False)
                    img_output = gr.Image(scale=1, label="生成结果", type="pil", image_mode="RGB", height=540, visible=True)                                       
                    user_input = gr.Textbox(show_label=False, placeholder="请在此输入文字...", lines=5, container=False, scale=3, interactive=True, visible=False)
                    audio_input = gr.Audio(source="microphone", label="输入音频", type="filepath", interactive=True)
                    with gr.Row():
                        submitBtn = gr.Button("提交", variant="primary", interactive=True)
                        # emptyBtn = gr.Button("清除",interactive=True)
        with gr.Tab("图生图"):
            mode1 = gr.Textbox(value=1, label="模式", visible=False)
            with gr.Row():
                with gr.Column(scale=2):
                    sr_device1 = gr.Textbox(value="CPU",label="推理设备", visible=True)
                    whisper_model1 = gr.Textbox(label="语音识别模型", value="whisper-medium")
                    with gr.Column(scale=2):
                        sd_device1 = gr.Textbox(value="iGPU",label="推理设备", visible=True)                 
                        sd_model_name1 = gr.Textbox(label="图像生成模型", value="LCM_Dreamshaper_v7")
                        param_template = gr.Dropdown(template_choices, label='风格', value=template_choices[0], interactive=True)
                        height1 = gr.Slider(256, 768, value=512, step=32, label="高度", interactive=True)
                        width1 = gr.Slider(256, 768, value=512, step=321, label="宽度", interactive=True)
                        steps1 = gr.Slider(1, 8, value=4, step=1, label="生成步数", interactive=True)
                        strength = gr.Slider(label="Strength", minimum=0, maximum=1, step=0.1, value=0.5, visible=False)
                        # seed1 = gr.Slider(0, 2048, value=outlen, step=1.0, label="生成种子", interactive=True)
                    with gr.Column(scale=2):                   
                        sr_latency1 = gr.Textbox(label="语音识别耗时（s）", visible=True)
                        sd_latency1 = gr.Textbox(label="图像生成耗时（s）", visible=True)
                with gr.Column(scale=8):
                    prompt_img1 = gr.Textbox(label = "提示词", show_label=True, visible=True)
                    # promptTranslated = gr.Textbox(label = "English", show_label=True, visible=True)
                    with gr.Row():
                        img_input1 = gr.Image(scale=1, label="输入图像", type="pil", image_mode="RGB", height=540, interactive=True)
                        img_output1 = gr.Image(scale=1, label="生成结果", type="pil", image_mode="RGB", height=540, visible=True)  
                    with gr.Row():                                       
                        user_input1 = gr.Textbox(show_label=False, placeholder="请在此输入文字...", lines=5, container=False, scale=5,interactive=True, visible=False)
                        audio_input1 = gr.Audio(source="microphone", label="输入音频", type="filepath", interactive=True)
                    with gr.Row():
                        submitBtn1 = gr.Button("提交", variant="primary",interactive=True)
                        # emptyBtn1 = gr.Button("清除",interactive=True)
        # with gr.Tab("Controlnet"):
        #     mode2 = gr.Textbox(value=2, label="模式", visible=False)
        #     with gr.Row():
        #         with gr.Column(scale=2):
        #             sr_device2 = gr.Textbox(value="CPU",label="推理设备", visible=True)
        #             whisper_model2 = gr.Textbox(label="语音识别模型", value="whisper-medium")
        #             with gr.Column(scale=2):
        #                 sd_device2 = gr.Textbox(value="iGPU",label="推理设备", visible=True)                 
        #                 sd_model_name2 = gr.Textbox(label="图像生成模型", value="Stable-Dififusion-v1.5")
        #                 controlnet_name2 = gr.Textbox(label="Controlnet模型", value="lllyasviel--sd-controlnet-canny")
        #                 height2 = gr.Slider(64, 1024, value=512, step=1, label="高度", interactive=True)
        #                 width2 = gr.Slider(64, 1024, value=512, step=1, label="宽度", interactive=True)
        #                 steps2 = gr.Slider(1, 100, value=20, step=1, label="生成步数", interactive=True)
        #                 # seed2 = gr.Slider(0, 2048, value=outlen, step=1.0, label="生成种子", interactive=True)
        #             with gr.Row():                   
        #                 sr_latency2 = gr.Textbox(label="语音识别耗时（s）", visible=True)
        #                 sd_latency2 = gr.Textbox(label="图像生成耗时（s）", visible=True)
        #         with gr.Column(scale=8):
        #             prompt_img2 = gr.Textbox(label = "提示词", show_label=True, visible=True)
        #             # promptTranslated = gr.Textbox(label = "English", show_label=True, visible=True)
        #             with gr.Row():
        #                 img_input2 = gr.Image(scale=1, label="输入图像", type="pil", image_mode="RGB", height=540, interactive=True)
        #                 img_output2 = gr.Image(scale=1, label="生成结果", type="pil", image_mode="RGB", height=540, visible=True)  
        #             with gr.Row():                                         
        #                 user_input2 = gr.Textbox(show_label=False, placeholder="请在此输入文字...", lines=5, container=False, scale=5,interactive=True, visible=False)
        #                 audio_input2 = gr.Audio(source="microphone", label="输入音频", type="filepath", interactive=True)
        #             with gr.Row():
        #                 submitBtn2 = gr.Button("提交", variant="primary",interactive=True)
        #                 # emptyBtn2 = gr.Button("清除",interactive=True)
        # with gr.Tab("Inpainting"):
        #     mode3 = gr.Textbox(value=3, label="模式", visible=False)
        #     with gr.Row():
        #         with gr.Column(scale=2):
        #             sr_device3 = gr.Textbox(value="CPU",label="推理设备", visible=True)
        #             whisper_model3 = gr.Textbox(label="语音识别模型", value="whisper-medium")
        #             with gr.Column(scale=2):
        #                 sd_device3 = gr.Textbox(value="iGPU",label="推理设备", visible=True)                 
        #                 sd_model_name3 = gr.Textbox(label="图像生成模型", value="Stable-Dififusion-v1.5")
        #                 height3 = gr.Slider(64, 1024, value=512, step=1, label="高度", interactive=True)
        #                 width3 = gr.Slider(64, 1024, value=512, step=1, label="宽度", interactive=True)
        #                 steps3 = gr.Slider(1, 100, value=20, step=1, label="生成步数", interactive=True)
        #                 # seed3 = gr.Slider(0, 2048, value=outlen, step=1.0, label="生成种子", interactive=True)
        #             with gr.Row():                   
        #                 sr_latency3 = gr.Textbox(label="语音识别耗时（s）", visible=True)
        #                 sd_latency3 = gr.Textbox(label="图像生成耗时（s）", visible=True)
        #         with gr.Column(scale=8):
        #             prompt_img3 = gr.Textbox(label = "提示词", show_label=True, visible=True)
        #             # promptTranslated = gr.Textbox(label = "English", show_label=True, visible=True)
        #             with gr.Row():
        #                 img_input3 = gr.Image(scale=1, label="输入图像", type="pil", tool="sketch", image_mode="RGB", height=540, interactive=True)
        #                 img_output3 = gr.Image(scale=1, label="生成结果", type="pil", image_mode="RGB", height=540, visible=True)  
        #             with gr.Row():                                        
        #                 user_input3 = gr.Textbox(show_label=False, placeholder="请在此输入文字...", lines=5, container=False, scale=5,interactive=True, visible=False)
        #                 audio_input3 = gr.Audio(source="microphone", label="输入音频", type="filepath", interactive=True)
        #             with gr.Row():
        #                 submitBtn3 = gr.Button("提交", variant="primary",interactive=True)
                        # emptyBtn3 = gr.Button("清除",interactive=True)
        # gr.Examples( [ "阳光沙滩",
        #                 "汽车，下雪的森林",
        #                 "高楼，夜晚，大厦",
        #                 "长河落日圆",
        #                 "大漠孤烟直",
        #                 "film still of a boy, smile, blond hair, photograph, highly detailed face, depth of field, moody light, golden hour, extremely detailed, Nikon D850, award winning photography",
        #                 "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        #                 "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        #                 "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece"],
        #                 user_input)

               


        # Initialize history and past_key_values for generator
        # history = gr.State([])


        # Action for submit/empty button
        submitBtn.click(predict, [mode, audio_input, user_input, param_template, height, width, steps],
                        [prompt_img, img_output, sr_latency, sd_latency], show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])
        # audio_input.stop_recording(predict, [mode, audio_input, user_input, height, width, steps],
        #                 [img_output, sr_latency, sd_latency], show_progress=True)

        submitBtn1.click(predict, [mode1, audio_input1, user_input1, param_template, height1, width1, steps1, img_input1, strength],
                        [prompt_img1, img_output1, sr_latency1, sd_latency1], show_progress=True)
        submitBtn1.click(reset_user_input, [], [user_input1])
        # audio_input1.stop_recording(predict, [mode1, audio_input1, user_input1, height1, width1, steps1, img_input1],
        #                 [img_output1, sr_latency1, sd_latency1], show_progress=True)

        # submitBtn2.click(predict, [mode2, audio_input2, user_input2, height2, width2, steps2, img_input2],
        #                 [img_output2, sr_latency2, sd_latency2], show_progress=True)
        # submitBtn2.click(reset_user_input, [], [user_input2])
        # # audio_input2.stop_recording(predict, [mode2, audio_input2, user_input2, height2, width2, steps2, img_input2],
        # #                 [img_output2, sr_latency2, sd_latency2], show_progress=True)

        # submitBtn3.click(predict, [mode3, audio_input3, user_input3, height3, width3, steps3, img_input3],
        #                 [img_output3, sr_latency3, sd_latency3], show_progress=True)
        # submitBtn3.click(reset_user_input, [], [user_input3])
        # audio_input3.stop_recording(predict, [mode3, audio_input3, user_input3, height3, width3, steps3, img_input3],
        #                 [img_output3, sr_latency3, sd_latency3], show_progress=True)

        # emptyBtn.click(reset_state, outputs=[chatbot, history, f_latency, a_latency], show_progress=True)

    # Launch the web app
    demo.queue().launch(share=False, inbrowser=True, server_port=7866)
