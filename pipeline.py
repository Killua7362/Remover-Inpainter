import numpy as np
import cv2
from transformers import SamConfig,SamProcessor,SamModel
import torch
from PIL import Image
from diffusers import  ControlNetModel, UniPCMultistepScheduler
from ultralytics import YOLO

import sys 
sys.path.append('./downloaded_models/ControlNetInpaint')
from src.pipeline_stable_diffusion_controlnet_inpaint import *


def main():
    image_path = ""
    input_image = Image.open(image_path)

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
    model = SamModel(config = model_config)
    model.load_state_dict(torch.load('saved_models/SAM/mito_model_checkpoint.pth'))
    model.to("cuda")

    image = np.array(input_image)
    model = YOLO('./results/50_epochs-/weights/last.pt')
    res = model.predict(image_path,conf=0.5)
    bounding_box = res[0]
    prompt = [bounding_box[i] for i in range(4)]
    
    inputs = processor(input_image,input_boxes=[[prompt]],return_tensors='pt')
    inputs = {k:v.to('cuda') for k,v in inputs.items()}
    model.eval()

    with torch.no_grad():
        outputs = model(**inputs,multimask_output=False)

    out = torch.sigmoid(outputs.pred_masks.squeeze(1)).cpu().numpy().squeeze()
    out = np.where(out > 0.999,1,0)
    out = np.stack((out,)*3, axis=-1)

    mask_image = cv2.normalize(out,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    input_image = input_image.resize((256,256),Image.NEAREST) # this is not a array

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision='fp16',
        controlnet = controlnet,
        torch_dtype=torch.float16,
    )

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.to('cuda')

    text_prompt="just fill up the space so that it matches the surroundings"

    canny_image = cv2.Canny(input_image, 100, 200)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)

    mask_image=Image.fromarray(mask_image)
    canny_image = Image.fromarray(canny_image)

    generator = torch.manual_seed(0)

    new_image = pipeline(
        text_prompt,
        num_inference_steps=100,
        generator=generator,
        image=input_image,
        control_image=canny_image,
        controlnet_conditioning_scale = 1,
        mask_image=mask_image
    ).images[0]

if __name__ == '__main__':
    main()