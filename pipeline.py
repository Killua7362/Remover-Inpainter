import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
import os
from transformers import SamConfig,SamProcessor,SamModel
import torch
import random
from PIL import Image
from utils.sam_utils import get_bounding_box
from diffusers import  ControlNetModel, UniPCMultistepScheduler

import sys 
sys.path.append('./downloaded_models/ControlNetInpaint')
from src.pipeline_stable_diffusion_controlnet_inpaint import *


def main():
    cfg = get_cfg()
    cfg.OUTPUT_DIR = "saved_models/detectron2"
    cfg.set_new_allowed(True)
    cfg.merge_from_file(os.path.join(cfg.OUTPUT_DIR,"config.yaml"))
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,"model_final.pth")
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 1000  
    cfg.SOLVER.STEPS = []        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  

    predictor = DefaultPredictor(cfg)

    image_path = ""
    input_image = Image.open(image_path)


    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
    model = SamModel(config = model_config)
    model.load_state_dict(torch.load('saved_models/SAM/mito_model_checkpoint.pth'))
    model.to("cuda")

    image = np.array(input_image)
    outputs = predictor(image)
    filter = [i for i,score in enumerate(outputs['instances'].scores) if score > 0.05]
    outputs['instances'] = outputs['instances'][filter]
    ground_truth_mask = np.zeros_like(outputs['instances'].pred_masks[0].cpu().numpy()).astype('uint8')

    for pred_mask in outputs['instances'].pred_masks:
        mask = pred_mask.cpu().numpy().astype('uint8') * 255
        ground_truth_mask += mask


    prompt = get_bounding_box(ground_truth_mask)
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