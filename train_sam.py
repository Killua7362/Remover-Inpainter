#import
import glob
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
from transformers import SamProcessor,SamModel
import monai
from datasets import Dataset as dictToDataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import statistics
from PIL import Image
from torch.optim import Adam

from utils.sam_utils import DataSetLoader,get_mask


def main():
    images_data_path = "dataset/images"
    mask_data_path = "dataset/labels_colored"

    images_path = []
    mask_path = []

    for path in glob.glob(images_data_path + '/*'):
        images_path.append(path)
        
    for path in glob.glob(mask_data_path + '/*'):
        mask_path.append(path)
        
    images_path.sort()
    mask_path.sort()

    images = [Image.open(path) for path in images_path ]
    masks = [get_mask(path) for path in mask_path]

    filtered_indices = {i for i,v in enumerate(np.array(masks)) if v.max() != 0}
    images = [img for i,img in enumerate(images) if i in filtered_indices] 
    masks = [img for i,img in enumerate(masks) if i in filtered_indices] 

    dataset_dict = {
        "image":images,
        "label":masks
    }

    dataset = dictToDataset.from_dict(dataset_dict)

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_dataset = DataSetLoader(dataset=dataset,processor=processor)
    train_dataloader = DataLoader(train_dataset,batch_size=2,shuffle=True,drop_last=False)

    model = SamModel.from_pretrained("facebook/sam-vit-base")

    for name,param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
            

    optimizer = Adam(model.parameters(),lr=1e-6,weight_decay=0.001)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    epochs = 1
    model.to("cuda")
    model.train()
    scaler = GradScaler()

    for epoch in range(epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
            with autocast():
                outputs = model(pixel_values=batch["pixel_values"].to("cuda"),
                                input_boxes=batch["input_boxes"].to("cuda"),
                                multimask_output=False)
                
                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to("cuda")
                loss = seg_loss(predicted_masks,ground_truth_masks.unsqueeze(1))
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_losses.append(loss.detach().item())
            
        
        print(f"For Epoch: {epoch} Epoch loss is: {statistics.mean(epoch_losses)}")


#torch.save(model.state_dict(), "saved_models/SAM/mito_model_checkpoint.pth")

if __name__ == '__main__':
    main()