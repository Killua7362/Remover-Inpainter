from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image

class DataSetLoader(Dataset):
    def __init__(self,dataset,processor):
        self.dataset = dataset
        self.processor = processor
    
    def __len__(self):
        return len(self.dataset["image"])

    def __getitem__(self,idx):
        item = self.dataset[idx]
        
        img = item["image"]
        
        mask = np.array(item["label"]) 
        prompt = get_bounding_box(mask)
        
        inputs = self.processor(img,input_boxes=[[prompt]],return_tensors='pt')
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        inputs["ground_truth_mask"] = mask
        
        return inputs
    
def get_mask(mask_path):
    image = Image.open(mask_path)
    image = np.array(image)
    remove = np.all(image == (180,222,44),axis=-1)
                
    image[remove] = [255,255,255]
    image[~remove] = [0,0,0]
    
    image = cv2.resize(image,(256,256),interpolation=cv2.INTER_NEAREST)    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    return Image.fromarray(image)

def get_bounding_box(ground_truth_mask):
    y_indices, x_indices = np.where(ground_truth_mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_mask.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox