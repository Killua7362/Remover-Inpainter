import os
import glob
from sklearn.model_selection import train_test_split
import cv2
import shutil
import numpy as np
import pandas as pd
import json

def get_image_mask(images_data,masks_data):
    images_path = []
    masks_path = []
    
    for path in glob.glob(images_data + '/*'):
        images_path.append(path)
        
    for path in glob.glob(masks_data+ '/*'):
        masks_path.append(path)
        
    return images_path,masks_path
    
def get_polygons(mask):
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    
    for cont in contours:
        if len(cont) > 2:
            poly = cont.reshape(-1).tolist()
            if len(poly) > 4:
                polygons.append(poly)
    return polygons

def process_data(images_path,masks_path,out_dir,category_map):
    annotations = []
    images = []
    image_id = 0
    ann_id = 0

    for img_path,m_path in zip(images_path,masks_path):
        print(image_id)
        image_id+=1
        img = cv2.imread(img_path)
        mask = cv2.imread(m_path,cv2.IMREAD_UNCHANGED)
        
        shutil.copy(img_path,os.path.join(out_dir,os.path.basename(img_path)))
        
        images.append({
            "id":image_id,
            "file_name":os.path.basename(img_path),
            "height":img.shape[0],
            "width":img.shape[1]
        })
        
        unique_values = np.unique(mask.reshape(-1,mask.shape[2]),axis=0)
        
        for value in unique_values:
            if not value.astype(bool).all() or category_map[tuple(value)][0] != 'foreground':
                continue
            object_mask = (mask == value).astype(np.uint8) * 255
            
            
            polygons = get_polygons(object_mask)
            for poly in polygons:
                ann_id+=1
                annotations.append({
                    "id":ann_id,
                    "image_id":image_id,
                    "category_id":1,
                    "segmentation":[poly],
                    "area":cv2.contourArea(np.array(poly).reshape(-1,2)),
                    "bbox":list(cv2.boundingRect(np.array(poly).reshape(-1,2))),
                    "iscrowd":0
                })
                
        coco_output = {
            "images":images,
            "annotations":annotations,
            "categories": [{"id":1,"name":"foreground"}]
        }
        
        with open(os.path.join(out_dir, 'coco_annotations.json'), 'w') as f:
            json.dump(coco_output, f)
        
def main():
    images_data_path = "dataset/images"
    mask_data_path = "dataset/labels_colored"
    labels_path = "dataset/labels_class_dict.csv"
    
    category_map = {}
    labels = pd.read_csv(labels_path).to_dict()
    
    i = 1
    for category,r,g,b in zip(labels['class_names'].values(),labels['r'].values(),labels['g'].values(),labels['b'].values()):
        category_map[(b,g,r)] = [category,i]
        i+=1
        
    out_dir= "dataset/coco"
    
    train_dir = os.path.join(out_dir,'train')
    test_dir= os.path.join(out_dir,'test')
    val_x_dir = os.path.join(out_dir,'val_x')
    val_y_dir = os.path.join(out_dir,'val_y')
    
    os.makedirs(train_dir,exist_ok=True)
    os.makedirs(test_dir,exist_ok=True)
    os.makedirs(val_x_dir,exist_ok=True)
    os.makedirs(val_y_dir,exist_ok=True)
    
    image_paths,mask_paths = get_image_mask(images_data_path,mask_data_path)
    
    train_image_paths,test_image_paths,train_mask_paths,test_mask_paths = train_test_split(image_paths,mask_paths,test_size=0.2,random_state=42)
    train_image_paths,val_image_paths,train_mask_paths,val_mask_paths = train_test_split(train_image_paths,train_mask_paths,test_size=0.1,random_state=42)
    process_data(train_image_paths,train_mask_paths,train_dir,category_map)
    process_data(test_image_paths,test_mask_paths,test_dir,category_map)
    
    for img,mask in zip( val_image_paths,val_mask_paths ):
        shutil.copy(img,os.path.join(val_x_dir,os.path.basename(img)))
        shutil.copy(mask,os.path.join(val_y_dir,os.path.basename(mask)))

    
if __name__ == '__main__':
    main()    