import os
import yaml
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog,DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo

def main():
    register_coco_instances("train_dataset",{},"dataset/coco/train/coco_annotations.json","dataset/coco/train")

    train_metadata = MetadataCatalog.get("train_dataset")
    train_dataset_dict = DatasetCatalog.get("train_dataset")

    cfg = get_cfg()
    cfg.OUTPUT_DIR = "saved_models/detectron2"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 1000  
    cfg.SOLVER.STEPS = []        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False) 
    trainer.train()

    config_path = 'saved_models/detectron2/config.yaml'

    with open(config_path,'w') as file:
        yaml.dump(cfg,file)

if __name__ == '__main__':
    main()