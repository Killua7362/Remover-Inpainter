from ultralytics import YOLO
model = YOLO('yolov8n-seg.yaml')  
model = YOLO('yolov8n-seg.pt')  

import yaml
with open("path to dataset yaml file", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

project = "./results"
name = '10_epochs'

result = model.train(
    data = "path to data yaml",
    project = project,
    name = name,
    epochs = 10,
    patience = 0,
    batch = 4,
    imgsz=512
)
#inference 
model = YOLO('./results/50_epochs-/weights/last.pt')
res = model.predict("",conf=0.5)
bounding_box = res[0]
bounding_box = [bounding_box[i] for i in range(4)]