from ultralytics import YOLO
import torch
torch.cuda.empty_cache()
# import torch
# print(torch.__version__)

def train_model():
    model = YOLO(r'yolov8s.pt')
    model.train(data=r'/root/home/k.papkov/train_ds/smokers.yaml', cfg=r'/root/home/k.papkov/train_ds/cfg.yaml')

if __name__ == '__main__':
    train_model()