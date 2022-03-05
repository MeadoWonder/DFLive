import torch


class DFDetector:
    def __init__(self, model_file='./data/resnet50-finetuned.pth'):
        self.model = torch.load(model_file)

    def detect(self, img):
        self.model()
