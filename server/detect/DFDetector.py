import os
import numpy as np
import cv2

import onnxruntime

from server.detect.timit_dataset import cut_roi


class DFDetector:
    def __init__(self, model_file=os.path.dirname(__file__)+'/data/detect_model.onnx'):
        print("detector loading...")
        self.session = onnxruntime.InferenceSession(model_file, providers=['CUDAExecutionProvider'], provider_options=[{'device_id': 1}])

    def detect(self, img, rect):
        img = cut_roi(img, rect)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        img = np.array([np.transpose(img, (2, 0, 1))], dtype=np.float32)

        inputs = {self.session.get_inputs()[0].name: img}
        output = self.session.run(['output'], inputs)[0][0]

        result = float(output[1] / (output[0] + output[1]) - 0.31) / 0.27
        return result
