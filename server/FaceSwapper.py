import os
import time
import base64
import heapq
from pathlib import Path
from multiprocessing import Queue, Process

import numpy as np
import cv2


ROOT = os.path.dirname(__file__)

extractor_gpu_idxs = [0]
merger_gpu_idxs = [1]
dfm_model_file = 'models/Kim_Jarrey.dfm'


class Frame:
    def __init__(self, idx, img):  # idx: 帧序号，小的排在前面
        self.idx = idx
        self.img = img

    def __lt__(self, other):
        return self.idx < other.idx


class FaceSwapper:
    def __init__(self):
        self.input_q = Queue(15)  # extractor输入队列
        self.mid_q = Queue(15)  # extractor和merger之间的队列
        self.output_q = Queue(15)  # merger输出队列
        self.heap = []  # 优先队列，多进程输出时对帧排序
        self.cnt = 0

        self.workers = []
        for gpu_idx in extractor_gpu_idxs:
            self.workers.append(ExtractWorker(self.input_q, self.mid_q, gpu_idx))
        for gpu_idx in merger_gpu_idxs:
            self.workers.append(MergeWorker(self.mid_q, self.output_q, gpu_idx))

        for w in self.workers:
            w.start()

        self.last_frame = np.zeros((320, 480, 3), np.uint8)  # 记录前一帧，掉帧时发送

    def swap(self, img_base64):
        self.cnt += 1
        header, frame = img_base64.split(',', 1)
        frame = base64.b64decode(frame)
        frame = np.frombuffer(frame, np.uint8)
        frame = Frame(self.cnt, cv2.imdecode(frame, cv2.IMREAD_COLOR))

        while self.input_q.full():
            self.input_q.get()
        self.input_q.put(frame)

        while not self.output_q.empty():
            heapq.heappush(self.heap, self.output_q.get())

        if len(self.heap) > 2:
            frame = heapq.heappop(self.heap).img
            self.last_frame = frame
        else:
            frame = self.last_frame

        frame = 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('ascii')
        return frame

    def terminate(self):
        for w in self.workers:
            w.kill()

        for w in self.workers:
            w.join()


class ExtractWorker(Process):
    def __init__(self, input_q, output_q, gpu_idx):
        Process.__init__(self)
        self.input_q = input_q
        self.output_q = output_q
        self.gpu_idx = gpu_idx

    def run(self):
        print("extractor loading...")

        from DeepFaceLab.core.leras import nn
        nn.initialize_main_env()

        from DeepFaceLab.mainscripts.Extractor import ExtractSubprocessor
        from xlib.onnxruntime.device import get_available_devices_info
        from server.YoloV5Face import YoloV5Face

        data = ExtractSubprocessor.Data()

        device_info = get_available_devices_info(include_cpu=False)[self.gpu_idx]
        print('rects_extractor on: ' + str(device_info))
        rects_extractor = YoloV5Face(device_info)  # 人脸检测（输出方框）

        landmarks_extractor = cv2.face.createFacemarkLBF()  # 人脸特征点提取
        landmarks_extractor.loadModel(ROOT + "/models/lbfmodel.yaml")

        print("extractor loaded")

        while True:
            frame = self.input_q.get()

            data = ExtractSubprocessor.Cli.rects_stage(data, frame.img, 1, rects_extractor)
            data.rects = [(l, t, r-l, b-t) for (l, t, r, b) in data.rects[0]]

            if len(data.rects) == 0:  # 无人脸，直接返回
                data.landmarks = [None]
            else:
                _, data.landmarks = landmarks_extractor.fit(frame.img, np.array(data.rects))

            while self.output_q.full():
                self.output_q.get()
            self.output_q.put((frame, data.landmarks[0]))


class MergeWorker(Process):
    def __init__(self, input_q, output_q, gpu_idx):
        Process.__init__(self)
        self.input_q = input_q
        self.output_q = output_q
        self.gpu_idx = gpu_idx

    def run(self):
        print("merger loading...")

        from DeepFaceLab.core.leras import nn
        nn.initialize_main_env()

        from DeepFaceLab.merger import MergerConfigMasked
        from xlib.onnxruntime.device import get_available_devices_info
        from server.MergeMasked2 import MergeMasked2
        from server.DFMModel import DFMModel

        # 选择模型、设备
        saved_models_path = Path(os.path.join(ROOT, dfm_model_file))
        device_info = get_available_devices_info(include_cpu=False)[self.gpu_idx]
        print('merger on: ' + str(device_info))

        model = DFMModel(saved_models_path, device_info)
        predictor_func = model.convert

        # 调节合成参数
        predictor_input_shape = (256, 256, 3)
        cfg = MergerConfigMasked(face_type=4)
        cfg.add_blur_mask_modifier(30)

        print("merger loaded")

        while True:
            print("mid_queue: " + str(self.input_q.qsize()))
            frame, landmarks = self.input_q.get()

            if landmarks is not None:
                time0 = time.time() * 1000
                frame.img = MergeMasked2(predictor_func,
                                         predictor_input_shape,
                                         face_enhancer_func=None,
                                         xseg_256_extract_func=None,
                                         cfg=cfg,
                                         img_bgr_uint8=frame.img,
                                         landmarks_list=landmarks)
                print("merge: " + str(time.time()*1000 - time0))

            while self.output_q.full():
                self.output_q.get()
            self.output_q.put(frame)
