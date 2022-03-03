from pathlib import Path
from typing import Tuple

import numpy as np
from xlib import onnxruntime as lib_ort
from xlib.image import ImageProcessor
from xlib.onnxruntime.device import ORTDeviceInfo


class DFMModel:
    def __init__(self, model_path: Path, device: ORTDeviceInfo = None):
        if device is None:
            device = lib_ort.get_cpu_device_info()
        self._model_path = model_path

        sess = self._sess = lib_ort.InferenceSession_with_device(str(model_path), device)

        inputs = sess.get_inputs()

        if len(inputs) == 0:
            raise Exception(f'Invalid model {model_path}')
        else:
            if 'in_face' not in inputs[0].name:
                raise Exception(f'Invalid model {model_path}')
            else:
                self._input_height, self._input_width = inputs[0].shape[1:3]
                self._model_type = 1
                if len(inputs) == 2:
                    if 'morph_value' not in inputs[1].name:
                        raise Exception(f'Invalid model {model_path}')
                    self._model_type = 2
                elif len(inputs) > 2:
                    raise Exception(f'Invalid model {model_path}')

    def get_model_path(self) -> Path:
        return self._model_path

    def get_input_res(self) -> Tuple[int, int]:
        return self._input_width, self._input_height

    def has_morph_value(self) -> bool:
        return self._model_type == 2

    def convert(self, img, morph_factor=0.75):
        """
         img    np.ndarray  HW,HWC,NHWC uint8,float32

         morph_factor   float   used if model supports it

        returns

         img        NHW3  same dtype as img
         celeb_mask NHW1  same dtype as img
         face_mask  NHW1  same dtype as img
        """

        ip = ImageProcessor(img)

        N, H, W, C = ip.get_dims()
        dtype = ip.get_dtype()

        img = ip.resize( (self._input_width,self._input_height) ).ch(3).to_ufloat32().get_image('NHWC')

        if self._model_type == 1:
            out_face_mask, out_celeb, out_celeb_mask = self._sess.run(None, {'in_face:0': img})
        elif self._model_type == 2:
            out_face_mask, out_celeb, out_celeb_mask = self._sess.run(None, {'in_face:0': img, 'morph_value:0': np.float32([morph_factor]) })

        out_celeb      = ImageProcessor(out_celeb).resize((W, H)).ch(3).to_dtype(dtype).get_image('NHWC')
        out_celeb_mask = ImageProcessor(out_celeb_mask).resize((W, H)).ch(1).to_dtype(dtype).get_image('NHWC')
        out_face_mask  = ImageProcessor(out_face_mask).resize((W, H)).ch(1).to_dtype(dtype).get_image('NHWC')

        return out_celeb[0], out_celeb_mask[0], out_face_mask[0]
