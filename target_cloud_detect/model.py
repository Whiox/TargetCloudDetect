import math

import numpy as np
import tensorflow as tf

from keras.models import load_model

from PIL import Image, ImageDraw
from io import BytesIO

from importlib.resources import files


class Model:
    def __init__(self, model_path: str = None) -> None:
        if model_path is None:
            model_path = (files("target_cloud_detect") / "model" / "cloud_target_detect.keras")

        self.model = load_model(
            model_path,
            custom_objects={"resize_to_fixed_shape": self.resize_to_fixed_shape},
            compile=False
        )

    def resize_to_fixed_shape(self, x) -> tf.Tensor:
        return tf.image.resize(x, (7, 7))

    @staticmethod
    def draw_thought_cloud(image, x, y, alpha=1.0) -> Image.Image:
        image = image.convert("RGBA")
        img_w, img_h = image.size

        bx = img_w / 2
        rx = img_w * 0.72

        oval_top = -img_h * 0.10
        margin = max(img_h * 0.10, 18)
        bottom = min(img_h * 0.24, y - margin)
        bottom = max(bottom, img_h * 0.06)

        cy = (oval_top + bottom) / 2
        ry = (bottom - oval_top) / 2

        edge_x = float(img_w) if x < img_w / 2 else 0.0

        kx = (edge_x - bx) / rx
        y_bot = cy + ry * math.sqrt(max(0.0, 1.0 - kx * kx))
        y_top = y_bot - ry

        base_y = (y_top + y_bot) / 2
        ux = x - edge_x
        uy = y - base_y
        length = math.hypot(ux, uy) + 1e-6

        tip_x = x - ux / length * 12
        tip_y = y - uy / length * 12

        mask = Image.new("L", image.size, int(255 * alpha))
        draw = ImageDraw.Draw(mask)

        draw.polygon(
            [(edge_x, y_top), (tip_x, tip_y), (edge_x, y_bot)],
            fill=0,
        )

        draw.ellipse(
            (bx - rx, cy - ry, bx + rx, cy + ry),
            fill=0,
        )

        image.putalpha(mask)
        return image

    def predict_x_y_conf(self, image) -> tuple[int, int, float]:
        image = image.convert("RGB")
        w, h = image.size
        resized = np.array(image.resize((224, 224))).astype(np.float32)
        predict = self.model.predict(resized[None, ...], verbose=0)
        fm_size = predict.shape[1]
        objectless = predict[0, :, :, -1]
        fy, fx = np.unravel_index(np.argmax(objectless), objectless.shape)
        dx = float(np.clip(predict[0, fy, fx, 0], 0.0, 1.0))
        dy = float(np.clip(predict[0, fy, fx, 1], 0.0, 1.0))
        x_norm = (fx + dx) / fm_size
        y_norm = (fy + dy) / fm_size
        conf = objectless[fy, fx]
        x = x_norm * w
        y = y_norm * h
        return int(x), int(y), float(conf)

    def predict_from_bytes(self, file_bytes) -> Image.Image:
        byte_stream = BytesIO(file_bytes)
        image = Image.open(byte_stream)
        x, y, conf = self.predict_x_y_conf(image)
        result = self.draw_thought_cloud(image, x=x, y=y)
        return result

    def predict_from_file(self, file) -> Image.Image:
        image = Image.open(file)
        x, y, conf = self.predict_x_y_conf(image)
        result = self.draw_thought_cloud(image, x=x, y=y)
        return result
