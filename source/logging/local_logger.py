import os
from typing import Dict

import jsonlines
import numpy as np
from PIL import Image


class LocalLogger:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.metrics_file = os.path.join(folder_path, "metrics.jsonl")
        self.images_folder = os.path.join(folder_path, "images")
        os.makedirs(self.folder_path, exist_ok=True)

    def log_metrics(self, metrics: Dict[str, float], step: int):
        metrics = dict(metrics)
        metrics["step"] = step
        with jsonlines.open(self.metrics_file, mode="a") as writer:
            writer.write(metrics)

    def log_images(self, images: Dict[str, np.ndarray], step: int):
        step_folder = os.path.join(self.images_folder, f"{step:05d}")
        os.makedirs(step_folder, exist_ok=True)
        for image_name, image in list(images.items()):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
                image.save(os.path.join(step_folder, f"{image_name}.JPEG"))
