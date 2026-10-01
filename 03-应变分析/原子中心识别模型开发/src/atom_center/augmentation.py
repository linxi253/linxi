"""Training-only grayscale transform, driven by the trainer's seeded Python RNG."""
import random
import numpy as np


class GrayscaleAugment:
    def __init__(self, brightness=.15, contrast=.15, probability=.5):
        self.brightness, self.contrast, self.probability = brightness, contrast, probability

    def __call__(self, labels):
        if random.random() >= self.probability:
            return labels
        image = labels["img"]
        gray = image[..., 0] if image.ndim == 3 else image
        gain = 1+random.uniform(-self.contrast, self.contrast)
        offset = random.uniform(-self.brightness, self.brightness)*255
        result = np.clip((gray.astype(np.float32)-127.5)*gain+127.5+offset, 0, 255).astype(np.uint8)
        labels["img"] = np.repeat(result[..., None], image.shape[2], axis=2) if image.ndim == 3 else result
        return labels
