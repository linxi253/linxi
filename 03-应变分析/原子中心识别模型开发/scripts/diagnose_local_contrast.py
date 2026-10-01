"""Train/validation-only contrast ablation; raw refinement input is preserved."""
from pathlib import Path
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from atom_center.backends import onnx_pipeline
from atom_center.pipeline import DetectionPipeline
from atom_center.image_io import normalize_percentile
from atom_center.evaluation import evaluate_dataset
from atom_center.storage import write_json


class ContrastBackend:
    def __init__(self, base, mode):
        self.base, self.mode = base, mode
        self.name, self.model_sha256 = base.name, base.model_sha256

    def predict(self, image):
        if self.mode == 'background32':
            transformed = image - gaussian_filter(image, 32.)
        else:
            norm = np.rint(normalize_percentile(image)*255).astype('uint8')
            enhanced = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(norm)
            transformed = (norm.astype(float)+enhanced)/2.
        result = self.base.predict(transformed)
        self.last_diagnostics = dict(getattr(self.base, 'last_diagnostics', {}))
        self.last_diagnostics['contrast_ablation'] = self.mode
        return result


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'runs/generalization-20260910/local_contrast'
    out.mkdir(exist_ok=False)
    base = onnx_pipeline(root/'runs/generalization-20260910/adaptive_blob-onnx/model_manifest.json')
    rows = []
    for mode in ('background32', 'clahe_blend'):
        pipe = DetectionPipeline(ContrastBackend(base.backend, mode), base.config)
        row = {'mode': mode}
        for split in ('train', 'val'):
            r = evaluate_dataset(pipe, root/'data/processed/real_workflow_20260909_v2', split=split, max_distance_px=2.)
            write_json(out/f'{mode}_{split}.json', r)
            row[split] = {k:r[k] for k in ('tp','fp','fn','precision','recall','f1','rmse_px')}
        rows.append(row)
        print(row, flush=True)
    write_json(out/'comparison.json', {'test_used':False, 'rows':rows})


if __name__ == '__main__':
    main()
