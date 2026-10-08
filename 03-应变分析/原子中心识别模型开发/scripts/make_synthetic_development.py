"""Make isolated synthetic annotations; no real performance claim or production registration."""
from pathlib import Path
import argparse
import numpy as np
import tifffile
from atom_center.annotations import AnnotationProject
from atom_center.storage import write_json

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--groups", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    projects = []
    yy, xx = np.mgrid[:args.size, :args.size]
    for index in range(args.groups):
        folder = root/f"batch-{index}"
        images = folder/"images"
        images.mkdir(parents=True)
        rng = np.random.default_rng(200+index)
        points = [(float(x+rng.uniform(-.3, .3)), float(y+rng.uniform(-.3, .3)))
                  for y in np.arange(12.5, args.size-10, 20.)
                  for x in np.arange(12.25, args.size-10, 20.)]
        if index == args.groups-1:
            points = []  # Reviewed background is intentionally preserved.
        raw = 1500.+rng.normal(0, 6, xx.shape)
        for x, y in points:
            raw += rng.uniform(4000, 6000)*np.exp(-((xx-x)**2+(yy-y)**2)/(2*1.2**2))
        tifffile.imwrite(images/"field.tif", np.rint(raw).astype(np.uint16), photometric="minisblack")
        project = AnnotationProject.create(folder/"annotation_project.json", image_root=images, modality="haadf_stem")
        document = project.load_document(project.records[0])
        document.points_xy = points
        document.coverage_regions_xyxy = [(0., 0., float(args.size), float(args.size))]
        document.metadata.update(sample_id="synthetic-only", acquisition_id=f"simulation-{index}",
                                 data_origin="synthetic")
        document.review_status = "reviewed"
        project.save_document(document)
        projects.append(str(project.project_path))
    write_json(root/"synthetic_projects.json", {"purpose": "software_smoke_only", "projects": projects})
    print(root/"synthetic_projects.json")


if __name__ == "__main__":
    main()
