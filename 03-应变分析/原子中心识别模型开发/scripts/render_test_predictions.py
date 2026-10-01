"""Run a fixed ONNX bundle on reserved test images and render measured centers."""
from pathlib import Path
import argparse
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from atom_center.backends import onnx_pipeline
from atom_center.image_io import load_image, normalize_percentile
from atom_center.source_identity import verify_source
from atom_center.storage import read_json, write_json, new_artifact_directory


def text_lines(draw, text, xy, font, fill, max_width):
    x, y = xy
    line = ""
    for char in text:
        if line and draw.textlength(line + char, font=font) > max_width:
            draw.text((x, y), line, font=font, fill=fill)
            line, y = char, y + font.size + 6
        else:
            line += char
    draw.text((x, y), line, font=font, fill=fill)
    return y + font.size + 6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--manifest", type=Path)
    model_source.add_argument("--run", type=Path, help="Preview the fixed best_points checkpoint with PyTorch")
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", default="固定候选模型")
    parser.add_argument("--title", default="测试集 · 固定候选模型预测")
    args = parser.parse_args()
    reservation = read_json(args.reservation)
    if args.manifest:
        pipeline = onnx_pipeline(args.manifest)
        refinement = dict(pipeline.backend.manifest.refinement)
    else:
        from atom_center.training import checkpoint_for_run
        from atom_center.backends import TorchBackend
        from atom_center.pipeline import DetectionPipeline
        from atom_center.configuration import pipeline_config
        manifest, checkpoint, digest = checkpoint_for_run(args.run, checkpoint='best_points.pt')
        backend = TorchBackend(checkpoint,expected_sha256=digest,contract=manifest['contract'],inference=manifest['config']['inference'],device='cuda:0')
        pipeline = DetectionPipeline(backend,pipeline_config(manifest['config']))
        refinement = dict(manifest['config']['refinement'])
    font_path = "C:/Windows/Fonts/msyh.ttc"
    font = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 19)
    conf = pipeline.backend.inference["conf"]
    raw_forward = pipeline.backend.raw
    tile_scores = []

    def measured_raw(tensor):
        output = raw_forward(tensor)
        tile_scores.append(float(np.max(output[:, 4])))
        return output

    pipeline.backend.raw = measured_raw
    rows = []
    with new_artifact_directory(args.output.resolve()) as target:
        panels = []
        for index, source in enumerate(reservation["records"], 1):
            path, identity = Path(source["path"]), source["source_identity"]
            verify_source(path, identity)
            raw = load_image(path, normalize=False, preserve_dtype=True,
                             series_index=identity["series_index"], frame_index=identity["frame_index"]).image
            tile_scores.clear()
            started = time.perf_counter()
            result = pipeline.detect(raw)
            seconds = time.perf_counter() - started
            verify_source(path, identity)
            pixels = np.rint(normalize_percentile(raw)*255).astype(np.uint8)
            picture = Image.fromarray(pixels).convert("RGB")
            mark = ImageDraw.Draw(picture)
            for x, y in result.points:
                mark.ellipse((x-4, y-4, x+4, y+4), outline="#00ff88", width=2)
                mark.line((x-2, y, x+2, y), fill="#00ff88", width=1)
                mark.line((x, y-2, x, y+2), fill="#00ff88", width=1)
            count = len(result.points)
            score = max(tile_scores)
            name = f"{index:02d}_{path.stem}_prediction"
            width, header = max(900, picture.width), 145
            canvas = Image.new("RGB", (width, picture.height+header), "#111827")
            canvas.paste(picture, ((width-picture.width)//2, header))
            draw = ImageDraw.Draw(canvas)
            y = text_lines(draw, path.name, (18, 10), font, "#f3f4f6", width-36)
            draw.text((18, y), f"{args.model_label} | 阈值 {conf:.2f} | 检测到 {count} 个原子中心", font=font,
                      fill="#fbbf24" if count == 0 else "#00ff88")
            draw.text((18, y+35), f"网络最高分 {score:.6f} | 绿色圆圈表示预测中心；0 点时无圆圈", font=small, fill="#cbd5e1")
            canvas.save(target/f"{name}.png")
            payload = {"source_image": str(path), "source_identity": identity,
                "model_manifest": str(args.manifest.resolve()) if args.manifest else None,
                "source_run": str(args.run.resolve()) if args.run else None, "model_sha256": result.model_sha256,
                "inference": pipeline.backend.inference, "refinement": refinement,
                "points_xy": result.points.tolist(), "confidences": result.confidences.tolist(),
                "metadata": dict(result.metadata), "network_max_score_before_filtering": score,
                "tile_network_max_scores": list(tile_scores), "elapsed_seconds": seconds,
                "overlay": f"{name}.png", "annotation_kind": "model_prediction_not_ground_truth"}
            write_json(target/f"{name}.json", payload)
            rows.append({"name": path.name, "points": count, "network_max_score": score,
                         "overlay": f"{name}.png", "prediction": f"{name}.json", "seconds": seconds})
            panel = Image.new("RGB", (880, 650), "#111827")
            preview = picture.copy()
            preview.thumbnail((844, 490), Image.Resampling.LANCZOS)
            panel.paste(preview, ((880-preview.width)//2, 140+(490-preview.height)//2))
            pd = ImageDraw.Draw(panel)
            py = text_lines(pd, f"{index:02d}  {path.name}", (18, 12), font, "#f3f4f6", 844)
            pd.text((18, py+3), f"检测 {count} 点 | 阈值 {conf:.2f} | 网络最高分 {score:.6f}", font=font,
                    fill="#fbbf24" if count == 0 else "#00ff88")
            panels.append(panel)
            print(f"{path.name}: points={count}, max_score={score:.6f}", flush=True)
        cols, panel_w, panel_h = 2, 880, 650
        overview = Image.new("RGB", (cols*panel_w, 100+((len(panels)+cols-1)//cols)*panel_h), "#090e18")
        draw = ImageDraw.Draw(overview)
        draw.text((20, 12), args.title, font=ImageFont.truetype(font_path, 32), fill="white")
        draw.text((20, 59), f"固定阈值 {conf:.2f}，未调参 | 绿色圆圈为预测中心；无圆圈表示未检出 | 无人工真值", font=font, fill="#cbd5e1")
        for i, panel in enumerate(panels):
            overview.paste(panel, ((i%cols)*panel_w, 100+(i//cols)*panel_h))
        overview.save(target/"overview.png")
        write_json(target/"summary.json", {"purpose": "user_requested_test_preview",
            "model_sha256": pipeline.backend.model_sha256, "fixed_confidence": conf,
            "model_label": args.model_label, "title": args.title,
            "threshold_tuned_on_test": False, "ground_truth_available": False,
            "source_files_unchanged": True, "display_normalization": "full image percentile 1/99; inference uses manifest tile preprocessing",
            "note": "Network maximum includes all output candidates before filtering; it is not a localization accuracy metric.",
            "images": rows})
        lines = ["# 当前模型测试图", "", f"固定置信度阈值：{conf:.2f}。绿色圆圈为模型预测，不是人工真值。", "",
                 "![总览](overview.png)", ""]
        for row in rows:
            lines.append(f"- [{row['name']}]({row['overlay']})：{row['points']} 点；[坐标 JSON]({row['prediction']})")
        (target/"README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
