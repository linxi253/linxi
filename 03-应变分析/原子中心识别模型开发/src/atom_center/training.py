"""Audited, isolated Ultralytics training runs with explicit continuation."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import yaml

from .configuration import validate_config, model_contract
from .data_workflow import verify_dataset, DatasetAuditError
from .model_manifest import sha256_file
from .storage import write_json, read_json, content_digest, relative_path, resolve_path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def software_versions():
    result = {"python": sys.version.split()[0]}
    for name in ("atom-center", "torch", "torchvision", "ultralytics", "onnx", "onnxruntime",
                 "numpy", "scipy", "opencv-python", "tifffile", "Pillow", "PyYAML"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result


def code_provenance():
    package = Path(__file__).resolve().parent
    files = {p.relative_to(package).as_posix(): sha256_file(p) for p in sorted(package.rglob("*.py"))}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=package,
                                          stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "source_files": files, "source_sha256": content_digest(files)}


def preflight(data, config, *, initialization_sha256=None):
    validate_config(config, ultralytics=True)
    manifest, root = verify_dataset(data)
    if manifest["modality"] != config["modality"]:
        raise DatasetAuditError("dataset and training modality differ")
    if (manifest["tile_size"] != config["training"]["imgsz"]
            or manifest.get("preprocessing_contract") != model_contract(config)["contract"]):
        raise DatasetAuditError("dataset tile size/preprocessing differs; rebuild the dataset")
    norm = config["image"]["normalization"]
    if any(manifest["normalization"][k] != norm[k] for k in ("method", "low", "high")):
        raise DatasetAuditError("dataset normalization differs; rebuild the dataset")
    margin = int(manifest["max_train_atoms_per_crop"]*1.25)+32
    if config["inference"]["max_det"] < margin:
        raise ValueError(f"max_det needs training-density headroom: >= {margin}")
    import torch
    device = str(config["training"]["device"])
    if device != "cpu":
        if not device.isdigit() or not torch.cuda.is_available() or int(device) >= torch.cuda.device_count():
            raise ValueError("device must be cpu or an available single CUDA device index")
    model = config["training"]["model"]
    if model in {"yolov8n.yaml", "yolov8s.yaml", "yolov8m.yaml"}:
        initialization = {"kind": "seeded_random", "architecture": model, "seed": config["seed"]}
    else:
        path = Path(model).resolve()
        if path.suffix != ".pt" or not path.is_file() or not initialization_sha256:
            raise ValueError("use built-in yolov8[n/s/m].yaml, or a trusted local .pt plus --init-sha256")
        if sha256_file(path) != initialization_sha256:
            raise ValueError("initialization checkpoint hash differs")
        initialization = {"kind": "local_checkpoint", "source": str(path), "sha256": initialization_sha256}
    return manifest, root, initialization


@contextmanager
def run_lock(run):
    path = run/".running.lock"
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ValueError(f"run is locked: {path}; after a process crash verify it has stopped before removing the lock") from error
    try:
        import os
        handle.write(f"pid={os.getpid()} started={utc_now()}\n")
        handle.close()
        yield
    finally:
        handle.close()
        path.unlink(missing_ok=True)


class TrainingPaused(Exception):
    pass


def _state_dict_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        data = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(data.dtype).encode())
        digest.update(str(data.shape).encode())
        digest.update(data.tobytes())
    return digest.hexdigest()


def _execute(run, manifest, state, *, resume=False, stop_after_epochs=None):
    # Imports below this boundary do not affect CPU ONNX users.
    import torch
    from copy import copy
    from .training_dataset import VerifiedDataset
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from .augmentation import GrayscaleAugment
    config = manifest["config"]
    point_selector = None
    if config.get("point_validation", {}).get("enabled", False):
        from .point_selection import PointCheckpointSelector
        point_selector = PointCheckpointSelector(run, dataset_sha256=manifest["dataset_content_sha256"],
            config=config, contract=manifest["contract"], resume=resume)
        # Use the resolved path, including when a byte-identical dataset moved.
        data_root = Path(yaml.safe_load((run/"data.resolved.yaml").read_text(encoding="utf-8"))["path"])

    class AtomTrainer(DetectionTrainer):
        def check_resume(self, overrides):
            super().check_resume(overrides)
            if self.resume:
                # A copied checkpoint retains paths to its old run even when
                # that directory still exists. Use this verified run's paths.
                for key in ("data", "project", "name", "save_dir", "exist_ok"):
                    setattr(self.args, key, overrides[key])

        def read_results_csv(self):
            # Checkpoint metadata only needs numeric CSV columns. The installed
            # Polars runtime fails its CPU feature check on this Windows host.
            import csv
            if not self.csv.exists():
                return {}
            with self.csv.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                result = {key.strip(): [] for key in reader.fieldnames or []}
                for row in reader:
                    for key, value in row.items():
                        result[key.strip()].append(float(value))
            return result

        def build_dataset(self, img_path, mode="train", batch=None):
            dataset = VerifiedDataset(img_path=img_path, imgsz=self.args.imgsz,
                batch_size=batch, augment=mode == "train", hyp=copy(self.args),
                rect=False, cache=None, single_cls=False, stride=32, pad=0.,
                prefix=f"{mode}: ", task="detect", classes=None, data=self.data, fraction=1.)
            if mode == "train":
                dataset.transforms.insert(0, GrayscaleAugment(**config["augmentation"]))
            return dataset

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            if not resume:
                write_json(run/"initialization.json", {**manifest["initialization"],
                    "model_state_sha256": _state_dict_digest(model), "resolved_architecture": model.yaml})
            return model

    weights_dir = run/"training"/"weights"

    def record_checkpoints(trainer):
        state["completed_epochs"] = trainer.epoch+1
        state["actual_batch"] = trainer.batch_size
        state["checkpoints"] = {
            name: {"path": relative_path(weights_dir/name, run), "sha256": sha256_file(weights_dir/name)}
            for name in ("best.pt", "last.pt") if (weights_dir/name).is_file()}
        if point_selector and point_selector.state["best"]:
            best = point_selector.state["best"]
            state["checkpoints"]["best_points.pt"] = {"path": best["path"], "sha256": best["sha256"]}
            state["point_selection"] = {"epoch": best["epoch"], "metrics": best["metrics"],
                                         "record": "point_validation/selection.json"}
        state["updated_utc"] = utc_now()
        if trainer.device.type == "cuda":
            state["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(trainer.device)
            state["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(trainer.device)
        write_json(run/"state.json", state)

    def on_save(trainer):
        # Keep a resumable copy before Ultralytics strips optimizers during final_eval.
        shutil.copy2(weights_dir/"last.pt", weights_dir/"resume.pt")
        state["resume_checkpoint"] = {"path": relative_path(weights_dir/"resume.pt", run),
                                      "sha256": sha256_file(weights_dir/"resume.pt")}
        epoch = trainer.epoch+1
        if point_selector and point_selector.due(epoch, final=bool(trainer.stop)
                or epoch >= trainer.epochs or (stop_after_epochs is not None and epoch >= stop_after_epochs)):
            point_selector.evaluate(weights_dir/"last.pt", epoch, data_root, device=str(trainer.device))
        record_checkpoints(trainer)
        if stop_after_epochs is not None and trainer.epoch+1 >= stop_after_epochs:
            raise TrainingPaused()

    overrides = {**config["training"], "seed": config["seed"], "max_det": config["inference"]["max_det"],
        "iou": config["inference"]["iou"], "data": str(run/"data.resolved.yaml"),
        "project": str(run), "name": "training", "exist_ok": True,
        "save_dir": str(run/"training"), "save": True, "val": True, "pretrained": False}
    if resume:
        checkpoint = resolve_path(state["resume_checkpoint"]["path"], run)
        if not checkpoint.is_relative_to(run) or sha256_file(checkpoint) != state["resume_checkpoint"]["sha256"]:
            raise ValueError("resume checkpoint hash changed")
        overrides["resume"] = str(checkpoint)
    trainer = None
    state["status"] = "running"
    state.setdefault("attempts", []).append({"started_utc": utc_now(), "resume": resume,
                                            "from_completed_epochs": state.get("completed_epochs", 0)})
    write_json(run/"state.json", state)
    try:
        if config["training"]["device"] != "cpu":
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(int(config["training"]["device"]))
        trainer = AtomTrainer(overrides=overrides)
        write_json(run/"ultralytics_args.json", vars(trainer.args))
        trainer.add_callback("on_model_save", on_save)
        trainer.train()
        record_checkpoints(trainer)
        state["status"] = "completed"
    except TrainingPaused:
        state["status"] = "paused"
    except (Exception, KeyboardInterrupt) as error:
        state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        state["error"] = str(error)
        (run/"error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        state["updated_utc"] = utc_now()
        write_json(run/"state.json", state)
        if trainer is not None:
            for key in ("train_loader", "test_loader"):
                loader = getattr(trainer, key, None)
                if hasattr(loader, "close"):
                    loader.close()
    return state


def train_run(data, run, config, *, initialization_sha256=None, stop_after_epochs=None):
    dataset, data_root, initialization = preflight(data, config, initialization_sha256=initialization_sha256)
    if stop_after_epochs is not None and not 0 < stop_after_epochs < config["training"]["epochs"]:
        raise ValueError("stop_after_epochs must be between 1 and epochs-1")
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "purpose": dataset["purpose"], "created_utc": utc_now(),
        "config": config, "dataset_content_sha256": dataset["content_sha256"],
        "dataset": relative_path(data_root, run), "initialization": initialization,
        "software": software_versions(), "code": code_provenance(), "contract": model_contract(config)}
    manifest["content_sha256"] = content_digest(manifest)
    write_json(run/"run_manifest.json", manifest)
    shutil.copy2(data_root/"dataset_manifest.json", run/"dataset_manifest.snapshot.json")
    resolved_data = {"path": str(data_root), **{k: f"images/{k}" for k in dataset["group_counts"]},
                     "names": {0: "atom_column"}}
    (run/"data.resolved.yaml").write_text(yaml.safe_dump(resolved_data, allow_unicode=True), encoding="utf-8")
    write_json(run/"effective_config.json", config)
    with run_lock(run):
        return _execute(run, manifest, {"status": "created", "completed_epochs": 0},
                        stop_after_epochs=stop_after_epochs)


def read_run(run):
    run = Path(run).resolve()
    manifest = read_json(run/"run_manifest.json")
    payload = {k: v for k, v in manifest.items() if k != "content_sha256"}
    if content_digest(payload) != manifest["content_sha256"]:
        raise ValueError("run manifest changed")
    state = read_json(run/"state.json")
    return run, manifest, state


def resume_run(run, *, data=None):
    run, manifest, state = read_run(run)
    with run_lock(run):
        if state["status"] == "completed" or state.get("completed_epochs", 0) >= manifest["config"]["training"]["epochs"]:
            raise ValueError("run has completed; start a new experiment for more training")
        if not state.get("resume_checkpoint"):
            raise ValueError("no complete epoch checkpoint is available for resume")
        if manifest["code"]["source_sha256"] != code_provenance()["source_sha256"]:
            raise ValueError("training source code changed; resume requires the same implementation")
        versions = software_versions()
        if any(versions.get(k) != v for k, v in manifest["software"].items()):
            raise ValueError("training dependency versions changed")
        dataset, data_root = verify_dataset(data or resolve_path(manifest["dataset"], run))
        if dataset["content_sha256"] != manifest["dataset_content_sha256"]:
            raise DatasetAuditError("resume dataset fingerprint differs")
        config = manifest["config"]
        validate_config(config, ultralytics=True)
        # The run can move together with a copied dataset; absolute runtime paths are rebuilt.
        (run/"data.resolved.yaml").write_text(yaml.safe_dump(
            {"path": str(data_root), **{k: f"images/{k}" for k in dataset["group_counts"]},
             "names": {0: "atom_column"}}, allow_unicode=True), encoding="utf-8")
        return _execute(run, manifest, state, resume=True)


def checkpoint_for_run(run, checkpoint="best.pt"):
    run, manifest, state = read_run(run)
    if checkpoint not in state.get("checkpoints", {}):
        raise ValueError(f"checkpoint unavailable: {checkpoint}")
    entry = state["checkpoints"][checkpoint]
    path = resolve_path(entry["path"], run)
    if not path.is_relative_to(run) or sha256_file(path) != entry["sha256"]:
        raise ValueError("run checkpoint missing or changed")
    return manifest, path, entry["sha256"]
