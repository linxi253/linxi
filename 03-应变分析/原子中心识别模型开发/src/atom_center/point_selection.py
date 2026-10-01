"""Select saved checkpoints using full-resolution validation point metrics only."""
from pathlib import Path
import math
import os
import shutil

from .storage import content_digest,read_json,write_json
from .model_manifest import sha256_file


def point_metric_rank(metrics):
    """F1 first, recall second, matched-point RMSE last; no-match is worst RMSE."""
    tp,fp,fn=(metrics[k] for k in ("tp","fp","fn"))
    if any(type(x) is not int or x < 0 for x in (tp,fp,fn)):
        raise ValueError("point counts must be nonnegative integers")
    denominator=2*tp+fp+fn
    f1=2*tp/denominator if denominator else 1.
    recall=tp/(tp+fn) if tp+fn else 1.
    rmse=metrics["rmse_px"]
    if tp and (not isinstance(rmse,(int,float)) or not math.isfinite(rmse) or rmse < 0):
        raise ValueError("matched points require finite nonnegative RMSE")
    return f1,recall,-float(rmse) if tp else -math.inf


class PointCheckpointSelector:
    """Records refer to the serialized checkpoint actually evaluated and copied.

    Epoch-specific candidates are immutable. The state file atomically selects
    one, so an interrupted promotion cannot corrupt the previous best selection.
    """
    def __init__(self,run,*,dataset_sha256,config,contract,resume=False):
        self.run=Path(run).resolve()
        self.directory=self.run/"point_validation"
        self.directory.mkdir(exist_ok=True)
        self.path=self.directory/"selection.json"
        self.contract={"dataset_sha256":dataset_sha256,"split":"val","config":config,
                       "contract":contract,"ranking":["max_f1","max_recall","min_matched_rmse"],
                       "checkpoint_semantics":"serialized_fp16_ema_loaded_as_fp32"}
        self.fingerprint=content_digest(self.contract)
        if self.path.exists():
            if not resume:
                raise ValueError("point selection already exists for new training run")
            self.state=read_json(self.path)
            if self.state["contract_sha256"] != self.fingerprint:
                raise ValueError("point selection contract changed")
            self._verify_best()
        else:
            self.state={"schema_version":1,"contract":self.contract,"contract_sha256":self.fingerprint,
                        "history":[],"best":None}

    def _verify_best(self):
        best=self.state["best"]
        if best:
            path=(self.run/best["path"]).resolve()
            if not path.is_relative_to(self.directory) or sha256_file(path)!=best["sha256"]:
                raise ValueError("point-selected checkpoint changed or escaped run directory")

    def due(self,epoch,*,final=False):
        settings=self.contract["config"]["point_validation"]
        return epoch==1 or epoch%settings["interval_epochs"]==0 or final

    def consider(self,checkpoint,epoch,metrics):
        if (metrics.get("split")!="val" or metrics.get("dataset_content_sha256")!=self.contract["dataset_sha256"]
                or metrics.get("match_distance_px")!=self.contract["config"]["point_validation"]["match_distance_px"]):
            raise ValueError("point checkpoint selection requires the configured validation dataset and distance")
        rank=point_metric_rank(metrics)
        digest=sha256_file(checkpoint)
        # Recovered training may repeat an epoch after an interruption. Keep
        # each checkpoint identity instead of silently overwriting its evidence.
        report_name=f"epoch_{epoch:04d}_{digest[:12]}.json"
        write_json(self.directory/report_name,metrics)
        record={"epoch":epoch,"checkpoint_sha256":digest,"report":f"point_validation/{report_name}",
                "metrics":{k:metrics[k] for k in ("tp","fp","fn","f1","recall","precision","rmse_px")}}
        self._verify_best()
        best=self.state["best"]
        improved=best is None or rank>point_metric_rank(best["metrics"])
        if improved:
            name=f"candidate_{epoch:04d}_{digest[:12]}.pt"
            target=self.directory/name
            temporary=self.directory/f".{name}.tmp"
            shutil.copy2(checkpoint,temporary)
            if sha256_file(temporary)!=digest:
                raise ValueError("point checkpoint changed during promotion")
            os.replace(temporary,target)
            self.state["best"]={"epoch":epoch,"path":f"point_validation/{name}","sha256":digest,
                                "metrics":record["metrics"],"report":record["report"]}
        self.state["history"].append(record)
        write_json(self.path,self.state)
        return improved

    def evaluate(self,checkpoint,epoch,data_root,*,device):
        from .backends import TorchBackend
        from .pipeline import DetectionPipeline
        from .configuration import pipeline_config
        from .evaluation import evaluate_dataset
        config=self.contract["config"]
        backend=TorchBackend(checkpoint,expected_sha256=sha256_file(checkpoint),contract=self.contract["contract"],
                             inference=config["inference"],device=device)
        pipe=DetectionPipeline(backend,pipeline_config(config))
        metrics=evaluate_dataset(pipe,data_root,split="val",max_distance_px=config["point_validation"]["match_distance_px"])
        improved=self.consider(checkpoint,epoch,metrics)
        print(f"Point validation epoch {epoch}: F1={metrics['f1']:.6f}, recall={metrics['recall']:.6f}, "
              f"RMSE={metrics['rmse_px']}, best_updated={improved}",flush=True)
        return metrics
