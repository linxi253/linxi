"""Same fine-tuning protocol as fixed-box arm; only supervision boxes differ."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from atom_center.configuration import load_config
from atom_center.training import train_run,checkpoint_for_run
from atom_center.storage import write_json
from optimize_generalization import evaluate

def main():
    root=Path(__file__).resolve().parents[1]
    _,checkpoint,digest=checkpoint_for_run(root/'runs/point-selection-20260909',checkpoint='best_points.pt')
    config=load_config(root/'configs/experiments/haadf_point_selected_150.yaml',overrides={
        'training.epochs':150,'training.patience':151,'training.model':str(checkpoint),
        'training.lr0':.00015,'training.fliplr':.5,'training.flipud':.5,
        'training.degrees':0.,'training.translate':.025,'training.scale':.15,
        'augmentation.probability':.5})
    run=root/'runs/generalization-20260910/spacing150'
    train_run(root/'data/processed/real_workflow_20260910_spacing',run,config,initialization_sha256=digest)
    result=evaluate(run,'spacing150')
    write_json(run.parent/'spacing_comparison.json',result)

if __name__=='__main__':main()
