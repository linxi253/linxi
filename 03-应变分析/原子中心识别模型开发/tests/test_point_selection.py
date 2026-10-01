from pathlib import Path
import pytest
from atom_center.configuration import load_config,model_contract
from atom_center.point_selection import PointCheckpointSelector,point_metric_rank
from atom_center.storage import read_json
from atom_center.model_manifest import sha256_file


def metrics(tp,fp,fn,rmse):
    return {'split':'val','dataset_content_sha256':'a'*64,'match_distance_px':2.,
            'tp':tp,'fp':fp,'fn':fn,'rmse_px':rmse,'f1':2*tp/(2*tp+fp+fn),
            'recall':tp/(tp+fn),'precision':tp/(tp+fp) if tp+fp else 1.}


def selector(run,resume=False):
    config=load_config(overrides={'point_validation.enabled':True})
    return PointCheckpointSelector(run,dataset_sha256='a'*64,config=config,contract=model_contract(config),resume=resume)


def test_rank_does_not_reward_missing_difficult_points():
    assert point_metric_rank(metrics(90,5,10,.8))>point_metric_rank(metrics(10,0,90,.01))
    assert point_metric_rank(metrics(90,5,10,.4))>point_metric_rank(metrics(90,5,10,.8))
    assert point_metric_rank(metrics(1,0,99,1.))>point_metric_rank(metrics(0,0,100,None))


def test_selection_promotes_exact_saved_bytes_and_resumes_without_resetting_best(tmp_path):
    selected=selector(tmp_path)
    checkpoint=tmp_path/'last.pt';checkpoint.write_bytes(b'first saved EMA')
    assert selected.consider(checkpoint,1,metrics(90,5,10,.8))
    first=read_json(selected.path)['best']
    assert (tmp_path/first['path']).read_bytes()==checkpoint.read_bytes()
    checkpoint.write_bytes(b'next saved EMA with worse F1')
    assert not selected.consider(checkpoint,5,metrics(80,1,20,.1))
    resumed=selector(tmp_path,resume=True)
    assert resumed.state['best']==first
    checkpoint.write_bytes(b'better saved EMA')
    assert resumed.consider(checkpoint,10,metrics(95,3,5,.6))
    assert resumed.state['best']['sha256']==sha256_file(checkpoint)
    assert len(resumed.state['history'])==3
    assert (tmp_path/first['path']).read_bytes()==b'first saved EMA'
    assert selected.due(1) and selected.due(5) and selected.due(7,final=True) and not selected.due(7)


def test_selection_rejects_test_data_changed_dataset_or_checkpoint(tmp_path):
    selected=selector(tmp_path);checkpoint=tmp_path/'last.pt';checkpoint.write_bytes(b'weights')
    valid=metrics(90,5,10,.8)
    for bad in ({**valid,'split':'test'},{**valid,'dataset_content_sha256':'b'*64},{**valid,'match_distance_px':5.}):
        with pytest.raises(ValueError,match='validation dataset'):selected.consider(checkpoint,1,bad)
    selected.consider(checkpoint,1,valid)
    (tmp_path/selected.state['best']['path']).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='checkpoint changed'):selector(tmp_path,resume=True)


@pytest.mark.parametrize('key,value',[
    ('enabled','true'),('interval_epochs',0),('interval_epochs',True),
    ('match_distance_px',float('nan')),('match_distance_px',0),('match_distance_px',True),
])
def test_point_validation_invalid_config_rejected(key,value):
    with pytest.raises(ValueError,match='point_validation'):
        load_config(overrides={f'point_validation.{key}':value})
