from pathlib import Path
import sys
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.train_multimodal import train
from multimodal.heads import predict_head

def test_export_parity_and_training_only_normalization(tmp_path):
    rng=np.random.default_rng(42)
    x=rng.normal(size=(24,12)).astype(np.float32); y=np.array([0,1]*12)
    split={"train":np.arange(16),"validation":np.arange(16,20),"test":np.arange(20,24)}
    weights,probs,temp,history=train(x,y,split,42,3)
    np.testing.assert_allclose(weights["mean"],x[:16].mean(0))
    np.savez_compressed(tmp_path/"weights.npz",**weights)
    (tmp_path/"metadata.json").write_text(json.dumps({"pipeline_id":"test","modality":"fusion","temperature":temp,"labels":["a","b"],"run_id":"test","task":"emotion"}))
    output=predict_head(tmp_path,x[20,:6],x[20,6:],"test")
    np.testing.assert_allclose(list(output["probabilities"].values()),probs[20],atol=1e-6)
