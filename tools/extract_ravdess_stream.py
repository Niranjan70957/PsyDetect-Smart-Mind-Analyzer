"""Overlap resumable feature extraction with verified RAVDESS archive downloads."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import psutil
from multimodal.models import Analyzer
from tools.feature_cache import valid_cache, save_features, save_report

def main():
    model=Analyzer(); model.load()
    dest=ROOT/"data"/"features"; dest.mkdir(parents=True,exist_ok=True)
    done=set(); failures=[]; started=time.perf_counter(); peak=0; last_message=0
    while len(done)<1440:
        files=sorted((ROOT/"data"/"ravdess").glob("Actor_*/01-01-*.mp4"))
        for path in files:
            if path.stem in done: continue
            target=dest/(path.stem+".npz")
            if valid_cache(target, model.pipeline_id):
                done.add(path.stem); continue
            try:
                f=model.extract(path)
                save_features(target, f)
            except Exception as exc:
                failures.append({"sample_id":path.stem,"error":str(exc)})
                print("Extraction failed:",path.stem,str(exc),flush=True)
            done.add(path.stem); peak=max(peak,psutil.Process().memory_info().rss)
            if len(done)%20==0:
                print(f"Features: {len(done)}/1440, exclusions {len(failures)}",flush=True)
                save_report(dest/"stream_progress.json", {"processed":len(done),"failures":failures,"seconds":time.perf_counter()-started,"peak_sampled_rss_mb":peak/1048576,"pipeline_id":model.pipeline_id})
        if len(done)<1440:
            if time.perf_counter()-last_message>60:
                print(f"Waiting for verified archives: {len(done)}/1440",flush=True); last_message=time.perf_counter()
            time.sleep(10)
    save_report(dest/"extraction_report.json", {"valid":len(done)-len(failures),"failures":failures,"seconds":time.perf_counter()-started,"peak_sampled_rss_mb":peak/1048576,"pipeline_id":model.pipeline_id})
    print("Feature extraction complete",flush=True)

if __name__=="__main__": main()
