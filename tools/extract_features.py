"""Resumable cache using the exact deployed feature extractor."""
import argparse
import csv
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
    parser=argparse.ArgumentParser()
    parser.add_argument("manifest",type=Path)
    parser.add_argument("--output",type=Path,default=ROOT/"data"/"features")
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    rows=list(csv.DictReader(args.manifest.open()))
    if args.limit: rows=rows[:args.limit]
    args.output.mkdir(parents=True,exist_ok=True)
    model=Analyzer(); model.load()
    started=time.perf_counter(); failures=[]; ok=0; peak=0
    for index,row in enumerate(rows):
        target=args.output/(row["sample_id"]+".npz")
        if not target.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("Unsafe sample id")
        if valid_cache(target, model.pipeline_id):
            ok+=1; continue
        try:
            features=model.extract(ROOT/row["media_path"])
            save_features(target, features)
            ok+=1
        except Exception as exc:
            failures.append({"sample_id":row["sample_id"],"error":str(exc)})
        peak=max(peak,psutil.Process().memory_info().rss)
        if index%10==0: print(f"{index+1}/{len(rows)} processed; valid={ok}; failures={len(failures)}",flush=True)
    report={"valid":ok,"failures":failures,"seconds":time.perf_counter()-started,"peak_sampled_rss_mb":peak/1048576,"pipeline_id":model.pipeline_id}
    save_report(args.output/"extraction_report.json", report)
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
