"""Train CPU heads on paired cached features; keep subjects out of other splits."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score, average_precision_score, brier_score_loss)
from multimodal.heads import softmax

def validate_manifest(rows, task):
    needed={"sample_id","subject_id","session_id","media_path","label","label_source","split"}
    if not rows or not needed.issubset(rows[0]):
        raise ValueError(f"Manifest requires {sorted(needed)}")
    seen={}; samples=set(); media=set(); subject_labels={}
    for row in rows:
        if any(not row[k].strip() for k in needed): raise ValueError("Empty manifest field")
        if row["split"] not in {"train","validation","test"}: raise ValueError("Unknown split")
        if row["sample_id"] in samples or row["media_path"] in media: raise ValueError("Duplicate sample/media")
        if Path(row["sample_id"]).name != row["sample_id"]: raise ValueError("Unsafe sample id")
        samples.add(row["sample_id"]); media.add(row["media_path"])
        prior=seen.setdefault(row["subject_id"],row["split"])
        if prior!=row["split"]: raise ValueError("Subject leakage between splits")
        if task=="depression" and (row["label"] not in {"lower","elevated"} or
            any(x in row["label_source"].lower() for x in ("acted","synthetic","ravdess","emotion"))):
            raise ValueError("Depression training requires real lower/elevated labels with depression provenance")
        if task=="depression" and subject_labels.setdefault(row["subject_id"],row["label"])!=row["label"]:
            raise ValueError("Participant labels must be consistent for depression evaluation")
    if {r["split"] for r in rows}!={"train","validation","test"}: raise ValueError("All three splits are required")

def metrics(y, probs, labels, binary=False, threshold=.5):
    pred=(probs[:,1]>=threshold).astype(int) if binary else probs.argmax(1)
    result={"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
        "classification_report":classification_report(y,pred,labels=list(range(len(labels))),target_names=labels,output_dict=True,zero_division=0),
        "confusion_matrix":confusion_matrix(y,pred,labels=list(range(len(labels)))).tolist()}
    if binary and len(np.unique(y))==2:
        tn,fp,fn,tp=np.asarray(result["confusion_matrix"]).ravel()
        result.update(sensitivity=float(tp/max(tp+fn,1)),specificity=float(tn/max(tn+fp,1)),
            roc_auc=float(roc_auc_score(y,probs[:,1])),pr_auc=float(average_precision_score(y,probs[:,1])),
            brier=float(brier_score_loss(y,probs[:,1])))
    return result

def bootstrap(y,probs,groups,binary,threshold):
    rng=np.random.default_rng(42); ids=np.unique(groups); values=[]
    for _ in range(500):
        idx=np.concatenate([np.flatnonzero(groups==s) for s in rng.choice(ids,len(ids),replace=True)])
        pred=(probs[idx,1]>=threshold).astype(int) if binary else probs[idx].argmax(1)
        values.append(f1_score(y[idx],pred,average="macro",zero_division=0))
    return dict(zip(["lower","upper"],map(float,np.percentile(values,[2.5,97.5]))))

def participants(y,probs,groups):
    ids=np.unique(groups)
    return np.array([y[groups==s][0] for s in ids]),np.stack([probs[groups==s].mean(0) for s in ids]),ids

def train(X,y,indices,seed,epochs):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    tr,va=indices["train"],indices["validation"]
    mean=X[tr].mean(0); scale=np.maximum(X[tr].std(0),1e-6)
    x=torch.tensor(np.clip((X-mean)/scale,-10,10),dtype=torch.float32)
    target=torch.tensor(y,dtype=torch.long)
    net=torch.nn.Sequential(torch.nn.Linear(X.shape[1],128),torch.nn.ReLU(),torch.nn.Dropout(.3),torch.nn.Linear(128,len(np.unique(y))))
    opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001)
    best=float("inf"); stale=0; history=[]; best_state=None
    for epoch in range(epochs):
        net.train(); losses=[]
        order=np.random.default_rng(seed+epoch).permutation(tr)
        for start in range(0,len(order),32):
            batch=order[start:start+32]; opt.zero_grad()
            loss=torch.nn.functional.cross_entropy(net(x[batch]),target[batch]); loss.backward(); opt.step(); losses.append(loss.item())
        net.eval()
        with torch.no_grad(): val=torch.nn.functional.cross_entropy(net(x[va]),target[va]).item()
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"validation_loss":val})
        if val<best-1e-5: best=val; best_state=copy.deepcopy(net.state_dict()); stale=0
        else: stale+=1
        if stale>=8: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad(): logits=net(x).numpy()
    # One scalar temperature, selected only using validation labels.
    grid=np.geomspace(.5,5,80)
    temperature=float(min(grid,key=lambda t:-np.log(softmax(logits[va]/t)[np.arange(len(va)),y[va]]+1e-12).mean()))
    weights={"mean":mean,"scale":scale,"w1":net[0].weight.detach().numpy(),"b1":net[0].bias.detach().numpy(),
             "w2":net[3].weight.detach().numpy(),"b2":net[3].bias.detach().numpy()}
    return weights,softmax(logits/temperature),temperature,history

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("manifest",type=Path)
    parser.add_argument("--features",type=Path,default=ROOT/"data"/"features")
    parser.add_argument("--task",choices=["emotion","depression"],default="emotion")
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(); torch.set_num_threads(4)
    all_rows=list(csv.DictReader(args.manifest.open())); validate_manifest(all_rows,args.task)
    rows=[]; audio=[]; face=[]; versions=set(); excluded=[]; durations=[]
    for row in all_rows:
        p=args.features/(row["sample_id"]+".npz")
        if not p.exists(): excluded.append(row["sample_id"]); continue
        with np.load(p,allow_pickle=False) as f:
            audio.append(f["audio"]); face.append(f["face"]); versions.add(str(f["pipeline_id"]))
            durations.append(json.loads(str(f["quality"]))["duration_seconds"])
        rows.append(row)
    validate_manifest(rows,args.task)
    if len(versions)!=1: raise ValueError("Mixed feature extractors")
    labels=["lower","elevated"] if args.task=="depression" else sorted({r["label"] for r in rows})
    y=np.array([labels.index(r["label"]) for r in rows]); groups=np.array([r["subject_id"] for r in rows])
    indices={s:np.array([i for i,r in enumerate(rows) if r["split"]==s]) for s in ("train","validation","test")}
    for split,idx in indices.items():
        if set(y[idx])!=set(range(len(labels))): raise ValueError(f"Missing classes in {split}")
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output=args.output or ROOT/"saved_models"/"multimodal"/args.task
    results=ROOT/"results"/"multimodal"/run_id; results.mkdir(parents=True,exist_ok=True)
    report={"task":args.task,"run_id":run_id,"labels":labels,"excluded":excluded,
        "manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "splits":{s:{"recordings":len(idx),"people":len(set(groups[idx]))} for s,idx in indices.items()},"models":{}}
    (results/"manifest.csv").write_bytes(args.manifest.read_bytes())
    matrices={"audio":np.stack(audio),"face":np.stack(face),"fusion":np.concatenate([audio,face],axis=1)}
    started=time.perf_counter()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    binary=args.task=="depression"
    for modality,X in matrices.items():
        weights,probs,temperature,history=train(X,y,indices,42,args.epochs)
        threshold=.5
        if binary:
            va=indices["validation"]
            vy,vp,_=participants(y[va],probs[va],groups[va])
            threshold=float(max(np.arange(.1,.901,.01),key=lambda t:f1_score(vy,vp[:,1]>=t,zero_division=0)))
        test=indices["test"]
        ty,tp,tg=participants(y[test],probs[test],groups[test]) if binary else (y[test],probs[test],groups[test])
        scores=metrics(ty,tp,labels,binary,threshold)
        scores["evaluation_unit"]="participant" if binary else "recording, grouped by actor"
        scores["macro_f1_bootstrap_95ci"]=bootstrap(ty,tp,tg,binary,threshold)
        scores["duration_groups"]={}
        for low,high in ((2,10),(10,30),(30,60),(60,121)):
            idx=np.array([i for i in test if low<=durations[i]<high],dtype=int)
            if len(idx):
                dy,dp,_=participants(y[idx],probs[idx],groups[idx]) if binary else (y[idx],probs[idx],groups[idx])
                scores["duration_groups"][f"{low}-{high}s"]={"recordings":len(idx),**metrics(dy,dp,labels,binary,threshold)}
        scores["temperature"]=temperature; scores["threshold"]=threshold
        scores["recorded_sex_subgroups"]={}
        for sex in sorted({r.get("recorded_sex","") for r in rows}-{ "" }):
            idx=np.array([i for i in test if rows[i].get("recorded_sex")==sex])
            if len(idx):
                sy,sp,_=participants(y[idx],probs[idx],groups[idx]) if binary else (y[idx],probs[idx],groups[idx])
                scores["recorded_sex_subgroups"][sex]=metrics(sy,sp,labels,binary,threshold)
        report["models"][modality]=scores
        dest=output/modality; dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/"weights.npz",**weights)
        meta={"run_id":run_id,"pipeline_id":next(iter(versions)),"labels":labels,"task":args.task,
            "modality":modality,"temperature":temperature,"threshold":threshold,
            "deployment_approved":False,"duration_validated":False,"evaluation_path":str(results.relative_to(ROOT)),
            "training":{"seed":42,"max_epochs":args.epochs,"epochs_run":len(history),"hidden_units":128,"dropout":.3,"batch_size":32,"learning_rate":.001,"frozen_encoders":True}}
        (dest/"metadata.json").write_text(json.dumps(meta,indent=2))
        (results/f"{modality}_history.json").write_text(json.dumps(history,indent=2))
        np.savez_compressed(results/f"{modality}_predictions.npz",labels=y[test],probabilities=probs[test],subjects=groups[test])
        fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(scores["confusion_matrix"]); fig.colorbar(im,ax=ax)
        ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,xlabel="Predicted",ylabel="Actual",title=f"{modality}: held-out people")
        plt.setp(ax.get_xticklabels(),rotation=45,ha="right"); fig.tight_layout(); fig.savefig(results/f"{modality}_confusion.png"); plt.close(fig)
        fig,ax=plt.subplots(); ax.plot([h["train_loss"] for h in history],label="train"); ax.plot([h["validation_loss"] for h in history],label="validation"); ax.legend(); fig.savefig(results/f"{modality}_loss.png"); plt.close(fig)
        if binary:
            from sklearn.calibration import calibration_curve
            observed,predicted=calibration_curve(ty,tp[:,1],n_bins=5,strategy="quantile")
            fig,ax=plt.subplots(); ax.plot(predicted,observed,"o-"); ax.plot([0,1],[0,1],"--"); ax.set(xlabel="Predicted probability",ylabel="Observed fraction"); fig.savefig(results/f"{modality}_calibration.png"); plt.close(fig)
        print(modality,scores["accuracy"],scores["macro_f1"],flush=True)
    report["training_seconds"]=time.perf_counter()-started
    report["fusion_improved_macro_f1"]=report["models"]["fusion"]["macro_f1"]>max(report["models"][m]["macro_f1"] for m in ("audio","face"))
    (results/"evaluation.json").write_text(json.dumps(report,indent=2))
    print("Evaluation:",results)

if __name__=="__main__": main()
