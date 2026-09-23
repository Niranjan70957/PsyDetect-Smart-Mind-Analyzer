"""Fetch the original RAVDESS speech videos and write a fixed actor split."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import zipfile
import requests
from concurrent.futures import ThreadPoolExecutor

def fetch_archive(item, dest):
    target=dest/item["key"]
    if not target.exists():
        partial=target.with_suffix(".partial")
        offset=partial.stat().st_size if partial.exists() else 0
        url=f"https://zenodo.org/records/1188976/files/{item['key']}?download=1"
        print(f"Downloading {item['key']} from {offset}/{item['size']} bytes",flush=True)
        with partial.open("ab") as f:
            while offset<item["size"]:
                end=min(offset+8*1024*1024-1,item["size"]-1)
                for attempt in range(3):
                    try:
                        r=requests.get(url,headers={"Range":f"bytes={offset}-{end}"},timeout=(20,45))
                        r.raise_for_status()
                        if r.status_code!=206 or len(r.content)!=end-offset+1:
                            raise ValueError("Server did not honor bounded range")
                        break
                    except (requests.RequestException,ValueError):
                        if attempt==2: raise
                f.write(r.content); f.flush(); offset=end+1
                print(f"{item['key']}: {offset*100/item['size']:.0f}%",flush=True)
        partial.replace(target)
    digest=hashlib.md5()
    with target.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): digest.update(chunk)
    if item["checksum"]!="md5:"+digest.hexdigest(): raise ValueError(f"Checksum mismatch: {target}")
    with zipfile.ZipFile(target) as z:
        for name in z.namelist():
            if Path(name).name.startswith("01-") and name.endswith(".mp4"):
                output=(dest/name).resolve()
                if not output.is_relative_to(dest.resolve()): raise ValueError("Unsafe archive path")
                if not output.exists(): z.extract(name,dest)

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--actors", type=int, nargs="+", default=list(range(1,25)))
    args = parser.parse_args()
    dest = ROOT / "data" / "ravdess"
    dest.mkdir(parents=True, exist_ok=True)
    if args.download:
        response = requests.get("https://zenodo.org/api/records/1188976", timeout=90)
        response.raise_for_status()
        record = response.json()
        (dest / "source.json").write_text(json.dumps(record, indent=2))
        files=[item for item in record["files"] if any(item["key"]==f"Video_Speech_Actor_{actor:02d}.zip" for actor in args.actors)]
        with ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(lambda item:fetch_archive(item,dest),files))
    rng = random.Random(42)
    odd, even = list(range(1,25,2)), list(range(2,25,2))
    rng.shuffle(odd); rng.shuffle(even)
    splits = {actor: split for group in (odd,even) for split, ids in
              (("train",group[:8]),("validation",group[8:10]),("test",group[10:])) for actor in ids}
    rows = []
    for path in sorted(dest.glob("Actor_*/*.mp4")):
        parts = path.stem.split("-")
        if parts[0] != "01" or parts[1] != "01":
            continue
        actor = int(parts[-1])
        rows.append({"sample_id":path.stem, "subject_id":f"actor-{actor:02d}", "session_id":path.stem,
            "media_path":str(path.relative_to(ROOT)), "label":LABELS[int(parts[2])-1],
            "label_source":"RAVDESS acted emotion", "split":splits[actor],
            "recorded_sex":"male" if actor%2 else "female"})
    if not rows:
        raise SystemExit("No AV speech videos found; run with --download.")
    with (dest / "manifest.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (dest / "splits.json").write_text(json.dumps(splits,indent=2))
    print(f"Prepared {len(rows)} recordings, {len(set(r['subject_id'] for r in rows))} actors.")

if __name__ == "__main__":
    main()
