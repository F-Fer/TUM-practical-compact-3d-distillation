"""Outdoor / object-centric training clips (no benchmark overlap): Mip-NeRF 360 (`images_4`, 9 scenes) and the
3DGS Tanks and Temples / Deep Blending set (4 scenes), extracted by HTTP range reads of the remote zips into
workspace/distill_data/arkit_frames/outdoor/<name>/ as JPEG q95 (same layout as prep_arkit.py, so train_res.py sees
them as extra videos). Frames are sorted by name (capture order).
  .venv/bin/python scripts/distill_steps/prep_outdoor.py
"""
import io, os, sys, zipfile
import requests
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_arkit import HTTPFile  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "workspace", "distill_data", "arkit_frames", "outdoor")
SRC = [("http://storage.googleapis.com/gresearch/refraw360/360_v2.zip", "images_4", "m360_"),
       ("https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip", "images", "tdb_")]
LOCAL = os.environ.get("DISTILL_ZIPS", "")  # dir with local copies of the zips (tum2: $SCRATCH/distill_data/zips)
s = requests.Session()
for url, sub, pre in SRC:
    local = os.path.join(LOCAL, os.path.basename(url)) if LOCAL else ""
    src = local if local and os.path.exists(local) else io.BufferedReader(HTTPFile(url, s), buffer_size=1 << 20)
    z = zipfile.ZipFile(src)
    scenes = {}
    for n in z.namelist():
        if n.lower().endswith((".jpg", ".png", ".jpeg")) and os.path.basename(os.path.dirname(n)) == sub:
            scenes.setdefault(os.path.dirname(os.path.dirname(n)), []).append(n)
    for sc, names in sorted(scenes.items()):
        d = os.path.join(OUT, pre + sc.strip("/").replace("/", "_"))
        if os.path.exists(f"{d}/DONE"):
            continue
        os.makedirs(d, exist_ok=True)
        names = sorted(names)
        step = max(1, len(names) // 300)
        for i, n in enumerate(names[::step]):
            img = Image.open(io.BytesIO(z.read(n))).convert("RGB")
            if max(img.size) > 1100:  # keep files small; the model input is 504 px anyway
                img.thumbnail((1100, 1100), Image.LANCZOS)
            img.save(os.path.join(d, f"{i:05d}.jpg"), quality=95)
        open(f"{d}/DONE", "w").write(f"{len(names[::step])} of {len(names)}\n")
        print(d, len(names[::step]), flush=True)
