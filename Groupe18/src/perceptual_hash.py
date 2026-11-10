"""64-bit DCT perceptual hashing with blockwise exact Hamming comparisons."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def phash_image(image: Image.Image) -> np.uint64:
    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    n = 32
    x = np.arange(n, dtype=np.float32)
    k = np.arange(n, dtype=np.float32)[:, None]
    dct = np.cos(np.pi * (2*x + 1) * k / (2*n)).astype(np.float32)
    dct[0] *= np.sqrt(1/n)
    dct[1:] *= np.sqrt(2/n)
    low = (dct @ pixels @ dct.T)[:8, :8].reshape(-1)
    med = np.median(low[1:])
    bits = low > med
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return np.uint64(value)


def phash(path: Path) -> np.uint64:
    with Image.open(path) as im:
        return phash_image(im)


def run(root: Path, config: dict):
    cfg = config["perceptual_hash"]
    threshold = int(cfg["hamming_distance_threshold"])
    block = int(cfg.get("block_size", 128))
    items=[]; hashes=[]; errors=[]
    t0=time.time()
    for split in ("train", "validation", "test"):
        manifest=json.loads((root/"data"/"selected"/f"{split}_ids.json").read_text(encoding="utf-8"))
        for iid in manifest["image_ids"]:
            path=root/"data"/"processed"/split/f"{iid:012d}.jpg"
            try:
                hashes.append(phash(path)); items.append({"image_id":int(iid),"split":split,"filename":path.name})
            except Exception as exc:
                errors.append({"image_id":int(iid),"split":split,"reason":str(exc)})
    hv=np.asarray(hashes,dtype=np.uint64)
    out=root/"reports"/"perceptual_hash"; out.mkdir(parents=True,exist_ok=True)
    npz_tmp=out/"phash_database.npz.tmp"
    with npz_tmp.open("wb") as f:
        np.savez_compressed(f, image_ids=np.asarray([x["image_id"] for x in items],np.int64),
                            splits=np.asarray([x["split"] for x in items]),
                            filenames=np.asarray([x["filename"] for x in items]), hashes=hv)
    npz_tmp.replace(out/"phash_database.npz")
    candidates=[]; within=Counter(); across=Counter(); dist=Counter()
    byte_view=hv.view(np.uint8)
    for i0 in range(0,len(hv),block):
        i1=min(i0+block,len(hv))
        # XOR and byte bit counts keep pairwise temporary memory bounded.
        xor=np.bitwise_xor(hv[i0:i1,None],hv[None,:])
        d=np.unpackbits(xor.view(np.uint8).reshape(i1-i0,len(hv),8),axis=2).sum(axis=2)
        ri,ci=np.where((d<=threshold) & (np.arange(i0,i1)[:,None] < np.arange(len(hv))[None,:]))
        for local,j in zip(ri.tolist(),ci.tolist()):
            i=i0+local; distance=int(d[local,j]); a,b=items[i],items[j]
            pair={"image_id_a":a["image_id"],"split_a":a["split"],"image_id_b":b["image_id"],"split_b":b["split"],
                  "hamming_distance":distance,"evidence":"64-bit DCT pHash; candidate threshold configured by project"}
            candidates.append(pair); dist[distance]+=1
            (within if a["split"]==b["split"] else across)[f"{a['split']}-{b['split']}"]+=1
        print(f"pHash compare rows {i1}/{len(hv)}; candidates={len(candidates)}",flush=True)
    pairs={"algorithm":cfg["algorithm"],"hash_bits":64,"threshold_hamming_inclusive":threshold,
           "images_hashed":len(items),"errors":errors,"pairs_within_split":dict(within),"pairs_across_splits":dict(across),
           "candidate_pair_count":len(candidates),"exact_phash_match_count":dist.get(0,0),
           "nonzero_perceptual_candidate_count":sum(v for k,v in dist.items() if k>0),
           "hamming_distance_counts":dict(sorted(dist.items())),"mean_candidate_distance":float(np.mean([p["hamming_distance"] for p in candidates])) if candidates else None,
           "threshold_interpretation":"Candidates only; not a declaration of duplicate identity. Verify visually before any action.",
           "elapsed_seconds":round(time.time()-t0,2),"candidates":candidates}
    (out/"duplicate_candidates.json").write_text(json.dumps(pairs,indent=2)+"\n",encoding="utf-8")
    (out/"hash_statistics.json").write_text(json.dumps({k:v for k,v in pairs.items() if k!="candidates"},indent=2)+"\n",encoding="utf-8")
    return {k:v for k,v in pairs.items() if k!="candidates"}
