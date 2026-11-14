# Leakage and quality audit — `g1-v1.1`

The frozen report is `artifacts/qc/qc_report.json` (also rendered as
`artifacts/qc/qc_report.md`). It records **40 checks, 40 PASS, 0 FAIL, and 0
WARN**. Details below reproduce every saved check and its detail field.

## All QC checks

| # | Status | Check | Saved detail |
|---:|---|---|---|
| 1 | PASS | taille split kb | `10000 vs 10000` |
| 2 | PASS | hash des IDs kb = splits_meta | *(empty)* |
| 3 | PASS | taille split val | `100 vs 100` |
| 4 | PASS | hash des IDs val = splits_meta | *(empty)* |
| 5 | PASS | taille split test | `100 vs 100` |
| 6 | PASS | hash des IDs test = splits_meta | *(empty)* |
| 7 | PASS | splits deux à deux disjoints | *(empty)* |
| 8 | PASS | images invalides (corrompues/absentes) | `0` |
| 9 | PASS | IDs uniques dans images.csv | *(empty)* |
| 10 | PASS | doublons exacts (sha256) inter/intra splits | `0 images` |
| 11 | PASS | pas de NaN dans les stats pixel | *(empty)* |
| 12 | PASS | KB : 5 captions / image | `min=5 max=5` |
| 13 | PASS | val : 5 références / image | *(empty)* |
| 14 | PASS | test : 5 références / image | *(empty)* |
| 15 | PASS | caption_id uniques (KB) | *(empty)* |
| 16 | PASS | ANTI-FUITE : aucune image val/test dans kb_captions | *(empty)* |
| 17 | PASS | ANTI-FUITE : aucun caption_id d'éval dans la KB | *(empty)* |
| 18 | PASS | captions brutes inchangées (raw != clean autorisé, raw non vide) | *(empty)* |
| 19 | PASS | info : refs val identiques à une caption KB (texte) | `1.0% — normal pour des captions génériques; pas une fuite d'image` |
| 20 | PASS | info : refs test identiques à une caption KB (texte) | `1.4% — normal pour des captions génériques; pas une fuite d'image` |
| 21 | PASS | métadonnées pour toutes les images | *(empty)* |
| 22 | PASS | attributs texte uniquement pour la KB | *(empty)* |
| 23 | PASS | embeddings clip-b32/kb | `shape=(10000, 512)` |
| 24 | PASS | embeddings clip-b32/val | `shape=(100, 512)` |
| 25 | PASS | embeddings clip-b32/test | `shape=(100, 512)` |
| 26 | PASS | index clip-b32 : ntotal=KB et auto-retrouvabilité | *(empty)* |
| 27 | PASS | embeddings siglip-b16/kb | `shape=(10000, 768)` |
| 28 | PASS | embeddings siglip-b16/val | `shape=(100, 768)` |
| 29 | PASS | embeddings siglip-b16/test | `shape=(100, 768)` |
| 30 | PASS | index siglip-b16 : ntotal=KB et auto-retrouvabilité | *(empty)* |
| 31 | PASS | embeddings dinov2-s/kb | `shape=(10000, 384)` |
| 32 | PASS | embeddings dinov2-s/val | `shape=(100, 384)` |
| 33 | PASS | embeddings dinov2-s/test | `shape=(100, 384)` |
| 34 | PASS | index dinov2-s : ntotal=KB et auto-retrouvabilité | *(empty)* |
| 35 | PASS | ANTI-FUITE : val jamais retrouvé pour lui-même (k=10) | *(empty)* |
| 36 | PASS | info : val avec near-duplicate dans la KB | `cos>=0.97: 0/100 \| dHash<=4: 0` |
| 37 | PASS | ANTI-FUITE : test jamais retrouvé pour lui-même (k=10) | *(empty)* |
| 38 | PASS | info : test avec near-duplicate dans la KB | `cos>=0.97: 0/100 \| dHash<=4: 0` |
| 39 | PASS | fumée val : NN-caption CIDEr > aléatoire | `NN=0.674 random=0.024 \| Jaccard classes@1=0.546` |
| 40 | PASS | fumée test : NN-caption CIDEr > aléatoire | `NN=0.622 random=0.033 \| Jaccard classes@1=0.506` |

## Near-duplicate screening and repair

The image audit in `artifacts/audit/image_audit.json` computes source-image
SHA-256 hashes for exact duplicates and perceptual difference hashes for
near-duplicates. It compares each validation/test image with the other
evaluation images and the KB. The configured dHash Hamming-distance threshold
is **4**. QC separately checks the primary encoder (SigLIP) nearest-neighbor
cosine threshold **>= 0.97**, and checks dHash distance **<= 4** against the KB.
For the frozen release, both checks found **0/100** flagged validation images
and **0/100** flagged test images in the KB; exact SHA-256 duplicates were
also zero.

The recorded repair in `artifacts/audit/dedup_repair.json` dropped:

- image `186275`: “copie identique (SHA-256) d'une image val/test”;
- image `463132`: “quasi-doublon (dHash <= 4) d'une image val/test”.

The repair added KB IDs `149053` and `474404`. The repair JSON records the
dropped and added ID lists but does not state a one-to-one mapping between
each dropped and added ID.

## Caption-text matches: generic text, not image/ID leakage

QC's text check applies `common.clean_caption` to validation/test raw reference
text and to KB `caption_clean` text, then tests membership in the KB caption
set. This is normalized text equality (lowercase, punctuation/whitespace
normalization), not a claim that the original raw strings are byte-identical.
It counts matching evaluation reference rows: **5 of 500 (1.0%) in validation**
and **7 of 500 (1.4%) in test**. The KB contains repeated generic caption text,
so those references produce 9 and 37 distinct evaluation-image/KB-image
mappings, respectively. All listed pairs have different evaluation and KB
image IDs. Matching text alone does not establish that the image or caption
record leaked.

The following pairs were extracted read-only from the frozen JSONL files using
the same `clean_caption` comparison as the QC notebook.

### Validation

| Evaluation image ID | Caption text | KB image ID |
|---:|---|---:|
| 40468 | `A man riding a wave on a surf board.` | 315162 |
| 104842 | `A man riding a skateboard at a skate park.` | 23050 |
| 104842 | `A man riding a skateboard at a skate park.` | 265267 |
| 164908 | `a person jumping a skate board in the air ` | 78583 |
| 164908 | `a person jumping a skate board in the air ` | 87681 |
| 164908 | `a person jumping a skate board in the air ` | 324872 |
| 164908 | `a person jumping a skate board in the air ` | 431530 |
| 272490 | `A computer keyboard and mouse sitting on a desk.` | 129832 |
| 566237 | `A person on a snowboard up in the air.` | 321182 |

### Test

| Evaluation image ID | Caption text | KB image ID |
|---:|---|---:|
| 43816 | `A batter getting ready to hit his pitch.` | 57827 |
| 62355 | `A baseball player holding a bat standing next to home plate.` | 48340 |
| 62355 | `A baseball player holding a bat standing next to home plate.` | 359232 |
| 261732 | `a man with a racket hits a tennis ball ` | 146420 |
| 390902 | `A man standing on a tennis court holding a racquet.` | 15153 |
| 390902 | `A man standing on a tennis court holding a racquet.` | 16509 |
| 390902 | `A man standing on a tennis court holding a racquet.` | 157656 |
| 390902 | `A man standing on a tennis court holding a racquet.` | 262413 |
| 390902 | `A man standing on a tennis court holding a racquet.` | 265135 |
| 390902 | `A man standing on a tennis court holding a racquet.` | 327299 |
| 390902 | `A man standing on a tennis court holding a racquet.` | 354656 |
| 390902 | `A man standing on a tennis court holding a racquet.` | 381492 |
| 390902 | `A man standing on a tennis court holding a racquet.` | 399968 |
| 390902 | `A man standing on a tennis court holding a racquet.` | 468825 |
| 390902 | `A man standing on a tennis court holding a racquet.` | 563945 |
| 390902 | `A man standing on a tennis court holding a racquet.` | 568210 |
| 390902 | `a man holding a tennis racquet on a tennis court` | 3157 |
| 390902 | `a man holding a tennis racquet on a tennis court` | 19552 |
| 390902 | `a man holding a tennis racquet on a tennis court` | 52193 |
| 390902 | `a man holding a tennis racquet on a tennis court` | 67854 |
| 390902 | `a man holding a tennis racquet on a tennis court` | 204337 |
| 390902 | `a man holding a tennis racquet on a tennis court` | 239456 |
| 390902 | `a man holding a tennis racquet on a tennis court` | 261477 |
| 390902 | `a man holding a tennis racquet on a tennis court` | 261984 |
| 390902 | `a man holding a tennis racquet on a tennis court` | 321991 |
| 390902 | `a man holding a tennis racquet on a tennis court` | 409964 |
| 390902 | `a man holding a tennis racquet on a tennis court` | 415684 |
| 390902 | `a man holding a tennis racquet on a tennis court` | 431933 |
| 390902 | `a man holding a tennis racquet on a tennis court` | 458089 |
| 390902 | `a man holding a tennis racquet on a tennis court` | 491323 |
| 390902 | `a man holding a tennis racquet on a tennis court` | 568259 |
| 428111 | `A man riding a skateboard up the side of a ramp.` | 272323 |
| 428111 | `A man riding a skateboard up the side of a ramp.` | 351221 |
| 428111 | `A man riding a skateboard up the side of a ramp.` | 462134 |
| 428111 | `A man riding a skateboard up the side of a ramp.` | 493926 |
| 428111 | `a man is in the air on a skateboard` | 201929 |
| 428111 | `a man is in the air on a skateboard` | 365220 |
