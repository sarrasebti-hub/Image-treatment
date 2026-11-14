# QC — official test set (1,000 images, COCO val2017), release g1-v1.2

**PASS** — 43 checks: 43 PASS, 0 FAIL, 0 SKIP (SKIP = needs arrays that are not versioned in Git; rerun locally after generating embeddings).

| # | Status | Check | Detail |
|---:|---|---|---|
| 1 | PASS | test split size = 1000 | 1000 rows |
| 2 | PASS | test image IDs are unique |  |
| 3 | PASS | all test images come from COCO val2017 |  |
| 4 | PASS | test ID hash matches split_manifest | 02fd4ec7bffd4b2b... |
| 5 | PASS | minimum image side >= 200 px |  |
| 6 | PASS | every test image has >= 1 object annotation |  |
| 7 | PASS | every test image has >= 5 COCO captions (first 5 kept as references) | 3 image(s) have more than 5 captions in COCO; only the first 5 are used |
| 8 | PASS | KB size = 10000 and validation size = 100 |  |
| 9 | PASS | ANTI-LEAK: test ∩ KB = ∅ | 0 shared |
| 10 | PASS | ANTI-LEAK: test ∩ validation = ∅ | 0 shared |
| 11 | PASS | test ∩ legacy 100-image test = ∅ | 0 shared |
| 12 | PASS | 5000 reference rows | 5000 rows |
| 13 | PASS | references cover exactly the test images |  |
| 14 | PASS | 5 references per test image | min=5 max=5 |
| 15 | PASS | reference caption_id are unique |  |
| 16 | PASS | no empty reference caption |  |
| 17 | PASS | ANTI-LEAK: no test image has captions in the KB |  |
| 18 | PASS | ANTI-LEAK: no test caption_id in the KB |  |
| 19 | PASS | info: test references textually identical to a KB caption | 132/5000 rows (2.6%) on 112 images; generic captions of DIFFERENT images, not same-image leakage |
| 20 | PASS | metadata exists for every test image | 1000 rows |
| 21 | PASS | metadata image IDs are unique |  |
| 22 | PASS | info: object classes covered by the test set | 80/80 |
| 23 | PASS | frozen file unchanged: artifacts/splits/test.json | content identical (hash compared modulo LF/CRLF line endings) |
| 24 | PASS | frozen file unchanged: artifacts/splits/val.json | content identical (hash compared modulo LF/CRLF line endings) |
| 25 | PASS | frozen file unchanged: artifacts/splits/kb.json | content identical (hash compared modulo LF/CRLF line endings) |
| 26 | PASS | frozen file unchanged: reports/tables/encoder_comparison_test.csv | content identical (hash compared modulo LF/CRLF line endings) |
| 27 | PASS | results cover 5 configurations x 1000 queries |  |
| 28 | PASS | every configuration beats the random-caption baseline (CIDEr) | clip-b32/img2img=0.498; clip-b32/img2txt=0.575; dinov2-s/img2img=0.531; siglip-b16/img2img=0.532; siglip-b16/img2txt=0.639 |
| 29 | PASS | embeddings clip-b32/test_1000 shape/dtype | shape=(1000, 512) |
| 30 | PASS | embeddings clip-b32/test_1000 finite and L2-normalised |  |
| 31 | PASS | embeddings clip-b32/test_1000 IDs match split |  |
| 32 | PASS | ANTI-LEAK clip-b32: no test image retrieved as itself in KB |  |
| 33 | PASS | info clip-b32: test images with a KB near-duplicate (cos >= 0.97) | 1/1000 |
| 34 | PASS | embeddings siglip-b16/test_1000 shape/dtype | shape=(1000, 768) |
| 35 | PASS | embeddings siglip-b16/test_1000 finite and L2-normalised |  |
| 36 | PASS | embeddings siglip-b16/test_1000 IDs match split |  |
| 37 | PASS | ANTI-LEAK siglip-b16: no test image retrieved as itself in KB |  |
| 38 | PASS | info siglip-b16: test images with a KB near-duplicate (cos >= 0.97) | 0/1000 |
| 39 | PASS | embeddings dinov2-s/test_1000 shape/dtype | shape=(1000, 384) |
| 40 | PASS | embeddings dinov2-s/test_1000 finite and L2-normalised |  |
| 41 | PASS | embeddings dinov2-s/test_1000 IDs match split |  |
| 42 | PASS | ANTI-LEAK dinov2-s: no test image retrieved as itself in KB |  |
| 43 | PASS | info dinov2-s: test images with a KB near-duplicate (cos >= 0.97) | 0/1000 |
