# Test 1000 experiment

This experiment is isolated from the frozen `test` split in `artifacts/`.

Commands:

```powershell
python scripts/build_test_1000_split.py
python scripts/prepare_test_1000.py
python scripts/build_test_1000_metadata.py
python scripts/build_test_1000_embeddings.py
python scripts/verify_test_1000.py
```

Run one encoder or all encoders:

```powershell
python scripts/build_test_1000_embeddings.py --encoder clip-b32
python scripts/build_test_1000_embeddings.py --encoder siglip-b16
python scripts/build_test_1000_embeddings.py --encoder dinov2-s
```

Run all remaining stages:

```powershell
python scripts/run_test_1000.py
```

To resume from an existing stage:

```powershell
python scripts/run_test_1000.py --from-stage embeddings
```

The generated outputs remain under `experiments/test_1000_val2017/`.
