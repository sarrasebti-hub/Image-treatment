"""Clean the captioned COCO pools and create the knowledge-base, validation and test splits."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import csv
import hashlib
import io
import json
import random
import shutil
import time

from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'cleaned_dataset'
SEED = 42
ImageFile.LOAD_TRUNCATED_IMAGES = False


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def inspect_image(task):
    pool, item = task
    path = ROOT / pool / pool / item['file_name']
    row = dict(source=pool, image_id=item['id'], file_name=item['file_name'],
               status='valid', reason='', sha256='', pixel_sha256='', width='', height='')
    if not path.is_file():
        row.update(status='missing', reason='Expected annotated image is missing')
        return row
    try:
        data = path.read_bytes()
        row['sha256'] = hashlib.sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            rgb = image.convert('RGB')
            row['width'], row['height'] = rgb.size
            if rgb.size != (item['width'], item['height']):
                raise ValueError('Image dimensions disagree with COCO annotation')
            digest = hashlib.sha256(f'{rgb.width}x{rgb.height}:RGB:'.encode())
            digest.update(rgb.tobytes())
            row['pixel_sha256'] = digest.hexdigest()
    except Exception as exc:
        row.update(status='corrupt', reason=f'{type(exc).__name__}: {exc}')
    return row


def main():
    start = time.monotonic()
    if OUT.exists():
        raise SystemExit(f'Refusing to overwrite existing output: {OUT}')
    OUT.mkdir()
    docs = {}
    tasks = []
    for pool in ('train2017', 'val2017'):
        doc = json.loads((ROOT / 'annotations_trainval2017' / 'annotations' /
                          f'captions_{pool}.json').read_text(encoding='utf-8'))
        docs[pool] = doc
        tasks.extend((pool, item) for item in sorted(doc['images'], key=lambda x: x['id']))
    print(f'Auditing {len(tasks):,} annotated images from train2017 and val2017.', flush=True)
    rows = []
    seen_bytes, seen_pixels = {}, {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, row in enumerate(executor.map(inspect_image, tasks), 1):
            if row['status'] == 'valid':
                original = seen_bytes.get(row['sha256']) or seen_pixels.get(row['pixel_sha256'])
                if original:
                    row.update(status='duplicate', reason=f"Same bytes or decoded RGB pixels as {original}")
                else:
                    name = f"{row['source']}/{row['file_name']}"
                    seen_bytes[row['sha256']] = name
                    seen_pixels[row['pixel_sha256']] = name
            rows.append(row)
            if i % 5000 == 0 or i == len(tasks):
                print(f'Checked {i:,}/{len(tasks):,}; {time.monotonic()-start:.0f}s elapsed', flush=True)
    captions = {}
    for pool, doc in docs.items():
        mapping = {}
        for annotation in doc['annotations']:
            mapping.setdefault(annotation['image_id'], []).append(annotation)
        captions[pool] = mapping
    for row in rows:
        if row['status'] == 'valid' and not captions[row['source']].get(row['image_id']):
            row.update(status='missing_captions', reason='No reference captions available')
    fields = list(rows[0])
    for name, data in [('cleaning_audit.csv', rows),
                       ('excluded_images.csv', [r for r in rows if r['status'] != 'valid'])]:
        with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    rng = random.Random(SEED)
    train = [r for r in rows if r['status'] == 'valid' and r['source'] == 'train2017']
    test_pool = [r for r in rows if r['status'] == 'valid' and r['source'] == 'val2017']
    kb = sorted(rng.sample(train, 10000), key=lambda r: r['image_id'])
    validation = sorted(rng.sample(kb, 100), key=lambda r: r['image_id'])
    test = sorted(rng.sample(test_pool, 100), key=lambda r: r['image_id'])
    splits = {'knowledge_base': kb, 'validation': validation, 'test': test}
    for name, selected in splits.items():
        folder = OUT / name
        (folder / 'images').mkdir(parents=True)
        pool = selected[0]['source']
        ids = {r['image_id'] for r in selected}
        manifest = []
        for row in selected:
            source = ROOT / pool / pool / row['file_name']
            destination = folder / 'images' / row['file_name']
            shutil.copy2(source, destination)
            if hashlib.sha256(destination.read_bytes()).hexdigest() != row['sha256']:
                raise RuntimeError(f'Copy checksum mismatch: {destination}')
            manifest.append(dict(image_id=row['image_id'], file_name=row['file_name'],
                                 image_path=f'images/{row["file_name"]}',
                                 source=f'{pool}/{pool}/{row["file_name"]}',
                                 sha256=row['sha256'], pixel_sha256=row['pixel_sha256'],
                                 captions=[a['caption'] for a in captions[pool][row['image_id']]]))
        save_json(folder / 'image_ids.json', sorted(ids))
        save_json(folder / 'manifest.json', manifest)
        doc = docs[pool]
        subset = {key: value for key, value in doc.items() if key not in ('images', 'annotations')}
        subset['images'] = [item for item in doc['images'] if item['id'] in ids]
        subset['annotations'] = [a for a in doc['annotations'] if a['image_id'] in ids]
        save_json(folder / 'captions.json', subset)
        print(f'Created {name}: {len(selected):,} images', flush=True)
    kb_ids = {r['image_id'] for r in kb}
    val_ids = {r['image_id'] for r in validation}
    test_ids = {r['image_id'] for r in test}
    assert len(kb_ids) == 10000 and len(val_ids) == len(test_ids) == 100
    assert val_ids <= kb_ids and not kb_ids & test_ids
    for field in ('sha256', 'pixel_sha256'):
        assert len({r[field] for r in kb + test}) == 10100
    summary = {
        'seed': SEED,
        'audit_scope': 'All annotated train2017 and val2017 images; test2017 and unlabeled2017 are unused.',
        'audited_images': len(rows),
        'audit_status_counts': dict(Counter(r['status'] for r in rows)),
        'source_status_counts': {pool: dict(Counter(r['status'] for r in rows if r['source'] == pool)) for pool in docs},
        'splits': {name: {'images': len(selected),
                          'captions': sum(len(captions[r['source']][r['image_id']]) for r in selected)}
                   for name, selected in splits.items()},
        'unique_selected_images': 10100,
        'validation_is_subset_of_knowledge_base': True,
        'test_disjoint_by_id_file_hash_and_pixel_hash': True,
        'copied_files_sha256_verified': True,
        'duplicate_policy': 'Keep first valid image in source order train2017 then val2017, ascending image ID; compare SHA256 of file bytes and decoded RGB pixels including dimensions.',
        'near_duplicates': 'Similarity-based near-duplicate detection is not performed; crops, resizes and recompressions with different decoded pixels are not treated as exact duplicates.',
        'raw_sources_modified': False,
    }
    save_json(OUT / 'summary.json', summary)
    (OUT / 'README.md').write_text('''# Clean dataset and splits

Only data cleaning and splitting were performed. No exploration, charts, embeddings, resizing or model work was done.

- `knowledge_base/images/`: 10,000 clean images from COCO train2017.
- `validation/images/`: 100 images drawn from that same knowledge base. These are intentional copies for convenient use, not 100 additional unique images.
- `test/images/`: exactly 100 images from COCO val2017, separate from the knowledge base by image ID, file SHA256 and decoded RGB SHA256.
- Each split has `image_ids.json`, `manifest.json` (paths, hashes and original captions), and `captions.json` in COCO format. Original captions are preserved, including any image with more than five captions.
- `cleaning_audit.csv`: one result for every annotated image in train2017 and val2017.
- `excluded_images.csv`: missing, corrupt, duplicate or captionless images excluded from selection. Duplicate rows identify the retained original.
- `summary.json`: measured audit counts and verified split checks.

Cleaning reads and fully decodes every annotated train2017 and val2017 image. It rejects missing files, unreadable/truncated images and annotation dimension mismatches. Exact duplicate files and identical decoded RGB images are reduced to one representative before sampling. Similar-looking scenes, crops, resizes and lossy recompressions with different pixels are outside this exact-duplicate check. Raw source folders remain unchanged; cleaning exclusions apply to this output dataset. Unused test2017 and unlabeled2017 pools are outside the audit scope.

Selection uses Python random seed 42 and uniform random sampling. No common instructor-provided test ID list was present; this is a locally generated test split, not a claim of the shared split for all groups. Keep these IDs fixed for subsequent work.

Following the Phase 1 instructions, the validation set is drawn from within the 10,000 knowledge-base images and the test set contains exactly 100 images. The project description also mentions 100 additional validation images and 1,000 test images; we followed the Phase 1 instructions.

For later validation evaluation, exclude the validation query image and its reference captions from its retrieval candidates. Test captions are evaluation references only and must not enter the knowledge base.

To reproduce into a fresh output folder, run `python clean_and_split.py` from the repository root with Pillow installed. The script refuses to overwrite an existing cleaned_dataset folder.
''', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Finished in {time.monotonic()-start:.0f}s. Output: {OUT}', flush=True)


if __name__ == '__main__':
    main()
