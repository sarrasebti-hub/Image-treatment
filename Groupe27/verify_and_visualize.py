"""Verify the delivered splits and create descriptive matplotlib charts."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv
import hashlib
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'cleaned_dataset'
VIS = DATA / 'visualizations'
ImageFile.LOAD_TRUNCATED_IMAGES = False
COLORS = ['#2463a6', '#e39728', '#339981']
NAMES = ['knowledge_base', 'validation', 'test']


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def check_copy(task):
    name, row = task
    path = DATA / name / row['image_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], str(path)
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        im.load()
        rgb = im.convert('RGB')
        h = hashlib.sha256(f'{rgb.width}x{rgb.height}:RGB:'.encode())
        h.update(rgb.tobytes())
        assert h.hexdigest() == row['pixel_sha256'], str(path)
        return name, row['image_id'], rgb.width, rgb.height


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finish(fig, filename):
    fig.tight_layout()
    fig.savefig(VIS / filename, dpi=170, facecolor='white', bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


def main():
    VIS.mkdir(exist_ok=True)
    manifests = {name: load(DATA / name / 'manifest.json') for name in NAMES}
    ids = {name: set(load(DATA / name / 'image_ids.json')) for name in NAMES}
    for name, expected in zip(NAMES, [10000, 100, 100]):
        records = manifests[name]
        coco = load(DATA / name / 'captions.json')
        assert len(records) == len(ids[name]) == expected
        assert {r['image_id'] for r in records} == ids[name] == {r['id'] for r in coco['images']}
        assert {p.name for p in (DATA / name / 'images').iterdir()} == {r['file_name'] for r in records}
        grouped = {}
        for a in coco['annotations']:
            assert a['image_id'] in ids[name]
            grouped.setdefault(a['image_id'], []).append(a['caption'])
        assert all(r['captions'] == grouped[r['image_id']] for r in records)
    assert ids['validation'] <= ids['knowledge_base']
    assert not ids['knowledge_base'] & ids['test']
    unique = manifests['knowledge_base'] + manifests['test']
    for key in ['sha256', 'pixel_sha256']:
        assert len({r[key] for r in unique}) == 10100
    print('Rechecking all 10,200 delivered files, including 100 intentional validation copies...', flush=True)
    tasks = [(name, row) for name in NAMES for row in manifests[name]]
    with ThreadPoolExecutor(max_workers=8) as executor:
        dimensions = {(name, iid): (w, h) for name, iid, w, h in executor.map(check_copy, tasks)}
    with (DATA / 'cleaning_audit.csv').open(encoding='utf-8') as stream:
        audit = list(csv.DictReader(stream))
    status = Counter(r['status'] for r in audit)
    assert len(audit) == 123287
    assert status == Counter(valid=123272, duplicate=15)
    accepted = {(r['source'], int(r['image_id'])) for r in audit if r['status'] == 'valid'}
    assert all((('val2017' if name == 'test' else 'train2017'), r['image_id']) in accepted for name, r in tasks)
    verification = dict(passed=True, checked_delivered_files=len(tasks),
                        unique_images=10100, source_pool_images_audited=len(audit),
                        exact_duplicates_excluded=status['duplicate'], missing_images=0, corrupt_images=0,
                        all_copies_fully_decoded=True, all_file_and_pixel_hashes_match=True,
                        split_counts_correct=True, validation_inside_knowledge_base=True,
                        test_disjoint_by_id_file_hash_and_pixel_hash=True,
                        captions_and_manifests_consistent=True,
                        near_duplicate_detection_performed=False,
                        scope='Delivered dataset verified; original train2017/val2017 audit log checked. Unused test2017/unlabeled2017 not audited.')
    save(DATA / 'verification.json', verification)
    print('Verification passed. Reading existing COCO object annotations for charts...', flush=True)
    object_counts, category_counts, category_names = {}, {}, {}
    for pool, selected in [('train2017', ids['knowledge_base']), ('val2017', ids['test'])]:
        split_name = 'knowledge_base' if pool == 'train2017' else 'test'
        subset_path = DATA / split_name / 'instances.json'
        source_path = ROOT / 'annotations_trainval2017' / 'annotations' / f'instances_{pool}.json'
        doc = load(subset_path if subset_path.exists() else source_path)
        assert selected <= {row['id'] for row in doc['images']}
        category_names.update({r['id']: r['name'] for r in doc['categories']})
        counts = {iid: 0 for iid in selected}
        cats = {iid: Counter() for iid in selected}
        for a in doc['annotations']:
            iid = a['image_id']
            if iid in selected:
                counts[iid] += 1
                cats[iid][a['category_id']] += 1
        object_counts[pool], category_counts[pool] = counts, cats
        del doc
    rows, lengths, summary = [], {}, {}
    all_categories = []
    for name in NAMES:
        pool = 'val2017' if name == 'test' else 'train2017'
        lengths[name] = []
        category_total = Counter()
        for record in manifests[name]:
            iid = record['image_id']
            lens = [len(re.findall(r"\b\w+(?:['\u2019]\w+)*\b", caption)) for caption in record['captions']]
            lengths[name].extend(lens)
            w, h = dimensions[name, iid]
            rows.append(dict(split=name, image_id=iid, width=w, height=h,
                             annotated_object_count=object_counts[pool][iid], caption_count=len(lens),
                             mean_caption_words=round(float(np.mean(lens)), 3)))
            category_total.update(category_counts[pool][iid])
        all_categories.extend(dict(split=name, category_id=cid, category=category_names[cid],
                                   annotated_instances=category_total[cid]) for cid in sorted(category_names))
        objects = [object_counts[pool][iid] for iid in ids[name]]
        summary[name] = dict(images=len(ids[name]), captions=len(lengths[name]),
                             mean_caption_words=round(float(np.mean(lengths[name])), 3),
                             median_caption_words=float(np.median(lengths[name])),
                             min_caption_words=min(lengths[name]), max_caption_words=max(lengths[name]),
                             annotated_objects=sum(objects),
                             mean_annotated_objects_per_image=round(float(np.mean(objects)), 3),
                             min_annotated_objects=min(objects), max_annotated_objects=max(objects),
                             categories_present=sum(v > 0 for v in category_total.values()))
    write_csv(VIS / 'per_image_statistics.csv', rows)
    write_csv(VIS / 'category_counts.csv', all_categories)
    save(VIS / 'statistics_summary.json', summary)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    labels = ['Knowledge base', 'Validation\n(subset of KB)', 'Test\n(separate)']
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, [10000, 100, 100], color=COLORS)
    ax.bar_label(bars, padding=4)
    ax.set_ylim(0, 11500)
    ax.set_ylabel('Image memberships')
    ax.set_title('Delivered splits: 10,100 unique images')
    fig.text(.5, .01, 'Validation overlaps the knowledge base; do not add it as a separate unique set.', ha='center', fontsize=9)
    finish(fig, '01_split_sizes.png')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(['Accepted unique', 'Excluded duplicates'], [123272, 15], color=[COLORS[0], COLORS[1]])
    axes[0].set_ylabel('Images'); axes[0].set_title('Source-pool audit (123,287 images)')
    axes[0].bar_label(axes[0].containers[0], padding=3); axes[0].set_ylim(0, 140000)
    axes[1].bar(['Duplicates', 'Missing', 'Corrupt'], [15, 0, 0], color=[COLORS[1], '#bd5555', '#888888'])
    axes[1].bar_label(axes[1].containers[0], padding=3); axes[1].set_ylim(0, 18)
    axes[1].set_title('Excluded-image detail'); axes[1].set_ylabel('Images')
    finish(fig, '02_cleaning_results.png')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    bins = np.arange(0, max(max(v) for v in lengths.values()) + 3, 2)
    for ax, name, color in zip(axes, NAMES, COLORS):
        ax.hist(lengths[name], bins=bins, color=color, edgecolor='white')
        ax.set_title(f'{name.replace("_", " ").title()} ({len(lengths[name]):,} captions)')
        ax.set_xlabel('Words per human caption'); ax.set_ylabel('Captions')
    fig.suptitle('Original caption lengths', y=1.02)
    finish(fig, '03_caption_length_histograms.png')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    max_objects = max(r['annotated_object_count'] for r in rows)
    for ax, name, color in zip(axes, NAMES, COLORS):
        values = [r['annotated_object_count'] for r in rows if r['split'] == name]
        ax.hist(values, bins=np.arange(-.5, max_objects + 1.5), color=color)
        ax.set_title(f'{name.replace("_", " ").title()} ({len(values):,} images)')
        ax.set_xlabel('Annotated objects per image'); ax.set_ylabel('Images')
    fig.suptitle('Scene complexity proxy: COCO instance counts', y=1.02)
    finish(fig, '04_objects_per_image_histograms.png')
    fig, ax = plt.subplots(figsize=(9, 7))
    top = sorted([r for r in all_categories if r['split'] == 'knowledge_base'], key=lambda r: r['annotated_instances'], reverse=True)[:20][::-1]
    ax.barh([r['category'] for r in top], [r['annotated_instances'] for r in top], color=COLORS[0])
    ax.set_xlabel('Annotated instances'); ax.set_title('Top 20 object categories in the 10,000-image knowledge base')
    finish(fig, '05_object_categories.png')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    kb_rows = [r for r in rows if r['split'] == 'knowledge_base']
    for ax, field in zip(axes, ['width', 'height']):
        ax.hist([r[field] for r in kb_rows], bins=25, color=COLORS[0], edgecolor='white')
        ax.set_xlabel(f'Image {field} (pixels)'); ax.set_ylabel('Images')
        ax.set_title(f'Knowledge base: {field}')
    finish(fig, '06_image_dimensions.png')
    print(json.dumps(summary, indent=2), flush=True)
    print('Saved six charts, underlying statistics and verification.json.', flush=True)


if __name__ == '__main__':
    main()
