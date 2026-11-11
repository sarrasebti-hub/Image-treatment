import json
import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from scipy import stats
import numpy as np

def evaluate_captions(generated_csv, ground_truth_json):
    print("Loading data...")
    # Load generated captions (Assuming columns: image_id, generated_caption)
    df = pd.read_csv(generated_csv)
    
    # Load ground truth
    with open(ground_truth_json, 'r') as f:
        gt_data = json.load(f)
        
    # Format for pycocoevalcap: Dictionary {image_id: [captions]}
    gts = {}
    for item in gt_data:
        img_id = item['image_id']
        if img_id not in gts:
            gts[img_id] = []
        gts[img_id].append(item['caption_original'])
        
    # Format generated captions
    res = {}
    for index, row in df.iterrows():
        res[int(row['image_id'])] = [row['generated_caption']]
        
    # Only evaluate images that are in both
    common_ids = set(gts.keys()).intersection(set(res.keys()))
    gts = {k: gts[k] for k in common_ids}
    res = {k: res[k] for k in common_ids}
    
    print(f"Evaluating on {len(common_ids)} images...")

    # Initialize metrics
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr")
    ]

    final_scores = {}
    for scorer, method in scorers:
        print(f"Computing {method}...")
        score, scores = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                final_scores[m] = sc
        else:
            final_scores[method] = score

    # Print results
    print("\n--- Evaluation Results ---")
    for metric, score in final_scores.items():
        print(f"{metric}: {score:.4f}")

    print("\n--- Statistical Test Preparation ---")
    print("To run the Wilcoxon test later, you will compare two arrays of per-image scores.")
    
    return final_scores

if __name__ == "__main__":
    print("Evaluation script is ready! Waiting for Membre 3's generated captions in Phase 2.")