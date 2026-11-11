import json
import evaluate

def run_evaluation(json_filepath="final_rag_vs_baseline_evaluation.json"): 
    with open(json_filepath, "r", encoding="utf-8") as file:
        records = json.load(file)

    baseline_preds = [item["baseline_caption"] for item in records]
    rag_preds = [item["rag_caption"] for item in records]
    references = [item["ground_truth"] for item in records]

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    base_bleu = bleu.compute(predictions=baseline_preds, references=references)
    base_rouge = rouge.compute(predictions=baseline_preds, references=references)

    rag_bleu = bleu.compute(predictions=rag_preds, references=references)
    rag_rouge = rouge.compute(predictions=rag_preds, references=references)

    print("=========================================")
    print("       EVALUATION METRIC RESULTS         ")
    print("=========================================")
    print(f"Vanilla Baseline -> BLEU: {base_bleu['bleu']:.4f} | ROUGE-L: {base_rouge['rougeL']:.4f}")
    print(f"RAG-Enhanced   -> BLEU: {rag_bleu['bleu']:.4f} | ROUGE-L: {rag_rouge['rougeL']:.4f}")
    print("=========================================")

if __name__ == "__main__":
    run_evaluation()
