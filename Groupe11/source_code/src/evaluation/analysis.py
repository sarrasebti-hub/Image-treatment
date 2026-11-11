import pandas as pd
import numpy as np
from scipy import stats

def analyze_hallucinations(csv_file):
    """
    Membre 4 Task: Analyze object, attribute, and relation hallucinations.
    (Assuming the CSV has a column 'hallucination_type' from manual review or an LLM evaluator)
    """
    print("\n--- Hallucination Analysis ---")
    df = pd.read_csv(csv_file)
    
    # Count hallucinations
    if 'hallucination_type' in df.columns:
        counts = df['hallucination_type'].value_counts()
        total = len(df)
        print(f"Total Images Analyzed: {total}")
        for h_type, count in counts.items():
            print(f"{h_type} Hallucination Rate: {(count/total)*100:.2f}%")
    else:
        print("Note: 'hallucination_type' column not found in CSV. Manual review needed.")

def run_statistical_tests(baseline_scores, rag_scores):
    """
    Membre 4 Task: Shapiro-Wilk normality test and Wilcoxon/Paired T-test.
    """
    print("\n--- Statistical Tests ---")
    
    # 1. Shapiro-Wilk Test (Normality)
    stat_b, p_b = stats.shapiro(baseline_scores)
    stat_r, p_r = stats.shapiro(rag_scores)
    print(f"Shapiro-Wilk (Baseline): p={p_b:.4f} (Normal if > 0.05)")
    print(f"Shapiro-Wilk (RAG): p={p_r:.4f} (Normal if > 0.05)")
    
    # 2. Paired T-test or Wilcoxon
    if p_b > 0.05 and p_r > 0.05:
        print("Distributions are normal. Running Paired T-test...")
        t_stat, p_val = stats.ttest_rel(baseline_scores, rag_scores)
        print(f"T-test: statistic={t_stat:.4f}, p-value={p_val:.4e}")
    else:
        print("Distributions are NOT normal. Running Wilcoxon signed-rank test...")
        w_stat, p_val = stats.wilcoxon(baseline_scores, rag_scores)
        print(f"Wilcoxon: statistic={w_stat:.4f}, p-value={p_val:.4e}")
        
    # 3. Bonferroni Correction
    print(f"Bonferroni corrected alpha (for 5 comparisons): {0.05/5:.5f}")

def run_ablations():
    """
    Membre 4 Task: Ablation study on Top-K retrieval.
    """
    print("\n--- Ablation Study (Top-K) ---")
    # Simulated data (In Phase 3, Membre 2 will provide real data)
    top_k_values = [1, 3, 5, 10]
    print(f"Top-K tested: {top_k_values}")
    print("Metrics (CIDEr) will be plotted here comparing K=1,3,5,10.")

if __name__ == "__main__":
    print("Membre 4 Analysis Script Ready!")
    # Example test data
    baseline = np.random.normal(0.5, 0.1, 100)
    rag = baseline + np.random.normal(0.05, 0.05, 100) # RAG is slightly better
    
    run_statistical_tests(baseline, rag)