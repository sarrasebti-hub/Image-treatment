import matplotlib.pyplot as plt

methods = [
    "CLIP Reranking",
    "MMR Lambda 0.7",
    "MMR Lambda 0.9"
]

relevance = [0.3235, 0.2966, 0.3208]
diversity = [0.2295, 0.4415, 0.2956]

plt.figure(figsize=(9, 6))

plt.scatter(relevance, diversity, s=120)

for i, method in enumerate(methods):
    plt.annotate(
        method,
        (relevance[i], diversity[i]),
        xytext=(7, 7),
        textcoords="offset points"
    )

plt.xlabel("Mean Relevance")
plt.ylabel("Mean Intra-List Diversity")

plt.title("Pareto Analysis: Relevance vs Diversity")

plt.grid(True)
plt.tight_layout()

output = "dataset/embeddings/mmr_pareto_plot.png"

plt.savefig(output, dpi=300)
plt.show()

print("Pareto plot saved to:", output)