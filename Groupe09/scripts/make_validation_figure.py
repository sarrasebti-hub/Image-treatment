import sys, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
plt.rcParams.update({'font.family':'Inter','font.size':9,'axes.edgecolor':'#c3c2b7','axes.labelcolor':'#52514e',
                     'xtick.color':'#52514e','ytick.color':'#0b0b0b'})
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/preprocessing_release/validation_selection.csv")
names = {'current_clean':'current cleaning','keep_apostrophes':'keep apostrophes','no_stopwords':'no stop words'}
def label(c):
    if c.startswith('hybrid_full_rrf'):
        k, tok = c[len('hybrid_full_rrf'):].split('__bm25:')
        return f'Hybrid, k={k}, BM25 {names[tok]}'
    if c.startswith('bm25_only__'): return 'BM25 only, ' + names[c.split('__')[1]]
    return {'dense_only_rrf60':'Dense only (image + caption), k=60',
            'image_to_image_only':'Image → image only','image_to_caption_only':'Image → caption only'}[c]
df = df.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.0, 5.4), dpi=192)
fig.patch.set_facecolor('#fcfcfb'); ax.set_facecolor('#fcfcfb')
cur = 'hybrid_full_rrf60__bm25:current_clean'
for i, r in df.iterrows():
    sig = bool(r.significant_vs_current)
    col = '#eb6834' if sig else '#2a78d6'
    ax.plot([r.ci95_low, r.ci95_high], [i, i], color=col, lw=2, solid_capstyle='round')
    ax.plot(r.delta_vs_current, i, ms=6, color=col, mec='#fcfcfb', mew=1.5, marker='s' if sig else 'o')
ax.axvline(0, color='#52514e', lw=1, ls=(0,(3,3)))
ax.set_yticks(range(len(df)))
ax.set_yticklabels([label(c) + ('  (current)' if c == cur else '') for c in df.config])
for t, c in zip(ax.get_yticklabels(), df.config):
    if c == cur: t.set_fontweight('bold')
ax.set_xlabel('Change in Jaccard@5 vs current setup (95% bootstrap CI)')
ax.grid(axis='x', color='#e7e6e1', lw=0.8); ax.set_axisbelow(True)
for s in ('top','right','left'): ax.spines[s].set_visible(False)
ax.tick_params(axis='y', length=0)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],color='#2a78d6',marker='o',lw=2,label='CI includes 0 (no real difference)'),
                   Line2D([],[],color='#eb6834',marker='s',lw=2,label='Significantly worse')],
          loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(sys.argv[2] if len(sys.argv) > 2 else "outputs/figures/validation_selection.png", facecolor=fig.get_facecolor())
