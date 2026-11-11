"""Export a standalone research figure from already frozen evaluation reports."""
import json
from pathlib import Path

import numpy as np


def plot_report(selection_path, audit_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    selection = json.loads(Path(selection_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    families = ['ball', 'set', 'gnn']
    names = ['Ball only', 'Players, no messages', 'GNN']
    colors = ['#8294a7', '#d39b43', '#206b91']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    for ax, title, results, baseline in [
        (axes[0], 'Validation: 3 matches', selection['validation'], selection['validation']['gnn']['ensemble']['baseline_rmse']),
        (axes[1], 'Test: 3 matches', audit['models'], audit['training_mean_reference']['rmse'])]:
        rmse = [results[k]['ensemble']['rmse'] for k in families]
        ax.bar(np.arange(3), rmse, color=colors, alpha=.85, label='Three-seed ensemble')
        for i, k in enumerate(families):
            points = [r['rmse'] for r in results[k]['individual']]
            ax.scatter(i + np.linspace(-.12, .12, len(points)), points, color='#20242a', s=18, zorder=3)
        ax.axhline(baseline, color='#8c3030', linestyle='--', linewidth=1.2, label='Training-mean reference')
        ax.set(title=title, ylabel='RMSE (signed xG proxy)', xticks=range(3), xticklabels=names)
        ax.tick_params(axis='x', labelrotation=18)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=7, loc='lower left')
    comparisons = [('gnn_vs_mean', 'vs constant'), ('gnn_vs_ball', 'vs ball'), ('gnn_vs_set', 'vs no messages')]
    for i, (key, label) in enumerate(comparisons):
        r = audit['comparisons'][key]
        point = r['mse_difference'] * 1e5
        low, high = np.array(r['ci95']) * 1e5
        axes[2].plot([low, high], [i, i], color='#206b91', linewidth=2)
        axes[2].scatter(point, i, color='#206b91', s=35)
    axes[2].axvline(0, color='#555', linestyle='--', linewidth=1)
    axes[2].set(yticks=range(3), yticklabels=[r[1] for r in comparisons],
                xlabel='GNN minus reference MSE (×10⁻⁵)\nNegative favours GNN', title='Paired test differences')
    axes[2].spines[['top', 'right']].set_visible(False)
    fig.suptitle('Open Graph EPV — frozen comparison on SkillCorner tracking', fontsize=13)
    fig.supxlabel('Targets: geometry-xG trained on StatsBomb. Dots on bars: individual seeds.\n'
                  'Intervals: exact match bootstrap; only three test matches. No causal or transfer-calibration claim.', fontsize=8)
    logo = Path(__file__).resolve().parents[2] / 'assets' / 'statsbomb-logo.png'
    if logo.exists():
        brand = fig.add_axes([.89, .92, .09, .06])
        brand.imshow(plt.imread(logo))
        brand.axis('off')
    for extension in ('png', 'pdf'):
        path = output / f'epv_evaluation.{extension}'
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180)
    plt.close(fig)
