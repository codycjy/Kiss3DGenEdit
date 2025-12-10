"""
P2P Batch Results Visualization
Generates elegant academic-style plots for experimental analysis

Usage: python visualize_p2p_results.py
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from math import pi
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = Path("results/p2p_batch_results/evaluation_summary")
OUTPUT_DIR = Path("results/p2p_batch_results/visualization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette: elegant, distinguishable, colorblind-friendly
COLORS = {
    'qk_img': '#2E86AB',    # Steel blue
    'qk_full': '#A23B72',   # Berry
    'full': '#F18F01',      # Amber
}

EDIT_MODE_LABELS = {
    'qk_img': 'QK-Image',
    'qk_full': 'QK-Full',
    'full': 'Full'
}

# ============================================================================
# Style Setup
# ============================================================================
def setup_style():
    """Configure matplotlib for elegant academic style"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Georgia'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
    })
    sns.set_palette([COLORS['qk_img'], COLORS['qk_full'], COLORS['full']])

# ============================================================================
# Data Loading
# ============================================================================
def load_data():
    """Load all data files"""
    # Find the most recent files
    json_files = list(DATA_DIR.glob("best_params_*.json"))
    if not json_files:
        raise FileNotFoundError("No data files found in evaluation_summary/")

    timestamp = json_files[0].stem.split('_')[-2] + '_' + json_files[0].stem.split('_')[-1]

    # Load JSON data
    with open(DATA_DIR / f"best_params_{timestamp}.json", 'r') as f:
        best_params = json.load(f)

    # Load CSV data
    summary_df = pd.read_csv(DATA_DIR / f"summary_{timestamp}.csv")
    comparison_df = pd.read_csv(DATA_DIR / f"comparison_by_entry_{timestamp}.csv")

    return best_params, summary_df, comparison_df

# ============================================================================
# Plot 1: Edit Mode Comparison Bar Chart
# ============================================================================
def plot_edit_mode_comparison(summary_df):
    """Grouped bar chart comparing edit modes across key metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = [
        ('avg_lpips', 'LPIPS (lower is better)', True),
        ('avg_tgt_consistency', 'Target Consistency (%)', False),
        ('avg_tgt_semantic', 'Target Semantic (%)', False),
        ('avg_mvclip_improvement', 'MV-CLIP Improvement', False)
    ]

    # Group by edit_mode
    grouped = summary_df.groupby('edit_mode').agg({
        'avg_lpips': 'mean',
        'avg_tgt_consistency': 'mean',
        'avg_tgt_semantic': 'mean',
        'avg_mvclip_improvement': 'mean'
    }).reset_index()

    for ax, (metric, label, invert) in zip(axes.flat, metrics):
        modes = grouped['edit_mode'].tolist()
        values = grouped[metric].tolist()
        colors = [COLORS[m] for m in modes]

        bars = ax.bar([EDIT_MODE_LABELS[m] for m in modes], values, color=colors,
                      edgecolor='white', linewidth=1.5)

        ax.set_ylabel(label)
        ax.set_xlabel('Edit Mode')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}' if val < 1 else f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

        if invert:
            ax.invert_yaxis()

    fig.suptitle('Edit Mode Comparison Across Metrics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_edit_mode_comparison.png')
    plt.close()
    print("  Saved: 01_edit_mode_comparison.png")

# ============================================================================
# Plot 2: Tau Effect Line Plots
# ============================================================================
def plot_tau_effect(summary_df):
    """Line plots showing effect of tau on metrics"""
    # Filter to diff_tau category which has multiple tau values
    tau_data = summary_df[summary_df['category'] == 'diff_tau'].copy()

    if len(tau_data) == 0:
        print("  Skipped: 02_tau_effect_analysis.png (insufficient tau variation data)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = [
        ('avg_lpips', 'LPIPS'),
        ('avg_tgt_consistency', 'Target Consistency (%)'),
        ('avg_tgt_semantic', 'Target Semantic (%)'),
        ('avg_mvclip_improvement', 'MV-CLIP Improvement')
    ]

    for ax, (metric, label) in zip(axes.flat, metrics):
        for mode in tau_data['edit_mode'].unique():
            mode_data = tau_data[tau_data['edit_mode'] == mode].sort_values('tau')
            ax.plot(mode_data['tau'], mode_data[metric],
                   marker='o', markersize=8, linewidth=2,
                   color=COLORS.get(mode, '#333333'),
                   label=EDIT_MODE_LABELS.get(mode, mode))

        ax.set_xlabel(r'$\tau$ (Threshold)')
        ax.set_ylabel(label)
        ax.legend(loc='best')

    fig.suptitle(r'Effect of $\tau$ on Evaluation Metrics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_tau_effect_analysis.png')
    plt.close()
    print("  Saved: 02_tau_effect_analysis.png")

# ============================================================================
# Plot 3: Correlation Heatmap
# ============================================================================
def plot_correlation_heatmap(comparison_df):
    """Correlation matrix of all numeric metrics"""
    numeric_cols = ['src_consistency', 'src_semantic', 'tgt_consistency', 'tgt_semantic',
                    'lpips', 'mvclip_src_src', 'mvclip_src_tgt', 'mvclip_tgt_src',
                    'mvclip_tgt_tgt', 'mvclip_improvement', 'mvclip_preservation']

    # Filter to existing columns
    available_cols = [c for c in numeric_cols if c in comparison_df.columns]
    corr_matrix = comparison_df[available_cols].corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                ax=ax)

    ax.set_title('Metrics Correlation Matrix', fontsize=14, fontweight='bold', pad=20)

    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_metrics_correlation.png')
    plt.close()
    print("  Saved: 03_metrics_correlation.png")

# ============================================================================
# Plot 3b: Focused Correlation Analysis (Key Meaningful Correlations)
# ============================================================================
def plot_focused_correlations(comparison_df):
    """
    Focused visualization of key meaningful correlations (all r > 0.5):
    1. MVCLIP internal correlation (mvclip_src_src ↔ mvclip_src_tgt): r=0.82
    2. Cross-modal consistency (src_consistency ↔ mvclip_src_src): r=0.56
    3. tgt_semantic ↔ mvclip_improvement: r=0.55
    4. tgt_semantic ↔ mvclip_preservation: r=0.52
    """
    fig = plt.figure(figsize=(14, 10))

    # Define the key correlation pairs (all r > 0.5)
    key_correlations = [
        {
            'pair': ('mvclip_src_src', 'mvclip_src_tgt'),
            'title': 'A. MVCLIP Internal Consistency',
            'interpretation': 'Source MVCLIP features\nstable before/after editing',
            'category': 'Preservation Validation'
        },
        {
            'pair': ('src_consistency', 'mvclip_src_src'),
            'title': 'B. AI Score × MVCLIP',
            'interpretation': 'AI consistency validates\nMVCLIP measurements',
            'category': 'Cross-modal Validation'
        },
        {
            'pair': ('tgt_semantic', 'mvclip_improvement'),
            'title': 'C. Semantic × MVCLIP Improvement',
            'interpretation': 'Target semantic quality\ncorrelates with MVCLIP gain',
            'category': 'Edit Quality'
        },
        {
            'pair': ('tgt_semantic', 'mvclip_preservation'),
            'title': 'D. Semantic × MVCLIP Preservation',
            'interpretation': 'Semantic quality linked to\nfeature preservation',
            'category': 'Edit Quality'
        },
    ]

    # Create 2x2 scatter plots
    for idx, corr_info in enumerate(key_correlations):
        ax = fig.add_subplot(2, 2, idx + 1)

        x_col, y_col = corr_info['pair']

        # Check if columns exist
        if x_col not in comparison_df.columns or y_col not in comparison_df.columns:
            ax.text(0.5, 0.5, f'Data not available\n({x_col}, {y_col})',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Get clean data (drop rows where either x or y is NaN)
        valid_mask = comparison_df[x_col].notna() & comparison_df[y_col].notna()
        x_data = comparison_df.loc[valid_mask, x_col]
        y_data = comparison_df.loc[valid_mask, y_col]

        if len(x_data) < 2:
            ax.text(0.5, 0.5, f'Insufficient data\n({x_col}, {y_col})',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Calculate correlation
        corr_val = x_data.corr(y_data)

        # Scatter by edit mode
        for mode in comparison_df['edit_mode'].unique():
            mode_mask = valid_mask & (comparison_df['edit_mode'] == mode)
            mode_x = comparison_df.loc[mode_mask, x_col]
            mode_y = comparison_df.loc[mode_mask, y_col]
            if len(mode_x) > 0:
                ax.scatter(mode_x, mode_y,
                          c=COLORS.get(mode, '#333333'),
                          label=EDIT_MODE_LABELS.get(mode, mode),
                          alpha=0.7, s=60, edgecolors='white', linewidth=0.5)

        # Add regression line
        z = np.polyfit(x_data.values, y_data.values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        ax.plot(x_line, p(x_line), '--', color='#555555', alpha=0.8, linewidth=1.5)

        # Labels and formatting
        label_map = {
            'mvclip_src_src': 'MVCLIP (Src→Src)',
            'mvclip_src_tgt': 'MVCLIP (Src→Tgt)',
            'mvclip_improvement': 'MVCLIP Improvement',
            'mvclip_preservation': 'MVCLIP Preservation',
            'src_consistency': 'Source Consistency (%)',
            'tgt_semantic': 'Target Semantic (%)',
        }
        x_label = label_map.get(x_col, x_col.replace('_', ' ').title())
        y_label = label_map.get(y_col, y_col.replace('_', ' ').title())

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(corr_info['title'], fontweight='bold', fontsize=12)

        # Add correlation value and interpretation
        corr_color = '#2E86AB' if corr_val > 0 else '#A23B72'
        ax.annotate(f'r = {corr_val:.2f}',
                   xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=14, fontweight='bold',
                   color=corr_color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=corr_color, alpha=0.9))

        # Add interpretation text
        ax.annotate(corr_info['interpretation'],
                   xy=(0.05, 0.05), xycoords='axes fraction',
                   ha='left', va='bottom', fontsize=9, style='italic',
                   color='#666666',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f8f8',
                            edgecolor='#cccccc', alpha=0.9))

        if idx == 0:
            ax.legend(loc='upper left', fontsize=9)

    fig.suptitle('Key Meaningful Correlations Analysis', fontsize=15, fontweight='bold', y=1.02)

    # Add category legend at bottom
    fig.text(0.5, -0.02,
             'Categories: Preservation Validation | Edit Magnitude Coherence | Cross-modal Validation',
             ha='center', fontsize=10, style='italic', color='#666666')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03b_focused_correlations.png')
    plt.close()
    print("  Saved: 03b_focused_correlations.png")


# ============================================================================
# Plot 3c: Two-Panel Correlation (A and B only)
# ============================================================================
def plot_focused_correlations_ac(comparison_df):
    """
    Two-panel visualization with key correlations:
    A. MVCLIP Internal Consistency (mvclip_src_src ↔ mvclip_src_tgt)
    B. Semantic × MVCLIP Improvement (tgt_semantic ↔ mvclip_improvement)
    """
    fig = plt.figure(figsize=(12, 5))

    key_correlations = [
        {
            'pair': ('mvclip_src_src', 'mvclip_src_tgt'),
            'title': 'A. MVCLIP Internal Consistency',
            'interpretation': 'Source MVCLIP features\nstable before/after editing',
        },
        {
            'pair': ('tgt_semantic', 'mvclip_improvement'),
            'title': 'B. Semantic × MVCLIP Improvement',
            'interpretation': 'Target semantic quality\ncorrelates with MVCLIP gain',
        },
    ]

    for idx, corr_info in enumerate(key_correlations):
        ax = fig.add_subplot(1, 2, idx + 1)

        x_col, y_col = corr_info['pair']

        if x_col not in comparison_df.columns or y_col not in comparison_df.columns:
            ax.text(0.5, 0.5, f'Data not available\n({x_col}, {y_col})',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        valid_mask = comparison_df[x_col].notna() & comparison_df[y_col].notna()
        x_data = comparison_df.loc[valid_mask, x_col]
        y_data = comparison_df.loc[valid_mask, y_col]

        if len(x_data) < 2:
            ax.text(0.5, 0.5, f'Insufficient data\n({x_col}, {y_col})',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        corr_val = x_data.corr(y_data)

        for mode in comparison_df['edit_mode'].unique():
            mode_mask = valid_mask & (comparison_df['edit_mode'] == mode)
            mode_x = comparison_df.loc[mode_mask, x_col]
            mode_y = comparison_df.loc[mode_mask, y_col]
            if len(mode_x) > 0:
                ax.scatter(mode_x, mode_y,
                          c=COLORS.get(mode, '#333333'),
                          label=EDIT_MODE_LABELS.get(mode, mode),
                          alpha=0.7, s=60, edgecolors='white', linewidth=0.5)

        z = np.polyfit(x_data.values, y_data.values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        ax.plot(x_line, p(x_line), '--', color='#555555', alpha=0.8, linewidth=1.5)

        label_map = {
            'mvclip_src_src': 'MVCLIP (Src→Src)',
            'mvclip_src_tgt': 'MVCLIP (Src→Tgt)',
            'mvclip_improvement': 'MVCLIP Improvement',
            'tgt_semantic': 'Target Semantic (%)',
        }
        x_label = label_map.get(x_col, x_col.replace('_', ' ').title())
        y_label = label_map.get(y_col, y_col.replace('_', ' ').title())

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(corr_info['title'], fontweight='bold', fontsize=12)

        corr_color = '#2E86AB' if corr_val > 0 else '#A23B72'
        ax.annotate(f'r = {corr_val:.2f}',
                   xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=14, fontweight='bold',
                   color=corr_color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=corr_color, alpha=0.9))

        # ax.annotate(corr_info['interpretation'],
        #            xy=(0.05, 0.05), xycoords='axes fraction',
        #            ha='left', va='bottom', fontsize=9, style='italic',
        #            color='#666666',
        #            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f8f8',
        #                     edgecolor='#cccccc', alpha=0.9))

        if idx == 0:
            ax.legend(loc='upper left', fontsize=9)

    fig.suptitle('Key Correlations Analysis', fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03c_focused_correlations_ac.png')
    plt.close()
    print("  Saved: 03c_focused_correlations_ac.png")


# ============================================================================
# Plot 4: Radar/Spider Chart
# ============================================================================
def plot_radar_chart(summary_df):
    """Radar chart for multi-dimensional edit mode comparison"""
    # Aggregate by edit mode
    metrics = ['avg_tgt_consistency', 'avg_tgt_semantic', 'avg_mvclip_improvement',
               'avg_src_consistency', 'avg_src_semantic']

    grouped = summary_df.groupby('edit_mode')[metrics].mean()

    # Normalize to [0, 1]
    normalized = (grouped - grouped.min()) / (grouped.max() - grouped.min() + 1e-8)

    # For LPIPS (not in metrics), we'd invert it - skip for now

    labels = ['Target\nConsistency', 'Target\nSemantic', 'MV-CLIP\nImprovement',
              'Source\nConsistency', 'Source\nSemantic']

    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Close the plot

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for mode in normalized.index:
        values = normalized.loc[mode].tolist()
        values += values[:1]  # Close the plot

        ax.plot(angles, values, 'o-', linewidth=2,
               color=COLORS.get(mode, '#333333'),
               label=EDIT_MODE_LABELS.get(mode, mode))
        ax.fill(angles, values, alpha=0.25, color=COLORS.get(mode, '#333333'))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    ax.set_ylim(0, 1)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Edit Mode Performance Radar', fontsize=14, fontweight='bold', y=1.08)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_radar_comparison.png')
    plt.close()
    print("  Saved: 04_radar_comparison.png")

# ============================================================================
# Plot 5: Per-Entry Performance Heatmap
# ============================================================================
def plot_entry_heatmap(comparison_df):
    """Heatmap showing performance across entries and experiments"""
    # Pivot to create heatmap data
    pivot_data = comparison_df.pivot_table(
        index='entry_name',
        columns='experiment',
        values='tgt_semantic',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlGnBu',
                linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Target Semantic (%)'})

    ax.set_xlabel('Experiment Configuration')
    ax.set_ylabel('Entry (Source → Target)')
    ax.set_title('Target Semantic Score by Entry and Experiment',
                fontsize=14, fontweight='bold', pad=20)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_entry_performance_heatmap.png')
    plt.close()
    print("  Saved: 05_entry_performance_heatmap.png")

# ============================================================================
# Plot 6: Trade-off Scatter Plot
# ============================================================================
def plot_tradeoff_scatter(comparison_df):
    """Scatter plot showing LPIPS vs Target Semantic trade-off"""
    fig, ax = plt.subplots(figsize=(10, 8))

    for mode in comparison_df['edit_mode'].unique():
        mode_data = comparison_df[comparison_df['edit_mode'] == mode]

        # Size by tau (handle negative tau)
        tau_values = mode_data['tau'].values
        sizes = (tau_values - tau_values.min() + 0.1) * 100 + 50

        scatter = ax.scatter(mode_data['lpips'], mode_data['tgt_semantic'],
                           c=COLORS.get(mode, '#333333'),
                           s=sizes, alpha=0.7,
                           label=EDIT_MODE_LABELS.get(mode, mode),
                           edgecolors='white', linewidth=0.5)

    ax.set_xlabel('LPIPS (Preservation, lower is better)')
    ax.set_ylabel('Target Semantic (%) (Editing Quality, higher is better)')
    ax.set_title('Trade-off: Preservation vs Editing Quality',
                fontsize=14, fontweight='bold')

    # Add legend
    ax.legend(title='Edit Mode', loc='best')

    # Add annotation for ideal region
    ax.annotate('Ideal Region\n(Low LPIPS, High Semantic)',
               xy=(0.02, 80), fontsize=10, style='italic',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_tradeoff_analysis.png')
    plt.close()
    print("  Saved: 06_tradeoff_analysis.png")

# ============================================================================
# Plot 7: Distribution Box/Violin Plots
# ============================================================================
def plot_distributions(comparison_df):
    """Violin plots showing metric distributions by edit mode"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = [
        ('lpips', 'LPIPS'),
        ('tgt_consistency', 'Target Consistency (%)'),
        ('tgt_semantic', 'Target Semantic (%)'),
        ('mvclip_improvement', 'MV-CLIP Improvement')
    ]

    for ax, (metric, label) in zip(axes.flat, metrics):
        # Create violin plot
        parts = ax.violinplot(
            [comparison_df[comparison_df['edit_mode'] == m][metric].dropna()
             for m in ['qk_img', 'qk_full', 'full'] if m in comparison_df['edit_mode'].unique()],
            showmeans=True, showmedians=True
        )

        # Color the violins
        colors_list = [COLORS.get(m, '#333333') for m in ['qk_img', 'qk_full', 'full']
                       if m in comparison_df['edit_mode'].unique()]
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_list[i])
            pc.set_alpha(0.7)

        modes_present = [m for m in ['qk_img', 'qk_full', 'full']
                        if m in comparison_df['edit_mode'].unique()]
        ax.set_xticks(range(1, len(modes_present) + 1))
        ax.set_xticklabels([EDIT_MODE_LABELS[m] for m in modes_present])
        ax.set_ylabel(label)
        ax.set_xlabel('Edit Mode')

    fig.suptitle('Metric Distributions by Edit Mode', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_metric_distributions.png')
    plt.close()
    print("  Saved: 07_metric_distributions.png")

# ============================================================================
# Plot 8: Best Parameters Summary
# ============================================================================
def plot_best_params(best_params):
    """Table-style visualization of best parameters"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Extract best params data
    bp = best_params.get('best_params', {})

    table_data = []
    metrics = ['best_lpips', 'best_tgt_consistency', 'best_tgt_semantic',
               'best_mvclip_tgt_tgt', 'best_mvclip_improvement']
    metric_labels = ['Best LPIPS', 'Best Target Consistency', 'Best Target Semantic',
                    'Best MV-CLIP (tgt-tgt)', 'Best MV-CLIP Improvement']

    for metric, label in zip(metrics, metric_labels):
        if metric in bp:
            info = bp[metric]
            table_data.append([
                label,
                EDIT_MODE_LABELS.get(info.get('edit_mode', 'N/A'), info.get('edit_mode', 'N/A')),
                f"{info.get('tau', 'N/A')}",
                f"{info.get('value', 0):.4f}"
            ])

    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=['Metric', 'Best Edit Mode', 'Best Tau', 'Value'],
            loc='center',
            cellLoc='center',
            colColours=['#E6E6E6'] * 4
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # Style the cells
        for i in range(len(table_data) + 1):
            for j in range(4):
                cell = table[i, j]
                cell.set_edgecolor('#CCCCCC')
                if i == 0:
                    cell.set_text_props(fontweight='bold')
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(color='white', fontweight='bold')

    ax.set_title('Best Parameters Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_best_params_summary.png')
    plt.close()
    print("  Saved: 08_best_params_summary.png")

# ============================================================================
# Plot 9: Entry Comparison Bars
# ============================================================================
def plot_entry_comparison_bars(comparison_df):
    """Faceted bar chart comparing entries"""
    entries = comparison_df['entry_name'].unique()
    n_entries = len(entries)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flat

    for i, entry in enumerate(entries[:6]):  # Max 6 entries
        ax = axes[i]
        entry_data = comparison_df[comparison_df['entry_name'] == entry]

        # Group by edit mode
        grouped = entry_data.groupby('edit_mode')['tgt_semantic'].mean()

        modes = grouped.index.tolist()
        values = grouped.values
        colors = [COLORS.get(m, '#333333') for m in modes]

        bars = ax.bar([EDIT_MODE_LABELS.get(m, m) for m in modes], values,
                     color=colors, edgecolor='white', linewidth=1)

        ax.set_title(entry, fontsize=11, fontweight='bold')
        ax.set_ylabel('Target Semantic (%)')
        ax.set_ylim(0, 100)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Target Semantic by Entry and Edit Mode', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '09_entry_comparison_bars.png')
    plt.close()
    print("  Saved: 09_entry_comparison_bars.png")

# ============================================================================
# Plot 10: Category Comparison
# ============================================================================
def plot_category_comparison(summary_df):
    """Faceted comparison across categories"""
    categories = summary_df['category'].unique()

    fig, axes = plt.subplots(1, len(categories), figsize=(5*len(categories), 5), sharey=True)

    if len(categories) == 1:
        axes = [axes]

    category_labels = {
        'diff_tau': 'Different Tau',
        'diff_editmode_diff_identity': 'Diff Mode, Diff Identity',
        'diff_editmode_same_identity': 'Diff Mode, Same Identity'
    }

    for ax, cat in zip(axes, categories):
        cat_data = summary_df[summary_df['category'] == cat]

        # Group by edit mode
        grouped = cat_data.groupby('edit_mode')['avg_tgt_semantic'].mean()

        modes = grouped.index.tolist()
        values = grouped.values
        colors = [COLORS.get(m, '#333333') for m in modes]

        bars = ax.bar([EDIT_MODE_LABELS.get(m, m) for m in modes], values,
                     color=colors, edgecolor='white', linewidth=1)

        ax.set_title(category_labels.get(cat, cat), fontsize=11, fontweight='bold')
        ax.set_xlabel('Edit Mode')
        if ax == axes[0]:
            ax.set_ylabel('Average Target Semantic (%)')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9)

    fig.suptitle('Category-wise Performance Comparison', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '10_category_facet_comparison.png')
    plt.close()
    print("  Saved: 10_category_facet_comparison.png")

# ============================================================================
# Plot 10b: Category Comparison (Right Two Only)
# ============================================================================
def plot_category_comparison_right2(summary_df):
    """Faceted comparison showing only diff_editmode categories"""
    # Only include the two rightmost categories
    target_categories = ['diff_editmode_diff_identity', 'diff_editmode_same_identity']
    filtered_df = summary_df[summary_df['category'].isin(target_categories)]

    if len(filtered_df) == 0:
        print("  Skipped: 10b_category_facet_right2.png (no matching categories)")
        return

    categories = [c for c in target_categories if c in filtered_df['category'].unique()]

    fig, axes = plt.subplots(1, len(categories), figsize=(5*len(categories), 5), sharey=True)

    if len(categories) == 1:
        axes = [axes]

    category_labels = {
        'diff_editmode_diff_identity': 'Diff Mode, Diff Identity',
        'diff_editmode_same_identity': 'Diff Mode, Same Identity'
    }

    for ax, cat in zip(axes, categories):
        cat_data = filtered_df[filtered_df['category'] == cat]

        # Group by edit mode
        grouped = cat_data.groupby('edit_mode')['avg_tgt_semantic'].mean()

        modes = grouped.index.tolist()
        values = grouped.values
        colors = [COLORS.get(m, '#333333') for m in modes]

        bars = ax.bar([EDIT_MODE_LABELS.get(m, m) for m in modes], values,
                     color=colors, edgecolor='white', linewidth=1)

        ax.set_title(category_labels.get(cat, cat), fontsize=11, fontweight='bold')
        ax.set_xlabel('Edit Mode')
        if ax == axes[0]:
            ax.set_ylabel('Average Target Semantic (%)')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9)

    fig.suptitle('Category-wise Performance Comparison', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '10b_category_facet_right2.png')
    plt.close()
    print("  Saved: 10b_category_facet_right2.png")

# ============================================================================
# Main Execution
# ============================================================================
def main():
    print("=" * 60)
    print("P2P Batch Results Visualization")
    print("=" * 60)

    # Setup
    setup_style()
    print("\n[1/3] Loading data...")
    best_params, summary_df, comparison_df = load_data()
    print(f"  Loaded {len(summary_df)} summary rows, {len(comparison_df)} comparison rows")

    # Generate plots
    print("\n[2/3] Generating visualizations...")

    plot_edit_mode_comparison(summary_df)
    plot_tau_effect(summary_df)
    plot_correlation_heatmap(comparison_df)
    plot_focused_correlations(comparison_df)
    plot_focused_correlations_ac(comparison_df)
    plot_radar_chart(summary_df)
    plot_entry_heatmap(comparison_df)
    plot_tradeoff_scatter(comparison_df)
    plot_distributions(comparison_df)
    plot_best_params(best_params)
    plot_entry_comparison_bars(comparison_df)
    plot_category_comparison(summary_df)
    plot_category_comparison_right2(summary_df)

    # Summary
    print("\n[3/3] Complete!")
    print(f"\nAll visualizations saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
