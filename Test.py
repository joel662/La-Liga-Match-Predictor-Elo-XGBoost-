import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10})

# Load data
df_scenarios = pd.read_csv("all_scenarios_results.csv")
df_ablation = pd.read_csv("ablation_results.csv")
df_rolling = pd.read_csv("rolling_backtest_results.csv")

# 1. Best Accuracy per Scenario
fig, ax = plt.subplots(figsize=(10, 6))
best_acc = df_scenarios.loc[df_scenarios.groupby('Scenario')['% Correct'].idxmax()]
# Sort by accuracy for better visualization
best_acc = best_acc.sort_values('% Correct', ascending=False)

sns.barplot(data=best_acc, x='% Correct', y='Scenario', palette='viridis', ax=ax)
ax.set_title('Best Model Accuracy per Scenario (2025/26 Holdout)', fontsize=14, weight='bold')
ax.set_xlabel('Accuracy (%)', fontsize=12)
ax.set_ylabel('League / Scenario', fontsize=12)
for i, v in enumerate(best_acc['% Correct']):
    ax.text(v + 0.2, i, f"{v:.1f}%", va='center')
ax.set_xlim(45, 60)
plt.tight_layout()
plt.savefig('best_accuracy.png', dpi=150)
plt.show()

# 2. Accuracy vs. Draw Recall Trade-off
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df_scenarios, x='Draw Recall %', y='% Correct', hue='Scenario', alpha=0.7, s=80, ax=ax)
ax.set_title('Accuracy vs. Draw Recall Trade-off', fontsize=14, weight='bold')
ax.set_xlabel('Draw Recall (%)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('acc_vs_draw_recall.png', dpi=150)
plt.show()

# 3. Ablation Study - Feature Set Comparison
fig, ax = plt.subplots(figsize=(12, 6))
# Filter to just the 5 main leagues for clarity
abl_leagues = df_ablation[~df_ablation['Scenario'].str.contains('Combined')].copy()

# UNIFY HYBRID COLORS: Remove the specific weights from the labels 
# so seaborn assigns them all the same category and color in the legend
abl_leagues['Feature Set'] = abl_leagues['Feature Set'].apply(
    lambda x: 'Full ML + Elo blend' if 'Elo blend' in str(x) else x
)

sns.barplot(data=abl_leagues, x='Scenario', y='Accuracy %', hue='Feature Set', palette='Set2', ax=ax)
ax.set_title('Ablation Study: Feature Set Contribution to Accuracy', fontsize=14, weight='bold')
ax.set_xlabel('League', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(35, 60)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Feature Set")
plt.tight_layout()
plt.savefig('ablation_chart.pdf', dpi=300)
# 4. Rolling Backtest Stability
fig, ax = plt.subplots(figsize=(10, 6))
best_rolling = df_rolling.loc[df_rolling.groupby(['Scenario', 'Season'])['% Correct'].idxmax()]
# Exclude combined leagues to focus on the 5 individual ones requested by backtest
sns.lineplot(data=best_rolling, x='Season', y='% Correct', hue='Scenario', marker='o', markersize=8, linewidth=2, ax=ax)
ax.set_title('Temporal Stability: Best Accuracy Across Historical Seasons', fontsize=14, weight='bold')
ax.set_xlabel('Season', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('rolling_stability.png', dpi=150)
plt.show()