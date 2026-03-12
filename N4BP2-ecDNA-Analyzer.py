# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines import CoxPHFitter
import os
import sys

print("="*70)
print("🔬 ACC (Adrenocortical Carcinoma) Survival Analysis for N4BP2")
print("="*70)

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# ------------------------------------------------------------
# 1. Check file existence
# ------------------------------------------------------------
expr_file = "TCGA.ACC.sampleMap_HiSeqV2.gz"
clin_file = "survival_ACC_survival.txt"

if not os.path.exists(expr_file):
    print(f"❌ Error: {expr_file} not found!")
    sys.exit()
if not os.path.exists(clin_file):
    print(f"❌ Error: {clin_file} not found!")
    sys.exit()

print("✅ All files found.")

# ------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------
expr = pd.read_csv(expr_file, compression='gzip', sep='\t', index_col=0)
clin = pd.read_csv(clin_file, sep='\t')
print(f"✅ Expression: {expr.shape[0]} genes, {expr.shape[1]} samples")
print(f"✅ Clinical: {clin.shape[0]} patients")

# ------------------------------------------------------------
# 3. Find N4BP2
# ------------------------------------------------------------
n4bp2_genes = [g for g in expr.index if 'N4BP2' in g.upper()]
if not n4bp2_genes:
    print("❌ N4BP2 not found")
    sys.exit()
gene = n4bp2_genes[0]
print(f"✅ Using gene: {gene}")

# ------------------------------------------------------------
# 4. Split by median
# ------------------------------------------------------------
expr_values = expr.loc[gene]
threshold = expr_values.median()
high_patients = expr_values[expr_values > threshold].index
low_patients = expr_values[expr_values <= threshold].index
print(f"✅ High: {len(high_patients)} patients")
print(f"✅ Low: {len(low_patients)} patients")

# ------------------------------------------------------------
# 5. Merge clinical data
# ------------------------------------------------------------
clin_high = clin[clin['sample'].isin(high_patients)].copy()
clin_low = clin[clin['sample'].isin(low_patients)].copy()
clin_high['group'] = 'High'
clin_low['group'] = 'Low'
merged = pd.concat([clin_high, clin_low], ignore_index=True)

# Remove missing values
merged = merged.dropna(subset=['OS.time', 'OS'])
print(f"✅ After cleaning: {merged.shape[0]} patients")

if merged.shape[0] == 0:
    print("❌ No patients left")
    sys.exit()

clin_high_clean = merged[merged['group'] == 'High']
clin_low_clean = merged[merged['group'] == 'Low']
print(f"✅ High after cleaning: {len(clin_high_clean)}")
print(f"✅ Low after cleaning: {len(clin_low_clean)}")

# ------------------------------------------------------------
# 6. Survival analysis
# ------------------------------------------------------------
high_t = clin_high_clean['OS.time']
high_e = clin_high_clean['OS']
low_t = clin_low_clean['OS.time']
low_e = clin_low_clean['OS']

# Log-rank test
results = logrank_test(high_t, low_t, event_observed_A=high_e, event_observed_B=low_e)
p_value = results.p_value
print(f"\n📊 Log-rank p-value = {p_value:.4f}")

# Median survival
kmf_high = KaplanMeierFitter()
kmf_low = KaplanMeierFitter()
kmf_high.fit(high_t, high_e)
kmf_low.fit(low_t, low_e)

median_high = kmf_high.median_survival_time_
median_low = kmf_low.median_survival_time_
print(f"📈 Median survival - High: {median_high:.0f} days")
print(f"📈 Median survival - Low: {median_low:.0f} days")

# Cox model
merged_cox = merged.copy()
merged_cox['group_binary'] = (merged_cox['group'] == 'High').astype(int)
cph = CoxPHFitter()
cph.fit(merged_cox[['OS.time', 'OS', 'group_binary']], duration_col='OS.time', event_col='OS')
hr = np.exp(cph.params_.iloc[0])
hr_ci_lower = np.exp(cph.confidence_intervals_.iloc[0, 0])
hr_ci_upper = np.exp(cph.confidence_intervals_.iloc[0, 1])
print(f"⚕️ Hazard Ratio: {hr:.2f} (95% CI: {hr_ci_lower:.2f}-{hr_ci_upper:.2f})")

# ------------------------------------------------------------
# 7. Kaplan-Meier plot with risk table
# ------------------------------------------------------------
fig, (ax, ax_table) = plt.subplots(2, 1, figsize=(10, 8),
                                    gridspec_kw={'height_ratios': [3, 1]})

# High group
kmf_high.plot_survival_function(ax=ax, color='#E64B35', show_censors=True,
                                censor_styles={'ms': 6, 'marker': 's'},
                                ci_alpha=0.2, label=f'High N4BP2 (n={len(high_t)})')

# Low group
kmf_low.plot_survival_function(ax=ax, color='#4DBBD5', show_censors=True,
                               censor_styles={'ms': 6, 'marker': 's'},
                               ci_alpha=0.2, label=f'Low N4BP2 (n={len(low_t)})')

# Statistical annotation
stats_text = f'p = {p_value:.4f}\nHR = {hr:.2f} (95% CI: {hr_ci_lower:.2f}-{hr_ci_upper:.2f})'
ax.text(0.6, 0.15, stats_text, transform=ax.transAxes,
        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.set_title('Overall Survival in ACC by N4BP2 Expression', fontweight='bold')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Survival Probability')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

# Risk table
add_at_risk_counts(kmf_high, kmf_low, ax=ax_table, xticks=ax.get_xticks())
ax_table.set_xlabel('')
ax_table.set_ylabel('At risk', fontweight='bold')
ax_table.grid(False)

plt.tight_layout()
plt.savefig('survival_ACC_improved.png', dpi=300, bbox_inches='tight')
plt.savefig('survival_ACC_improved.pdf', format='pdf', bbox_inches='tight')
print("✅ Plots saved (PNG + PDF)")
plt.show()

# =============================================================================
# OV (Ovarian Cancer) Analysis
# =============================================================================
print("\n" + "="*70)
print("🔬 OV (Ovarian Cancer) Survival Analysis for N4BP2")
print("="*70)

# ------------------------------------------------------------
# 1. Check OV files
# ------------------------------------------------------------
expr_ov_file = "TCGA.OV.sampleMap_HiSeqV2.gz"
clin_ov_file = "survival_OV_survival.txt"

if not os.path.exists(expr_ov_file):
    print(f"❌ Error: {expr_ov_file} not found!")
    sys.exit()
if not os.path.exists(clin_ov_file):
    print(f"❌ Error: {clin_ov_file} not found!")
    sys.exit()

print("✅ OV files found.")

# ------------------------------------------------------------
# 2. Load OV data
# ------------------------------------------------------------
expr_ov = pd.read_csv(expr_ov_file, compression='gzip', sep='\t', index_col=0)
clin_ov = pd.read_csv(clin_ov_file, sep='\t')
print(f"✅ OV Expression: {expr_ov.shape[0]} genes, {expr_ov.shape[1]} samples")
print(f"✅ OV Clinical: {clin_ov.shape[0]} patients")

# ------------------------------------------------------------
# 3. Find N4BP2 in OV
# ------------------------------------------------------------
n4bp2_ov = [g for g in expr_ov.index if 'N4BP2' in g.upper()]
if not n4bp2_ov:
    print("❌ N4BP2 not found in OV")
    sys.exit()
gene_ov = n4bp2_ov[0]
print(f"✅ Using gene: {gene_ov}")

# ------------------------------------------------------------
# 4. Split by median in OV
# ------------------------------------------------------------
expr_values_ov = expr_ov.loc[gene_ov]
threshold_ov = expr_values_ov.median()
high_patients_ov = expr_values_ov[expr_values_ov > threshold_ov].index
low_patients_ov = expr_values_ov[expr_values_ov <= threshold_ov].index
print(f"✅ OV High: {len(high_patients_ov)} patients")
print(f"✅ OV Low: {len(low_patients_ov)} patients")

# ------------------------------------------------------------
# 5. Merge OV clinical data
# ------------------------------------------------------------
clin_high_ov = clin_ov[clin_ov['sample'].isin(high_patients_ov)].copy()
clin_low_ov = clin_ov[clin_ov['sample'].isin(low_patients_ov)].copy()
clin_high_ov['group'] = 'High'
clin_low_ov['group'] = 'Low'
merged_ov = pd.concat([clin_high_ov, clin_low_ov], ignore_index=True)

# Remove missing values (CRITICAL FOR OV)
merged_ov = merged_ov.dropna(subset=['OS.time', 'OS'])
print(f"✅ OV After cleaning: {merged_ov.shape[0]} patients")

if merged_ov.shape[0] == 0:
    print("❌ No OV patients left")
    sys.exit()

clin_high_clean_ov = merged_ov[merged_ov['group'] == 'High']
clin_low_clean_ov = merged_ov[merged_ov['group'] == 'Low']
print(f"✅ OV High after cleaning: {len(clin_high_clean_ov)}")
print(f"✅ OV Low after cleaning: {len(clin_low_clean_ov)}")

# ------------------------------------------------------------
# 6. OV Survival analysis
# ------------------------------------------------------------
high_t_ov = clin_high_clean_ov['OS.time']
high_e_ov = clin_high_clean_ov['OS']
low_t_ov = clin_low_clean_ov['OS.time']
low_e_ov = clin_low_clean_ov['OS']

# Log-rank test
results_ov = logrank_test(high_t_ov, low_t_ov, event_observed_A=high_e_ov, event_observed_B=low_e_ov)
p_value_ov = results_ov.p_value
print(f"\n📊 OV Log-rank p-value = {p_value_ov:.4f}")

# Median survival
kmf_high_ov = KaplanMeierFitter()
kmf_low_ov = KaplanMeierFitter()
kmf_high_ov.fit(high_t_ov, high_e_ov)
kmf_low_ov.fit(low_t_ov, low_e_ov)

median_high_ov = kmf_high_ov.median_survival_time_
median_low_ov = kmf_low_ov.median_survival_time_
print(f"📈 OV Median survival - High: {median_high_ov:.0f} days")
print(f"📈 OV Median survival - Low: {median_low_ov:.0f} days")

# Cox model
merged_cox_ov = merged_ov.copy()
merged_cox_ov['group_binary'] = (merged_cox_ov['group'] == 'High').astype(int)
cph_ov = CoxPHFitter()
cph_ov.fit(merged_cox_ov[['OS.time', 'OS', 'group_binary']], duration_col='OS.time', event_col='OS')
hr_ov = np.exp(cph_ov.params_.iloc[0])
hr_ci_lower_ov = np.exp(cph_ov.confidence_intervals_.iloc[0, 0])
hr_ci_upper_ov = np.exp(cph_ov.confidence_intervals_.iloc[0, 1])
print(f"⚕️ OV Hazard Ratio: {hr_ov:.2f} (95% CI: {hr_ci_lower_ov:.2f}-{hr_ci_upper_ov:.2f})")

# ------------------------------------------------------------
# 7. OV Kaplan-Meier plot
# ------------------------------------------------------------
fig_ov, (ax_ov, ax_table_ov) = plt.subplots(2, 1, figsize=(10, 8),
                                             gridspec_kw={'height_ratios': [3, 1]})

# High group
kmf_high_ov.plot_survival_function(ax=ax_ov, color='#E64B35', show_censors=True,
                                    censor_styles={'ms': 6, 'marker': 's'},
                                    ci_alpha=0.2, label=f'High N4BP2 OV (n={len(high_t_ov)})')

# Low group
kmf_low_ov.plot_survival_function(ax=ax_ov, color='#4DBBD5', show_censors=True,
                                   censor_styles={'ms': 6, 'marker': 's'},
                                   ci_alpha=0.2, label=f'Low N4BP2 OV (n={len(low_t_ov)})')

# Statistical annotation
stats_text_ov = f'p = {p_value_ov:.4f}\nHR = {hr_ov:.2f} (95% CI: {hr_ci_lower_ov:.2f}-{hr_ci_upper_ov:.2f})'
ax_ov.text(0.6, 0.15, stats_text_ov, transform=ax_ov.transAxes,
           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax_ov.set_title('Overall Survival in OV by N4BP2 Expression', fontweight='bold')
ax_ov.set_xlabel('Time (days)')
ax_ov.set_ylabel('Survival Probability')
ax_ov.legend()
ax_ov.grid(True, linestyle='--', alpha=0.6)

# Risk table
add_at_risk_counts(kmf_high_ov, kmf_low_ov, ax=ax_table_ov, xticks=ax_ov.get_xticks())
ax_table_ov.set_xlabel('')
ax_table_ov.set_ylabel('At risk', fontweight='bold')
ax_table_ov.grid(False)

plt.tight_layout()
plt.savefig('survival_OV_improved.png', dpi=300, bbox_inches='tight')
plt.savefig('survival_OV_improved.pdf', format='pdf', bbox_inches='tight')
print("✅ OV plots saved (PNG + PDF)")
plt.show()