import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt
print("[OK] N4BP2-ecDNA Analyzer started")
# Generate synthetic data np.random.seed(42) n_samples = 1000 n_genes = 10
gene_names = ['N4BP2', 'TP53', 'BRCA1', 'EGFR', 'MYC', 'KRAS', 'PTEN', 'ATM', 'CDKN2A', 'RB1']
n_samples = 1000
n_genes = 10
X = np.random.rand(n_samples, n_genes) * 10
prob = 1 / (1 + np.exp(- (X[:, 0] - 5)))
prob = 1 / (1 + np.exp(- (X[:, 0] - 5)))
y = (prob > 0.5).astype(int)
print("Target distribution:", np.bincount(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X_train, y_train)
y_pred = model.predict(X_test) 
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")
importance = model.feature_importances_ 
for name, imp in zip(gene_names, importance):print(f"{name}: {imp:.4f}")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:50])
shap.summary_plot(shap_values, X_test[:50], feature_names=gene_names, show=False) 
plt.savefig("shap_summary.png")
print("[OK] SHAP plot saved as shap_summary.png")