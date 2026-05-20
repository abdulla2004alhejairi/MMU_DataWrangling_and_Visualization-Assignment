#Part 2 — Feature Selection


import pandas as pd
from pathlib import Path

WDBC_DATA = r"C:\Users\anson\Downloads\Assignment\wdbc.data"
WDBC_NAMES = r"C:\Users\anson\Downloads\Assignment\wdbc.names"

# Column names for WDBC (from UCI repository)
cols = ["id","diagnosis",
        "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
        "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
        "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"]

wdbc = pd.read_csv(WDBC_DATA, header=None, names=cols)
wdbc.head()


import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline

# Prepare X, y
y = (wdbc["diagnosis"] == "M").astype(int)   # Malignant -> 1, Benign -> 0
X = wdbc.drop(columns=["id","diagnosis"]).copy()

# Simple imputation (dataset normally has no NAs; if any, fill with median)
X = X.fillna(X.median())
print("X shape:", X.shape, "y positive rate:", y.mean())

#Forward Selection (Linear Regression, k_features = 4)

from itertools import combinations

def forward_selection(X, y, k_features=4, cv=5, random_state=42):
    remaining = list(X.columns)
    selected = []
    best_scores = []
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    for _ in range(k_features):
        best_feat, best_score = None, -np.inf
        for feat in remaining:
            feats = selected + [feat]
            pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
            score = cross_val_score(pipe, X[feats], y, cv=kfold, scoring="r2").mean()
            if score > best_score:
                best_score, best_feat = score, feat
        selected.append(best_feat)
        remaining.remove(best_feat)
        best_scores.append(best_score)
        print(f"Step {len(selected)}: added '{best_feat}' | CV R^2 = {best_score:.4f}")
    return selected, best_scores

selected_feats, cv_scores = forward_selection(X, y, k_features=4)
print("\nSelected features (k=4):", selected_feats)

# Filter Method (Correlation with target > 0.5)

corrs = {}
for c in X.columns:
    corrs[c] = np.corrcoef(X[c], y)[0,1]
corrs = pd.Series(corrs).sort_values(key=np.abs, ascending=False)
selected_filter = list(corrs[corrs.abs() > 0.5].index)
print("Top correlations:\n", corrs.head(10))
print("\nSelected by filter (|corr| > 0.5):", selected_filter)


# Embedded Method — Lasso Regression

from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
import pandas as pd

# Silence only the specific convergence warning if it still appears after tweaks
warnings.filterwarnings("ignore", category=ConvergenceWarning)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cross-validated alpha with higher iteration budget and finer alpha grid
lasso_cv = LassoCV(
    cv=5,
    n_alphas=200,       # denser search grid
    max_iter=20000,     # larger budget for convergence
    tol=1e-4,           # standard tolerance (can relax to 2e-4 if needed)
    random_state=42
).fit(X_scaled, y)

lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=20000, tol=1e-4).fit(X_scaled, y)
coef = pd.Series(lasso.coef_, index=X.columns)

selected_lasso = list(coef[coef != 0].index)
print("Chosen alpha:", lasso_cv.alpha_)
print("Selected by Lasso (non-zero coef):", selected_lasso[:20],
      ("... ({} total)".format(len(selected_lasso)) if len(selected_lasso) > 20 else ""))

# Optional: show top 15 features by absolute coefficient magnitude
print("\nTop by |coef|:")
print(coef.abs().sort_values(ascending=False).head(15))
