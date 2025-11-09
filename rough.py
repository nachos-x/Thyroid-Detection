#!/usr/bin/env python3
# --------------------------------------------------------------
# Hypothyroid Diagnosis – hypothyroid.csv (FULLY FIXED)
# --------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import pinv
from numpy.linalg import cond as condition
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings, os
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

# --------------------------------------------------------------
# 1. Load local dataset
# --------------------------------------------------------------
path = "hypothyroid.csv"
if not os.path.exists(path):
    print(f"Error: '{path}' not found. Put the CSV in the same folder.")
    exit(1)

df = pd.read_csv(path)
print(f"Loaded: {df.shape[0]} samples, {df.shape[1]} columns")

# --------------------------------------------------------------
# 2. Target & raw features
# --------------------------------------------------------------
df.replace('?', np.nan, inplace=True)                     # '?' → NaN
df['binaryClass'] = (df['binaryClass'] == 'N').astype(int) # N=1 (disease), P=0
y = df['binaryClass'].values
X_raw = df.drop('binaryClass', axis=1)

# --------------------------------------------------------------
# 3. ONE-HOT + FORCE ALL TO NUMERIC (the critical fix)
# --------------------------------------------------------------
# numeric lab columns that exist in the file
lab_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
for c in lab_cols:
    if c in X_raw.columns:
        X_raw[c] = pd.to_numeric(X_raw[c], errors='coerce')

# one-hot encode everything that is still object
cat_cols = X_raw.select_dtypes(include='object').columns
X = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)

# ----> FORCE EVERY column to numeric (0/1 or float) <----
for col in X.columns:
    if X[col].dtype == 'object':                     # leftover t/f strings
        X[col] = X[col].map(lambda v: 1 if str(v).strip().lower() in {'t','true','1','yes'} else 0)
    X[col] = pd.to_numeric(X[col], errors='coerce')  # final safety
X.fillna(0, inplace=True)                            # any parsing NaNs → 0
X = X.astype('float32')                              # PyTorch-ready
print(f"After ONE-HOT + numeric conversion: {X.shape[1]} features, dtype={X.dtypes.unique()}")

# --------------------------------------------------------------
# 4. Impute *only* the lab values (TBG will be dropped automatically)
# --------------------------------------------------------------
available_labs = [c for c in lab_cols if c in X.columns]
if available_labs:
    imputer = SimpleImputer(strategy='mean')
    X[available_labs] = imputer.fit_transform(X[available_labs])
else:
    print("Warning: No lab columns to impute.")
X_filled = X.copy()
print(f"X_filled ready: {X_filled.shape}  dtype={X_filled.dtypes.unique()}")

# --------------------------------------------------------------
# 5. Dimensionality reduction
# --------------------------------------------------------------
# 5a. Inner-similarity (drop |ρ|>0.8)
corr = X_filled.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.8)]
X_inner = X_filled.drop(to_drop, axis=1)
print(f"Inner → {X_inner.shape[1]} features (dropped {len(to_drop)})")

# 5b. Target-similarity (top-10 by mutual info)
mi = mutual_info_classif(X_inner, y, random_state=42)
k = min(10, X_inner.shape[1])
top_idx = mi.argsort()[-k:][::-1]
X_target = X_inner.iloc[:, top_idx]
print(f"Target → {X_target.shape[1]} features")
X_reduced = X_target

# --------------------------------------------------------------
# 6. Size reduction – hierarchical clustering
# --------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)

Z = linkage(X_scaled, method='weighted', metric='euclidean')
clusters = fcluster(Z, t=1.5, criterion='distance')   # tweak t if you want more/less reduction

unique = np.unique(clusters)
keep_idx = []
for cid in unique:
    mask = clusters == cid
    centroid = X_scaled[mask].mean(axis=0)
    dists = np.linalg.norm(X_scaled[mask] - centroid, axis=1)
    keep_idx.append(np.where(mask)[0][dists.argmin()])

X_final = X_reduced.iloc[keep_idx].reset_index(drop=True)
y_final = y[keep_idx]
print(f"Size reduction → {X_final.shape[0]} samples (kept {len(keep_idx)})")

# --------------------------------------------------------------
# 7. Train / test split (now guaranteed float32)
# --------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X_final.values, y_final, test_size=0.2, stratify=y_final, random_state=42)

X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)
X_te_t = torch.tensor(X_te, dtype=torch.float32)
y_te_t = torch.tensor(y_te, dtype=torch.float32).view(-1, 1)

# --------------------------------------------------------------
# 8. ALGORITHMS (unchanged)
# --------------------------------------------------------------
# 8.1 Batch Least Squares (BLS) – RIDGE (stable & high accuracy)
lambda_reg = 0.01
XTX = X_tr.T @ X_tr + lambda_reg * np.eye(X_tr.shape[1])
XTy = X_tr.T @ y_tr
W_bls = np.linalg.solve(XTX, XTy)    # <-- stable solver
b_bls = (y_tr - X_tr @ W_bls).mean()
logits = X_te @ W_bls + b_bls
y_pred_bls = (torch.sigmoid(torch.tensor(logits)) > 0.5).float()
acc_bls = accuracy_score(y_te, y_pred_bls)
print(f"[BLS]  Test acc: {acc_bls:.1%}")

# 8.2 Iterative Neural Network (INN)
class INN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 1)
        nn.init.xavier_uniform_(self.lin.weight)
    def forward(self, x):
        return torch.sigmoid(self.lin(x))

inn = INN(X_tr.shape[1])
opt = optim.SGD(inn.parameters(), lr=0.05, momentum=0.9)
criterion = nn.BCELoss()
for _ in range(300):
    opt.zero_grad()
    loss = criterion(inn(X_tr_t), y_tr_t)
    loss.backward()
    opt.step()
with torch.no_grad():
    y_pred_inn = (inn(X_te_t) > 0.5).float()
acc_inn = accuracy_score(y_te_t, y_pred_inn)
print(f"[INN]  Test acc: {acc_inn:.1%}")

# 8.3 Least Squares with Linear Constraints (LSLC)
class LSLC(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, 1))
        self.b = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return torch.sigmoid(x @ self.w + self.b)

lslc = LSLC(X_tr.shape[1])
opt_lslc = optim.SGD(lslc.parameters(), lr=0.02)
for _ in range(400):
    opt_lslc.zero_grad()
    loss = nn.BCELoss()(lslc(X_tr_t), y_tr_t)
    loss.backward()
    opt_lslc.step()
    with torch.no_grad():
        lslc.w.clamp_(min=0.0)               # non-negativity
with torch.no_grad():
    y_pred_lslc = (lslc(X_te_t) > 0.5).float()
acc_lslc = accuracy_score(y_te_t, y_pred_lslc)
print(f"[LSLC] Test acc: {acc_lslc:.1%}")

# 8.4 Imperialistic Competitive Algorithm (ICA)
def sigmoid(z): return 1/(1+np.exp(-z))

def mse_error(X, y, p):
    w, b = p[:-1], p[-1]
    return np.mean((sigmoid(X @ w + b) - y)**2)

def ica_optimize(X, y, pop=60, iters=30, dim=None):
    if dim is None:
        dim = X.shape[1]  # Use actual feature count
    population = np.random.uniform(-2, 2, (pop, dim+1))
    costs = np.array([mse_error(X, y, p) for p in population])
    for _ in range(iters):
        idx = np.argsort(costs)
        population, costs = population[idx], costs[idx]
        n_imp = max(1, pop//10)
        imperialists = population[:n_imp]
        imp_costs    = costs[:n_imp]
        new_pop = population.copy()
        for i in range(n_imp, pop):
            imp = imperialists[np.random.randint(n_imp)]
            beta = np.random.uniform(0.1, 0.9)
            new_pop[i] = population[i] + beta*(imp - population[i])
            if np.random.rand() < 0.05:
                new_pop[i] += np.random.normal(0, 0.2, new_pop[i].shape)
        new_costs = np.array([mse_error(X, y, p) for p in new_pop])
        for i in range(n_imp):
            best_col_idx = n_imp + np.argmin(new_costs[n_imp:])
            if new_costs[best_col_idx] < imp_costs[i]:
                imperialists[i], new_pop[best_col_idx] = new_pop[best_col_idx], imperialists[i]
                imp_costs[i], new_costs[best_col_idx] = new_costs[best_col_idx], imp_costs[i]
        population, costs = new_pop, new_costs
    best = population[np.argmin(costs)]
    w, b = best[:-1], best[-1]
    pred = sigmoid(X_te @ w + b)
    y_pred = (pred > 0.5).astype(float)
    acc = accuracy_score(y_te, y_pred)
    print(f"[ICA]  Test acc: {acc:.1%}")
    return best

# CALL IT WITH REDUCED FEATURES
ica_optimize(X_tr, y_tr, dim=X_tr.shape[1])

# --------------------------------------------------------------
# 9. Final summary
# --------------------------------------------------------------
print("\n=== FINAL SUMMARY ===")
print(f"BLS   : {acc_bls:.1%}")
print(f"INN   : {acc_inn:.1%}")
print(f"LSLC  : {acc_lslc:.1%}")
# ICA printed inside function