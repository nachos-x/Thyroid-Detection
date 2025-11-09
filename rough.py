#!/usr/bin/env python3
# --------------------------------------------------------------
# Hypothyroid Diagnosis – hypothyroid.csv (PAPER-ACCURATE INN + METRICS)
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
from sklearn.metrics import accuracy_score, precision_score, recall_score
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
df.replace('?', np.nan, inplace=True)
df['binaryClass'] = (df['binaryClass'] == 'N').astype(int)  # N=1 (disease), P=0
y = df['binaryClass'].values
X_raw = df.drop('binaryClass', axis=1)

# --------------------------------------------------------------
# 3. ONE-HOT + FORCE ALL TO NUMERIC
# --------------------------------------------------------------
lab_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
for c in lab_cols:
    if c in X_raw.columns:
        X_raw[c] = pd.to_numeric(X_raw[c], errors='coerce')

cat_cols = X_raw.select_dtypes(include='object').columns
X = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].map(lambda v: 1 if str(v).strip().lower() in {'t','true','1','yes'} else 0)
    X[col] = pd.to_numeric(X[col], errors='coerce')
X.fillna(0, inplace=True)
X = X.astype('float32')
print(f"After ONE-HOT + numeric conversion: {X.shape[1]} features")

# --------------------------------------------------------------
# 4. Impute lab values (WHOLE-BATCH MEAN)
# --------------------------------------------------------------
available_labs = [c for c in lab_cols if c in X.columns]
if available_labs:
    imputer = SimpleImputer(strategy='mean')
    X[available_labs] = imputer.fit_transform(X[available_labs])
X_filled = X.copy()
print(f"X_filled ready: {X_filled.shape}")

# --------------------------------------------------------------
# 5. Dimensionality reduction
# --------------------------------------------------------------
corr = X_filled.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.8)]
X_inner = X_filled.drop(to_drop, axis=1)
print(f"Inner → {X_inner.shape[1]} features (dropped {len(to_drop)})")

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
clusters = fcluster(Z, t=1.5, criterion='distance')
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
# 7. Train / test split
# --------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X_final.values, y_final, test_size=0.2, stratify=y_final, random_state=42)

X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)
X_te_t = torch.tensor(X_te, dtype=torch.float32)
y_te_t = torch.tensor(y_te, dtype=torch.float32).view(-1, 1)

# --------------------------------------------------------------
# 8. ALGORITHMS
# --------------------------------------------------------------

# 8.1 Batch Least Squares (BLS)
lambda_reg = 0.01
XTX = X_tr.T @ X_tr + lambda_reg * np.eye(X_tr.shape[1])
XTy = X_tr.T @ y_tr
W_bls = np.linalg.solve(XTX, XTy)
b_bls = (y_tr - X_tr @ W_bls).mean()
logits_test = X_te @ W_bls + b_bls
y_pred_bls = (torch.sigmoid(torch.tensor(logits_test)) > 0.5).float()
acc_bls = accuracy_score(y_te, y_pred_bls)
logits_train = X_tr @ W_bls + b_bls
y_train_pred_bls = (torch.sigmoid(torch.tensor(logits_train)) > 0.5).float()
train_acc_bls = accuracy_score(y_tr, y_train_pred_bls)
print(f"[BLS] Train acc: {train_acc_bls:.1%} | Test acc: {acc_bls:.1%}")

# --------------------------------------------------------------
# 8.2 PAPER-ACCURATE INCREMENTAL NEURAL NETWORK (INN)
# --------------------------------------------------------------

class PageHinkley:
    def __init__(self, delta=0.005, lambda_=20, alpha=0.99):
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.m_t = 0
        self.sum = 0
        self.x_mean = 0
        self.t = 0
        self.drift_detected = False

    def update(self, error):
        self.t += 1
        self.x_mean = self.alpha * self.x_mean + (1 - self.alpha) * error
        self.sum = self.alpha * self.sum + (error - self.x_mean - self.delta)
        self.m_t = max(self.m_t, self.sum)
        self.drift_detected = (self.m_t - self.sum) > self.lambda_
        return self.drift_detected


class INN:
    def __init__(self, E1=0.3, E2=0.5, sigma_init=1.0, lr=0.1,
                 ph_delta=0.005, ph_lambda=20, ph_alpha=0.99):
        self.E1 = E1
        self.E2 = E2
        self.sigma_init = sigma_init
        self.lr = lr
        self.ph = PageHinkley(delta=ph_delta, lambda_=ph_lambda, alpha=ph_alpha)
        self.neurons = []  # [center, sigma, w_out, err_sum, hits]

    def _gaussian(self, x, c, sigma):
        return np.exp(-np.sum((x - c)**2) / (2 * sigma**2 + 1e-8))

    def _find_winner(self, x):
        if not self.neurons:
            return None
        dists = [np.linalg.norm(x - n[0]) for n in self.neurons]
        return np.argmin(dists)

    def _add_neuron(self, x, y):
        center = x.copy()
        sigma = self.sigma_init
        w_out = y.item() if isinstance(y, np.ndarray) else y
        self.neurons.append([center, sigma, w_out, 0.0, 0])

    def _adapt_winner(self, idx, x, y, pred):
        n = self.neurons[idx]
        c, sigma, w_out, err_sum, hits = n
        hits += 1
        err_sum += abs(y - pred)
        # Update center
        c = (1 - 1/hits) * c + (1/hits) * x
        # Update output weight
        error = y - pred
        w_out += self.lr * error
        self.neurons[idx] = [c, sigma, w_out, err_sum, hits]

    def predict(self, X):
        if not self.neurons:
            return np.full(len(X), 0.5)
        preds = []
        for x in X:
            if not self.neurons:
                preds.append(0.5)
                continue
            winner_idx = self._find_winner(x)
            c, sigma, w_out, _, _ = self.neurons[winner_idx]
            act = self._gaussian(x, c, sigma)
            pred = act * w_out
            preds.append(pred)
        return np.array(preds).reshape(-1, 1)

    def update(self, x, y):
        x = x.ravel()
        y = y.item() if isinstance(y, np.ndarray) else y

        if len(self.neurons) == 0:
            self._add_neuron(x, y)
            return

        winner_idx = self._find_winner(x)
        c, sigma, w_out, _, _ = self.neurons[winner_idx]
        act = self._gaussian(x, c, sigma)
        pred = act * w_out
        error = abs(y - pred)

        # Drift detection
        if self.ph.update(error):
            print("Drift detected! Resetting network...")
            self.neurons = []
            self.ph.reset()
            self._add_neuron(x, y)
            return

        dist_to_winner = np.linalg.norm(x - c)
        if error > self.E1 and dist_to_winner > self.E2:
            self._add_neuron(x, y)
        else:
            self._adapt_winner(winner_idx, x, y, pred)


# Train INN online
inn = INN(E1=0.3, E2=0.5, sigma_init=1.0, lr=0.1)
indices = np.random.permutation(len(X_tr))
for i in indices:
    inn.update(X_tr[i:i+1], y_tr[i:i+1])

# --------------------------------------------------------------
# PRINT INN ARCHITECTURE
# --------------------------------------------------------------
# def print_inn_architecture(inn):
#     print("\n" + "="*60)
#     print("INCREMENTAL NEURAL NETWORK (INN) ARCHITECTURE")
#     print("="*60)
#     print(f"Input dimension      : {X_tr.shape[1]}")
#     print(f"Number of neurons    : {len(inn.neurons)}")
#     print(f"Output activation    : RBF × linear (Gaussian kernel)")
#     print("-"*60)
#     if len(inn.neurons) == 0:
#         print("No hidden neurons grown yet.")
#     else:
#         print(f"{'ID':>3} | {'Center (norm)':>12} | {'Sigma':>8} | {'Weight':>8} | {'Hits':>6} | {'AvgErr':>8}")
#         print("-"*60)
#         for idx, (center, sigma, w_out, err_sum, hits) in enumerate(inn.neurons):
#             center_norm = np.linalg.norm(center)
#             avg_err = err_sum / hits if hits > 0 else 0.0
#             print(f"{idx:>3} | {center_norm:>12.4f} | {sigma:>8.3f} | {w_out:>8.4f} | {hits:>6} | {avg_err:>8.4f}")
#     print("="*60 + "\n")

# # Call it
# print_inn_architecture(inn)

# Predict
y_pred_inn_train = (inn.predict(X_tr) > 0.5).astype(float)
y_pred_inn_test = (inn.predict(X_te) > 0.5).astype(float)
train_acc_inn = accuracy_score(y_tr, y_pred_inn_train)
acc_inn = accuracy_score(y_te, y_pred_inn_test)
print(f"[INN] Train acc: {train_acc_inn:.1%} | Test acc: {acc_inn:.1%} | Neurons: {len(inn.neurons)}")

# --------------------------------------------------------------
# 8.3 Least Squares with Linear Constraints (LSLC)
# --------------------------------------------------------------
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
        lslc.w.clamp_(min=0.0)

with torch.no_grad():
    y_pred_lslc = (lslc(X_te_t) > 0.5).float()
    acc_lslc = accuracy_score(y_te_t, y_pred_lslc)
    y_train_pred_lslc = (lslc(X_tr_t) > 0.5).float()
    train_acc_lslc = accuracy_score(y_tr_t, y_train_pred_lslc)
print(f"[LSLC] Train acc: {train_acc_lslc:.1%} | Test acc: {acc_lslc:.1%}")

# --------------------------------------------------------------
# 8.4 Imperialistic Competitive Algorithm (ICA)
# --------------------------------------------------------------
def sigmoid(z): return 1/(1+np.exp(-np.clip(z, -500, 500)))
def mse_error(X, y, p):
    w, b = p[:-1].reshape(-1,1), p[-1]
    return np.mean((sigmoid(X @ w + b) - y.reshape(-1,1))**2)

def ica_optimize(X_train, y_train, X_test, y_test, pop=60, iters=30, dim=None):
    if dim is None: dim = X_train.shape[1]
    population = np.random.uniform(-2, 2, (pop, dim+1))
    costs = np.array([mse_error(X_train, y_train, p) for p in population])
    for _ in range(iters):
        idx = np.argsort(costs)
        population, costs = population[idx], costs[idx]
        n_imp = max(1, pop//10)
        imperialists = population[:n_imp]
        imp_costs = costs[:n_imp]
        new_pop = population.copy()
        for i in range(n_imp, pop):
            imp = imperialists[np.random.randint(n_imp)]
            beta = np.random.uniform(0.1, 0.9)
            new_pop[i] = population[i] + beta * (imp - population[i])
            if np.random.rand() < 0.05:
                new_pop[i] += np.random.normal(0, 0.2, new_pop[i].shape)
        new_costs = np.array([mse_error(X_train, y_train, p) for p in new_pop])
        for i in range(n_imp):
            best_col_idx = n_imp + np.argmin(new_costs[n_imp:])
            if new_costs[best_col_idx] < imp_costs[i]:
                imperialists[i], new_pop[best_col_idx] = new_pop[best_col_idx], imperialists[i]
                imp_costs[i], new_costs[best_col_idx] = new_costs[best_col_idx], new_costs[best_col_idx]
        population, costs = new_pop, new_costs
    best = population[np.argmin(costs)]
    w, b = best[:-1].reshape(-1,1), best[-1]
    pred_test = sigmoid(X_test @ w + b)
    y_pred_test = (pred_test > 0.5).astype(float).flatten()
    test_acc = accuracy_score(y_test, y_pred_test)
    pred_train = sigmoid(X_train @ w + b)
    y_pred_train = (pred_train > 0.5).astype(float).flatten()
    train_acc = accuracy_score(y_train, y_pred_train)
    print(f"[ICA] Train acc: {train_acc:.1%} | Test acc: {test_acc:.1%}")
    return train_acc, test_acc, y_pred_test

train_acc_ica, acc_ica, y_pred_ica = ica_optimize(X_tr, y_tr, X_te, y_te, dim=X_tr.shape[1])

# --------------------------------------------------------------
# 9. Final summary
# --------------------------------------------------------------
print("\n=== FINAL SUMMARY ===")
print(f"BLS : Train {train_acc_bls:.1%} | Test {acc_bls:.1%}")
print(f"INN : Train {train_acc_inn:.1%} | Test {acc_inn:.1%}")
print(f"LSLC : Train {train_acc_lslc:.1%} | Test {acc_lslc:.1%}")
print(f"ICA : Train {train_acc_ica:.1%} | Test {acc_ica:.1%}")

# --------------------------------------------------------------
# 10. Precision & Recall
# --------------------------------------------------------------
print("\n=== PRECISION & RECALL ===")
models = [
    ("BLS", y_te, y_pred_bls.numpy().flatten()),
    ("INN", y_te, y_pred_inn_test.flatten()),
    ("LSLC", y_te_t.numpy().flatten(), y_pred_lslc.numpy().flatten()),
    ("ICA", y_te, y_pred_ica)
]
for name, y_true, y_pred in models:
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    print(f"{name:5} → Precision: {prec:.3f} | Recall: {rec:.3f}")