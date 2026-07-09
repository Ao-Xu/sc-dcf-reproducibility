import os
import time
import math
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_linnerud
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(ROOT, "data_cache")
os.makedirs(DATA, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
})

COLORS = {
    "SC-DCF-LS": "#1f77b4",
    "Independent DCF-LS": "#ff7f0e",
    "kNN": "#7f7f7f",
    "MO-RF": "#2ca02c",
    "MO-Ridge": "#9467bd",
    "PLS": "#8c564b",
}
MARKERS = {
    "SC-DCF-LS": "o",
    "Independent DCF-LS": "s",
    "kNN": "^",
    "MO-RF": "D",
    "MO-Ridge": "v",
    "PLS": "P",
}


def afpc(X, k, first_index=0):
    n = X.shape[0]
    k = int(max(1, min(k, n)))
    start = time.perf_counter()
    centers = [first_index]
    diff = X - X[first_index]
    min_d2 = np.sum(diff * diff, axis=1)
    while len(centers) < k:
        idx = int(np.argmax(min_d2))
        if idx in centers:
            break
        centers.append(idx)
        diff = X - X[idx]
        d2 = np.sum(diff * diff, axis=1)
        min_d2 = np.minimum(min_d2, d2)
    elapsed = time.perf_counter() - start
    radius = float(np.sqrt(np.max(min_d2))) if len(min_d2) else 0.0
    return np.array(centers, dtype=int), elapsed, radius


def choose_k(n, intrinsic_dim=2, max_k=80):
    return int(max(4, min(max_k, math.ceil(n ** (intrinsic_dim / (2.0 + intrinsic_dim))))))


def center_features(X, centers, feature_mode="full"):
    # DCF-inspired shared-center Lipschitz dictionary:
    # constant + triangular radial features + negative distance features.
    D = np.sqrt(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    R = np.median(D) + 1e-8
    tent = np.maximum(0.0, R - D)
    negdist = -D
    if feature_mode == "full":
        blocks = [np.ones((X.shape[0], 1)), tent, negdist]
    elif feature_mode == "tent":
        blocks = [np.ones((X.shape[0], 1)), tent]
    elif feature_mode == "negdist":
        blocks = [np.ones((X.shape[0], 1)), negdist]
    else:
        raise ValueError("unknown feature_mode={}".format(feature_mode))
    return np.hstack(blocks)


def estimate_shared_peak_mib(n, d, m, k):
    q = 1 + 2 * k
    bytes_est = 8.0 * (
        n * d +        # standardized covariates
        n * m +        # standardized responses
        4 * n * k +    # distance, tent, negative-distance, and final radial dictionary blocks
        n * q +        # final dictionary matrix
        q * q + q * m  # ridge normal equations and coefficients
    )
    return bytes_est / (1024.0 ** 2)


class SharedCenterDCFPrototype(object):
    def __init__(self, intrinsic_dim=2, alpha=1e-2, max_k=80, feature_mode="full", center_rule="afpc", random_state=0):
        self.intrinsic_dim = intrinsic_dim
        self.alpha = alpha
        self.max_k = max_k
        self.feature_mode = feature_mode
        self.center_rule = center_rule
        self.random_state = random_state

    def fit(self, X, Y):
        self.scaler_x = StandardScaler().fit(X)
        self.scaler_y = StandardScaler().fit(Y)
        Xs = self.scaler_x.transform(X)
        Ys = self.scaler_y.transform(Y)
        self.k = choose_k(X.shape[0], self.intrinsic_dim, self.max_k)
        start_centers = time.perf_counter()
        if self.center_rule == "afpc":
            idx, self.afpc_time_, self.cover_radius_ = afpc(Xs, self.k)
            self.center_indices_ = idx
            self.centers_ = Xs[idx]
        elif self.center_rule == "random":
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(Xs.shape[0], size=self.k, replace=False)
            self.center_indices_ = idx
            self.centers_ = Xs[idx]
            D = np.sqrt(((Xs[:, None, :] - self.centers_[None, :, :]) ** 2).sum(axis=2))
            self.cover_radius_ = float(np.max(np.min(D, axis=1)))
            self.afpc_time_ = time.perf_counter() - start_centers
        elif self.center_rule == "kmeans":
            km = KMeans(n_clusters=self.k, random_state=self.random_state, n_init=5, max_iter=100)
            km.fit(Xs)
            self.center_indices_ = np.array([], dtype=int)
            self.centers_ = km.cluster_centers_
            D = np.sqrt(((Xs[:, None, :] - self.centers_[None, :, :]) ** 2).sum(axis=2))
            self.cover_radius_ = float(np.max(np.min(D, axis=1)))
            self.afpc_time_ = time.perf_counter() - start_centers
        else:
            raise ValueError("unknown center_rule={}".format(self.center_rule))
        dict_start = time.perf_counter()
        Phi = center_features(Xs, self.centers_, self.feature_mode)
        self.dict_time_ = time.perf_counter() - dict_start
        start = time.perf_counter()
        self.model_ = Ridge(alpha=self.alpha, fit_intercept=False)
        self.model_.fit(Phi, Ys)
        self.fit_time_ = time.perf_counter() - start
        self.total_time_ = self.afpc_time_ + self.dict_time_ + self.fit_time_
        self.active_pieces_ = int(np.sum(np.linalg.norm(self.model_.coef_[:, 1:1+self.k], axis=0) > 1e-8))
        return self

    def predict(self, X):
        Xs = self.scaler_x.transform(X)
        Phi = center_features(Xs, self.centers_, self.feature_mode)
        Ys = self.model_.predict(Phi)
        return self.scaler_y.inverse_transform(Ys)


class IndependentCenterDCFPrototype(object):
    def __init__(self, intrinsic_dim=2, alpha=1e-2, max_k=80, feature_mode="full"):
        self.intrinsic_dim = intrinsic_dim
        self.alpha = alpha
        self.max_k = max_k
        self.feature_mode = feature_mode

    def fit(self, X, Y):
        self.scaler_x = StandardScaler().fit(X)
        self.scaler_y = StandardScaler().fit(Y)
        Xs = self.scaler_x.transform(X)
        Ys = self.scaler_y.transform(Y)
        self.models_ = []
        self.centers_ = []
        self.afpc_time_ = 0.0
        self.dict_time_ = 0.0
        self.fit_time_ = 0.0
        self.cover_radius_ = 0.0
        self.k = choose_k(X.shape[0], self.intrinsic_dim, self.max_k)
        for a in range(Y.shape[1]):
            idx, t_afpc, radius = afpc(Xs, self.k)
            centers = Xs[idx]
            dict_start = time.perf_counter()
            Phi = center_features(Xs, centers, self.feature_mode)
            self.dict_time_ = getattr(self, "dict_time_", 0.0) + (time.perf_counter() - dict_start)
            start = time.perf_counter()
            model = Ridge(alpha=self.alpha, fit_intercept=False)
            model.fit(Phi, Ys[:, a])
            self.fit_time_ += time.perf_counter() - start
            self.afpc_time_ += t_afpc
            self.cover_radius_ = max(self.cover_radius_, radius)
            self.models_.append(model)
            self.centers_.append(centers)
        self.total_time_ = self.afpc_time_ + self.dict_time_ + self.fit_time_
        self.active_pieces_ = self.k
        return self

    def predict(self, X):
        Xs = self.scaler_x.transform(X)
        preds = []
        for model, centers in zip(self.models_, self.centers_):
            Phi = center_features(Xs, centers, self.feature_mode)
            preds.append(model.predict(Phi))
        Ys = np.vstack(preds).T
        return self.scaler_y.inverse_transform(Ys)


def make_lowdim_data(n, m=5, d_star=2, d=20, noise=0.15, corr=0.0, seed=0):
    rng = np.random.RandomState(seed)
    Z = rng.uniform(-1.0, 1.0, size=(n, d_star))
    A = rng.normal(size=(d_star, d))
    A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-8
    X = Z.dot(A) + 0.03 * rng.normal(size=(n, d))
    F = []
    for a in range(m):
        z0 = Z[:, 0]
        z1 = Z[:, 1 % d_star]
        val = np.sin((a + 1) * z0) + 0.5 * np.cos((a + 2) * z1)
        if d_star > 2:
            val += 0.25 * Z[:, 2]
        F.append(val)
    F = np.vstack(F).T
    Sigma = (1.0 - corr) * np.eye(m) + corr * np.ones((m, m))
    L = np.linalg.cholesky(Sigma)
    E = rng.normal(size=(n, m)).dot(L.T) * noise
    return X, F + E


def summed_mse(y_true, y_pred):
    return float(np.mean(np.sum((y_true - y_pred) ** 2, axis=1)))


def max_coord_mse(y_true, y_pred):
    return float(np.max(np.mean((y_true - y_pred) ** 2, axis=0)))


def average_r2(y_true, y_pred):
    sse = np.sum((y_true - y_pred) ** 2, axis=0)
    sst = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2, axis=0)
    keep = sst > 1e-8
    if not np.any(keep):
        return np.nan
    return float(np.mean(1.0 - sse[keep] / sst[keep]))


def add_metrics(row, Yte, pred, Ytr):
    row["summed_mse"] = summed_mse(Yte, pred)
    row["max_coord_mse"] = max_coord_mse(Yte, pred)
    sy = StandardScaler().fit(Ytr)
    Yte_s = sy.transform(Yte)
    pred_s = sy.transform(pred)
    row["std_summed_mse"] = summed_mse(Yte_s, pred_s)
    row["avg_r2"] = average_r2(Yte, pred)
    return row


def eval_methods(X, Y, intrinsic_dim=2, seed=0, include_baselines=True):
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.35, random_state=seed)
    methods = [
        ("SC-DCF-LS", SharedCenterDCFPrototype(intrinsic_dim=intrinsic_dim, alpha=1e-2)),
        ("Independent DCF-LS", IndependentCenterDCFPrototype(intrinsic_dim=intrinsic_dim, alpha=1e-2)),
    ]
    if include_baselines:
        methods.extend([
            ("kNN", KNeighborsRegressor(n_neighbors=max(3, int(np.sqrt(Xtr.shape[0]))))),
            ("MO-RF", RandomForestRegressor(n_estimators=25, random_state=seed, n_jobs=1)),
            ("MO-Ridge", Ridge(alpha=1.0)),
            ("PLS", PLSRegression(n_components=max(1, min(5, Xtr.shape[1], Ytr.shape[1])))),
        ])
    rows = []
    for name, model in methods:
        start = time.perf_counter()
        if name in ("kNN", "MO-RF", "MO-Ridge", "PLS"):
            sx = StandardScaler().fit(Xtr)
            sy = StandardScaler().fit(Ytr)
            model.fit(sx.transform(Xtr), sy.transform(Ytr))
            pred = sy.inverse_transform(model.predict(sx.transform(Xte)))
            total = time.perf_counter() - start
            afpc_time = 0.0
            fit_time = total
            k = 0
            radius = np.nan
            active = 0
        else:
            model.fit(Xtr, Ytr)
            pred = model.predict(Xte)
            total = model.total_time_
            afpc_time = model.afpc_time_
            fit_time = model.fit_time_
            k = model.k
            radius = model.cover_radius_
            active = model.active_pieces_
        row = {
            "method": name,
            "total_time": total,
            "afpc_time": afpc_time,
            "fit_time": fit_time,
            "K": k,
            "cover_radius": radius,
            "active_pieces": active,
        }
        rows.append(add_metrics(row, Yte, pred, Ytr))
    return rows


def experiment_rate():
    rows = []
    for n in [1000, 2500, 5000, 10000]:
        for rep in range(6):
            print("rate n={} rep={}".format(n, rep), flush=True)
            X, Y = make_lowdim_data(n, m=5, d_star=2, d=20, noise=0.15, seed=1000+n+rep)
            for r in eval_methods(X, Y, intrinsic_dim=2, seed=rep, include_baselines=True):
                r.update({"experiment": "rate", "n": n, "m": 5, "rep": rep})
                rows.append(r)
    return pd.DataFrame(rows)


def experiment_scaling():
    rows = []
    for m in [5, 20, 50, 100]:
        for rep in range(5):
            print("scaling m={} rep={}".format(m, rep), flush=True)
            X, Y = make_lowdim_data(10000, m=m, d_star=2, d=50, noise=0.15, seed=2000+10*m+rep)
            Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.35, random_state=rep)
            model = SharedCenterDCFPrototype(intrinsic_dim=2, alpha=1e-2)
            model.fit(Xtr, Ytr)
            pred = model.predict(Xte)
            base = {
                "K": model.k,
                "cover_radius": model.cover_radius_,
                "active_pieces": model.active_pieces_,
                "experiment": "scaling",
                "n": 10000,
                "m": m,
                "rep": rep,
            }
            shared = dict(base)
            shared.update({
                "method": "SC-DCF-LS",
                "total_time": model.total_time_,
                "afpc_time": model.afpc_time_,
                "fit_time": model.fit_time_,
            })
            rows.append(add_metrics(shared, Yte, pred, Ytr))

            indep_model = IndependentCenterDCFPrototype(intrinsic_dim=2, alpha=1e-2)
            indep_model.fit(Xtr, Ytr)
            indep_pred = indep_model.predict(Xte)
            independent = dict(base)
            independent.update({
                "method": "Independent DCF-LS",
                "total_time": indep_model.total_time_,
                "afpc_time": indep_model.afpc_time_,
                "fit_time": indep_model.fit_time_,
                "cover_radius": indep_model.cover_radius_,
                "active_pieces": indep_model.active_pieces_,
            })
            rows.append(add_metrics(independent, Yte, indep_pred, Ytr))
    return pd.DataFrame(rows)


def experiment_correlation():
    rows = []
    for corr in [0.0, 0.5, 0.9]:
        for rep in range(5):
            print("correlation corr={} rep={}".format(corr, rep), flush=True)
            X, Y = make_lowdim_data(5000, m=20, d_star=2, d=50, noise=0.2, corr=corr, seed=3000+int(100*corr)+rep)
            for r in eval_methods(X, Y, intrinsic_dim=2, seed=rep, include_baselines=True):
                r.update({"experiment": "correlation", "n": 5000, "m": 20, "corr": corr, "rep": rep})
                rows.append(r)
    return pd.DataFrame(rows)


def eval_fixed_split_methods(Xtr, Xte, Ytr, Yte, rep, intrinsic_dim=2):
    methods = [
        ("SC-DCF-LS", SharedCenterDCFPrototype(intrinsic_dim=intrinsic_dim, alpha=1e-1, max_k=80)),
        ("Independent DCF-LS", IndependentCenterDCFPrototype(intrinsic_dim=intrinsic_dim, alpha=1e-1, max_k=80)),
        ("kNN", KNeighborsRegressor(n_neighbors=max(3, int(np.sqrt(Xtr.shape[0]))))),
        ("MO-RF", RandomForestRegressor(n_estimators=60, random_state=rep, n_jobs=1)),
        ("MO-Ridge", Ridge(alpha=1.0)),
        ("PLS", PLSRegression(n_components=max(1, min(5, Xtr.shape[1], Ytr.shape[1])))),
    ]
    rows = []
    for name, model in methods:
        start = time.perf_counter()
        if name in ("kNN", "MO-RF", "MO-Ridge", "PLS"):
            sx = StandardScaler().fit(Xtr)
            sy = StandardScaler().fit(Ytr)
            model.fit(sx.transform(Xtr), sy.transform(Ytr))
            pred = sy.inverse_transform(model.predict(sx.transform(Xte)))
            total = time.perf_counter() - start
            afpc_time = 0.0
            fit_time = total
            k = 0
        else:
            model.fit(Xtr, Ytr)
            pred = model.predict(Xte)
            total = model.total_time_
            afpc_time = model.afpc_time_
            fit_time = model.fit_time_
            k = model.k
        row = {
            "method": name,
            "total_time": total,
            "afpc_time": afpc_time,
            "fit_time": fit_time,
            "K": k,
            "rep": rep,
        }
        rows.append(add_metrics(row, Yte, pred, Ytr))
    return rows


def download_cached(url, filename):
    path = os.path.join(DATA, filename)
    if not os.path.exists(path):
        data = urllib.request.urlopen(url, timeout=90).read()
        with open(path, "wb") as f:
            f.write(data)
    return path


def frame_from_arff(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype.kind in "OSU":
            values = df[col].map(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            codes, _ = pd.factorize(values)
            df[col] = codes.astype(float)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_mulan_arff(files, target_count):
    base = "https://downloads.sourceforge.net/project/mulan/datasets/multi-target%20regression%20datasets/"
    if isinstance(files, str):
        files = [files]
    frames = []
    for filename in files:
        path = download_cached(base + filename, filename)
        frames.append(frame_from_arff(path))
    df = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    values = df.values.astype(float)
    X = values[:, :-target_count]
    Y = values[:, -target_count:]
    valid_y = np.isfinite(Y).all(axis=1)
    X = X[valid_y]
    Y = Y[valid_y]
    keep = np.mean(np.isfinite(X), axis=0) >= 0.75
    X = X[:, keep]
    X = SimpleImputer(strategy="median").fit_transform(X)
    return X, Y


def experiment_real_mulan(dataset_key, display_name, files, target_count, intrinsic_dim, folds=5):
    X, Y = load_mulan_arff(files, target_count)
    rows = []
    kf = KFold(n_splits=folds, shuffle=True, random_state=101 + len(display_name))
    for rep, (tr, te) in enumerate(kf.split(X)):
        print("{} fold={}".format(dataset_key, rep), flush=True)
        for r in eval_fixed_split_methods(X[tr], X[te], Y[tr], Y[te], rep, intrinsic_dim=intrinsic_dim):
            r.update({
                "experiment": dataset_key,
                "dataset": display_name,
                "n": X.shape[0],
                "d": X.shape[1],
                "m": Y.shape[1],
            })
            rows.append(r)
    return pd.DataFrame(rows)


def experiment_real_linnerud():
    data = load_linnerud()
    X = data.data.astype(float)
    Y = data.target.astype(float)
    rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for rep, (tr, te) in enumerate(kf.split(X)):
        print("real_linnerud fold={}".format(rep), flush=True)
        for r in eval_fixed_split_methods(X[tr], X[te], Y[tr], Y[te], rep, intrinsic_dim=2):
            r.update({"experiment": "real_linnerud", "dataset": "Linnerud", "n": X.shape[0], "d": X.shape[1], "m": Y.shape[1]})
            rows.append(r)
    return pd.DataFrame(rows)


def experiment_real_energy():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    df = pd.read_excel(url)
    X = df.iloc[:, :-2].values.astype(float)
    Y = df.iloc[:, -2:].values.astype(float)
    rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=7)
    for rep, (tr, te) in enumerate(kf.split(X)):
        print("real_energy fold={}".format(rep), flush=True)
        for r in eval_fixed_split_methods(X[tr], X[te], Y[tr], Y[te], rep, intrinsic_dim=3):
            r.update({"experiment": "real_energy", "dataset": "Energy", "n": X.shape[0], "d": X.shape[1], "m": Y.shape[1]})
            rows.append(r)
    return pd.DataFrame(rows)


def load_communities_crime_unnormalized():
    url = "https://archive.ics.uci.edu/static/public/211/communities%2Band%2Bcrime%2Bunnormalized.zip"
    cache = os.path.join(DATA, "communities_crime_unnormalized.zip")
    if not os.path.exists(cache):
        data = urllib.request.urlopen(url, timeout=60).read()
        with open(cache, "wb") as f:
            f.write(data)
    with zipfile.ZipFile(cache) as zf:
        with zf.open("CommViolPredUnnormalizedData.txt") as fh:
            df = pd.read_csv(fh, header=None, na_values="?")

    # Columns 0--4 contain community identifiers and administrative fields.
    # Columns 128 onward are crime counts/rates, so they are excluded from X
    # to avoid leakage into the multi-output crime-rate targets.
    X = df.iloc[:, 5:128].apply(pd.to_numeric, errors="coerce").values.astype(float)
    target_cols = [134, 136, 138, 140, 142, 144, 145, 146]
    Y = df.iloc[:, target_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    valid_y = np.isfinite(Y).all(axis=1)
    X = X[valid_y]
    Y = Y[valid_y]
    keep = np.mean(np.isfinite(X), axis=0) >= 0.75
    X = X[:, keep]
    X = SimpleImputer(strategy="median").fit_transform(X)
    return X, Y


def experiment_real_communities():
    X, Y = load_communities_crime_unnormalized()
    rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=23)
    for rep, (tr, te) in enumerate(kf.split(X)):
        print("real_communities fold={}".format(rep), flush=True)
        for r in eval_fixed_split_methods(X[tr], X[te], Y[tr], Y[te], rep, intrinsic_dim=4):
            r.update({"experiment": "real_communities", "dataset": "Communities", "n": X.shape[0], "d": X.shape[1], "m": Y.shape[1]})
            rows.append(r)
    return pd.DataFrame(rows)


def experiment_ablation():
    rows = []
    configs = [
        ("AFPC-full-K40", "afpc", "full", 40),
        ("AFPC-full-K80", "afpc", "full", 80),
        ("AFPC-full-K120", "afpc", "full", 120),
        ("AFPC-tent-K80", "afpc", "tent", 80),
        ("AFPC-negdist-K80", "afpc", "negdist", 80),
        ("random-full-K80", "random", "full", 80),
        ("kmeans-full-K80", "kmeans", "full", 80),
    ]
    for rep in range(5):
        print("ablation rep={}".format(rep), flush=True)
        X, Y = make_lowdim_data(5000, m=20, d_star=2, d=50, noise=0.15, seed=5000+rep)
        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.35, random_state=rep)
        for label, rule, mode, max_k in configs:
            model = SharedCenterDCFPrototype(intrinsic_dim=2, alpha=1e-2, max_k=max_k,
                                             feature_mode=mode, center_rule=rule, random_state=rep)
            model.fit(Xtr, Ytr)
            pred = model.predict(Xte)
            row = {
                "experiment": "ablation",
                "config": label,
                "method": "SC-DCF-LS",
                "center_rule": rule,
                "feature_mode": mode,
                "max_k": max_k,
                "n": X.shape[0],
                "m": Y.shape[1],
                "rep": rep,
                "total_time": model.total_time_,
                "afpc_time": model.afpc_time_,
                "fit_time": model.fit_time_,
                "K": model.k,
                "cover_radius": model.cover_radius_,
                "active_pieces": model.active_pieces_,
            }
            rows.append(add_metrics(row, Yte, pred, Ytr))
    return pd.DataFrame(rows)


def experiment_afpc_sanity():
    rows = []
    for m in [5, 20]:
        for rep in range(5):
            print("afpc_sanity m={} rep={}".format(m, rep), flush=True)
            X, Y = make_lowdim_data(10000, m=m, d_star=2, d=50, noise=0.15, seed=7000+10*m+rep)
            Xtr, _, _, _ = train_test_split(X, Y, test_size=0.35, random_state=rep)
            sx = StandardScaler().fit(Xtr)
            Xs = sx.transform(Xtr)
            k = choose_k(Xtr.shape[0], 2, 80)
            _, single_time, _ = afpc(Xs, k)
            start = time.perf_counter()
            for _ in range(m):
                afpc(Xs, k)
            actual = time.perf_counter() - start
            projected = m * single_time
            rows.append({
                "experiment": "afpc_sanity",
                "m": m,
                "rep": rep,
                "K": k,
                "single_afpc_time": single_time,
                "projected_afpc_time": projected,
                "actual_repeated_afpc_time": actual,
                "actual_projected_ratio": actual / max(projected, 1e-12),
            })
    return pd.DataFrame(rows)


def fit_shared_normal_equations(X, Y, k=80, alpha=1e-2):
    sx = StandardScaler().fit(X)
    sy = StandardScaler().fit(Y)
    Xs = sx.transform(X)
    Ys = sy.transform(Y)
    idx, afpc_time, radius = afpc(Xs, k)
    centers = Xs[idx]
    start = time.perf_counter()
    Phi = center_features(Xs, centers, "full")
    dict_time = time.perf_counter() - start
    start = time.perf_counter()
    gram = Phi.T.dot(Phi)
    rhs = Phi.T.dot(Ys)
    gram.flat[::gram.shape[0] + 1] += alpha * X.shape[0]
    coef = np.linalg.solve(gram, rhs)
    solve_time = time.perf_counter() - start
    return {
        "afpc_time": afpc_time,
        "dict_time": dict_time,
        "fit_time": solve_time,
        "total_time": afpc_time + dict_time + solve_time,
        "K": k,
        "cover_radius": radius,
        "q": Phi.shape[1],
        "estimated_peak_array_mib": estimate_shared_peak_mib(X.shape[0], X.shape[1], Y.shape[1], k),
        "coef_norm": float(np.linalg.norm(coef)),
    }


def time_independent_coordinate_subset(X, Y, k=80, alpha=1e-2, coord_count=5):
    sx = StandardScaler().fit(X)
    sy = StandardScaler().fit(Y)
    Xs = sx.transform(X)
    Ys = sy.transform(Y)
    coord_count = int(min(coord_count, Y.shape[1]))
    rows = []
    for a in range(coord_count):
        idx, afpc_time, radius = afpc(Xs, k)
        centers = Xs[idx]
        start = time.perf_counter()
        Phi = center_features(Xs, centers, "full")
        dict_time = time.perf_counter() - start
        start = time.perf_counter()
        gram = Phi.T.dot(Phi)
        rhs = Phi.T.dot(Ys[:, a])
        gram.flat[::gram.shape[0] + 1] += alpha * X.shape[0]
        coef = np.linalg.solve(gram, rhs)
        fit_time = time.perf_counter() - start
        rows.append({
            "afpc_time": afpc_time,
            "dict_time": dict_time,
            "fit_time": fit_time,
            "total_time": afpc_time + dict_time + fit_time,
            "cover_radius": radius,
            "coef_norm": float(np.linalg.norm(coef)),
        })
    return pd.DataFrame(rows)


def experiment_large_scale_stress():
    rows = []
    configs = [
        (50000, 50, 100, 80, 9001),
        (50000, 50, 200, 80, 9002),
    ]
    for n, d, m, k, seed in configs:
        print("stress n={} d={} m={} K={}".format(n, d, m, k), flush=True)
        X, Y = make_lowdim_data(n, m=m, d_star=2, d=d, noise=0.15, seed=seed)
        shared = fit_shared_normal_equations(X, Y, k=k, alpha=1e-2)
        rows.append({
            "experiment": "large_scale_stress",
            "method": "SC-DCF-LS",
            "n": n,
            "d": d,
            "m": m,
            "K": k,
            "q": shared["q"],
            "afpc_time": shared["afpc_time"],
            "dict_time": shared["dict_time"],
            "fit_time": shared["fit_time"],
            "total_time": shared["total_time"],
            "estimated_peak_array_mib": shared["estimated_peak_array_mib"],
            "coord_count_measured": m,
            "timing_note": "actual shared fit",
        })
        subset = time_independent_coordinate_subset(X, Y, k=k, alpha=1e-2, coord_count=5)
        per_coord = subset[["afpc_time", "dict_time", "fit_time", "total_time"]].mean()
        rows.append({
            "experiment": "large_scale_stress",
            "method": "Independent DCF-LS",
            "n": n,
            "d": d,
            "m": m,
            "K": k,
            "q": shared["q"],
            "afpc_time": per_coord["afpc_time"] * m,
            "dict_time": per_coord["dict_time"] * m,
            "fit_time": per_coord["fit_time"] * m,
            "total_time": per_coord["total_time"] * m,
            "estimated_peak_array_mib": shared["estimated_peak_array_mib"],
            "coord_count_measured": len(subset),
            "timing_note": "extrapolated from 5 actual coordinate fits",
        })
    return pd.DataFrame(rows)


def max_feature_values(Xs, centers, params):
    k = centers.shape[0]
    d = Xs.shape[1]
    P = params.reshape(k, d + 2)
    b = P[:, 0]
    w = P[:, 1:1 + d]
    u = P[:, -1]
    diff = Xs[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
    vals = b[None, :] + np.einsum("nkd,kd->nk", diff, w) + dist * u[None, :]
    return np.max(vals, axis=1)


class DirectMaxFeatureSCDCFSmall(object):
    def __init__(self, k=8, alpha=1e-3, maxiter=80):
        self.k = k
        self.alpha = alpha
        self.maxiter = maxiter

    def fit(self, X, Y):
        self.scaler_x = StandardScaler().fit(X)
        self.scaler_y = StandardScaler().fit(Y)
        Xs = self.scaler_x.transform(X)
        Ys = self.scaler_y.transform(Y)
        idx, self.afpc_time_, self.cover_radius_ = afpc(Xs, self.k)
        self.centers_ = Xs[idx]
        self.params_ = []
        start = time.perf_counter()
        p = self.k * (Xs.shape[1] + 2)
        for a in range(Ys.shape[1]):
            y = Ys[:, a]
            def obj(theta):
                pred = max_feature_values(Xs, self.centers_, theta)
                return float(np.mean((pred - y) ** 2) + self.alpha * np.mean(theta ** 2))
            init = np.zeros(p)
            init[0::(Xs.shape[1] + 2)] = np.mean(y)
            res = minimize(obj, init, method="L-BFGS-B", options={"maxiter": self.maxiter, "ftol": 1e-7})
            self.params_.append(res.x)
        self.fit_time_ = time.perf_counter() - start
        self.total_time_ = self.afpc_time_ + self.fit_time_
        return self

    def predict(self, X):
        Xs = self.scaler_x.transform(X)
        preds = [max_feature_values(Xs, self.centers_, theta) for theta in self.params_]
        Ys = np.vstack(preds).T
        return self.scaler_y.inverse_transform(Ys)


def experiment_direct_max_sanity():
    rows = []
    for n in [150, 300]:
        for rep in range(3):
            print("direct_max_sanity n={} rep={}".format(n, rep), flush=True)
            X, Y = make_lowdim_data(n, m=2, d_star=2, d=5, noise=0.10, seed=9500 + n + rep)
            Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.35, random_state=rep)
            methods = [
                ("direct max-feature SC-DCF", DirectMaxFeatureSCDCFSmall(k=8, alpha=1e-3, maxiter=80)),
                ("SC-DCF-LS", SharedCenterDCFPrototype(intrinsic_dim=2, alpha=1e-2, max_k=8)),
                ("kNN", KNeighborsRegressor(n_neighbors=max(3, int(np.sqrt(Xtr.shape[0]))))),
            ]
            for name, model in methods:
                start = time.perf_counter()
                if name == "kNN":
                    sx = StandardScaler().fit(Xtr)
                    sy = StandardScaler().fit(Ytr)
                    model.fit(sx.transform(Xtr), sy.transform(Ytr))
                    pred = sy.inverse_transform(model.predict(sx.transform(Xte)))
                    total = time.perf_counter() - start
                else:
                    model.fit(Xtr, Ytr)
                    pred = model.predict(Xte)
                    total = model.total_time_
                row = {
                    "experiment": "direct_max_sanity",
                    "n": n,
                    "d": 5,
                    "m": 2,
                    "rep": rep,
                    "method": name,
                    "total_time": total,
                    "K": 8 if name != "kNN" else 0,
                }
                rows.append(add_metrics(row, Yte, pred, Ytr))
    return pd.DataFrame(rows)


def mean_se(df, group_cols):
    metrics = ["summed_mse", "max_coord_mse", "std_summed_mse", "avg_r2",
               "total_time", "afpc_time", "fit_time", "K"]
    out = df.groupby(group_cols)[metrics].agg(["mean", "std", "count"]).reset_index()
    # flatten
    out.columns = ["_".join([c for c in col if c]) for col in out.columns.values]
    for metric in metrics:
        out[metric + "_se"] = out[metric + "_std"] / np.sqrt(out[metric + "_count"].clip(lower=1))
    return out


def write_latex_table(df, path, columns, caption):
    sub = df[columns].copy()
    for c in sub.columns:
        if sub[c].dtype.kind in "fc":
            sub[c] = sub[c].map(lambda x: "" if pd.isnull(x) else "{:.4g}".format(x))
    with open(path, "w", encoding="utf-8") as f:
        f.write("% " + caption + "\n")
        f.write(sub.to_latex(index=False, escape=False))


def fmt_mean_se(row, metric):
    mean = row[metric + "_mean"]
    se = row[metric + "_se"]
    if pd.isnull(mean):
        return ""
    if abs(mean) >= 100:
        return "{:.1f} ({:.1f})".format(mean, se)
    if abs(mean) >= 10:
        return "{:.2f} ({:.2f})".format(mean, se)
    if abs(mean) >= 1:
        return "{:.3f} ({:.3f})".format(mean, se)
    return "{:.4f} ({:.4f})".format(mean, se)


def write_compact_table(df, path, group_cols, metrics, caption):
    rows = []
    for _, row in df.iterrows():
        out = {}
        for c in group_cols:
            out[c] = row[c]
        for metric in metrics:
            out[metric] = fmt_mean_se(row, metric)
        rows.append(out)
    sub = pd.DataFrame(rows)
    with open(path, "w", encoding="utf-8") as f:
        f.write("% " + caption + "\n")
        f.write(sub.to_latex(index=False, escape=False))


def make_speedup_table(scaling):
    shared = scaling[scaling.method == "SC-DCF-LS"].set_index("m")
    indep = scaling[scaling.method == "Independent DCF-LS"].set_index("m")
    rows = []
    for m in sorted(set(shared.index).intersection(indep.index)):
        sc_time = shared.loc[m, "total_time_mean"]
        indep_time = indep.loc[m, "total_time_mean"]
        afpc = shared.loc[m, "afpc_time_mean"]
        rows.append({
            "m": m,
            "sc_total_time": sc_time,
            "independent_actual_time": indep_time,
            "speedup": indep_time / sc_time,
            "afpc_share": afpc / sc_time,
        })
    return pd.DataFrame(rows)


def add_real_speedup(real_main):
    rows = []
    for dataset, sub in real_main.groupby("dataset"):
        base = float(sub[sub.method == "Independent DCF-LS"]["total_time_mean"].iloc[0])
        for _, row in sub.iterrows():
            out = row.to_dict()
            out["speedup_vs_independent"] = base / float(row["total_time_mean"])
            rows.append(out)
    return pd.DataFrame(rows)


def write_simple_latex(df, path, caption):
    with open(path, "w", encoding="utf-8") as f:
        f.write("% " + caption + "\n")
        f.write(df.to_latex(index=False, escape=False))


def plot_outputs(rate, scaling):
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for method in ["SC-DCF-LS", "kNN", "MO-RF"]:
        sub = rate[rate.method == method].sort_values("n")
        if len(sub):
            ax.errorbar(sub["n"], sub["summed_mse_mean"], yerr=sub["summed_mse_se"],
                        marker=MARKERS[method], color=COLORS[method],
                        linewidth=1.8, capsize=2.5, label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample size n")
    ax.set_ylabel("summed MSE")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_rate_mse_main.pdf"))
    fig.savefig(os.path.join(OUT, "fig_rate_mse_main.png"), dpi=240)
    fig.savefig(os.path.join(OUT, "fig_rate_mse.pdf"))
    fig.savefig(os.path.join(OUT, "fig_rate_mse.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for method in ["SC-DCF-LS", "Independent DCF-LS", "kNN", "MO-RF"]:
        sub = rate[rate.method == method].sort_values("n")
        if len(sub):
            ax.errorbar(sub["n"], sub["total_time_mean"], yerr=sub["total_time_se"],
                        marker=MARKERS[method], color=COLORS[method],
                        linewidth=1.8, capsize=2.5, label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample size n")
    ax.set_ylabel("total time (seconds)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_rate_runtime_main.pdf"))
    fig.savefig(os.path.join(OUT, "fig_rate_runtime_main.png"), dpi=240)
    fig.savefig(os.path.join(OUT, "fig_rate_runtime.pdf"))
    fig.savefig(os.path.join(OUT, "fig_rate_runtime.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for method in ["SC-DCF-LS", "Independent DCF-LS"]:
        sub = scaling[scaling.method == method].sort_values("m")
        if len(sub):
            ax.errorbar(sub["m"], sub["total_time_mean"], yerr=sub["total_time_se"],
                        marker=MARKERS[method], color=COLORS[method],
                        linewidth=1.8, capsize=2.5, label=method)
    ax.set_xlabel("number of outputs m")
    ax.set_ylabel("total time (seconds)")
    ax.set_yscale("log")
    ax.legend(frameon=True)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_runtime_m_log.pdf"))
    fig.savefig(os.path.join(OUT, "fig_runtime_m_log.png"), dpi=240)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_runtime_m.pdf"))
    fig.savefig(os.path.join(OUT, "fig_runtime_m.png"), dpi=200)
    plt.close(fig)

    shared = scaling[scaling.method == "SC-DCF-LS"].set_index("m")
    indep = scaling[scaling.method == "Independent DCF-LS"].set_index("m")
    common = sorted(set(shared.index).intersection(set(indep.index)))
    speedup = [indep.loc[m, "total_time_mean"] / shared.loc[m, "total_time_mean"] for m in common]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(common, speedup, marker="o", color=COLORS["SC-DCF-LS"], linewidth=2.0)
    slope = speedup[0] * np.array(common) / float(common[0])
    ax.plot(common, slope, linestyle="--", color="#b0b0b0", linewidth=1.2, label="linear reference")
    for x, y in zip(common, speedup):
        ax.annotate("{:.1f}x".format(y), xy=(x, y), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("number of outputs m")
    ax.set_ylabel("speedup over independent centers")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_speedup_annotated.pdf"))
    fig.savefig(os.path.join(OUT, "fig_speedup_annotated.png"), dpi=240)
    fig.savefig(os.path.join(OUT, "fig_speedup_m.pdf"))
    fig.savefig(os.path.join(OUT, "fig_speedup_m.png"), dpi=200)
    plt.close(fig)


def plot_correlation(corr):
    methods = ["SC-DCF-LS", "kNN", "MO-RF"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for method in methods:
        sub = corr[corr.method == method].sort_values("corr")
        base = float(sub[sub["corr"] == 0.0]["summed_mse_mean"].iloc[0])
        rel = sub["summed_mse_mean"] / base
        rel_se = sub["summed_mse_se"] / base
        axes[0].errorbar(sub["corr"], rel, yerr=rel_se,
                         marker=MARKERS[method], color=COLORS[method],
                         linewidth=1.8, capsize=2.5, label=method)
        axes[1].errorbar(sub["corr"], sub["total_time_mean"], yerr=sub["total_time_se"],
                         marker=MARKERS[method], color=COLORS[method],
                         linewidth=1.8, capsize=2.5, label=method)
    axes[0].set_xlabel("noise correlation")
    axes[0].set_ylabel("relative summed MSE")
    axes[1].set_xlabel("noise correlation")
    axes[1].set_ylabel("total time (seconds)")
    axes[1].set_yscale("log")
    axes[0].grid(True, linewidth=0.3, alpha=0.5)
    axes[1].grid(True, which="both", linewidth=0.3, alpha=0.5)
    axes[0].axhline(1.0, color="#b0b0b0", linewidth=0.8, linestyle="--")
    axes[0].legend(fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_correlation_stability_main.pdf"))
    fig.savefig(os.path.join(OUT, "fig_correlation_stability_main.png"), dpi=240)
    fig.savefig(os.path.join(OUT, "fig_correlation_stability.pdf"))
    fig.savefig(os.path.join(OUT, "fig_correlation_stability.png"), dpi=200)
    plt.close(fig)


def main():
    all_frames = []
    real_mulan_fns = [
        lambda: experiment_real_mulan("real_water_quality", "WaterQuality", "water-quality.arff", 14, 3),
        lambda: experiment_real_mulan("real_edm", "EDM", "edm.arff", 2, 2),
        lambda: experiment_real_mulan("real_atp1d", "ATP1d", "atp1d.arff", 6, 4),
        lambda: experiment_real_mulan("real_atp7d", "ATP7d", "atp7d.arff", 6, 4),
        lambda: experiment_real_mulan("real_rf1", "RF1", ["rf1-train.arff", "rf1-test.arff"], 8, 4),
    ]
    experiment_fns = [
        experiment_rate,
        experiment_scaling,
        experiment_large_scale_stress,
        experiment_correlation,
        experiment_real_energy,
        experiment_real_communities,
    ] + real_mulan_fns + [
        experiment_direct_max_sanity,
        experiment_real_linnerud,
        experiment_ablation,
        experiment_afpc_sanity,
    ]
    for fn in experiment_fns:
        print("starting {}".format(fn.__name__), flush=True)
        df = fn()
        all_frames.append(df)
    raw = pd.concat(all_frames, ignore_index=True)
    raw.to_csv(os.path.join(OUT, "raw_results.csv"), index=False)

    rate = mean_se(raw[raw.experiment == "rate"], ["n", "method"])
    scaling = mean_se(raw[raw.experiment == "scaling"], ["m", "method"])
    corr = mean_se(raw[raw.experiment == "correlation"], ["corr", "method"])
    direct_max = mean_se(raw[raw.experiment == "direct_max_sanity"], ["n", "method"])
    real_energy = mean_se(raw[raw.experiment == "real_energy"], ["method"])
    real_communities = mean_se(raw[raw.experiment == "real_communities"], ["method"])
    real_linnerud = mean_se(raw[raw.experiment == "real_linnerud"], ["method"])
    real_experiments = [
        "real_energy",
        "real_communities",
        "real_water_quality",
        "real_edm",
        "real_atp1d",
        "real_atp7d",
        "real_rf1",
    ]
    real_all = mean_se(raw[raw.experiment.isin(real_experiments + ["real_linnerud"])], ["experiment", "dataset", "method"])
    real_main = mean_se(raw[raw.experiment.isin(real_experiments)], ["dataset", "method"])
    ablation = mean_se(raw[raw.experiment == "ablation"], ["config", "feature_mode", "max_k"])
    ablation = mean_se(raw[raw.experiment == "ablation"], ["config", "center_rule", "feature_mode", "max_k"])
    rate_final = rate[rate.n == 10000].copy()
    speedup = make_speedup_table(scaling)
    stress = raw[raw.experiment == "large_scale_stress"].copy()
    stress_rows = []
    for (n, d, m, k), sub in stress.groupby(["n", "d", "m", "K"]):
        shared = sub[sub.method == "SC-DCF-LS"].iloc[0]
        independent = sub[sub.method == "Independent DCF-LS"].iloc[0]
        for _, row in sub.iterrows():
            out = row.to_dict()
            out["speedup_vs_shared"] = float(independent["total_time"]) / float(shared["total_time"])
            stress_rows.append(out)
    stress = pd.DataFrame(stress_rows)
    dataset_dims = raw[raw.experiment.isin(real_experiments)][["dataset", "n", "d", "m"]].dropna().drop_duplicates()
    real_main = pd.merge(real_main, dataset_dims, on="dataset", how="left")
    afpc_sanity = raw[raw.experiment == "afpc_sanity"].groupby(["m"])[
        ["single_afpc_time", "projected_afpc_time", "actual_repeated_afpc_time", "actual_projected_ratio"]
    ].agg(["mean", "std", "count"]).reset_index()
    afpc_sanity.columns = ["_".join([c for c in col if c]) for col in afpc_sanity.columns.values]
    for c in ["single_afpc_time", "projected_afpc_time", "actual_repeated_afpc_time", "actual_projected_ratio"]:
        afpc_sanity[c + "_se"] = afpc_sanity[c + "_std"] / np.sqrt(afpc_sanity[c + "_count"].clip(lower=1))

    rate.to_csv(os.path.join(OUT, "rate_summary.csv"), index=False)
    scaling.to_csv(os.path.join(OUT, "scaling_summary.csv"), index=False)
    corr.to_csv(os.path.join(OUT, "correlation_summary.csv"), index=False)
    direct_max.to_csv(os.path.join(OUT, "direct_max_sanity_summary.csv"), index=False)
    stress.to_csv(os.path.join(OUT, "large_scale_stress_summary.csv"), index=False)
    real_energy.to_csv(os.path.join(OUT, "real_energy_summary.csv"), index=False)
    real_communities.to_csv(os.path.join(OUT, "real_communities_summary.csv"), index=False)
    real_linnerud.to_csv(os.path.join(OUT, "real_linnerud_summary.csv"), index=False)
    real_all.to_csv(os.path.join(OUT, "real_summary.csv"), index=False)
    ablation.to_csv(os.path.join(OUT, "ablation_summary.csv"), index=False)
    rate_final.to_csv(os.path.join(OUT, "rate_n10000_summary.csv"), index=False)
    speedup.to_csv(os.path.join(OUT, "scaling_speedup_summary.csv"), index=False)
    afpc_sanity.to_csv(os.path.join(OUT, "afpc_sanity_summary.csv"), index=False)

    write_latex_table(rate, os.path.join(OUT, "table_rate.tex"),
                      ["n", "method", "summed_mse_mean", "max_coord_mse_mean", "total_time_mean", "K_mean"],
                      "Rate experiment summary")
    write_latex_table(scaling, os.path.join(OUT, "table_scaling.tex"),
                      ["m", "method", "summed_mse_mean", "afpc_time_mean", "total_time_mean", "K_mean"],
                      "Scaling experiment summary")
    write_latex_table(corr, os.path.join(OUT, "table_correlation.tex"),
                      ["corr", "method", "summed_mse_mean", "max_coord_mse_mean", "total_time_mean"],
                      "Correlated noise experiment summary")
    write_latex_table(direct_max, os.path.join(OUT, "table_direct_max_sanity.tex"),
                      ["n", "method", "summed_mse_mean", "total_time_mean", "K_mean"],
                      "Direct max-feature sanity experiment summary")
    write_latex_table(real_energy, os.path.join(OUT, "table_real_energy.tex"),
                      ["method", "summed_mse_mean", "max_coord_mse_mean", "total_time_mean"],
                      "Energy Efficiency real-data summary")
    write_latex_table(real_communities, os.path.join(OUT, "table_real_communities.tex"),
                      ["method", "summed_mse_mean", "max_coord_mse_mean", "total_time_mean"],
                      "Communities and Crime real-data summary")
    write_latex_table(real_linnerud, os.path.join(OUT, "table_real_linnerud.tex"),
                      ["method", "summed_mse_mean", "max_coord_mse_mean", "total_time_mean"],
                      "Linnerud real-data summary")
    write_latex_table(real_all, os.path.join(OUT, "table_real.tex"),
                      ["experiment", "dataset", "method", "summed_mse_mean", "max_coord_mse_mean", "total_time_mean"],
                      "Combined real-data summary")

    write_compact_table(rate, os.path.join(OUT, "table_rate_compact.tex"),
                        ["n", "method"], ["summed_mse", "max_coord_mse", "total_time", "K"],
                        "Rate experiment summary with standard errors")
    write_compact_table(scaling, os.path.join(OUT, "table_scaling_compact.tex"),
                        ["m", "method"], ["summed_mse", "afpc_time", "total_time", "K"],
                        "Scaling experiment summary with standard errors")
    write_compact_table(corr, os.path.join(OUT, "table_correlation_compact.tex"),
                        ["corr", "method"], ["summed_mse", "max_coord_mse", "total_time"],
                        "Correlated noise experiment summary with standard errors")
    write_compact_table(direct_max, os.path.join(OUT, "table_direct_max_sanity_compact.tex"),
                        ["n", "method"], ["summed_mse", "total_time", "K"],
                        "Direct max-feature sanity experiment summary with standard errors")
    write_compact_table(real_energy, os.path.join(OUT, "table_real_energy_compact.tex"),
                        ["method"], ["summed_mse", "max_coord_mse", "total_time"],
                        "Energy Efficiency real-data summary with standard errors")
    write_compact_table(real_communities, os.path.join(OUT, "table_real_communities_compact.tex"),
                        ["method"], ["summed_mse", "max_coord_mse", "total_time"],
                        "Communities and Crime real-data summary with standard errors")
    write_compact_table(real_linnerud, os.path.join(OUT, "table_real_linnerud_compact.tex"),
                        ["method"], ["summed_mse", "max_coord_mse", "total_time"],
                        "Linnerud real-data summary with standard errors")
    write_compact_table(ablation, os.path.join(OUT, "table_ablation_compact.tex"),
                        ["config", "center_rule", "feature_mode", "max_k"], ["summed_mse", "max_coord_mse", "total_time", "K"],
                        "SC-DCF-LS dictionary and K ablation with standard errors")
    write_compact_table(rate_final, os.path.join(OUT, "table_rate_n10000_compact.tex"),
                        ["method"], ["summed_mse", "total_time", "K"],
                        "Low-dimensional simulation n=10000 summary with standard errors")

    real_main = add_real_speedup(real_main)
    real_main.to_csv(os.path.join(OUT, "real_standardized_summary.csv"), index=False)
    write_compact_table(real_main, os.path.join(OUT, "table_real_standardized_compact.tex"),
                        ["dataset", "n", "d", "m", "method"], ["std_summed_mse", "avg_r2", "total_time"],
                        "Real-data standardized summary with standard errors")
    real_pub = []
    for _, row in real_main.iterrows():
        real_pub.append({
            "dataset": row["dataset"],
            "n": int(row["n"]),
            "d": int(row["d"]),
            "m": int(row["m"]),
            "method": row["method"],
            "std_summed_mse": fmt_mean_se(row, "std_summed_mse"),
            "avg_r2": fmt_mean_se(row, "avg_r2"),
            "total_time": fmt_mean_se(row, "total_time"),
            "speedup": "{:.2f}".format(row["speedup_vs_independent"]),
        })
    write_simple_latex(pd.DataFrame(real_pub), os.path.join(OUT, "table_real_standardized_speedup.tex"),
                       "Real-data standardized summary with speedup")

    speed_fmt = speedup.copy()
    for c in speed_fmt.columns:
        if c == "m":
            continue
        speed_fmt[c] = speed_fmt[c].map(lambda x: "{:.2f}".format(x))
    write_simple_latex(speed_fmt, os.path.join(OUT, "table_scaling_speedup.tex"),
                       "Scaling speedup summary")

    stress_pub = []
    if len(stress):
        for _, row in stress.iterrows():
            stress_pub.append({
                "n": int(row["n"]),
                "d": int(row["d"]),
                "m": int(row["m"]),
                "K": int(row["K"]),
                "method": row["method"],
                "afpc_time": "{:.2f}".format(row["afpc_time"]),
                "dict_time": "{:.2f}".format(row["dict_time"]),
                "ridge_time": "{:.2f}".format(row["fit_time"]),
                "total_time": "{:.2f}".format(row["total_time"]),
                "est_peak_mib": "{:.0f}".format(row["estimated_peak_array_mib"]),
                "speedup": "{:.1f}".format(row["speedup_vs_shared"]),
                "note": row["timing_note"],
            })
    write_simple_latex(pd.DataFrame(stress_pub), os.path.join(OUT, "table_large_scale_stress.tex"),
                       "Large-scale stress-test timing summary")

    sanity_fmt = pd.DataFrame({
        "m": afpc_sanity["m"],
        "projected_afpc_time": [fmt_mean_se(row, "projected_afpc_time") for _, row in afpc_sanity.iterrows()],
        "actual_repeated_afpc_time": [fmt_mean_se(row, "actual_repeated_afpc_time") for _, row in afpc_sanity.iterrows()],
        "actual_projected_ratio": [fmt_mean_se(row, "actual_projected_ratio") for _, row in afpc_sanity.iterrows()],
    })
    write_simple_latex(sanity_fmt, os.path.join(OUT, "table_afpc_sanity.tex"),
                       "Actual repeated AFPC timing sanity check")

    plot_outputs(rate, scaling)
    plot_correlation(corr)

    print("Wrote results to", OUT)


if __name__ == "__main__":
    main()
