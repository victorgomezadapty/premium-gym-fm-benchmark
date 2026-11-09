"""First Kumo trial on the synthetic gym dataset (100% fake data).

Backends (auto-selected by which key is set):
  KUMO_API_KEY   -> KumoRFM via `kumoai.rfm` (pip install kumoai)
  NVIDIA_API_KEY -> NVIDIA hosted Kumo Relational (build.nvidia.com/nvidia/kumo-relational):
                    requests are built locally with `sdm_client` (structured-data-models-client)
                    and POSTed with `Authorization: Bearer` to the API Catalog endpoint.
  none           -> dry backend: builds the graph and the requests locally, sends nothing.

Usage:
  export NVIDIA_API_KEY=<your key>      # or KUMO_API_KEY=<your key>; never commit it
  python kumo_trial.py   (options: backend auto, kumo, nvidia or dry; n members; ctx rows; stat median or mean)

Honest evaluation: tables are cut at the anchor date (2025-07-01) before they reach Kumo,
so the model never sees the future; ground truth comes from the full tables. The churn
result is compared with a LightGBM baseline scored on the same members and anchor.
"""
import os, sys, json, argparse, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
ANCHOR = pd.Timestamp("2025-07-01")
TIME_COL = dict(members="join_date", contracts="start_date", payments="payment_time", visits="swipe_time",
                trainers="hire_date", pt_packages="purchase_time", pt_sessions="session_time")
TABLES = ["members","plans","contracts","payments","visits","trainers","pt_packages","pt_sessions"]
QUERIES = {
 "churn_30d": ("PREDICT COUNT(visits.*, 0, 30, days) = 0 FOR EACH members.member_id "
               "WHERE COUNT(visits.*, -30, 0, days) > 0", "binary"),
 "revenue_30d": ("PREDICT SUM(payments.net_amount, 0, 30, days) FOR EACH members.member_id", "regression"),
 "visits_30d": ("PREDICT COUNT(visits.*, 0, 30, days) FOR EACH members.member_id", "regression"),
}
TAG = ""; ERRORS = {}
NV_URL = "https://ai.api.nvidia.com/v1/structured-data/nvidia/kumo-relational/predictions"

def load():
    full = {n: pd.read_parquet(f"{D}/{n}.parquet") for n in TABLES}
    cut = {n: (df[df[TIME_COL[n]] < ANCHOR].copy() if n in TIME_COL else df.copy()) for n, df in full.items()}
    return full, cut

def truth(full, ids):
    v = full["visits"]; v = v[v.result == "Allowed"]; p = full["payments"]; p = p[~p.is_refund]
    win = lambda df, c, a, b: df[(df[c] >= ANCHOR + pd.Timedelta(days=a)) & (df[c] < ANCHOR + pd.Timedelta(days=b))].groupby("member_id")
    nv = win(v, "swipe_time", 0, 30).size().reindex(ids).fillna(0)
    rev = win(p, "payment_time", 0, 30).net_amount.sum().reindex(ids).fillna(0)
    return {"churn_30d": (nv == 0).astype(int), "revenue_30d": rev, "visits_30d": nv}

def pick_members(full, n, seed=0):
    v = full["visits"]; v = v[(v.result == "Allowed") & (v.swipe_time >= ANCHOR - pd.Timedelta(days=30)) & (v.swipe_time < ANCHOR)]
    act = np.array(sorted(v.member_id.unique()))
    return [str(x) for x in np.random.default_rng(seed).choice(act, min(n, len(act)), replace=False)]

def score(kind, y, pred):
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    if kind == "binary":
        k = max(1, len(y)//10); top = np.argsort(-pred)[:k]
        return dict(roc_auc=round(roc_auc_score(y, pred), 4), pr_auc=round(average_precision_score(y, pred), 4),
                    precision_top10=round(y[top].mean(), 4), base_rate=round(y.mean(), 4))
    return dict(mae=round(mean_absolute_error(y, pred), 2), mae_naive_mean=round(mean_absolute_error(y, np.full_like(y, y.mean())), 2),
                truth_mean=round(y.mean(), 2), pred_mean=round(pred.mean(), 2))

# KumoRFM (kumoai)
def run_kumo(cut, ids):
    import kumoai.rfm as rfm
    rfm.init()  # reads KUMO_API_KEY (and optional RFM_API_URL)
    g = rfm.Graph.from_data(cut, verbose=False)
    g["pt_packages"].primary_key = "package_id"; g["pt_sessions"].primary_key = "session_id"
    g.link("pt_sessions", "package_id", "pt_packages"); g.validate()
    model = rfm.KumoRFM(g, verbose=False)
    out = {}
    for name, (q, kind) in QUERIES.items():
        t0 = time.time(); df = model.predict(q, indices=ids, anchor_time=ANCHOR, verbose=False)
        df.to_csv(f"{OUT}/kumo_{name}.csv", index=False)
        ent = [c for c in df.columns if c.upper() in ("ENTITY", "MEMBER_ID")][0]
        if kind == "binary":
            col = [c for c in df.columns if "PROB" in c.upper() and "TRUE" in c.upper()][0]
        else:
            col = [c for c in df.columns if "PRED" in c.upper()][0]
        out[name] = (pd.Series(df[col].values, index=df[ent].astype(str)).reindex(ids), time.time() - t0)
    return out

# NVIDIA hosted Kumo Relational
def build_requests(cut, ids, n_ctx=1000, stat="median"):
    import sdm_client as sc, warnings
    warnings.filterwarnings("ignore")
    sch = sc.Schema.from_data(cut)
    reqs = {}
    for name, (q, kind) in QUERIES.items():
        extra = {"prediction_statistic": stat} if kind == "regression" else {}
        r = sc.PredictiveQuery(q, sch).materialize(indices=ids, num_context_samples=n_ctx, anchor_time=ANCHOR, **extra)
        reqs[name] = r
    return reqs

def run_nvidia(reqs, ids):
    import requests
    from sdm_client.adapters.relational import RelationalAdapter
    key = os.environ["NVIDIA_API_KEY"]; out = {}; adapter = RelationalAdapter()
    for name, r in reqs.items():
        kind = QUERIES[name][1]; preds = {}; t0 = time.time()
        # Envelopes exactly as the official sdm_client would send them (drops the engine's default
        # inference_config such as output_type=median, sets output.prediction_statistic).
        fields = ("prediction", "probabilities") if kind == "binary" else ("prediction",)
        for i, pl in enumerate(adapter.prediction_plan(r, fields)):
            body = json.dumps(pl, default=str); resp = None
            for attempt in range(4):
                try:
                    resp = requests.post(NV_URL, headers={"Authorization": f"Bearer {key}", "Accept": "application/json",
                                         "Content-Type": "application/json"}, data=body, timeout=300)
                    if resp.status_code in (429, 500, 502, 503, 504):
                        print(f"  {name}: HTTP {resp.status_code}, retry {attempt+1}", flush=True); time.sleep(20*(attempt+1)); continue
                    break
                except requests.exceptions.RequestException as e:
                    print(f"  {name}: {type(e).__name__}, retry {attempt+1}", flush=True); resp = None; time.sleep(20*(attempt+1))
            if resp is None:
                ERRORS[name] = "connection failed after 4 attempts"; break
            if resp.status_code != 200:
                ERRORS[name] = f"HTTP {resp.status_code}: {resp.text[:300]}"; break
            js = resp.json(); json.dump(js, open(f"{OUT}/nvidia_{name}_{i}{TAG}.json", "w"))
            it = pl["predict"]["instance_table"]; cols = it["columns"]
            ecol = [c for c in cols if c == "__kumo_entity_ref_0" or c.endswith("member_id")]
            ecol = cols.index(ecol[0]) if ecol else None
            for item in js.get("predictions", js.get("items", [])):
                idx = item.get("row_index", item.get("index"))
                row = it["rows"][idx]
                mid = str(row[ecol]) if ecol is not None else ids[idx]
                if kind == "binary":
                    pr = item.get("probabilities")
                    p = pr.get("true", pr.get("True")) if isinstance(pr, dict) else (pr[-1] if pr else float(item["prediction"] in (True, "true")))
                    preds[mid] = float(p)
                else:
                    pv = item["prediction"]; preds[mid] = float(pv["mean"] if isinstance(pv, dict) and "mean" in pv else pv)
        print(f"  {name}: {len(preds)} predictions in {time.time()-t0:.1f}s", flush=True)
        out[name] = (pd.Series(preds, dtype=float).reindex(ids), time.time() - t0)
    return out

# baseline on same members/anchor
def baseline_same_members(ids):
    import lightgbm as lgb
    X = pd.read_parquet(f"{D}/churn_features.parquet")
    num = [c for c in X.columns if c not in ("member_id", "anchor", "label", "plan_type")]
    X[num] = X[num].fillna({k: 0 for k in num if k not in ("days_since_visit", "days_to_contract_end")}).fillna(-1)
    X = pd.get_dummies(X, columns=["plan_type"], dummy_na=True); feats = [c for c in X.columns if c not in ("member_id", "anchor", "label")]
    tr = X[X.anchor <= ANCHOR - pd.Timedelta(days=31)]  # label window closes before the anchor
    te = X[(X.anchor == ANCHOR) & X.member_id.isin(ids)].set_index("member_id")
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    m.fit(tr[feats].astype(float), tr.label)
    return pd.Series(m.predict_proba(te[feats].astype(float))[:, 1], index=te.index).reindex(ids)

def baseline_regression(full, ids):
    """LightGBM regressors (L1 objective, MAE-optimal) on the same monthly feature table."""
    import lightgbm as lgb
    X = pd.read_parquet(f"{D}/churn_features.parquet")
    v = full["visits"]; v = v[v.result == "Allowed"]; p = full["payments"]; p = p[~p.is_refund]
    codes = {m_: i for i, m_ in enumerate(full["members"].member_id)}
    vt, vc = v.swipe_time.values, v.member_id.map(codes).values
    pt_, pc, pa = p.payment_time.values, p.member_id.map(codes).values, p.net_amount.values
    X = X.sort_values(["anchor", "member_id"]).reset_index(drop=True)
    X["revenue_30d"] = 0.0; X["visits_30d"] = 0.0
    xc = X.member_id.map(codes).values
    for t in X.anchor.unique():
        t0 = np.datetime64(pd.Timestamp(t)); t1 = t0 + np.timedelta64(30, "D"); rows = (X.anchor == t).values
        mv = (vt >= t0) & (vt < t1); mp = (pt_ >= t0) & (pt_ < t1)
        X.loc[rows, "visits_30d"] = np.bincount(vc[mv], minlength=len(codes))[xc[rows]]
        X.loc[rows, "revenue_30d"] = np.bincount(pc[mp], weights=pa[mp], minlength=len(codes))[xc[rows]]
    ys = ("revenue_30d", "visits_30d")
    num = [c for c in X.columns if c not in ("member_id", "anchor", "label", "plan_type", "revenue_30d", "visits_30d")]
    X[num] = X[num].fillna({k: 0 for k in num if k not in ("days_since_visit", "days_to_contract_end")}).fillna(-1)
    X = pd.get_dummies(X, columns=["plan_type"], dummy_na=True)
    feats = [c for c in X.columns if c not in ("member_id", "anchor", "label", "revenue_30d", "visits_30d")]
    tr = X[X.anchor <= ANCHOR - pd.Timedelta(days=31)]; te = X[(X.anchor == ANCHOR) & X.member_id.isin(ids)].set_index("member_id")
    out = {}
    for k_ in ys:
        m = lgb.LGBMRegressor(objective="l1", n_estimators=400, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
        m.fit(tr[feats].astype(float), tr[k_]); out[k_] = pd.Series(m.predict(te[feats].astype(float)), index=te.index).reindex(ids)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backend", default="auto"); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--ctx", type=int, default=1000); ap.add_argument("--stat", default="median"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); global TAG; TAG = a.tag
    b = a.backend
    if b == "auto":
        b = "kumo" if os.getenv("KUMO_API_KEY") else "nvidia" if os.getenv("NVIDIA_API_KEY") else "dry"
    full, cut = load(); ids = pick_members(full, a.n); y = truth(full, ids)
    print(f"backend={b} anchor={ANCHOR.date()} members_scored={len(ids)} churn_rate={y['churn_30d'].mean():.3f}")
    report = {"backend": b, "anchor": str(ANCHOR.date()), "n": len(ids)}
    base = baseline_same_members(ids)
    report["lightgbm_churn_same_members"] = score("binary", y["churn_30d"], base.values)
    for k_, pr in baseline_regression(full, ids).items():
        report[f"lightgbm_{k_}_same_members"] = score("regression", y[k_], pr.values)
        report[f"lightgbm_{k_}_same_members"]["mae_predict_zero"] = round(float(np.abs(y[k_]).mean()), 2)
    report["context_rows"] = a.ctx; report["prediction_statistic"] = a.stat
    if b == "dry":
        reqs = build_requests(cut, ids)
        report["dry_run"] = {k: dict(calls=len(r.payloads), context_rows=r.context_rows, predict_rows=r.predict_rows,
                                     payload_mb=round(sum(len(json.dumps(p, default=str)) for p in r.payloads)/1e6, 2)) for k, r in reqs.items()}
        report["note"] = "No KUMO_API_KEY or NVIDIA_API_KEY set: requests built locally, nothing sent."
    else:
        res = run_kumo(cut, ids) if b == "kumo" else run_nvidia(build_requests(cut, ids, a.ctx, a.stat), ids)
        for name, (pred, secs) in res.items():
            ok = pred.notna()
            if not ok.any(): report[f"kumo_{name}"] = {"error": ERRORS.get(name, "no predictions")}; continue
            report[f"kumo_{name}"] = dict(score(QUERIES[name][1], y[name][ok.values], pred[ok].values), seconds=round(secs, 1), scored=int(ok.sum()))
    if ERRORS: report["errors"] = ERRORS
    print(json.dumps(report, indent=2)); json.dump(report, open(f"{OUT}/kumo_trial_{b}{TAG}.json", "w"), indent=2)

if __name__ == "__main__":
    main()
