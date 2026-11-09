"""Fair comparison: Kumo gets the same information LightGBM had (synthetic data only).

Variants (same 300 test members, anchor 2025-07-01, tables cut at the anchor):
  R_multi    Kumo Relational, raw tables, context sampled from the same monthly anchors LightGBM
             trained on (2022-03-01 .. 2025-05-01), merged into one request.
  R_feat     Same, plus LightGBM's engineered features as columns on each member row
             (computed per member *at that row's own anchor*, so no leakage).
  T_same     Kumo Tabular on LightGBM's feature table, the same rows Kumo Relational got as context.
  T_10k      Kumo Tabular, 10,000 random rows of LightGBM's training table (endpoint cap).
  LGB_full   LightGBM on all training rows.   LGB_same  LightGBM on the same rows Kumo got.
Fairness fixes vs the first trial: visits = Allowed swipes only (denied swipes moved to their
own table) and refunded payments removed, so Kumo's targets equal the evaluation labels;
all three queries use the same population filter as LightGBM (visited in the prior 30 days).
Usage: NVIDIA_API_KEY must be set. python fair_trial.py (options: per-anchor rows per month, default 77; only a comma list of variants)
"""
import os, json, time, argparse, warnings, copy
import numpy as np, pandas as pd, requests
import sdm_client as sc
from sdm_client.adapters.relational import RelationalAdapter
from sdm_client.adapters.tabular import TabularAdapter
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
import lightgbm as lgb
import kumo_trial as k
warnings.filterwarnings("ignore")
OUT = k.OUT; ANCHOR = k.ANCHOR
URL = {"rel": "https://ai.api.nvidia.com/v1/structured-data/nvidia/kumo-relational/predictions",
       "tab": "https://ai.api.nvidia.com/v1/structured-data/nvidia/kumo-tabular/predictions"}
WHERE = " WHERE COUNT(visits.*, -30, 0, days) > 0"
Q = {"churn_30d": ("PREDICT COUNT(visits.*, 0, 30, days) = 0 FOR EACH members.member_id" + WHERE, "binary"),
     "revenue_30d": ("PREDICT SUM(payments.net_amount, 0, 30, days) FOR EACH members.member_id" + WHERE, "regression"),
     "visits_30d": ("PREDICT COUNT(visits.*, 0, 30, days) FOR EACH members.member_id" + WHERE, "regression")}
TRAIN_ANCHORS = pd.date_range("2022-03-01", "2025-05-01", freq="MS")   # = LightGBM's training anchors
FEATS_NUM = ["visits_7d","visits_30d","visits_90d","denied_30d","days_since_visit","spend_90d","spend_total",
             "n_membership_payments","days_to_contract_end","n_contracts","pt_sessions_30d","pt_paid_ever","tenure_days","age","trend"]

STAT = os.getenv("FAIR_STAT", "mean")   # "median" matches LightGBM's L1 (MAE) objective
TASKS = os.getenv("FAIR_TASKS", "")

def tables():
    full, cut = k.load(); ids = k.pick_members(full, 300)
    for d in (full, cut):
        v = d["visits"]; d["denied_swipes"] = v[v.result == "Denied"].rename(columns={"visit_id": "denied_id"}).reset_index(drop=True)
        d["visits"] = v[v.result == "Allowed"].drop(columns=["result", "deny_reason"]).reset_index(drop=True)
        d["payments"] = d["payments"][~d["payments"].is_refund].drop(columns=["is_refund"]).reset_index(drop=True)
    return full, cut, ids

def feature_table(full):
    """LightGBM's monthly feature table + the three labels (vectorised)."""
    X = pd.read_parquet(f"{k.D}/churn_features.parquet").sort_values(["anchor", "member_id"]).reset_index(drop=True)
    codes = {m: i for i, m in enumerate(full["members"].member_id)}
    v, p = full["visits"], full["payments"]
    vt, vc = v.swipe_time.values, v.member_id.map(codes).values
    pt, pc, pa = p.payment_time.values, p.member_id.map(codes).values, p.net_amount.values
    xc = X.member_id.map(codes).values; X["revenue_30d"] = 0.0; X["visits_30d_next"] = 0.0
    for t in X.anchor.unique():
        t0 = np.datetime64(pd.Timestamp(t)); t1 = t0 + np.timedelta64(30, "D"); rows = (X.anchor == t).values
        mv = (vt >= t0) & (vt < t1); mp = (pt >= t0) & (pt < t1)
        X.loc[rows, "visits_30d_next"] = np.bincount(vc[mv], minlength=len(codes))[xc[rows]]
        X.loc[rows, "revenue_30d"] = np.bincount(pc[mp], weights=pa[mp], minlength=len(codes))[xc[rows]]
    X = X.rename(columns={"label": "churn_30d"}); X["visits_next"] = X.pop("visits_30d_next")
    X[FEATS_NUM] = X[FEATS_NUM].astype(float)
    for c in FEATS_NUM:
        if c not in ("days_since_visit", "days_to_contract_end"): X[c] = X[c].fillna(0)
    X["plan_type"] = X.plan_type.fillna("none")
    return X

LABEL = {"churn_30d": "churn_30d", "revenue_30d": "revenue_30d", "visits_30d": "visits_next"}

def post(kind, payload, tag):
    key = os.environ["NVIDIA_API_KEY"]; body = json.dumps(payload, default=str)
    mb = len(body) / 1e6; t0 = time.time(); last = None
    for attempt in range(4):
        try:
            r = requests.post(URL[kind], data=body, timeout=420, headers={"Authorization": f"Bearer {key}",
                              "Accept": "application/json", "Content-Type": "application/json"})
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"; print(f"   {tag}: {last}, retry", flush=True); time.sleep(20 * (attempt + 1)); continue
            if r.status_code != 200:
                return None, f"HTTP {r.status_code}: {r.text[:300]}", mb, time.time() - t0
            js = r.json(); json.dump(js, open(f"{OUT}/fair_{tag}.json", "w"))
            return js, None, mb, time.time() - t0
        except requests.exceptions.RequestException as e:
            last = type(e).__name__; print(f"   {tag}: {last}, retry", flush=True); time.sleep(20 * (attempt + 1))
    return None, f"failed after retries ({last})", mb, time.time() - t0

def parse(js, kind):
    out = {}
    for it in js["predictions"]:
        if kind == "binary":
            pr = it["probabilities"]; out[it["row_index"]] = float(pr.get("true", pr.get("True", pr.get("1", list(pr.values())[-1]))))
        else:
            pv = it["prediction"]; out[it["row_index"]] = float(pv["mean"] if isinstance(pv, dict) else pv)
    return out

def relational_payload(cut, ids, name, per_anchor, feats=None, seed=0):
    """Merge context gathered at each training anchor into one request (instance/node ids re-numbered)."""
    q, kind = Q[name]; sch = sc.Schema.from_data(cut); opts = {"prediction_statistic": STAT} if kind == "regression" else {}
    base = None; ctx_rows = []; inst_off = 0; node_off = 0
    ctx = None
    for i, A in enumerate(TRAIN_ANCHORS):
        r = sc.PredictiveQuery(q, sch).materialize(indices=ids, num_context_samples=per_anchor, anchor_time=ANCHOR,
                                                    context_anchor_time=A, random_seed=seed + i, **opts)
        pl = next(RelationalAdapter().prediction_plan(r, ("prediction", "probabilities") if kind == "binary" else ("prediction",)))
        if base is None:
            base = copy.deepcopy(pl); ctx = {"instance_table": copy.deepcopy(pl["context"]["instance_table"]), "related_tables": {}}
            ctx["instance_table"]["rows"] = []
            for t, tb in pl["context"]["related_tables"].items(): ctx["related_tables"][t] = {**tb, "rows": []}
        c = pl["context"]; ic = c["instance_table"]["columns"]; ii = ic.index("instance_id")
        rows = c["instance_table"]["rows"]
        if not rows: continue
        ai = ic.index("__kumo_anchor_time")   # the engine sometimes steps back past the first anchor; keep exact anchors only
        rows = [row for row in rows if pd.Timestamp(row[ai]).tz_localize(None) == A]
        if not rows: continue
        m = {row[ii]: row[ii] + inst_off for row in rows}
        for row in rows: row = list(row); row[ii] = m[row[ii]]; ctx["instance_table"]["rows"].append(row)
        mx_node = 0
        for t, tb in c["related_tables"].items():
            cols = tb["columns"]; ti = cols.index("instance_id"); ni = cols.index("__node_id") if "__node_id" in cols else None
            for row in tb["rows"]:
                if row[ti] not in m: continue
                row = list(row); row[ti] = m[row[ti]]
                if ni is not None: mx_node = max(mx_node, row[ni]); row[ni] += node_off
                ctx["related_tables"][t]["rows"].append(row)
        inst_off = max(m.values()) + 1; node_off += mx_node + 1
    base["context"] = ctx
    n_ctx = len(ctx["instance_table"]["rows"])   # mode is chosen by the engine from the per-anchor size; set it for the merged size
    if isinstance(base.get("inference"), dict) and n_ctx > 1000: base["inference"]["run_mode"] = "best"
    # predict half: re-number instance ids above the context, node ids above context nodes
    P = base["predict"]; pi = P["instance_table"]["columns"].index("instance_id")
    shift = inst_off + 10; nshift = node_off + 10
    P["instance_table"]["rows"] = [[*row[:pi], row[pi] + shift, *row[pi + 1:]] for row in P["instance_table"]["rows"]]
    for t, tb in P["related_tables"].items():
        cols = tb["columns"]; ti = cols.index("instance_id"); ni = cols.index("__node_id") if "__node_id" in cols else None
        new = []
        for row in tb["rows"]:
            row = list(row); row[ti] += shift
            if ni is not None: row[ni] += nshift
            new.append(row)
        tb["rows"] = new
    if feats is not None: add_features(base, feats)
    return base

def add_features(pl, X):
    """Add LightGBM's features (as of each instance's own anchor) to the members rows."""
    look = X.set_index(["member_id", "anchor"])
    mem = pl["schema"]["related_tables"]["members"]["columns"]
    for f in FEATS_NUM: mem[f"f_{f}"] = {"dtype": "float64", "stype": "numerical"}
    mem["f_plan_type"] = {"dtype": "string", "stype": "categorical"}
    for half in ("context", "predict"):
        it = pl[half]["instance_table"]; cols = it["columns"]
        ai, ei, ii = cols.index("__kumo_anchor_time"), cols.index("__kumo_entity_ref_0"), cols.index("instance_id")
        anc = {row[ii]: (row[ei], pd.Timestamp(row[ai]).tz_localize(None)) for row in it["rows"]}
        tb = pl[half]["related_tables"]["members"]; tb["columns"] = tb["columns"] + [f"f_{f}" for f in FEATS_NUM] + ["f_plan_type"]
        ti = tb["columns"].index("instance_id")
        new = []
        for row in tb["rows"]:
            key = anc.get(row[ti]); vals = [None] * (len(FEATS_NUM) + 1)
            if key in look.index:
                r = look.loc[key]; vals = [None if pd.isna(r[f]) else float(r[f]) for f in FEATS_NUM] + [str(r.plan_type)]
            new.append(list(row) + vals)
        tb["rows"] = new

def context_keys(pl):
    it = pl["context"]["instance_table"]; c = it["columns"]
    return pd.DataFrame({"member_id": [r[c.index("__kumo_entity_ref_0")] for r in it["rows"]],
                         "anchor": [pd.Timestamp(r[c.index("__kumo_anchor_time")]).tz_localize(None) for r in it["rows"]],
                         "kumo_target": [r[c.index("TARGET")] for r in it["rows"]]})

def metrics(kind, y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); ok = ~np.isnan(p); y, p = y[ok], p[ok]
    if kind == "binary":
        top = np.argsort(-p)[:max(1, len(y) // 10)]
        return dict(roc_auc=round(roc_auc_score(y, p), 4), pr_auc=round(average_precision_score(y, p), 4), precision_top10=round(y[top].mean(), 4), n=int(len(y)))
    return dict(mae=round(mean_absolute_error(y, p), 2), pred_mean=round(p.mean(), 2), truth_mean=round(y.mean(), 2), n=int(len(y)))

def lgb_fit_predict(kind, tr, te, label):
    F = FEATS_NUM + ["plan_type"]
    def prep(d):
        Z = d[F].copy(); Z["plan_type"] = Z.plan_type.astype("category").cat.set_categories(sorted(X_ALL.plan_type.unique())); return Z
    if kind == "binary":
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
        m.fit(prep(tr), tr[label]); return m.predict_proba(prep(te))[:, 1]
    m = lgb.LGBMRegressor(objective="l1", n_estimators=400, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    m.fit(prep(tr), tr[label]); return m.predict(prep(te))

def tabular_payload(kind, tr, te, label):
    F = FEATS_NUM + ["plan_type"]
    ctx = tr[F + [label]].reset_index(drop=True).copy(); prd = te[F].reset_index(drop=True).copy()
    if kind == "binary": ctx[label] = ctx[label].map({1: "true", 0: "false"})
    req = sc.Request(context=ctx, predict=prd, target=label, task="classification" if kind == "binary" else "regression")
    fields = ("prediction", "probabilities") if kind == "binary" else ("prediction",)
    pl = next(TabularAdapter().prediction_plan(req, fields))
    if kind == "regression": pl.setdefault("output", {})["prediction_statistic"] = STAT
    return pl

def main():
    global X_ALL
    ap = argparse.ArgumentParser(); ap.add_argument("--per-anchor", type=int, default=77); ap.add_argument("--only", default="")
    a = ap.parse_args(); only = set(a.only.split(",")) - {""}
    assert os.getenv("NVIDIA_API_KEY"), "NVIDIA_API_KEY not set"
    full, cut, ids = tables()
    X_ALL = X = feature_table(full)
    te = X[X.anchor == ANCHOR].set_index("member_id").reindex(ids).reset_index()
    tr_full = X[X.anchor <= ANCHOR - pd.Timedelta(days=31)]
    res = {"test_members": len(ids), "anchor": str(ANCHOR.date()), "train_rows_full": len(tr_full)}; lat = {}
    for name, (q, kind) in Q.items():
        if TASKS and name not in TASKS.split(","): continue
        lab = LABEL[name]; y = te[lab].values; R = res.setdefault(name, {})
        R["LGB_full"] = metrics(kind, y, lgb_fit_predict(kind, tr_full, te, lab))
        keys = None
        for var in ("R_multi", "R_feat"):
            if only and var not in only and "T_same" not in only and "LGB_same" not in only: continue
            pl = relational_payload(cut, ids, name, a.per_anchor, feats=X if var == "R_feat" else None)
            if keys is None:
                keys = context_keys(pl); chk = keys.merge(X[["member_id", "anchor", lab]], on=["member_id", "anchor"], how="left")
                agree = (chk.kumo_target.astype(float).round(2) == chk[lab].astype(float).round(2)).mean()
                R["context"] = dict(rows=len(keys), anchors=int(keys.anchor.nunique()), label_agreement_with_lgb_table=round(float(agree), 4),
                                    in_lgb_table=round(float(chk[lab].notna().mean()), 4))
            if only and var not in only: continue
            js, err, mb, secs = post("rel", pl, f"{var}_{name}_{STAT}")
            lat[f"{var}_{name}"] = dict(seconds=round(secs, 1), payload_mb=round(mb, 1))
            if err: R[var] = {"error": err}; print(name, var, err, flush=True); continue
            pr = parse(js, kind); P = pl["predict"]["instance_table"]; ei = P["columns"].index("__kumo_entity_ref_0")
            s = pd.Series({P["rows"][i][ei]: v for i, v in pr.items()}).reindex(ids)
            R[var] = metrics(kind, y, s.values); print(name, var, R[var], lat[f"{var}_{name}"], flush=True)
        same = keys.merge(X, on=["member_id", "anchor"], how="inner")
        R["LGB_same"] = metrics(kind, y, lgb_fit_predict(kind, same, te, lab)); R["LGB_same"]["train_rows"] = len(same)
        for var, trs in (("T_same", same), ("T_10k", tr_full.sample(10000, random_state=0))):
            if only and var not in only: continue
            pl = tabular_payload(kind, trs, te, lab)
            js, err, mb, secs = post("tab", pl, f"{var}_{name}_{STAT}")
            lat[f"{var}_{name}"] = dict(seconds=round(secs, 1), payload_mb=round(mb, 1))
            if err: R[var] = {"error": err}; print(name, var, err, flush=True); continue
            pr = parse(js, kind); R[var] = metrics(kind, y, [pr.get(i, np.nan) for i in range(len(te))]); print(name, var, R[var], lat[f"{var}_{name}"], flush=True)
        json.dump({**res, "latency": lat}, open(f"{OUT}/fair_results.json", "w"), indent=2, default=str)
    res["latency"] = lat
    json.dump(res, open(f"{OUT}/fair_results.json", "w"), indent=2, default=str); print(json.dumps(res, indent=1, default=str))

if __name__ == "__main__":
    main()
