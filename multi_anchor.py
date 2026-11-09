"""Multi anchor comparison with bootstrap confidence intervals (fictional data only).

For each monthly test anchor from 2025-01-01 to 2025-08-01:
  * the same 300 members (active in the 30 days before the anchor, picked with kumo_trial.pick_members)
    are scored by every model;
  * tables are cut at the anchor before they reach Kumo; labels come from the full tables;
  * LightGBM trains only on feature rows whose 30 day label window closes before the anchor
    (anchor <= A minus 31 days), and Kumo's in context examples come from those same training anchors.
Models: R_multi (Kumo Relational, raw tables), R_feat (same plus LightGBM features on member rows),
T_10k (Kumo Tabular, 10,000 random LightGBM training rows), LGB_full (LightGBM, all training rows),
LGB_same (LightGBM trained only on the rows Kumo Relational got as context).
Regression tasks ask Kumo for the median, which is what MAE rewards and what LightGBM's L1 objective fits.
95% intervals: member level cluster bootstrap over the pooled member months, paired across models.
Usage: NVIDIA_API_KEY must be set. python multi_anchor.py [--anchors 2025-01-01,...] [--boot 2000]
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
os.environ.setdefault("FAIR_STAT", "median")
import kumo_trial as k
import fair_trial as ft

CTX_TOTAL = 3000  # about the 2,990 context rows of fair_trial.py at 2025-07-01

def run_anchor(A, preds, log):
    k.ANCHOR = A; ft.ANCHOR = A
    ft.TRAIN_ANCHORS = pd.date_range("2022-03-01", A - pd.Timedelta(days=31), freq="MS")
    per = max(1, round(CTX_TOTAL / len(ft.TRAIN_ANCHORS)))
    full, cut, ids = ft.tables()
    X = ft.feature_table(full); ft.X_ALL = X
    te = X[X.anchor == A].set_index("member_id").reindex(ids).reset_index()
    assert te.churn_30d.notna().all(), "member missing from feature table"
    tr_full = X[X.anchor <= A - pd.Timedelta(days=31)]
    tag_a = A.strftime("%Y%m")
    for name, (q, kind) in ft.Q.items():
        lab = ft.LABEL[name]
        rec = pd.DataFrame({"anchor": A, "member_id": ids, "task": name, "y": te[lab].values})
        rec["LGB_full"] = ft.lgb_fit_predict(kind, tr_full, te, lab)
        keys = None
        for var in ("R_multi", "R_feat"):
            pl = ft.relational_payload(cut, ids, name, per, feats=X if var == "R_feat" else None)
            if keys is None:
                keys = ft.context_keys(pl)
                chk = keys.merge(X[["member_id", "anchor", lab]], on=["member_id", "anchor"], how="left")
                agree = float((chk.kumo_target.astype(float).round(2) == chk[lab].astype(float).round(2)).mean())
                log.append(dict(anchor=str(A.date()), task=name, context_rows=len(keys), context_anchors=int(keys.anchor.nunique()),
                                max_context_anchor=str(keys.anchor.max().date()), label_agreement=round(agree, 4)))
            js, err, mb, secs = ft.post("rel", pl, f"{var}_{name}_{ft.STAT}_{tag_a}")
            if err:
                raise RuntimeError(f"{A.date()} {name} {var}: {err}")
            pr = ft.parse(js, kind); P = pl["predict"]["instance_table"]; ei = P["columns"].index("__kumo_entity_ref_0")
            rec[var] = pd.Series({P["rows"][i][ei]: v for i, v in pr.items()}).reindex(ids).values
            print(A.date(), name, var, f"{secs:.1f}s", f"{mb:.1f}MB", flush=True)
        same = keys.merge(X, on=["member_id", "anchor"], how="inner")
        rec["LGB_same"] = ft.lgb_fit_predict(kind, same, te, lab)
        pl = ft.tabular_payload(kind, tr_full.sample(10000, random_state=0), te, lab)
        js, err, mb, secs = ft.post("tab", pl, f"T_10k_{name}_{ft.STAT}_{tag_a}")
        if err:
            raise RuntimeError(f"{A.date()} {name} T_10k: {err}")
        pr = ft.parse(js, kind); rec["T_10k"] = [pr.get(i, np.nan) for i in range(len(te))]
        print(A.date(), name, "T_10k", f"{secs:.1f}s", flush=True)
        preds.append(rec)

MODELS = ["R_multi", "R_feat", "T_10k", "LGB_full", "LGB_same"]

def metr(kind, y, p):
    if kind == "binary":
        top = np.argsort(-p, kind="stable")[:max(1, len(y) // 10)]
        return dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), precision_top10=y[top].mean())
    return dict(mae=mean_absolute_error(y, p))

def summarize(df, B, seed=0):
    out = {}
    for name, (q, kind) in ft.Q.items():
        d = df[df.task == name].dropna(subset=MODELS).reset_index(drop=True)
        y = d.y.values.astype(float); P = {m: d[m].values.astype(float) for m in MODELS}
        point = {m: metr(kind, y, P[m]) for m in MODELS}
        mem = d.member_id.values; um = np.unique(mem); groups = {u: np.where(mem == u)[0] for u in um}
        rng = np.random.default_rng(seed); boots = {m: [] for m in MODELS}
        for _ in range(B):
            idx = np.concatenate([groups[u] for u in rng.choice(um, len(um), replace=True)])
            if kind == "binary" and y[idx].min() == y[idx].max(): continue
            for m in MODELS: boots[m].append(metr(kind, y[idx], P[m][idx]))
        R = {"n_member_months": int(len(d)), "n_members": int(len(um)), "n_anchors": int(d.anchor.nunique()),
             "positive_rate" if kind == "binary" else "truth_mean": round(float(y.mean()), 4), "models": {}}
        for m in MODELS:
            R["models"][m] = {}
            for met, v in point[m].items():
                bs = np.array([b[met] for b in boots[m]])
                R["models"][m][met] = dict(value=round(float(v), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])
                if m != "LGB_full":
                    diff = bs - np.array([b[met] for b in boots["LGB_full"]])
                    R["models"][m][met]["minus_LGB_full"] = dict(value=round(float(v - point["LGB_full"][met]), 4),
                        ci95=[round(float(np.percentile(diff, 2.5)), 4), round(float(np.percentile(diff, 97.5)), 4)])
        R["per_anchor"] = {str(a.date()): {m: {kk: round(float(vv), 4) for kk, vv in metr(kind, g.y.values.astype(float), g[m].values.astype(float)).items()}
                                           for m in MODELS} for a, g in d.groupby("anchor")}
        out[name] = R
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default=",".join(str(x.date()) for x in pd.date_range("2025-01-01", "2025-08-01", freq="MS")))
    ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args(); path = f"{k.OUT}/multi_anchor_predictions.csv.gz"
    if not a.summarize_only:
        assert os.getenv("NVIDIA_API_KEY"), "NVIDIA_API_KEY not set"
        preds, log, failure = [], [], None
        for s in a.anchors.split(","):
            try: run_anchor(pd.Timestamp(s), preds, log)
            except Exception as e:
                failure = f"{type(e).__name__}: {str(e)[:400]}"; print("STOP:", failure, flush=True); break
        if preds: pd.concat(preds).to_csv(path, index=False)
        json.dump(dict(context_log=log, failure=failure, stat=ft.STAT, ctx_total=CTX_TOTAL), open(f"{k.OUT}/multi_anchor_run.json", "w"), indent=2)
    df = pd.read_csv(path, parse_dates=["anchor"]); df["member_id"] = df.member_id.astype(str)
    res = summarize(df, a.boot)
    json.dump(res, open(f"{k.OUT}/multi_anchor_summary.json", "w"), indent=2)
    for t, R in res.items():
        print(t, R["n_member_months"], {m: {kk: (vv["value"], vv["ci95"]) for kk, vv in x.items()} for m, x in R["models"].items()})

if __name__ == "__main__":
    main()
