"""Baseline: classic models trained on the synthetic data for the churn task used in Kumo.
Task: for each member with >=1 visit in the 30 days before the anchor date, predict
1 = NO visits in the next 30 days. Monthly anchors. Temporal split: train anchors
2022-03..2024-12, test anchors 2025-01..2025-08-01 (no random split, no leakage)."""
import os, pandas as pd, numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),"data")+os.sep
v=pd.read_parquet(D+"visits.parquet"); pay=pd.read_parquet(D+"payments.parquet"); c=pd.read_parquet(D+"contracts.parquet")
pl=pd.read_parquet(D+"plans.parquet"); m=pd.read_parquet(D+"members.parquet"); pk=pd.read_parquet(D+"pt_packages.parquet"); s=pd.read_parquet(D+"pt_sessions.parquet")
c=c.merge(pl[["plan_id","plan_type"]],on="plan_id")
va=v[v.result=="Allowed"]; vd=v[v.result=="Denied"]
anchors=pd.date_range("2022-03-01","2025-08-01",freq="MS")
def cnt(df,col,t,a,b):  # events in [t+a, t+b) days
    x=df[(df[col]>=t+pd.Timedelta(days=a))&(df[col]<t+pd.Timedelta(days=b))]; return x.groupby("member_id").size()
rows=[]
for t in anchors:
    f=pd.DataFrame(index=m.member_id)
    for a in (7,30,90): f[f"visits_{a}d"]=cnt(va,"swipe_time",t,-a,0)
    f["denied_30d"]=cnt(vd,"swipe_time",t,-30,0)
    last=va[va.swipe_time<t].groupby("member_id").swipe_time.max(); f["days_since_visit"]=(t-last).dt.days
    p=pay[(pay.payment_time<t)&~pay.is_refund]
    f["spend_90d"]=p[p.payment_time>=t-pd.Timedelta(days=90)].groupby("member_id").net_amount.sum()
    f["spend_total"]=p.groupby("member_id").net_amount.sum()
    f["n_membership_payments"]=p[p.product_type=="Membership"].groupby("member_id").size()
    cc=c[c.start_date<t].sort_values("start_date").groupby("member_id").last()
    f["days_to_contract_end"]=(cc.end_date-t).dt.days; f["plan_type"]=cc.plan_type
    f["n_contracts"]=c[c.start_date<t].groupby("member_id").size()
    f["pt_sessions_30d"]=cnt(s[s.is_completed],"session_time",t,-30,0)
    f["pt_paid_ever"]=pk[(pk.purchase_time<t)&pk.is_paid].groupby("member_id").size()
    mm=m.set_index("member_id"); f["tenure_days"]=(t-mm.join_date).dt.days; f["age"]=mm.age
    f=f[f.tenure_days>=0]
    f["trend"]=f.visits_30d.fillna(0)-f.visits_90d.fillna(0)/3
    f["label"]=(cnt(va,"swipe_time",t,0,30).reindex(f.index).fillna(0)==0).astype(int)
    f=f[f.visits_30d.fillna(0)>0]; f["anchor"]=t; rows.append(f.reset_index())
X=pd.concat(rows); X.to_parquet(D+"churn_features.parquet",index=False)
num=[c_ for c_ in X.columns if c_ not in("member_id","anchor","label","plan_type")]
X[num]=X[num].fillna({k:0 for k in num if k not in("days_since_visit","days_to_contract_end")}).fillna(-1)
X=pd.get_dummies(X,columns=["plan_type"],dummy_na=True)
feats=[c_ for c_ in X.columns if c_ not in("member_id","anchor","label")]
tr=X[X.anchor<"2025-01-01"]; te=X[X.anchor>="2025-01-01"]
def metrics(y,p):
    k=max(1,int(len(y)*0.1)); top=np.argsort(-p)[:k]
    return dict(roc_auc=round(roc_auc_score(y,p),4), pr_auc=round(average_precision_score(y,p),4), precision_top10=round(float(np.asarray(y)[top].mean()),4))
res={"n_train":len(tr),"n_test":len(te),"test_churn_rate":round(te.label.mean(),4)}
lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(tr[feats].astype(float),tr.label)
res["logistic_regression"]=metrics(te.label,lr.predict_proba(te[feats].astype(float))[:,1])
gb=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,random_state=0).fit(tr[feats].astype(float),tr.label)
res["lightgbm"]=metrics(te.label,gb.predict_proba(te[feats].astype(float))[:,1])
res["naive_days_since_visit"]=metrics(te.label,te.days_since_visit.values)
imp=pd.Series(gb.booster_.feature_importance("gain"),index=feats).sort_values(ascending=False)
res["lightgbm_top_features"]=list(imp.index[:6])
print(json.dumps(res,indent=2)); json.dump(res,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"results","baseline_results.json"),"w"),indent=2)
