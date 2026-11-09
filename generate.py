"""Fictional premium gym dataset generator. Every value below is invented for this benchmark.
Prices, plan mix, hours, seasonality, renewal rules and labels were chosen by hand as round,
illustrative numbers. None was measured from, or calibrated to, a real operator's data.
Seeded (42) and reproducible."""
import numpy as np, pandas as pd, os
rng = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(__file__), "data")
START, END = pd.Timestamp("2022-01-01"), pd.Timestamp("2025-08-31")
DAYS = pd.date_range(START, END, freq="D"); ND = len(DAYS)
N_MEMBERS = 2400
VAT = 0.10

# plans (dimension): invented round prices
plans = pd.DataFrame([
 # plan_id, name, type, months, net_price, weight
 (1, "Essential monthly", "monthly", 1, 150.00, 0.40),
 (2, "Signature monthly", "monthly", 1, 240.00, 0.18),
 (3, "Essential annual", "annual", 12, 1500.00, 0.16),
 (4, "Signature annual", "annual", 12, 2400.00, 0.10),
 (5, "Essential half year", "semiannual", 6, 810.00, 0.06),
 (6, "Guest trial", "trial", 0, 0.00, 0.06),
 (7, "Visitor 7 day pass", "tourist", 0, 90.00, 0.02),
 (8, "Visitor 3 day pass", "tourist", 0, 45.00, 0.02),
], columns=["plan_id","plan_name","plan_type","duration_months","net_price","w"])
PW = plans.w.values / plans.w.sum()
PLAN = plans.set_index("plan_id")

# helpers: a January peak and a summer dip, both invented
season = 1 + 0.13*np.cos(2*np.pi*(DAYS.day_of_year.values-15)/365)
DOW = np.array([1.20,1.15,1.10,1.05,0.85,0.80,0.85])  # Mon..Sun, invented weekly profile
season *= DOW[DAYS.day_of_week.values]
HOURS = np.arange(24)
# Invented two peak day: early morning and early evening
HP = 0.48*np.exp(-0.5*((HOURS-7)/1.5)**2) + 0.52*np.exp(-0.5*((HOURS-18)/1.8)**2); HP[HOURS<5] = 0; HP/=HP.sum()
PAYM = ["card","direct_debit","app_wallet","cash"]
PAYW = np.array([.45,.35,.15,.05])
SRC = ["front_desk","member_app","website"]; SRCW=[.40,.40,.20]
didx = lambda ts: int((ts-START).days)

# join dates: growth over time (more joins later), weekly noise
jw = np.linspace(1.0, 1.4, ND) * season
jw /= jw.sum()
join_idx = np.sort(rng.choice(ND-30, N_MEMBERS, p=jw[:ND-30]/jw[:ND-30].sum()))

members, contracts, payments, visits, pt_packages, pt_sessions = [],[],[],[],[],[]
trainers = pd.DataFrame({"trainer_id":[f"T{i:02d}" for i in range(1,19)],
    "role": rng.choice(["Coach","Personal Trainer","Physio"],18,p=[.3,.6,.1]),
    "hire_date": pd.to_datetime(rng.choice(pd.date_range("2020-01-01","2024-06-30"),18))})
tw = rng.dirichlet(np.ones(18)*2.0)
PTPK = [("Starter coaching",4,280.00,.30),("Regular coaching",8,520.00,.45),("Intensive coaching",16,960.00,.25)]
PTW = np.array([p[3] for p in PTPK]); PTW/=PTW.sum()
cid=pid=vid=kid=sid=0

for m in range(N_MEMBERS):
    mid = f"M{m+1:05d}"
    j = int(join_idx[m]); jdate = DAYS[j]
    age = int(np.clip(rng.normal(42.0, 12.0), 18, 75))
    home = rng.choice(["SITE_NORTH","SITE_SOUTH"], p=[.55,.45])
    corporate = rng.random() < 0.12
    eng = rng.lognormal(np.log(2.1), 0.5)          # latent visits/week
    loyalty = rng.beta(2.0, 1.4)                    # drives renewal
    pref_h = rng.choice(HOURS, p=HP)
    members.append(dict(member_id=mid, join_date=jdate.normalize(), age=age, home_club=home,
                        gender=rng.choice(["Female","Male"], p=[.52,.48]), is_corporate=corporate))
    # contract chain
    plan_id = int(rng.choice(plans.plan_id, p=PW))
    t = j; active = np.zeros(ND, bool); churn_day = None
    first = True
    while t < ND:
        P = PLAN.loc[plan_id]
        if P.plan_type == "trial": length = 7
        elif P.plan_type == "tourist": length = 7 if "7" in P.plan_name else 3
        else: length = int(round(P.duration_months*30.4))
        s = DAYS[t]; e = DAYS[min(t+length-1, ND-1)]
        cid += 1; contract_id = f"C{cid:06d}"
        contracts.append(dict(contract_id=contract_id, member_id=mid, plan_id=plan_id, start_date=s, end_date=DAYS[t+length-1] if t+length-1<ND else s+pd.Timedelta(days=length-1)))
        active[t:min(t+length, ND)] = True
        if P.net_price > 0:
            pid += 1
            disc = rng.random() < 0.05
            net = round(P.net_price*(0.90 if disc else 1), 2)
            payments.append(dict(payment_id=f"P{pid:07d}", member_id=mid, contract_id=contract_id, payment_time=s+pd.Timedelta(hours=int(rng.choice(HOURS,p=HP)), minutes=int(rng.integers(60))),
                product_type="Membership", plan_id=plan_id, net_amount=net, vat_amount=round(net*VAT,2), gross_amount=round(net*(1+VAT),2),
                payment_method=rng.choice(PAYM,p=PAYW), source=rng.choice(SRC,p=SRCW), has_discount=disc, is_refund=False))
            if first and P.plan_type=="monthly" and rng.random()<0.25:   # joining fee
                pid += 1; fee = float(rng.choice([50.0, 100.0]))
                payments.append(dict(payment_id=f"P{pid:07d}", member_id=mid, contract_id=contract_id, payment_time=s+pd.Timedelta(hours=11), product_type="AdminFee", plan_id=plan_id,
                    net_amount=fee, vat_amount=round(fee*VAT,2), gross_amount=round(fee*(1+VAT),2), payment_method=rng.choice(PAYM,p=PAYW), source=rng.choice(SRC,p=SRCW), has_discount=False, is_refund=False))
        first = False
        t_end = t + length
        if t_end >= ND: break
        # renewal decision: depends on loyalty and engagement
        if P.plan_type in ("trial","tourist"):
            p_ren = 0.30*loyalty + 0.20; next_plan = int(rng.choice([1,2,3,4],p=[.5,.2,.2,.1]))
        elif P.plan_type == "monthly":
            p_ren = np.clip(0.55 + 0.30*loyalty + 0.03*min(eng,3), 0.40, 0.97); next_plan = plan_id if rng.random()<0.90 else 3
        else:
            p_ren = np.clip(0.45 + 0.35*loyalty + 0.03*min(eng,3), 0.3, 0.92); next_plan = int(rng.choice([plan_id,1,3],p=[.6,.3,.1]))
        if rng.random() < p_ren:
            gap = int(rng.choice([0,0,0,0,1,7,14]))   # same-day/short-gap renewals
            if P.plan_type=="monthly" and rng.random()<0.03:  # freeze a month
                pid+=1
                payments.append(dict(payment_id=f"P{pid:07d}", member_id=mid, contract_id=contract_id, payment_time=DAYS[t_end-1]+pd.Timedelta(hours=9), product_type="Freeze", plan_id=plan_id,
                    net_amount=0.0, vat_amount=0.0, gross_amount=0.0, payment_method="card", source="front_desk", has_discount=False, is_refund=False))
                gap += 21
            t = t_end + gap; plan_id = next_plan
        else:
            churn_day = t_end
            # some come back after a long break (win-back)
            if rng.random() < 0.30:
                t = t_end + int(rng.integers(60, 300)); plan_id = int(rng.choice([1,3],p=[.7,.3])); loyalty *= 0.7; churn_day=None
            else: break
    # visits
    # engagement trend: decays before contract ends without renewal
    p = np.zeros(ND)
    p[active] = np.clip(eng/7.0, 0, 0.90)
    if churn_day is not None:
        w = np.arange(max(j, churn_day-30), churn_day)
        p[w] *= np.linspace(1.0, 0.40, len(w))
    # month-to-month variability and occasional 'off' months (travel, injury)
    mo = (DAYS.year.values-2022)*12 + DAYS.month.values-1
    mm_mult = rng.lognormal(0, 0.4, mo.max()+1) * (rng.random(mo.max()+1) > 0.06)
    p *= season * mm_mult[mo]
    p = np.clip(p, 0, 0.92)
    vd = np.nonzero(rng.random(ND) < p)[0]
    # denied attempts after churn or occasional debit
    den = []
    if churn_day is not None and churn_day < ND-1:
        k = rng.poisson(1.5); den = list(rng.integers(churn_day, min(ND, churn_day+21), k))
    for d in vd:
        h = int(np.clip(rng.normal(pref_h, 2.5), 5, 22)) if rng.random()<0.60 else int(rng.choice(HOURS,p=HP))
        vid += 1
        res, why = ("Allowed","")
        if rng.random() < 0.006: res, why = "Denied", rng.choice(["membership_paused","payment_overdue"],p=[.5,.5])
        visits.append((f"V{vid:08d}", mid, DAYS[d]+pd.Timedelta(hours=h, minutes=int(rng.integers(60))), res, why))
    for d in den:
        vid += 1; visits.append((f"V{vid:08d}", mid, DAYS[d]+pd.Timedelta(hours=int(rng.choice(HOURS,p=HP)), minutes=int(rng.integers(60))), "Denied", "no_active_membership"))
    # PT: free intro sessions, then paid packages
    intro = False
    if rng.random() < 0.35:
        kid += 1; d = min(j + int(rng.integers(0, 14)), ND-1); intro=True
        nm, q = ("Welcome assessment",1) if rng.random()<0.50 else ("Intro sessions (free)",2)
        pkg = f"K{kid:06d}"
        pt_packages.append(dict(package_id=pkg, member_id=mid, purchase_time=DAYS[d]+pd.Timedelta(hours=10), package_name=nm, sessions=q, net_amount=0.0, gross_amount=0.0, is_paid=False))
        for i in range(q):
            sid += 1; dd = min(d+i*5, ND-1)
            pt_sessions.append(dict(session_id=f"S{sid:07d}", package_id=pkg, member_id=mid, trainer_id=rng.choice(trainers.trainer_id, p=tw),
                session_time=DAYS[dd]+pd.Timedelta(hours=int(rng.choice(HOURS,p=HP))), training_name=nm if q==1 else "Intro session", is_completed=rng.random()<0.85, is_cancelled=rng.random()<0.02, used_net_amount=0.0))
    p_buy = 0.10 + 0.12*intro + 0.03*min(eng,4)
    if active.any() and rng.random() < p_buy:
        ad = np.nonzero(active)[0]
        d = int(ad[min(len(ad)-1, int(rng.integers(0, min(len(ad), 90))))])
        trainer = rng.choice(trainers.trainer_id, p=tw)
        nbuy = 1 + rng.poisson(0.8*loyalty)
        for b in range(nbuy):
            if d >= ND: break
            nm, q, price, _ = PTPK[int(rng.choice(len(PTPK), p=PTW))]
            kid += 1; pkg=f"K{kid:06d}"
            pt_packages.append(dict(package_id=pkg, member_id=mid, purchase_time=DAYS[d]+pd.Timedelta(hours=int(rng.choice(HOURS,p=HP))), package_name=nm, sessions=q, net_amount=price, gross_amount=round(price*(1+VAT),2), is_paid=True))
            pid += 1
            payments.append(dict(payment_id=f"P{pid:07d}", member_id=mid, contract_id=None, payment_time=DAYS[d]+pd.Timedelta(hours=16), product_type="Coaching", plan_id=None,
                net_amount=price, vat_amount=round(price*VAT,2), gross_amount=round(price*(1+VAT),2), payment_method=rng.choice(PAYM,p=PAYW), source=rng.choice(SRC,p=SRCW), has_discount=False, is_refund=False))
            per = rng.uniform(0.5, 2.0)  # sessions/week
            dd = d
            for i in range(q):
                dd += max(1, int(rng.exponential(7/per)))
                if dd >= ND: break
                sid += 1
                canc = rng.random()<0.03
                pt_sessions.append(dict(session_id=f"S{sid:07d}", package_id=pkg, member_id=mid, trainer_id=trainer if rng.random()<0.70 else rng.choice(trainers.trainer_id, p=tw),
                    session_time=DAYS[dd]+pd.Timedelta(hours=int(np.clip(rng.normal(pref_h,2.0),6,22))), training_name="Coaching session",
                    is_completed=not canc and rng.random()<0.95, is_cancelled=canc, used_net_amount=0.0 if canc else round(price/q,2)))
            d = dd + int(rng.integers(7, 45))

members = pd.DataFrame(members)
contracts = pd.DataFrame(contracts)
payments = pd.DataFrame(payments).sort_values("payment_time").reset_index(drop=True)
# a few refunded membership payments, invented rate
ri = payments[payments.product_type=="Membership"].sample(frac=0.004, random_state=1).index
payments.loc[ri,"is_refund"] = True
visits = pd.DataFrame(visits, columns=["visit_id","member_id","swipe_time","result","deny_reason"]).sort_values("swipe_time").reset_index(drop=True)
pt_packages = pd.DataFrame(pt_packages).sort_values("purchase_time").reset_index(drop=True)
pt_sessions = pd.DataFrame(pt_sessions).sort_values("session_time").reset_index(drop=True)
contracts["end_date"] = pd.to_datetime(contracts.end_date)
os.makedirs(OUT, exist_ok=True)
for name, df in dict(members=members, plans=plans.drop(columns="w"), contracts=contracts, payments=payments, visits=visits,
                     trainers=trainers, pt_packages=pt_packages, pt_sessions=pt_sessions).items():
    df.to_csv(f"{OUT}/{name}.csv", index=False); df.to_parquet(f"{OUT}/{name}.parquet", index=False)
    print(f"{name:12s} rows={len(df):>8,d} cols={df.shape[1]}")
