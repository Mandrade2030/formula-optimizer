import streamlit as st
import pandas as pd
import numpy as np
import random, json, warnings
from io import BytesIO
from math import sqrt
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

APP_VERSION = "2.6.0"
SCHEMA_VERSION = "2.0"
TOL = 0.05

st.set_page_config(page_title="Formula Optimizer - Mixture Laboratory V2.6", layout="wide")

# -------------------------
# Init state
# -------------------------
def default_variables():
    return [
        {"name":"A","base":53.28,"min":30.0,"max":70.0,"step":2.0,"locked":False},
        {"name":"B","base":0.10,"min":0.05,"max":1.0,"step":0.05,"locked":False},
        {"name":"C","base":0.30,"min":0.10,"max":1.0,"step":0.05,"locked":False},
        {"name":"D","base":29.13,"min":5.0,"max":30.0,"step":1.0,"locked":False},
        {"name":"E","base":14.26,"min":5.0,"max":30.0,"step":0.5,"locked":False},
        {"name":"F","base":2.93,"min":1.0,"max":3.0,"step":0.1,"locked":False},
    ]

def init_state():
    if "project" not in st.session_state:
        st.session_state.project = {"name":"Nuovo progetto", "notes":"", "created_at":datetime.now().isoformat(timespec="seconds"), "app_version":APP_VERSION, "schema_version":SCHEMA_VERSION}
    if "variables" not in st.session_state:
        st.session_state.variables = default_variables()
    if "trials" not in st.session_state:
        st.session_state.trials = []
    if "settings" not in st.session_state:
        st.session_state.settings = {"n_initial":8, "n_suggest":3, "candidate_pool":12000, "duplicate_threshold":0.03, "diversity_threshold":0.08, "exploration_weight":1.5, "local_radius":0.18, "include_base":True, "random_seed":42}
    if "component_count" not in st.session_state:
        st.session_state.component_count = len(st.session_state.variables)
init_state()

# -------------------------
# Helpers
# -------------------------
def names():
    return [v["name"] for v in st.session_state.variables]

def to_float(x, default=np.nan):
    try:
        if x is None or x == "": return default
        return float(x)
    except Exception:
        return default

def clamp(x, mn, mx): return min(max(float(x), float(mn)), float(mx))

def round_to_step(x, step):
    step = max(float(step), 1e-9)
    return round(round(float(x)/step)*step, 10)

def specs_feasible(specs):
    min_sum = sum(v["min"] for v in specs)
    max_sum = sum(v["max"] for v in specs)
    locked_sum = sum(v["base"] for v in specs if v.get("locked"))
    unlocked_min = sum(v["min"] for v in specs if not v.get("locked"))
    unlocked_max = sum(v["max"] for v in specs if not v.get("locked"))
    locked_ok = all(v["min"]-TOL <= v["base"] <= v["max"]+TOL for v in specs if v.get("locked"))
    total_ok = min_sum <= 100+TOL and max_sum >= 100-TOL
    locked_total_ok = locked_sum + unlocked_min <= 100+TOL and locked_sum + unlocked_max >= 100-TOL
    return total_ok and locked_ok and locked_total_ok, {"min_sum":min_sum,"max_sum":max_sum,"locked_sum":locked_sum,"locked_ok":locked_ok,"locked_total_ok":locked_total_ok}

def repair_mixture(raw, specs, max_iter=80):
    feasible, _ = specs_feasible(specs)
    if not feasible: return None
    vals=[]
    for x,v in zip(raw,specs):
        vals.append(float(v["base"]) if v.get("locked") else clamp(x,v["min"],v["max"]))
    unlocked=[i for i,v in enumerate(specs) if not v.get("locked")]
    if not unlocked:
        return vals if abs(sum(vals)-100) <= TOL else None
    for _ in range(max_iter):
        delta=100-sum(vals)
        if abs(delta) < 1e-8: break
        if delta>0:
            adj=[i for i in unlocked if vals[i] < specs[i]["max"]-1e-9]
            caps=[specs[i]["max"]-vals[i] for i in adj]
        else:
            adj=[i for i in unlocked if vals[i] > specs[i]["min"]+1e-9]
            caps=[vals[i]-specs[i]["min"] for i in adj]
        if not adj or sum(caps)<=0: return None
        for i,c in zip(adj,caps):
            vals[i]=clamp(vals[i]+delta*c/sum(caps), specs[i]["min"], specs[i]["max"])
    # round to steps
    for i,v in enumerate(specs):
        vals[i]=round_to_step(vals[i],v["step"])
        vals[i]=clamp(vals[i],v["min"],v["max"])
        if v.get("locked"): vals[i]=float(v["base"])
    # repair discrete residual
    for _ in range(300):
        residual=round(100-sum(vals),10)
        if abs(residual)<=TOL: break
        direction=1 if residual>0 else -1
        candidates=[]
        for i in unlocked:
            step=specs[i]["step"]
            nv=vals[i]+direction*step
            if specs[i]["min"]-1e-9 <= nv <= specs[i]["max"]+1e-9:
                candidates.append((abs(residual-direction*step), step, i, nv))
        if not candidates: break
        candidates.sort(key=lambda x:(x[0],x[1]))
        _,_,i,nv=candidates[0]
        vals[i]=round_to_step(nv,specs[i]["step"])
    if abs(sum(vals)-100)>0.15: return None
    return [round(float(x),6) for x in vals]

def norm_dist(a,b,specs):
    s=0
    for x,y,v in zip(a,b,specs):
        span=max(v["max"]-v["min"], v["step"], 1e-9)
        s += ((x-y)/span)**2
    return sqrt(s/len(specs))

def trial_df():
    cols=["ID","Iterazione","Source"]+names()+["Totale","Score"]
    if not st.session_state.trials: return pd.DataFrame(columns=cols)
    df=pd.DataFrame(st.session_state.trials)
    for c in cols:
        if c not in df.columns: df[c]=np.nan
    return df[cols]

def sync_trials(df):
    ns=names(); rows=[]
    for _,r in df.iterrows():
        vals=[to_float(r.get(n),0.0) for n in ns]
        score=to_float(r.get("Score"),np.nan)
        if sum(abs(v) for v in vals)<1e-12 and np.isnan(score):
            continue
        rec={"ID":int(to_float(r.get("ID"), len(rows)+1)), "Iterazione":int(to_float(r.get("Iterazione"),0)), "Source":str(r.get("Source","manual"))}
        for n,v in zip(ns,vals): rec[n]=round(float(v),4)
        rec["Totale"]=round(sum(vals),4)
        rec["Score"]=np.nan if np.isnan(score) else float(score)
        rows.append(rec)
    st.session_state.trials=rows

def scored_df():
    df=trial_df()
    if df.empty: return df
    df["Score"]=pd.to_numeric(df["Score"], errors="coerce")
    return df.dropna(subset=["Score"])

def next_id():
    if not st.session_state.trials: return 1
    return int(max(to_float(r.get("ID"),0) for r in st.session_state.trials))+1

def current_iter():
    if not st.session_state.trials: return 0
    return int(max(to_float(r.get("Iterazione"),0) for r in st.session_state.trials))

def existing_vectors():
    ns=names(); out=[]
    for r in st.session_state.trials:
        try: out.append([float(r[n]) for n in ns])
        except Exception: pass
    return out

def append_trials(cands, iteration, source):
    ns=names(); nid=next_id()
    for vals in cands:
        row={"ID":nid, "Iterazione":iteration, "Source":source}
        for n,v in zip(ns,vals): row[n]=round(float(v),4)
        row["Totale"]=round(sum(vals),4)
        row["Score"]=np.nan
        st.session_state.trials.append(row); nid += 1

def generate_initial(n):
    random.seed(int(st.session_state.settings["random_seed"]))
    specs=st.session_state.variables
    feasible,info=specs_feasible(specs)
    if not feasible:
        st.error(f"Vincoli non fattibili. Min sum={info['min_sum']:.2f}, Max sum={info['max_sum']:.2f}, Locked sum={info['locked_sum']:.2f}")
        return []
    cands=[]; seen=set()
    base=[v["base"] for v in specs]
    if st.session_state.settings.get("include_base",True):
        b=repair_mixture(base,specs)
        if b:
            cands.append(b); seen.add(tuple(b))
    attempts=0; radius=float(st.session_state.settings["local_radius"])
    while len(cands)<n and attempts<n*2500:
        attempts += 1
        raw=[]
        for v in specs:
            if v.get("locked"): raw.append(v["base"])
            else:
                span=(v["max"]-v["min"])*radius
                raw.append(v["base"]+random.uniform(-span,span))
        vals=repair_mixture(raw,specs)
        if not vals: continue
        key=tuple(vals)
        if key in seen: continue
        if all(norm_dist(vals,c,specs)>=st.session_state.settings["duplicate_threshold"] for c in cands):
            cands.append(vals); seen.add(key)
    return cands

def fallback_suggestions(n):
    specs=st.session_state.variables; sdf=scored_df()
    if sdf.empty: center=[v["base"] for v in specs]
    else:
        best=sdf.sort_values("Score",ascending=False).iloc[0]
        center=[float(best[v["name"]]) for v in specs]
    cands=[]; existing=existing_vectors(); attempts=0
    while len(cands)<n and attempts<n*2500:
        attempts += 1
        raw=[]
        mult=max(1,attempts//400+1)
        for c,v in zip(center,specs):
            raw.append(v["base"] if v.get("locked") else c+random.uniform(-2,2)*v["step"]*mult)
        vals=repair_mixture(raw,specs)
        if vals is None: continue
        if any(norm_dist(vals,e,specs)<st.session_state.settings["duplicate_threshold"] for e in existing): continue
        if any(norm_dist(vals,c,specs)<st.session_state.settings["diversity_threshold"] for c in cands): continue
        cands.append(vals)
    return cands

def optimize(n):
    specs=st.session_state.variables; ns=names(); sdf=scored_df()
    if len(sdf)<5:
        st.warning("Meno di 5 score disponibili: uso una generazione locale intorno alla migliore formula.")
        return fallback_suggestions(n)
    X=sdf[ns].astype(float).values; y=sdf["Score"].astype(float).values
    mins=np.array([v["min"] for v in specs]); spans=np.array([max(v["max"]-v["min"],v["step"],1e-9) for v in specs])
    Xn=(X-mins)/spans
    kernel=ConstantKernel(1.0,(1e-3,1e3))*RBF(length_scale=np.ones(len(specs)), length_scale_bounds=(1e-2,1e2))+WhiteKernel(noise_level=1e-5,noise_level_bounds=(1e-8,1e-1))
    gp=GaussianProcessRegressor(kernel=kernel,normalize_y=True,alpha=1e-6,random_state=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        try: gp.fit(Xn,y)
        except Exception as e:
            st.warning(f"Modello non stabile ({e}). Uso fallback locale.")
            return fallback_suggestions(n)
    best_y=float(np.max(y)); best=sdf.sort_values("Score",ascending=False).iloc[0]
    best_center=[float(best[v["name"]]) for v in specs]
    base_center=[v["base"] for v in specs]
    existing=existing_vectors(); pool=[]; attempts=0
    target_pool=int(st.session_state.settings["candidate_pool"]); kappa=float(st.session_state.settings["exploration_weight"])
    while len(pool)<target_pool and attempts<target_pool*12:
        attempts += 1; mode=random.random(); raw=[]
        for i,v in enumerate(specs):
            if v.get("locked"): raw.append(v["base"])
            elif mode<0.50:
                raw.append(best_center[i]+random.uniform(-0.16,0.16)*(v["max"]-v["min"]))
            elif mode<0.80:
                raw.append(base_center[i]+random.uniform(-0.22,0.22)*(v["max"]-v["min"]))
            else:
                raw.append(random.uniform(v["min"],v["max"]))
        vals=repair_mixture(raw,specs)
        if vals is None: continue
        if any(norm_dist(vals,e,specs)<st.session_state.settings["duplicate_threshold"] for e in existing): continue
        xn=((np.array(vals)-mins)/spans).reshape(1,-1)
        mu,std=gp.predict(xn, return_std=True); mu=float(mu[0]); std=float(std[0])
        d=min([norm_dist(vals,e,specs) for e in existing] or [1.0])
        acq=(mu-best_y)+kappa*std+5*d
        pool.append((acq,vals))
    if not pool: return fallback_suggestions(n)
    pool.sort(key=lambda z:z[0], reverse=True)
    selected=[]
    for _,vals in pool:
        if any(norm_dist(vals,s,specs)<st.session_state.settings["diversity_threshold"] for s in selected): continue
        selected.append(vals)
        if len(selected)>=n: break
    if len(selected)<n: selected += fallback_suggestions(n-len(selected))
    return selected[:n]

def influence():
    specs=st.session_state.variables; ns=names(); sdf=scored_df()
    if len(sdf)<5: return pd.DataFrame()
    X=sdf[ns].astype(float).values; y=sdf["Score"].astype(float).values
    mins=np.array([v["min"] for v in specs]); spans=np.array([max(v["max"]-v["min"],v["step"],1e-9) for v in specs])
    Xn=(X-mins)/spans
    try:
        gp=GaussianProcessRegressor(kernel=ConstantKernel(1.0)*RBF(length_scale=np.ones(len(specs)))+WhiteKernel(1e-5),normalize_y=True,random_state=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(Xn,y)
        center=np.mean(Xn,axis=0); rows=[]
        for i,name in enumerate(ns):
            p1=center.copy(); p2=center.copy()
            p1[i]=min(1.0,p1[i]+0.05); p2[i]=max(0.0,p2[i]-0.05)
            delta=abs(float(gp.predict([p1])[0]-gp.predict([p2])[0]))
            rows.append({"Variabile":name, "Indice influenza":round(delta,4)})
        return pd.DataFrame(rows).sort_values("Indice influenza",ascending=False)
    except Exception:
        return pd.DataFrame()

def to_json():
    data={"schema_version":SCHEMA_VERSION, "project":st.session_state.project, "variables":st.session_state.variables, "trials":st.session_state.trials, "settings":st.session_state.settings}
    return json.dumps(data,indent=2,ensure_ascii=False)

def load_json(f):
    data=json.load(f)
    st.session_state.project=data.get("project",st.session_state.project)
    st.session_state.variables=data.get("variables",st.session_state.variables)
    st.session_state.trials=data.get("trials",[])
    st.session_state.settings={**st.session_state.settings, **data.get("settings",{})}

def make_xlsx():
    output=BytesIO()
    with pd.ExcelWriter(output,engine="openpyxl") as w:
        pd.DataFrame([st.session_state.project]).to_excel(w,sheet_name="Project",index=False)
        trial_df().to_excel(w,sheet_name="Trials",index=False)
        pd.DataFrame(st.session_state.variables).to_excel(w,sheet_name="Variables",index=False)
    return output.getvalue()

# ==================
# STREAMLIT UI
# ==================
st.title("🧪 Formula Optimizer - Mixture Laboratory V2.6")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Impostazioni Globali")
    st.session_state.project["name"]=st.text_input("Nome Progetto:",st.session_state.project["name"])
    st.session_state.project["notes"]=st.text_area("Note:",st.session_state.project["notes"],height=80)
    st.markdown("---")
    st.header("📥 Importa/Esporta")
    col1,col2=st.columns(2)
    with col1:
        if st.button("📥 Carica JSON"):
            st.session_state.show_upload=True
    with col2:
        st.download_button("📤 Scarica JSON",to_json(),"project.json","application/json")
    if st.session_state.get("show_upload"):
        uploaded=st.file_uploader("Scegli file JSON",type="json")
        if uploaded:
            load_json(uploaded)
            st.session_state.show_upload=False
            st.rerun()
    st.download_button("📊 Scarica XLSX",make_xlsx(),"project.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown("---")
    st.header("🎛️ Parametri Ottimizzazione")
    st.session_state.settings["n_initial"]=st.number_input("N iniziale DOE:",1,100,st.session_state.settings["n_initial"])
    st.session_state.settings["n_suggest"]=st.number_input("N suggerimenti:",1,50,st.session_state.settings["n_suggest"])
    st.session_state.settings["candidate_pool"]=st.number_input("Pool candidati:",1000,50000,st.session_state.settings["candidate_pool"],step=1000)
    st.session_state.settings["duplicate_threshold"]=st.slider("Soglia duplicati:",0.001,0.1,st.session_state.settings["duplicate_threshold"],0.001)
    st.session_state.settings["diversity_threshold"]=st.slider("Soglia diversità:",0.01,0.2,st.session_state.settings["diversity_threshold"],0.01)
    st.session_state.settings["exploration_weight"]=st.slider("Peso esplorazione:",0.5,3.0,st.session_state.settings["exploration_weight"],0.1)
    st.session_state.settings["local_radius"]=st.slider("Raggio locale:",0.05,0.5,st.session_state.settings["local_radius"],0.01)
    st.session_state.settings["include_base"]=st.checkbox("Includi base",st.session_state.settings["include_base"])
    st.session_state.settings["random_seed"]=st.number_input("Seed random:",0,10000,st.session_state.settings["random_seed"])

# Tab
tab1,tab2,tab3,tab4=st.tabs(["🔧 Variabili","📋 Tabella Principale","🎯 Suggerimenti","📊 Analisi"])

with tab1:
    st.subheader("Gestione Variabili/Componenti")
    st.write(f"Totale componenti: **{len(st.session_state.variables)}**")
    st.markdown("---")
    cols=st.columns([1,1,1,1,1,1,1,1,1])
    var_list=st.session_state.variables
    for idx,v in enumerate(var_list):
        with st.expander(f"🔹 {v['name']}"):
            col1,col2=st.columns(2)
            with col1:
                v["base"]=st.number_input(f"Base {v['name']}:",format="%.4f",value=float(v["base"]),key=f"base_{idx}")
                v["min"]=st.number_input(f"Min {v['name']}:",format="%.4f",value=float(v["min"]),key=f"min_{idx}")
            with col2:
                v["max"]=st.number_input(f"Max {v['name']}:",format="%.4f",value=float(v["max"]),key=f"max_{idx}")
                v["step"]=st.number_input(f"Step {v['name']}:",format="%.6f",value=float(v["step"]),key=f"step_{idx}")
            v["locked"]=st.checkbox(f"Bloccato",value=v["locked"],key=f"locked_{idx}")
    if st.button("✅ SALVA IMPOSTAZIONI VARIABILI",key="save_vars"):
        st.success("Variabili salvate!")

with tab2:
    st.subheader("Tabella Principale")
    col1,col2,col3=st.columns([1,1,1])
    with col1:
        if st.button("🔄 Genera DOE Iniziale"):
            n=st.session_state.settings["n_initial"]
            cands=generate_initial(n)
            if cands:
                append_trials(cands,1,"Initial DOE")
                st.success(f"✅ Aggiunte {len(cands)} prove iniziali")
            else:
                st.error("Impossibile generare DOE")
    with col2:
        if st.button("📥 Importa Tabella CSV"):
            st.session_state.show_csv_upload=True
    with col3:
        st.download_button("📤 Esporta CSV",trial_df().to_csv(index=False),"trials.csv","text/csv")
    if st.session_state.get("show_csv_upload"):
        uploaded=st.file_uploader("Scegli file CSV",type="csv")
        if uploaded:
            df=pd.read_csv(uploaded)
            sync_trials(df)
            st.session_state.show_csv_upload=False
            st.success("CSV importato!")
            st.rerun()
    st.markdown("---")
    df=trial_df()
    edited_df=st.data_editor(df,use_container_width=True,hide_index=False,key="trials_editor")
    st.markdown("---")
    if st.button("💾 Salva modifiche tabella",key="save_table"):
        sync_trials(edited_df)
        st.success("Tabella salvata!")

with tab3:
    st.subheader("Generazione Suggerimenti")
    col1,col2=st.columns([1,1])
    with col1:
        if st.button("🎯 Genera Suggerimenti (Ottimizzati)"):
            n=st.session_state.settings["n_suggest"]
            cands=optimize(n)
            if cands:
                append_trials(cands,current_iter()+1,"Optimized")
                st.success(f"✅ Aggiunte {len(cands)} prove ottimizzate")
            else:
                st.error("Impossibile generare suggerimenti")
    with col2:
        if st.button("🎲 Genera Suggerimenti (Locali)"):
            n=st.session_state.settings["n_suggest"]
            cands=fallback_suggestions(n)
            if cands:
                append_trials(cands,current_iter()+1,"Local")
                st.success(f"✅ Aggiunte {len(cands)} prove locali")
            else:
                st.error("Impossibile generare suggerimenti")
    st.markdown("---")
    st.info("💡 Inserisci gli Score nella tabella principale, poi premi 'Salva modifiche tabella' prima di generare nuovi suggerimenti.")

with tab4:
    st.subheader("Analisi e Visualizzazioni")
    sdf=scored_df()
    if not sdf.empty:
        col1,col2,col3=st.columns(3)
        with col1:
            st.metric("Prove con Score",len(sdf))
        with col2:
            st.metric("Score Massimo",f"{sdf['Score'].max():.4f}")
        with col3:
            st.metric("Score Medio",f"{sdf['Score'].mean():.4f}")
        st.markdown("---")
        st.subheader("Influenza Variabili")
        inf=influence()
        if not inf.empty:
            st.dataframe(inf,use_container_width=True)
            st.bar_chart(inf.set_index("Variabile")["Indice influenza"])
        else:
            st.info("Servono almeno 5 prove con Score per calcolare l'influenza.")
        st.markdown("---")
        st.subheader("Andamento Score")
        st.line_chart(sdf[["ID","Score"]].set_index("ID"))
    else:
        st.info("Nessuna prova con Score disponibile.")
