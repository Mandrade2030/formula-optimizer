
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
        # skip fully blank manual rows
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
        gp=GaussianProcessRegressor(kernel=ConstantKernel(1.0)*RBF(length_scale=np.ones(len(specs)))+WhiteKernel(1e-5), normalize_y=True, random_state=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning); gp.fit(Xn,y)
        center=np.mean(Xn,axis=0); rows=[]
        for i,n in enumerate(ns):
            p1=center.copy(); p2=center.copy(); p1[i]=min(1,p1[i]+0.05); p2[i]=max(0,p2[i]-0.05)
            rows.append({"Variabile":n,"Indice influenza":round(abs(float(gp.predict([p1])[0]-gp.predict([p2])[0])),4)})
        return pd.DataFrame(rows).sort_values("Indice influenza",ascending=False)
    except Exception:
        return pd.DataFrame()

def project_json():
    return json.dumps({"schema_version":SCHEMA_VERSION,"project":st.session_state.project,"variables":st.session_state.variables,"trials":st.session_state.trials,"settings":st.session_state.settings}, indent=2, ensure_ascii=False)

def load_project(uploaded):
    data=json.load(uploaded)
    st.session_state.project=data.get("project",st.session_state.project)
    st.session_state.variables=data.get("variables",st.session_state.variables)
    st.session_state.trials=data.get("trials",[])
    st.session_state.settings={**st.session_state.settings, **data.get("settings",{})}
    st.session_state.component_count=len(st.session_state.variables)

def make_xlsx():
    bio=BytesIO()
    with pd.ExcelWriter(bio,engine="openpyxl") as w:
        pd.DataFrame([st.session_state.project]).to_excel(w,sheet_name="Project",index=False)
        pd.DataFrame(st.session_state.variables).to_excel(w,sheet_name="Variables",index=False)
        trial_df().to_excel(w,sheet_name="Trials",index=False)
        sdf=scored_df()
        if not sdf.empty: sdf.sort_values("Score",ascending=False).head(1).to_excel(w,sheet_name="Best",index=False)
        inf=influence()
        if not inf.empty: inf.to_excel(w,sheet_name="Influence",index=False)
    return bio.getvalue()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Formula Optimizer V2.6")
st.sidebar.caption("Mixture Laboratory - somma obbligatoria 100%")
st.sidebar.subheader("Progetto")
st.session_state.project["name"]=st.sidebar.text_input("Nome progetto",st.session_state.project.get("name","Nuovo progetto"))
st.session_state.project["notes"]=st.sidebar.text_area("Note",st.session_state.project.get("notes",""),height=80)
st.sidebar.subheader("Impostazioni DOE")
st.session_state.settings["n_initial"]=st.sidebar.number_input("Prove DOE iniziali",3,50,int(st.session_state.settings["n_initial"]),1)
st.session_state.settings["n_suggest"]=st.sidebar.number_input("Prove suggerite per ciclo",1,20,int(st.session_state.settings["n_suggest"]),1)
st.session_state.settings["candidate_pool"]=st.sidebar.number_input("Candidate pool",1000,50000,int(st.session_state.settings["candidate_pool"]),1000)
st.session_state.settings["local_radius"]=st.sidebar.slider("Raggio DOE locale",0.05,0.50,float(st.session_state.settings["local_radius"]),0.01)
st.session_state.settings["exploration_weight"]=st.sidebar.slider("Peso esplorazione",0.1,5.0,float(st.session_state.settings["exploration_weight"]),0.1)
st.session_state.settings["include_base"]=st.sidebar.checkbox("Includi formula base nel DOE", bool(st.session_state.settings.get("include_base",True)))

st.sidebar.subheader("Import")
json_file=st.sidebar.file_uploader("Carica progetto JSON",type=["json"])
if json_file and st.sidebar.button("Importa JSON"):
    load_project(json_file); st.sidebar.success("JSON caricato"); st.rerun()
csv_file=st.sidebar.file_uploader("Importa storico CSV",type=["csv"])
if csv_file and st.sidebar.button("Importa CSV"):
    df=pd.read_csv(csv_file); ns=names(); missing=[n for n in ns if n not in df.columns]
    if missing: st.sidebar.error(f"Mancano colonne: {missing}")
    else:
        rows=[]; sid=next_id()
        for idx,r in df.iterrows():
            vals=[float(r[n]) for n in ns]
            fixed=repair_mixture(vals,st.session_state.variables) or vals
            rec={"ID":int(r.get("ID",sid+idx)),"Iterazione":int(r.get("Iterazione",0)),"Source":"imported"}
            for n,v in zip(ns,fixed): rec[n]=round(float(v),4)
            rec["Totale"]=round(sum(fixed),4); rec["Score"]=to_float(r.get("Score"),np.nan)
            rows.append(rec)
        st.session_state.trials.extend(rows); st.sidebar.success(f"Importate {len(rows)} righe"); st.rerun()

# -------------------------
# Main
# -------------------------
st.title("Formula Optimizer - Mixture Laboratory V2.6")
st.markdown("**Versione stabile con salvataggio esplicito per componenti e tabella prove.**")

st.header("1. Componenti / variabili")
st.warning("Modifica i valori e premi **SALVA IMPOSTAZIONI VARIABILI**. I valori vengono applicati solo dopo il salvataggio.")

new_count=st.number_input("Numero componenti visibili",1,30,st.session_state.component_count,1)
if new_count != st.session_state.component_count:
    cur=list(st.session_state.variables)
    if new_count>len(cur):
        for i in range(len(cur),new_count): cur.append({"name":f"RM{i+1}","base":0.0,"min":0.0,"max":100.0,"step":0.1,"locked":False})
    else:
        cur=cur[:new_count]
    st.session_state.variables=cur; st.session_state.component_count=new_count; st.rerun()

with st.form("component_form", clear_on_submit=False):
    h=st.columns([1.4,1,1,1,1,0.7])
    for c,label in zip(h,["Nome","Base","Min","Max","Passo","Lock"]): c.markdown(f"**{label}**")
    edited=[]
    for i,v in enumerate(st.session_state.variables):
        c1,c2,c3,c4,c5,c6=st.columns([1.4,1,1,1,1,0.7])
        name=c1.text_input("Nome",str(v.get("name",f"RM{i+1}")),key=f"nm_{i}",label_visibility="collapsed")
        base=c2.number_input("Base",value=float(v.get("base",0.0)),step=0.01,format="%.4f",key=f"ba_{i}",label_visibility="collapsed")
        mn=c3.number_input("Min",value=float(v.get("min",0.0)),step=0.01,format="%.4f",key=f"mi_{i}",label_visibility="collapsed")
        mx=c4.number_input("Max",value=float(v.get("max",100.0)),step=0.01,format="%.4f",key=f"ma_{i}",label_visibility="collapsed")
        step=c5.number_input("Passo",value=float(v.get("step",0.1)),min_value=0.0001,step=0.01,format="%.4f",key=f"st_{i}",label_visibility="collapsed")
        lock=c6.checkbox("Lock",value=bool(v.get("locked",False)),key=f"lo_{i}",label_visibility="collapsed")
        edited.append({"name":name,"base":base,"min":mn,"max":mx,"step":step,"locked":lock})
    save_vars=st.form_submit_button("SALVA IMPOSTAZIONI VARIABILI",type="primary",use_container_width=True)

if save_vars:
    clean=[]; seen=set()
    for v in edited:
        name=str(v["name"]).strip()
        if not name or name.lower() in ["none","nan"]: continue
        if name in seen: st.error(f"Nome duplicato: {name}"); st.stop()
        if float(v["max"]) < float(v["min"]): st.error(f"{name}: Max < Min"); st.stop()
        seen.add(name); clean.append({"name":name,"base":float(v["base"]),"min":float(v["min"]),"max":float(v["max"]),"step":max(float(v["step"]),0.0001),"locked":bool(v["locked"])})
    if clean:
        old=set(names()); st.session_state.variables=clean; st.session_state.component_count=len(clean)
        if st.session_state.trials and old != set(names()): st.warning("Nomi variabili cambiati: le prove esistenti potrebbero non allinearsi.")
        st.success("Impostazioni variabili salvate."); st.rerun()

base_sum=sum(v["base"] for v in st.session_state.variables)
feasible,info=specs_feasible(st.session_state.variables)
c1,c2,c3,c4=st.columns(4)
c1.metric("Somma Base",f"{base_sum:.2f}%"); c2.metric("Min sum",f"{info['min_sum']:.2f}%"); c3.metric("Max sum",f"{info['max_sum']:.2f}%"); c4.metric("Fattibilità","OK" if feasible else "NO")
if abs(base_sum-100)>TOL: st.warning("La formula base non somma a 100%. Il DOE riparerà le formule, ma conviene correggere la base.")
if not feasible: st.error("Vincoli non fattibili: correggere Min/Max/Lock.")

st.header("2. Azioni")
a1,a2,a3,a4=st.columns(4)
with a1:
    if st.button("Genera DOE iniziale",type="primary",disabled=not feasible):
        st.session_state.trials=[]; cands=generate_initial(int(st.session_state.settings["n_initial"])); append_trials(cands,0,"initial_doe"); st.rerun()
with a2:
    if st.button("Ripara totali tabella"):
        ns=names(); rows=[]
        for row in st.session_state.trials:
            vals=[to_float(row.get(n),0.0) for n in ns]; fixed=repair_mixture(vals,st.session_state.variables)
            if fixed:
                for n,v in zip(ns,fixed): row[n]=round(v,4)
                row["Totale"]=round(sum(fixed),4)
            rows.append(row)
        st.session_state.trials=rows; st.rerun()
with a3:
    if st.button("Svuota prove"):
        st.session_state.trials=[]; st.rerun()
with a4:
    st.download_button("Salva JSON",project_json(),file_name=f"{st.session_state.project['name'].replace(' ','_')}_project.json",mime="application/json")

st.header("3. Tabella principale")
df=trial_df()
if df.empty: st.info("Genera DOE o importa uno storico CSV.")
else:
    st.info("Modifica score/formule, poi premi **Salva modifiche tabella**.")
    with st.form("trial_form", clear_on_submit=False):
        edited_df=st.data_editor(df,use_container_width=True,num_rows="dynamic",column_config={"ID":st.column_config.NumberColumn("ID",disabled=True),"Iterazione":st.column_config.NumberColumn("Iterazione",disabled=True),"Source":st.column_config.TextColumn("Source",disabled=True),"Totale":st.column_config.NumberColumn("Totale",disabled=True,format="%.4f"),"Score":st.column_config.NumberColumn("Score",format="%.4f")},key="trial_editor")
        save_trials=st.form_submit_button("Salva modifiche tabella",type="primary")
    if save_trials:
        sync_trials(edited_df); st.success("Modifiche tabella salvate."); st.rerun()

st.header("4. Generazione ciclo successivo")
sdf=scored_df()
g1,g2=st.columns([1,3])
with g1:
    if st.button("Genera nuove prove",disabled=not feasible or len(sdf)<3):
        cands=optimize(int(st.session_state.settings["n_suggest"])); append_trials(cands,current_iter()+1,"suggested"); st.rerun()
with g2:
    st.write(f"Score disponibili: **{len(sdf)}**")

st.header("5. Dashboard")
df=trial_df(); sdf=scored_df()
m=st.columns(5)
m[0].metric("Formule totali",len(df)); m[1].metric("Con score",len(sdf)); m[2].metric("Iterazione",current_iter())
if not sdf.empty:
    best=sdf.sort_values("Score",ascending=False).iloc[0]
    m[3].metric("Miglior score",f"{float(best['Score']):.2f}"); m[4].metric("ID migliore",int(best["ID"]))
    with st.expander("Migliore formulazione",expanded=True): st.dataframe(best[["ID","Iterazione"]+names()+["Totale","Score"]].to_frame().T,use_container_width=True)
    plot=sdf.sort_values("ID").copy(); plot["BestScore"]=plot["Score"].cummax(); st.line_chart(plot.set_index("ID")[["Score","BestScore"]])
inf=influence()
if not inf.empty:
    st.subheader("Importanza variabili (proxy)"); st.dataframe(inf,use_container_width=True)

st.header("6. Export")
e1,e2=st.columns(2)
with e1: st.download_button("Export CSV",trial_df().to_csv(index=False).encode("utf-8"),file_name=f"{st.session_state.project['name'].replace(' ','_')}_trials.csv",mime="text/csv")
with e2: st.download_button("Export XLSX",make_xlsx(),file_name=f"{st.session_state.project['name'].replace(' ','_')}_project.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
st.caption(f"Formula Optimizer - Mixture Laboratory V2.6 | Versione {APP_VERSION}")
