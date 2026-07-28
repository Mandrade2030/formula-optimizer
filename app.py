import streamlit as st
import pandas as pd
import numpy as np
import json
import random
from io import BytesIO
from math import sqrt
from datetime import datetime

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
import warnings

APP_VERSION = "2.0.0"
SCHEMA_VERSION = "2.0"
TOL = 0.05

st.set_page_config(page_title="Formula Optimizer - Mixture Laboratory V2", layout="wide")

# -----------------------------
# Session state initialization
# -----------------------------
def init_state():
    if "project" not in st.session_state:
        st.session_state.project = {
            "name": "Nuovo progetto",
            "notes": "",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "app_version": APP_VERSION,
            "schema_version": SCHEMA_VERSION,
        }
    if "variables" not in st.session_state:
        st.session_state.variables = [
            {"name": "A", "base": 53.28, "min": 30.0, "max": 70.0, "step": 2.0, "locked": False},
            {"name": "B", "base": 0.10, "min": 0.05, "max": 1.0, "step": 0.05, "locked": False},
            {"name": "C", "base": 0.30, "min": 0.10, "max": 1.0, "step": 0.05, "locked": False},
            {"name": "D", "base": 29.13, "min": 5.0, "max": 30.0, "step": 1.0, "locked": False},
            {"name": "E", "base": 14.26, "min": 5.0, "max": 30.0, "step": 0.5, "locked": False},
            {"name": "F", "base": 2.93, "min": 1.0, "max": 3.0, "step": 0.1, "locked": False},
        ]
    if "trials" not in st.session_state:
        st.session_state.trials = []
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "n_initial": 8,
            "n_suggest": 3,
            "candidate_pool": 8000,
            "duplicate_threshold": 0.03,
            "diversity_threshold": 0.08,
            "exploration_weight": 1.5,
            "local_radius": 0.18,
            "include_base": True,
            "random_seed": 42,
        }

init_state()

# -----------------------------
# Utility functions
# -----------------------------
def variable_names():
    return [v["name"] for v in st.session_state.variables]


def to_float(x, default=np.nan):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def round_to_step(x, step):
    if step <= 0:
        return float(x)
    return round(round(float(x) / step) * step, 10)


def clamp(x, mn, mx):
    return min(max(float(x), float(mn)), float(mx))


def specs_feasible(specs):
    min_sum = sum(v["min"] for v in specs)
    max_sum = sum(v["max"] for v in specs)
    locked_sum = sum(v["base"] for v in specs if v.get("locked"))
    unlocked_min_sum = sum(v["min"] for v in specs if not v.get("locked"))
    unlocked_max_sum = sum(v["max"] for v in specs if not v.get("locked"))
    locked_feasible = all(v["min"] - TOL <= v["base"] <= v["max"] + TOL for v in specs if v.get("locked"))
    total_feasible = min_sum <= 100 + TOL and max_sum >= 100 - TOL
    locked_total_feasible = (locked_sum + unlocked_min_sum <= 100 + TOL) and (locked_sum + unlocked_max_sum >= 100 - TOL)
    return total_feasible and locked_feasible and locked_total_feasible, {
        "min_sum": min_sum,
        "max_sum": max_sum,
        "locked_sum": locked_sum,
        "locked_total_feasible": locked_total_feasible,
        "locked_feasible": locked_feasible,
    }


def repair_mixture(raw_values, specs, max_iter=80):
    """Repair a candidate to satisfy bounds, locked variables and sum=100.
    Returns list of values or None if infeasible.
    """
    feasible, info = specs_feasible(specs)
    if not feasible:
        return None

    vals = []
    for x, v in zip(raw_values, specs):
        if v.get("locked"):
            vals.append(float(v["base"]))
        else:
            vals.append(clamp(x, v["min"], v["max"]))

    unlocked = [i for i, v in enumerate(specs) if not v.get("locked")]
    if not unlocked:
        return vals if abs(sum(vals) - 100) <= TOL else None

    # Iterative continuous repair
    for _ in range(max_iter):
        total = sum(vals)
        delta = 100 - total
        if abs(delta) <= 1e-7:
            break
        if delta > 0:
            adjustable = [i for i in unlocked if vals[i] < specs[i]["max"] - 1e-9]
            caps = [specs[i]["max"] - vals[i] for i in adjustable]
        else:
            adjustable = [i for i in unlocked if vals[i] > specs[i]["min"] + 1e-9]
            caps = [vals[i] - specs[i]["min"] for i in adjustable]
        if not adjustable:
            return None
        cap_sum = sum(caps)
        if cap_sum <= 0:
            return None
        for i, cap in zip(adjustable, caps):
            vals[i] += delta * (cap / cap_sum)
            vals[i] = clamp(vals[i], specs[i]["min"], specs[i]["max"])

    # Step rounding
    for i, v in enumerate(specs):
        vals[i] = round_to_step(vals[i], v["step"])
        vals[i] = clamp(vals[i], v["min"], v["max"])
        if v.get("locked"):
            vals[i] = float(v["base"])

    # Discrete residual repair using smallest steps
    for _ in range(200):
        residual = round(100 - sum(vals), 10)
        if abs(residual) <= TOL:
            break
        # Find candidate variable that can absorb one step in the residual direction
        direction = 1 if residual > 0 else -1
        candidates = []
        for i in unlocked:
            step = specs[i]["step"]
            if step <= 0:
                continue
            new_val = vals[i] + direction * step
            if specs[i]["min"] - 1e-9 <= new_val <= specs[i]["max"] + 1e-9:
                # Prefer steps that reduce residual most without overshooting too much
                score = abs(residual - direction * step)
                candidates.append((score, step, i, new_val))
        if not candidates:
            break
        candidates.sort(key=lambda x: (x[0], x[1]))
        _, _, i, new_val = candidates[0]
        vals[i] = round_to_step(new_val, specs[i]["step"])

    if abs(sum(vals) - 100) > 0.15:
        return None

    vals = [round(float(x), 6) for x in vals]
    return vals


def row_total(row):
    return round(sum(to_float(row.get(n), 0.0) for n in variable_names()), 4)


def candidate_key(values, specs):
    return tuple(round_to_step(v, s["step"]) for v, s in zip(values, specs))


def normalized_distance(a, b, specs):
    parts = []
    for x, y, s in zip(a, b, specs):
        span = max(s["max"] - s["min"], s["step"], 1e-9)
        parts.append(((x - y) / span) ** 2)
    return sqrt(sum(parts) / len(parts))


def existing_value_vectors():
    names = variable_names()
    vals = []
    for row in st.session_state.trials:
        try:
            vals.append([float(row[n]) for n in names])
        except Exception:
            pass
    return vals


def scored_dataframe():
    if not st.session_state.trials:
        return pd.DataFrame()
    df = pd.DataFrame(st.session_state.trials)
    if "Score" not in df.columns:
        return pd.DataFrame()
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    return df.dropna(subset=["Score"])


def next_trial_id():
    if not st.session_state.trials:
        return 1
    return int(max(to_float(r.get("ID"), 0) for r in st.session_state.trials)) + 1


def current_iteration():
    if not st.session_state.trials:
        return 0
    return int(max(to_float(r.get("Iterazione"), 0) for r in st.session_state.trials))


def append_trials(candidates, iteration, source):
    names = variable_names()
    next_id = next_trial_id()
    for vals in candidates:
        row = {"ID": next_id, "Iterazione": iteration, "Source": source}
        for n, v in zip(names, vals):
            row[n] = round(float(v), 4)
        row["Totale"] = round(sum(vals), 4)
        row["Score"] = np.nan
        st.session_state.trials.append(row)
        next_id += 1


def generate_initial_doe(n):
    random.seed(int(st.session_state.settings.get("random_seed", 42)))
    specs = st.session_state.variables
    feasible, info = specs_feasible(specs)
    if not feasible:
        st.error(f"Vincoli non fattibili: min_sum={info['min_sum']:.2f}, max_sum={info['max_sum']:.2f}, locked_sum={info['locked_sum']:.2f}")
        return []
    candidates = []
    existing = set()
    base_vals = [v["base"] for v in specs]
    base_repaired = repair_mixture(base_vals, specs)
    if st.session_state.settings.get("include_base", True) and base_repaired:
        candidates.append(base_repaired)
        existing.add(candidate_key(base_repaired, specs))
    attempts = 0
    radius = st.session_state.settings.get("local_radius", 0.18)
    while len(candidates) < n and attempts < n * 1000:
        attempts += 1
        vals = []
        # mixture-aware local perturbation around base
        for v in specs:
            if v.get("locked"):
                vals.append(v["base"])
            else:
                span = (v["max"] - v["min"]) * radius
                vals.append(v["base"] + random.uniform(-span, span))
        repaired = repair_mixture(vals, specs)
        if repaired is None:
            continue
        key = candidate_key(repaired, specs)
        if key in existing:
            continue
        if all(normalized_distance(repaired, c, specs) >= st.session_state.settings["duplicate_threshold"] for c in candidates):
            candidates.append(repaired)
            existing.add(key)
    return candidates


def fallback_local_suggestions(n):
    specs = st.session_state.variables
    scored = scored_dataframe()
    if scored.empty:
        center = [v["base"] for v in specs]
    else:
        best = scored.sort_values("Score", ascending=False).iloc[0]
        center = [float(best[v["name"]]) for v in specs]
    candidates = []
    existing = existing_value_vectors()
    attempts = 0
    while len(candidates) < n and attempts < n * 1500:
        attempts += 1
        vals = []
        for c, v in zip(center, specs):
            if v.get("locked"):
                vals.append(v["base"])
            else:
                vals.append(c + random.uniform(-2, 2) * v["step"] * max(1, attempts // 200 + 1))
        repaired = repair_mixture(vals, specs)
        if repaired is None:
            continue
        if any(normalized_distance(repaired, e, specs) < st.session_state.settings["duplicate_threshold"] for e in existing):
            continue
        if any(normalized_distance(repaired, s, specs) < st.session_state.settings["diversity_threshold"] for s in candidates):
            continue
        candidates.append(repaired)
    return candidates


def optimize_suggestions(n):
    specs = st.session_state.variables
    names = variable_names()
    scored = scored_dataframe()
    if len(scored) < 5:
        st.warning("Servono almeno 5 formulazioni con Score per usare il modello. Uso fallback DOE locale.")
        return fallback_local_suggestions(n)

    X = scored[names].astype(float).values
    y = scored["Score"].astype(float).values
    best_y = float(np.max(y))
    # Normalize X by variable ranges for GP stability
    mins = np.array([v["min"] for v in specs], dtype=float)
    spans = np.array([max(v["max"] - v["min"], v["step"], 1e-9) for v in specs], dtype=float)
    Xn = (X - mins) / spans

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(len(specs)), length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6, random_state=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        try:
            gp.fit(Xn, y)
        except Exception as e:
            st.warning(f"Modello GP non stabile ({e}). Uso fallback DOE locale.")
            return fallback_local_suggestions(n)

    existing = existing_value_vectors()
    pool = []
    candidate_pool = int(st.session_state.settings.get("candidate_pool", 8000))
    kappa = float(st.session_state.settings.get("exploration_weight", 1.5))
    attempts = 0
    # use current best as center for part of candidates, global random for the rest
    best_row = scored.sort_values("Score", ascending=False).iloc[0]
    best_center = [float(best_row[v["name"]]) for v in specs]
    base_center = [v["base"] for v in specs]

    while len(pool) < candidate_pool and attempts < candidate_pool * 10:
        attempts += 1
        mode = random.random()
        raw = []
        for idx, v in enumerate(specs):
            if v.get("locked"):
                raw.append(v["base"])
            else:
                if mode < 0.45:
                    center = best_center[idx]
                    radius = (v["max"] - v["min"]) * 0.20
                    raw.append(center + random.uniform(-radius, radius))
                elif mode < 0.75:
                    center = base_center[idx]
                    radius = (v["max"] - v["min"]) * 0.25
                    raw.append(center + random.uniform(-radius, radius))
                else:
                    raw.append(random.uniform(v["min"], v["max"]))
        repaired = repair_mixture(raw, specs)
        if repaired is None:
            continue
        if any(normalized_distance(repaired, e, specs) < st.session_state.settings["duplicate_threshold"] for e in existing):
            continue
        xn = ((np.array(repaired) - mins) / spans).reshape(1, -1)
        mu, std = gp.predict(xn, return_std=True)
        mu = float(mu[0]); std = float(std[0])
        min_dist = min([normalized_distance(repaired, e, specs) for e in existing] or [1.0])
        acquisition = (mu - best_y) + kappa * std + 5.0 * min_dist
        pool.append((acquisition, mu, std, min_dist, repaired))

    if not pool:
        return fallback_local_suggestions(n)
    pool.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for cand in pool:
        vals = cand[4]
        if any(normalized_distance(vals, s, specs) < st.session_state.settings["diversity_threshold"] for s in selected):
            continue
        selected.append(vals)
        if len(selected) >= n:
            break
    if len(selected) < n:
        selected += fallback_local_suggestions(n - len(selected))
    return selected[:n]


def variable_influence():
    specs = st.session_state.variables
    names = variable_names()
    scored = scored_dataframe()
    if len(scored) < 5:
        return pd.DataFrame()
    X = scored[names].astype(float).values
    y = scored["Score"].astype(float).values
    mins = np.array([v["min"] for v in specs], dtype=float)
    spans = np.array([max(v["max"] - v["min"], v["step"], 1e-9) for v in specs], dtype=float)
    Xn = (X - mins) / spans
    try:
        gp = GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(length_scale=np.ones(len(specs))) + WhiteKernel(1e-5), normalize_y=True, random_state=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(Xn, y)
        center = np.mean(Xn, axis=0)
        rows = []
        for i, name in enumerate(names):
            p1 = center.copy(); p2 = center.copy()
            p1[i] = min(1.0, p1[i] + 0.05)
            p2[i] = max(0.0, p2[i] - 0.05)
            delta = abs(float(gp.predict([p1])[0] - gp.predict([p2])[0]))
            rows.append({"Variabile": name, "Indice influenza": round(delta, 4)})
        return pd.DataFrame(rows).sort_values("Indice influenza", ascending=False)
    except Exception:
        return pd.DataFrame()


def project_to_json():
    data = {
        "schema_version": SCHEMA_VERSION,
        "project": st.session_state.project,
        "variables": st.session_state.variables,
        "trials": st.session_state.trials,
        "settings": st.session_state.settings,
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def load_project_json(uploaded):
    data = json.load(uploaded)
    st.session_state.project = data.get("project", st.session_state.project)
    st.session_state.variables = data.get("variables", st.session_state.variables)
    st.session_state.trials = data.get("trials", [])
    st.session_state.settings = {**st.session_state.settings, **data.get("settings", {})}


def trials_df():
    if not st.session_state.trials:
        cols = ["ID", "Iterazione", "Source"] + variable_names() + ["Totale", "Score"]
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(st.session_state.trials)
    desired = ["ID", "Iterazione", "Source"] + variable_names() + ["Totale", "Score"]
    for col in desired:
        if col not in df.columns:
            df[col] = np.nan
    df = df[desired]
    return df


def sync_trials_from_df(df):
    names = variable_names()
    rows = []
    for _, row in df.iterrows():
        rec = {}
        rec["ID"] = int(to_float(row.get("ID"), len(rows)+1))
        rec["Iterazione"] = int(to_float(row.get("Iterazione"), 0))
        rec["Source"] = str(row.get("Source", "manual"))
        vals = []
        for n in names:
            val = to_float(row.get(n), 0.0)
            rec[n] = round(float(val), 4)
            vals.append(float(val))
        rec["Totale"] = round(sum(vals), 4)
        score = to_float(row.get("Score"), np.nan)
        rec["Score"] = np.nan if np.isnan(score) else float(score)
        rows.append(rec)
    st.session_state.trials = rows


def make_xlsx():
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([st.session_state.project]).to_excel(writer, sheet_name="Project", index=False)
        pd.DataFrame(st.session_state.variables).to_excel(writer, sheet_name="Variables", index=False)
        trials_df().to_excel(writer, sheet_name="Trials", index=False)
        scored = scored_dataframe()
        if not scored.empty:
            best = scored.sort_values("Score", ascending=False).head(1)
            best.to_excel(writer, sheet_name="Best", index=False)
        infl = variable_influence()
        if not infl.empty:
            infl.to_excel(writer, sheet_name="Influence", index=False)
    return output.getvalue()

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Formula Optimizer V2")
st.sidebar.caption("Mixture Laboratory - somma obbligatoria 100%")

st.sidebar.subheader("Progetto")
st.session_state.project["name"] = st.sidebar.text_input("Nome progetto", st.session_state.project.get("name", "Nuovo progetto"))
st.session_state.project["notes"] = st.sidebar.text_area("Note", st.session_state.project.get("notes", ""), height=80)

st.sidebar.subheader("Impostazioni DOE")
st.session_state.settings["n_initial"] = st.sidebar.number_input("Prove DOE iniziali", min_value=3, max_value=50, value=int(st.session_state.settings["n_initial"]), step=1)
st.session_state.settings["n_suggest"] = st.sidebar.number_input("Prove suggerite per ciclo", min_value=1, max_value=20, value=int(st.session_state.settings["n_suggest"]), step=1)
st.session_state.settings["candidate_pool"] = st.sidebar.number_input("Candidate pool", min_value=1000, max_value=50000, value=int(st.session_state.settings["candidate_pool"]), step=1000)
st.session_state.settings["local_radius"] = st.sidebar.slider("Raggio DOE locale", 0.05, 0.50, float(st.session_state.settings["local_radius"]), 0.01)
st.session_state.settings["exploration_weight"] = st.sidebar.slider("Peso esplorazione", 0.1, 5.0, float(st.session_state.settings["exploration_weight"]), 0.1)
st.session_state.settings["include_base"] = st.sidebar.checkbox("Includi formula base nel DOE", value=bool(st.session_state.settings.get("include_base", True)))

st.sidebar.subheader("Import / Load")
json_file = st.sidebar.file_uploader("Carica progetto JSON", type=["json"])
if json_file is not None and st.sidebar.button("Importa JSON"):
    load_project_json(json_file)
    st.sidebar.success("Progetto JSON caricato.")
    st.rerun()

csv_file = st.sidebar.file_uploader("Importa storico CSV", type=["csv"])
if csv_file is not None and st.sidebar.button("Importa CSV"):
    df_imp = pd.read_csv(csv_file)
    names = variable_names()
    missing = [n for n in names if n not in df_imp.columns]
    if missing:
        st.sidebar.error(f"Colonne mancanti nel CSV: {missing}")
    else:
        start_id = next_trial_id()
        rows = []
        for idx, row in df_imp.iterrows():
            rec = {"ID": int(row.get("ID", start_id + idx)), "Iterazione": int(row.get("Iterazione", 0)), "Source": "imported"}
            vals = [float(row[n]) for n in names]
            repaired = repair_mixture(vals, st.session_state.variables)
            if repaired is None:
                repaired = vals
            for n, v in zip(names, repaired):
                rec[n] = round(float(v), 4)
            rec["Totale"] = round(sum(repaired), 4)
            rec["Score"] = to_float(row.get("Score"), np.nan)
            rows.append(rec)
        st.session_state.trials.extend(rows)
        st.sidebar.success(f"Importate {len(rows)} righe.")
        st.rerun()

# -----------------------------
# Main page
# -----------------------------
st.title("Formula Optimizer - Mixture Laboratory V2")
st.markdown("**MVP per DOE iterativo su formulazioni a miscela con vincolo nativo: somma = 100%.**")

# Variable editor
st.header("1. Componenti / variabili")
var_df = pd.DataFrame(st.session_state.variables)
var_df = st.data_editor(
    var_df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "name": st.column_config.TextColumn("Nome"),
        "base": st.column_config.NumberColumn("Base", format="%.4f"),
        "min": st.column_config.NumberColumn("Min", format="%.4f"),
        "max": st.column_config.NumberColumn("Max", format="%.4f"),
        "step": st.column_config.NumberColumn("Passo", format="%.4f", min_value=0.0001),
        "locked": st.column_config.CheckboxColumn("Lock"),
    },
    key="variable_editor",
)
# sanitize and save variables
clean_vars = []
for _, r in var_df.iterrows():
    name = str(r.get("name", "")).strip()
    if not name:
        continue
    clean_vars.append({
        "name": name,
        "base": float(to_float(r.get("base"), 0.0)),
        "min": float(to_float(r.get("min"), 0.0)),
        "max": float(to_float(r.get("max"), 100.0)),
        "step": max(float(to_float(r.get("step"), 0.1)), 0.0001),
        "locked": bool(r.get("locked", False)),
    })
if clean_vars:
    st.session_state.variables = clean_vars

base_sum = sum(v["base"] for v in st.session_state.variables)
feasible, feas_info = specs_feasible(st.session_state.variables)
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Somma formula base", f"{base_sum:.2f}%")
col_b.metric("Min sum", f"{feas_info['min_sum']:.2f}%")
col_c.metric("Max sum", f"{feas_info['max_sum']:.2f}%")
col_d.metric("Fattibilità", "OK" if feasible else "NO")
if abs(base_sum - 100) > TOL:
    st.warning("La formula base non somma esattamente a 100%. Il DOE riparerà i candidati, ma conviene controllare i valori Base.")
if not feasible:
    st.error("I vincoli min/max/lock rendono impossibile ottenere una formula con somma 100%. Correggere i limiti prima di generare DOE.")

# Actions
st.header("2. Azioni DOE")
act1, act2, act3, act4, act5 = st.columns(5)
with act1:
    if st.button("Genera DOE iniziale", type="primary", disabled=not feasible):
        st.session_state.trials = []
        candidates = generate_initial_doe(int(st.session_state.settings["n_initial"]))
        append_trials(candidates, iteration=0, source="initial_doe")
        st.success(f"Generate {len(candidates)} formulazioni iniziali.")
        st.rerun()
with act2:
    if st.button("Genera nuove prove", disabled=not feasible or len(scored_dataframe()) < 3):
        candidates = optimize_suggestions(int(st.session_state.settings["n_suggest"]))
        append_trials(candidates, iteration=current_iteration() + 1, source="suggested")
        st.success(f"Aggiunte {len(candidates)} nuove formulazioni alla tabella principale.")
        st.rerun()
with act3:
    if st.button("Ripara totali tabella"):
        names = variable_names()
        repaired_rows = []
        for row in st.session_state.trials:
            vals = [to_float(row.get(n), 0.0) for n in names]
            fixed = repair_mixture(vals, st.session_state.variables)
            if fixed is not None:
                for n, v in zip(names, fixed):
                    row[n] = round(v, 4)
                row["Totale"] = round(sum(fixed), 4)
            repaired_rows.append(row)
        st.session_state.trials = repaired_rows
        st.rerun()
with act4:
    if st.button("Svuota prove"):
        st.session_state.trials = []
        st.rerun()
with act5:
    st.download_button("Salva JSON", data=project_to_json(), file_name=f"{st.session_state.project['name'].replace(' ','_')}_project.json", mime="application/json")

# Dashboard
st.header("3. Dashboard")
df_trials = trials_df()
scored = scored_dataframe()
metric_cols = st.columns(5)
metric_cols[0].metric("Formulazioni totali", len(df_trials))
metric_cols[1].metric("Formulazioni con score", len(scored))
metric_cols[2].metric("Iterazione corrente", current_iteration())
if not scored.empty:
    best = scored.sort_values("Score", ascending=False).iloc[0]
    metric_cols[3].metric("Miglior score", f"{float(best['Score']):.2f}")
    metric_cols[4].metric("ID migliore", int(best["ID"]))
    with st.expander("Migliore formulazione corrente", expanded=True):
        best_view = best[["ID", "Iterazione"] + variable_names() + ["Totale", "Score"]].to_frame().T
        st.dataframe(best_view, use_container_width=True)
else:
    metric_cols[3].metric("Miglior score", "-")
    metric_cols[4].metric("ID migliore", "-")

# Main table
st.header("4. Tabella principale unica")
st.caption("Le nuove prove suggerite vengono aggiunte qui con Score vuoto. Inserisci lo score e genera il ciclo successivo.")
if df_trials.empty:
    st.info("Nessuna formulazione presente. Genera il DOE iniziale o importa uno storico CSV.")
else:
    edited = st.data_editor(
        df_trials,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "ID": st.column_config.NumberColumn("ID", disabled=True),
            "Iterazione": st.column_config.NumberColumn("Iterazione", disabled=True),
            "Source": st.column_config.TextColumn("Source", disabled=True),
            "Totale": st.column_config.NumberColumn("Totale", disabled=True, format="%.4f"),
            "Score": st.column_config.NumberColumn("Score", format="%.4f"),
        },
        key="trials_editor",
    )
    sync_trials_from_df(edited)
    totals = trials_df()["Totale"].astype(float)
    bad = totals[(totals - 100).abs() > 0.15]
    if len(bad) > 0:
        st.warning(f"Attenzione: {len(bad)} righe hanno totale fuori tolleranza. Usa 'Ripara totali tabella'.")

# Charts and influence
st.header("5. Analisi")
if not scored.empty:
    scored_plot = scored.sort_values("ID").copy()
    scored_plot["BestScore"] = scored_plot["Score"].cummax()
    st.subheader("Convergenza")
    st.line_chart(scored_plot.set_index("ID")[["Score", "BestScore"]])

infl = variable_influence()
if not infl.empty:
    st.subheader("Importanza variabili (proxy modello)")
    st.dataframe(infl, use_container_width=True)

# Export
st.header("6. Export")
exp1, exp2 = st.columns(2)
with exp1:
    st.download_button(
        "Export CSV prove",
        data=trials_df().to_csv(index=False).encode("utf-8"),
        file_name=f"{st.session_state.project['name'].replace(' ','_')}_trials.csv",
        mime="text/csv",
    )
with exp2:
    st.download_button(
        "Export XLSX progetto",
        data=make_xlsx(),
        file_name=f"{st.session_state.project['name'].replace(' ','_')}_project.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown("---")
st.caption(f"Formula Optimizer - Mixture Laboratory V2 | App version {APP_VERSION} | Offline/private deployment recommended for proprietary formulations.")
