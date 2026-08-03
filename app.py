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

APP_VERSION = "2.5.0"
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
        return float(x)
    except (ValueError, TypeError):
        return default
