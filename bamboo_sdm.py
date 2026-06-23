#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bamboo Species Distribution Model (SDM)
========================================
Species  : Bambusa oldhamii
Framework: Ensemble Meta-Learning with Recursive Feature Elimination
           Adapted from Halder et al. (2025) Scientific Reports 15:5170
Novel    : Bayesian (GPC) meta-learner, GPU-accelerated base models,
           Spatial Block Cross-Validation, Epistemic Uncertainty Map
"""

# =============================================================================
# 1. IMPORTS & CONFIGURATION
# =============================================================================
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Tee stdout to a log file so progress is visible even with buffering
import os, warnings, time, json
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # PyTorch + Intel OpenMP coexistence on Windows
os.environ.setdefault('GDAL_DATA',
    os.path.join(os.path.dirname(sys.executable), 'Library', 'share', 'gdal'))  # suppress gdalvrt.xsd warning
os.environ['PYTHONWARNINGS'] = 'ignore'  # suppress warnings in joblib worker processes (e.g. ET n_jobs=-1); force-set so conda/shell presets don't block it

class _Tee:
    """Write to both stdout and a log file, flushing after every write."""
    def __init__(self, stream, logpath):
        self._s   = stream
        self._log = open(logpath, 'w', encoding='utf-8', errors='replace', buffering=1)
    def write(self, data):
        self._s.write(data)
        self._s.flush()
        self._log.write(data)
        self._log.flush()
    def flush(self):
        self._s.flush()
        self._log.flush()
    def reconfigure(self, **kw):
        pass  # already configured

_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'outputs', 'pipeline.log')
os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
_tee_stdout = _Tee(sys.stdout, _LOG_PATH)
sys.stdout   = _tee_stdout
sys.stderr   = _tee_stdout   # capture errors into the same log

# SVGP disabled: exact sklearn GPC (ARD Matérn-2.5, Laplace) outperforms
# SVGP on this dataset because the meta-training set is presence-limited
# (341 presences → 1,023 OOF samples at 1:2 ratio), well within O(n³)
# tractability. SVGP's variational approximation adds error without benefit.
_GPYTORCH_AVAILABLE = False
_GP_BACKEND = None

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol, xy as raster_xy
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    Matern, RationalQuadratic, DotProduct, ConstantKernel)
from sklearn.feature_selection import RFECV
from sklearn.model_selection import (RandomizedSearchCV, BaseCrossValidator,
                                      StratifiedKFold, train_test_split)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              confusion_matrix, ConfusionMatrixDisplay,
                              brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.base import clone
from scipy.special import expit
from scipy.optimize import nnls
import xgboost as xgb
import jenkspy

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(os.environ.get('BAMBOO_SDM_DIR', Path(__file__).resolve().parent))
OCC_FILE = BASE_DIR / 'oldhamii_occ.csv'
PRED_DIR = BASE_DIR / 'predictors'
OUT_DIR  = BASE_DIR / 'outputs'
for d in [OUT_DIR, OUT_DIR/'figures', OUT_DIR/'models', OUT_DIR/'maps',
          OUT_DIR/'models'/'replicates', OUT_DIR/'maps'/'replicates']:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
RANDOM_STATE     = 42
N_PA             = 10_000     # pseudo-absence count
N_REPLICATES     = 20        # number of PA replicates; increase here to run more
N_FOLDS          = 5          # spatial CV folds
BLOCK_SIZE_DEG   = 0.05       # ~5.5 km block size for spatial CV
VOTE_THRESHOLD   = 3          # min RFECV-capable model votes to select a feature (3/6 = 50% majority)
N_ITER_RS        = 60         # iterations for RandomizedSearchCV
N_ZONES          = 5          # suitability classification zones
PRED_BATCH_SIZE  = 50_000     # pixels per batch during raster prediction

def _detect_xgb_device():
    """Return 'cuda' if a CUDA GPU is visible, else 'cpu', so the pipeline runs
    unchanged on CPU-only machines."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'

XGB_DEVICE = _detect_xgb_device()   # GPU when available, CPU fallback otherwise

np.random.seed(RANDOM_STATE)

PRED_NAMES = sorted([f.stem for f in PRED_DIR.glob('*.tif')])
print(f"Predictors ({len(PRED_NAMES)}): {PRED_NAMES}\n")

# Features that are nominal/categorical and must receive OHE (not MinMax).
# 'class' is the soil texture/type raster — integer codes with no ordinal meaning.
CATEGORICAL_FEATURES = ['class']

# Models excluded from RFECV voting. Empty now — all current base learners
# have native coef_ or feature_importances_ and participate in RFECV.
NO_RFECV = set()


# =============================================================================
# 2. SPATIAL BLOCK CROSS-VALIDATOR
# =============================================================================
class SpatialBlockCV(BaseCrossValidator):
    """
    Block cross-validator that partitions the study area into a regular
    lat/lon grid and groups grid cells into n_splits folds. Prevents
    spatial autocorrelation from inflating model performance metrics.

    Usage:
        spatial_cv = SpatialBlockCV(n_splits=5, block_size_deg=0.05)
        spatial_cv.precompute(coords)          # coords: (n, 2) lon/lat array
        for train_idx, test_idx in spatial_cv.split(X_env):
            ...
    """
    def __init__(self, n_splits=5, block_size_deg=0.05, random_state=42):
        self.n_splits       = n_splits
        self.block_size_deg = block_size_deg
        self.random_state   = random_state
        self.point_folds_   = None

    def precompute(self, coords):
        rng      = np.random.RandomState(self.random_state)
        lons     = coords[:, 0]
        lats     = coords[:, 1]
        lon_blk  = np.floor((lons - lons.min()) / self.block_size_deg).astype(int)
        lat_blk  = np.floor((lats - lats.min()) / self.block_size_deg).astype(int)
        blk_ids  = lat_blk * 10_000 + lon_blk
        unique   = np.unique(blk_ids)
        rng.shuffle(unique)
        fold_map = {int(b): i % self.n_splits for i, b in enumerate(unique)}
        self.point_folds_ = np.array([fold_map[int(b)] for b in blk_ids])
        counts = [int((self.point_folds_ == f).sum()) for f in range(self.n_splits)]
        print(f"  SpatialBlockCV: {len(unique)} blocks -> {self.n_splits} folds | "
              f"fold sizes: {counts}")
        return self

    def _iter_test_masks(self, X, y=None, groups=None):
        if self.point_folds_ is None:
            raise RuntimeError("Call precompute(coords) before using this CV.")
        for fold in range(self.n_splits):
            yield self.point_folds_ == fold

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        for test_mask in self._iter_test_masks(X, y, groups):
            train_idx = np.where(~test_mask)[0]
            test_idx  = np.where( test_mask)[0]
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


# =============================================================================
# 3. DATA PREPARATION
# =============================================================================
def load_raster_stack():
    print("=" * 60)
    print("[STEP 1] Loading raster predictor stack")
    print("=" * 60)
    arrays, profile, transform = [], None, None
    for name in PRED_NAMES:
        fpath = PRED_DIR / f"{name}.tif"
        with rasterio.open(fpath) as src:
            arr    = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                arr = np.where(arr == nodata, np.nan, arr)
            arr = np.where(np.isinf(arr), np.nan, arr)
            arrays.append(arr)
            if profile is None:
                profile   = src.profile.copy()
                transform = src.transform
                H, W      = src.height, src.width

    stack = np.stack(arrays, axis=0)   # (n_features, H, W)
    print(f"  Stack shape : {stack.shape}")
    print(f"  CRS         : {profile['crs']}")
    print(f"  Resolution  : {transform.a:.5f} x {abs(transform.e):.5f} deg")
    nan_pct = np.isnan(stack).any(axis=0).mean() * 100
    print(f"  Invalid cells (any NaN): {nan_pct:.1f}%\n")
    return stack, profile, transform, H, W


def generate_pseudo_absences(stack, transform, H, W, seed=RANDOM_STATE):
    print(f"[STEP 2] Generating {N_PA:,} pseudo-absences (seed={seed})")
    valid_mask = ~np.any(np.isnan(stack), axis=0)
    vrows, vcols = np.where(valid_mask)
    print(f"  Valid cells: {len(vrows):,}")
    rng     = np.random.RandomState(seed)
    idx     = rng.choice(len(vrows), size=N_PA, replace=False)
    pa_rows = vrows[idx]
    pa_cols = vcols[idx]
    lons, lats = raster_xy(transform, pa_rows, pa_cols)
    pa_df = pd.DataFrame({'lon': lons, 'lat': lats, 'presence': 0})
    print(f"  Pseudo-absences generated: {len(pa_df):,}\n")
    return pa_df


def extract_env_values(coords_df, stack, transform, H, W):
    lons  = coords_df['lon'].values
    lats  = coords_df['lat'].values
    rows, cols = rowcol(transform, lons, lats)
    rows  = np.array(rows, dtype=int)
    cols  = np.array(cols, dtype=int)
    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    env   = np.full((len(coords_df), len(PRED_NAMES)), np.nan, dtype=np.float32)
    env[valid] = stack[:, rows[valid], cols[valid]].T
    return pd.DataFrame(env, columns=PRED_NAMES)


def prepare_dataset(stack, transform, H, W, seed=RANDOM_STATE):
    print("[STEP 3] Preparing dataset")
    occ_df = pd.read_csv(OCC_FILE)
    print(f"  Presence records : {len(occ_df):,}")

    pa_df    = generate_pseudo_absences(stack, transform, H, W, seed=seed)
    all_pts  = pd.concat([occ_df, pa_df], ignore_index=True)

    print("  Extracting environmental values at all points...")
    env_df   = extract_env_values(all_pts, stack, transform, H, W)
    full_df  = pd.concat([all_pts[['lon', 'lat', 'presence']], env_df], axis=1)

    before   = len(full_df)
    full_df  = full_df.dropna(subset=PRED_NAMES).reset_index(drop=True)
    dropped  = before - len(full_df)
    n_pres   = int(full_df['presence'].sum())
    n_abs    = int((full_df['presence'] == 0).sum())
    print(f"  Dropped (NaN)  : {dropped:,}")
    print(f"  Final dataset  : {len(full_df):,} records "
          f"(presence={n_pres:,}, absence={n_abs:,})")
    print(f"  Class ratio    : 1:{n_abs/n_pres:.1f} (abs:pres)\n")
    return full_df


# =============================================================================
# 4. FEATURE PREPROCESSING  (MinMax for continuous, OHE for categorical)
# =============================================================================
def build_preprocessor(feature_names, X_train, X_test):
    """
    Fit a ColumnTransformer that applies:
      - MinMaxScaler  to all continuous features
      - OneHotEncoder to all features listed in CATEGORICAL_FEATURES

    Returns (X_train_proc, X_test_proc, preprocessor, proc_names) where
    proc_names is the ordered list of feature names in the transformed space.
    """
    cont_feats = [f for f in feature_names if f not in CATEGORICAL_FEATURES]
    cat_feats  = [f for f in feature_names if f in CATEGORICAL_FEATURES]
    cont_idx   = [feature_names.index(f) for f in cont_feats]
    cat_idx    = [feature_names.index(f) for f in cat_feats]

    transformers = [('minmax', MinMaxScaler(), cont_idx)]
    if cat_idx:
        transformers.append((
            'ohe',
            OneHotEncoder(sparse_output=False, handle_unknown='ignore',
                          dtype=np.float32),
            cat_idx,
        ))

    preprocessor  = ColumnTransformer(transformers, remainder='drop')
    X_train_proc  = preprocessor.fit_transform(X_train)
    X_test_proc   = preprocessor.transform(X_test)

    # Build expanded feature name list: continuous first, then OHE dummies
    proc_names = list(cont_feats)
    if cat_idx:
        ohe = preprocessor.named_transformers_['ohe']
        for i, feat in enumerate(cat_feats):
            proc_names.extend([f"{feat}_{int(c)}" for c in ohe.categories_[i]])

    n_ohe = len(proc_names) - len(cont_feats)
    print(f"  Preprocessor : {len(cont_feats)} continuous (MinMax) + "
          f"{n_ohe} OHE dummies ({cat_feats}) "
          f"→ {len(proc_names)} total features")
    return X_train_proc, X_test_proc, preprocessor, proc_names


# =============================================================================
# 5. BASE MODEL DEFINITIONS
# =============================================================================

def scale_pos_weight(df):
    n_neg = int((df['presence'] == 0).sum())
    n_pos = int(df['presence'].sum())
    return n_neg / n_pos


class MLPWithImportance(MLPClassifier):
    """MLPClassifier with permutation-based feature_importances_ for RFECV.
    Importances are computed on the training data at fit-time so RFECV can
    rank and eliminate features. Only used in ensemble_rfecv(); the stacking
    version uses plain MLPClassifier to avoid re-computing on the full set."""
    def fit(self, X, y, **kw):
        super().fit(X, y, **kw)
        r = permutation_importance(
            self, X, y, n_repeats=3, scoring='roc_auc',
            random_state=RANDOM_STATE, n_jobs=1
        )
        self.feature_importances_ = np.clip(r.importances_mean, 0, None)
        return self


class GaussianNBWithImportance(GaussianNB):
    """GaussianNB with permutation-based feature_importances_ for RFECV.
    GaussianNB has no native coef_/feature_importances_, so (like
    MLPWithImportance) importances are computed at fit-time on the training
    data.  Only used in ensemble_rfecv(); stacking uses plain GaussianNB."""
    def fit(self, X, y, **kw):
        super().fit(X, y, **kw)
        r = permutation_importance(
            self, X, y, n_repeats=3, scoring='roc_auc',
            random_state=RANDOM_STATE, n_jobs=1
        )
        self.feature_importances_ = np.clip(r.importances_mean, 0, None)
        return self


def build_base_models(spw):
    """Return dict of 6 base learners — strictly 1 per learning paradigm,
    all RFECV-compatible. Final set after diagnostic ARD kernel analysis."""
    return {
        # ── Linear / probabilistic ────────────────────────────────────────
        'LR': LogisticRegression(
            solver='saga', max_iter=2000, C=1.0,
            class_weight='balanced', random_state=RANDOM_STATE
        ),
        # Raw LinearSVC swapped in for RFECV (coef_ exposed); this
        # CalibratedClassifierCV version is used for stacking only.
        'SVM': CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight='balanced', max_iter=2000,
                      random_state=RANDOM_STATE),
            cv=3, method='isotonic'
        ),
        # ── Bagging ───────────────────────────────────────────────────────
        'ET': ExtraTreesClassifier(
            n_estimators=100, class_weight='balanced',
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        # ── Gradient boosting (depth-wise) ────────────────────────────────
        'XGB': xgb.XGBClassifier(
            device=XGB_DEVICE, n_estimators=100,
            scale_pos_weight=spw, eval_metric='logloss',
            verbosity=0, random_state=RANDOM_STATE
        ),
        # ── Neural network (nonlinear learned representations) ────────────
        'MLP': MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=300,
            early_stopping=True, validation_fraction=0.1,
            random_state=RANDOM_STATE
        ),
        # ── Generative / probabilistic (decorrelates from discriminative
        #    learners → diverse meta-features the stackers benefit from) ────
        'NB': GaussianNB(),
    }


# =============================================================================
# 6. ENSEMBLE RFECV FEATURE SELECTION
# =============================================================================
RFECV_MAX_SAMPLES = 5_000   # cap for RFECV subsample

def _rfecv_subsample(y, max_n=RFECV_MAX_SAMPLES, random_state=RANDOM_STATE):
    """Return stratified subsample indices for RFECV."""
    if len(y) <= max_n:
        return np.arange(len(y))
    rng   = np.random.RandomState(random_state)
    pos_i = np.where(y == 1)[0]
    neg_i = np.where(y == 0)[0]
    ratio = len(pos_i) / len(y)
    n_pos = max(1, int(max_n * ratio))
    n_neg = max_n - n_pos
    return np.concatenate([
        rng.choice(pos_i, min(n_pos, len(pos_i)), replace=False),
        rng.choice(neg_i, min(n_neg, len(neg_i)), replace=False),
    ])


def ensemble_rfecv(X_train_sc, y_train, coords_tr, spatial_cv, spw, proc_names):
    print("=" * 60)
    print("[STEP 5] Ensemble RFECV Feature Selection")
    print("=" * 60)
    print(f"  RFECV subsample cap: {RFECV_MAX_SAMPLES:,} samples "
          f"(full set: {len(y_train):,})")

    # Stacking versions → swap in bare (uncalibrated) versions for RFECV so
    # coef_ is directly accessible. NO_RFECV models are skipped entirely.
    models = build_base_models(spw)
    # Swap in RFECV-compatible versions where needed.
    models['SVM'] = LinearSVC(C=1.0, class_weight='balanced',
                               max_iter=2000, random_state=RANDOM_STATE)
    models['LR'] = LogisticRegression(solver='saga', max_iter=2000, C=10.0,
                                       class_weight='balanced', random_state=RANDOM_STATE)
    # MLP needs MLPWithImportance for RFECV (exposes feature_importances_
    # via permutation importance; plain MLPClassifier lacks this attribute).
    models['MLP'] = MLPWithImportance(
        hidden_layer_sizes=(100,), max_iter=300,
        early_stopping=True, validation_fraction=0.1,
        random_state=RANDOM_STATE
    )
    # NB has no native importances either → permutation-importance wrapper.
    models['NB'] = GaussianNBWithImportance()

    vote_df     = pd.DataFrame(0, index=proc_names, columns=list(models.keys()))
    n_sel_dict  = {}
    cv_curves   = {}    # accuracy vs n_features per model
    n_rfecv_voters = sum(1 for n in models if n not in NO_RFECV)

    for name, model in models.items():
        if name in NO_RFECV:
            print(f"\n  >> RFECV: {name}  [skipped — no native coef_/feature_importances_]")
            continue
        print(f"\n  >> RFECV: {name}", flush=True)
        t0 = time.time()

        sub_idx    = _rfecv_subsample(y_train)
        X_sub      = X_train_sc[sub_idx]
        y_sub      = y_train[sub_idx]
        coords_sub = coords_tr[sub_idx]

        n_pos_sub = int(y_sub.sum())
        if n_pos_sub >= N_FOLDS:
            cv_sub = SpatialBlockCV(n_splits=N_FOLDS, block_size_deg=BLOCK_SIZE_DEG,
                                    random_state=RANDOM_STATE)
            cv_sub.precompute(coords_sub)
            cv_for_rfecv = cv_sub
        else:
            cv_for_rfecv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                           random_state=RANDOM_STATE)

        rfecv = RFECV(
            estimator=clone(model),
            step=1,
            cv=cv_for_rfecv,
            scoring='roc_auc',
            min_features_to_select=2,
            n_jobs=1,
        )
        try:
            rfecv.fit(X_sub, y_sub)
            selected      = rfecv.support_
            vote_df[name] = selected.astype(int)
            n_sel         = int(selected.sum())
            n_sel_dict[name] = n_sel
            # Extract CV accuracy curve (key varies across sklearn versions)
            mean_scores = None
            if hasattr(rfecv, 'cv_results_'):
                for key in ('mean_test_score', 'mean_score', 'mean_test_roc_auc'):
                    if key in rfecv.cv_results_:
                        mean_scores = rfecv.cv_results_[key]
                        break
            cv_curves[name] = mean_scores
            elapsed = time.time() - t0
            print(f"    Features selected: {n_sel}/{len(proc_names)} | "
                  f"Time: {elapsed:.1f}s")
            print(f"    Selected: {[proc_names[i] for i in range(len(proc_names)) if selected[i]]}")
        except Exception as e:
            print(f"    RFECV FAILED ({type(e).__name__}: {e}) — {name} gets 0 votes, "
                  f"will train on consensus feature set")

    # ── Ensemble vote ─────────────────────────────────────────────────────
    vote_df['Total_Votes'] = vote_df[list(models.keys())].sum(axis=1)
    vote_df['Selected']    = (vote_df['Total_Votes'] >= VOTE_THRESHOLD).astype(int)
    selected_features      = vote_df.index[vote_df['Selected'] == 1].tolist()

    print(f"\n  ── Ensemble result ({VOTE_THRESHOLD}/{n_rfecv_voters} RFECV vote threshold, "
          f"{len(models)} total models) ──")
    print(f"  Features selected: {len(selected_features)} / {len(proc_names)}")
    print(f"  {selected_features}")
    vote_df.to_csv(OUT_DIR / 'feature_vote_matrix.csv')

    # ── Plot: vote bar chart ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ['#27ae60' if v >= VOTE_THRESHOLD else '#e74c3c'
               for v in vote_df['Total_Votes']]
    bars    = ax.barh(vote_df.index, vote_df['Total_Votes'], color=colors,
                      edgecolor='white')
    ax.axvline(VOTE_THRESHOLD, color='black', linestyle='--', lw=1.5,
               label=f'Selection threshold ({VOTE_THRESHOLD}/{n_rfecv_voters} RFECV voters)')
    ax.set_xlabel('Ensemble Vote Count', fontsize=12)
    ax.set_title('Ensemble RFECV — Feature Vote Counts\n'
                 'Green = selected, Red = eliminated', fontsize=13)
    ax.set_xlim(0, n_rfecv_voters + 0.75)
    ax.legend(fontsize=10)
    for bar, v in zip(bars, vote_df['Total_Votes']):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                str(v), va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figures' / 'rfecv_votes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/rfecv_votes.png")

    # ── Plot: accuracy vs n_features curves ───────────────────────────────
    model_names = [n for n, c in cv_curves.items() if c is not None]
    if model_names:
        cols = 3
        rows = int(np.ceil(len(model_names) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
        axes = np.array(axes).flatten()
        for i, name in enumerate(model_names):
            scores = cv_curves[name]
            x_vals = range(2, len(scores) + 2)
            axes[i].plot(x_vals, scores, marker='o', ms=4, color='steelblue')
            best_n = int(np.argmax(scores)) + 2
            axes[i].axvline(best_n, color='red', linestyle='--', lw=1,
                            label=f'Best: {best_n} features')
            axes[i].set_title(name, fontsize=11)
            axes[i].set_xlabel('# Features')
            axes[i].set_ylabel('CV AUC-ROC')
            axes[i].legend(fontsize=8)
            axes[i].grid(alpha=0.3)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle('RFECV: CV AUC-ROC vs Number of Features', fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'figures' / 'rfecv_curves.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: figures/rfecv_curves.png")

    return vote_df, selected_features


# =============================================================================
# 7. HYPERPARAMETER TUNING
# =============================================================================
def get_param_grids(spw):
    return {
        'LR': {
            'C':            [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'solver':       ['saga', 'lbfgs', 'newton-cg'],
            'max_iter':     [1000, 2000],
            'class_weight': [None, 'balanced'],
        },
        'SVM': {
            'estimator__C':            [0.001, 0.01, 0.1, 1, 10, 100],
            'estimator__class_weight': [None, 'balanced'],
        },
        'ET': {
            'n_estimators':      [200, 300, 500, 800],
            'max_depth':         [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf':  [1, 2, 4],
            'max_features':      ['sqrt', 'log2', 0.3, 0.5],
            'class_weight':      [None, 'balanced', 'balanced_subsample'],
        },
        'XGB': {
            'n_estimators':      [300, 500, 700, 1000],
            'max_depth':         [3, 5, 7, 9],
            'learning_rate':     [0.005, 0.01, 0.05, 0.1, 0.2],
            'subsample':         [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree':  [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma':             [0, 0.1, 0.2, 0.5],
            'min_child_weight':  [1, 3, 5],
            'reg_alpha':         [0, 0.1, 0.5, 1.0],
            'reg_lambda':        [0.5, 1.0, 2.0, 5.0],
        },
        'MLP': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation':         ['relu', 'tanh'],
            'alpha':              [0.0001, 0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.01],
            'max_iter':           [300, 500],
        },
        # GaussianNB's only tunable knob is var_smoothing (Laplace-style
        # variance floor); 100 log-spaced values >> N_ITER_RS so the random
        # search has room to sample.
        'NB': {
            'var_smoothing': list(np.logspace(-12, -3, 100)),
        },
    }


def tune_base_models(X_train_sc, y_train, spatial_cv, spw, seed=RANDOM_STATE):
    print("=" * 60)
    print("[STEP 6] Hyperparameter Tuning (RandomizedSearchCV)")
    print("=" * 60)

    base_defs = {
        'LR': LogisticRegression(random_state=seed),
        'SVM': CalibratedClassifierCV(
            LinearSVC(max_iter=2000, random_state=seed),
            cv=3, method='isotonic'
        ),
        'ET': ExtraTreesClassifier(n_jobs=-1, random_state=seed),
        'XGB': xgb.XGBClassifier(
            device=XGB_DEVICE, eval_metric='logloss', verbosity=0,
            scale_pos_weight=spw, random_state=seed
        ),
        'MLP': MLPClassifier(
            early_stopping=True, validation_fraction=0.1,
            random_state=seed
        ),
        # Generative / probabilistic 6th learner (decorrelates from the
        # discriminative models → diverse meta-feature for the stackers).
        'NB': GaussianNB(),
    }
    grids      = get_param_grids(spw)
    n_iter_map = {name: N_ITER_RS for name in base_defs}
    best_models  = {}
    best_params  = {}
    best_scores  = {}

    for name, model in base_defs.items():
        print(f"\n  ▶ Tuning {name}...")
        t0 = time.time()
        try:
            rs = RandomizedSearchCV(
                estimator=model,
                param_distributions=grids[name],
                n_iter=n_iter_map[name],
                cv=spatial_cv,
                scoring='roc_auc',
                n_jobs=1,
                random_state=seed,
                refit=True,
                error_score=0.0,
            )
            rs.fit(X_train_sc, y_train)
            best_models[name] = rs.best_estimator_
            best_params[name] = rs.best_params_
            best_scores[name] = rs.best_score_
            print(f"    Best CV AUC: {rs.best_score_:.4f} | Time: {time.time()-t0:.1f}s")
            print(f"    Params     : {rs.best_params_}")
        except Exception as e:
            print(f"    FAILED ({e}) — skipping {name}")
            best_scores[name] = 0.0

    # Guard against silent model-drop: a tuning exception above removes a base
    # learner from best_models WITHOUT raising, which would quietly shrink the
    # ensemble (this is exactly how the NB-missing run happened).  Fail loudly
    # so a dropped learner can never slip into a full run unnoticed.
    missing = [n for n in base_defs if n not in best_models]
    if missing:
        raise RuntimeError(
            f"[tune_base_models] {len(missing)} base learner(s) failed tuning and "
            f"were dropped: {missing}. Expected all {len(base_defs)} "
            f"({list(base_defs.keys())}). Fix the cause before running — a partial "
            f"base-learner set silently changes the stacking meta-features.")

    with open(OUT_DIR / 'best_hyperparameters.json', 'w') as f:
        json.dump({k: {pk: str(pv) for pk, pv in v.items()}
                   for k, v in best_params.items()}, f, indent=2)
    print("\n  Saved: best_hyperparameters.json")
    return best_models, best_params, best_scores


# =============================================================================
# 8. TRADITIONAL MEAN ENSEMBLE  (biomod2-style)
# =============================================================================
class MeanEnsemble:
    """
    Traditional SDM ensemble: unweighted mean of base model probabilities.
    Analogous to biomod2's 'mean' ensemble; used as the baseline for
    comparison against the Bayesian GPC meta-learner.
    """
    def __init__(self, estimators):
        self.estimators = estimators   # list of (name, fitted_model) tuples

    def predict_proba(self, X):
        probs = np.stack(
            [mdl.predict_proba(X)[:, 1] for _, mdl in self.estimators], axis=1
        )
        p = probs.mean(axis=1)
        return np.column_stack([1 - p, p])


# =============================================================================
# 8b. TSS-WEIGHTED MEAN ENSEMBLE
# =============================================================================
class WeightedMeanEnsemble:
    """
    TSS-weighted mean of base model probabilities.
    Weights are proportional to each base model's TSS on the test set.
    Provides a stronger baseline than unweighted mean: GPC must beat a
    performance-weighted ensemble to establish its contribution.
    """
    def __init__(self, estimators, weights):
        self.estimators = estimators        # list of (name, fitted_model)
        weights = np.array(weights, dtype=np.float64)
        weights = np.maximum(weights, 1e-6) # guard against zero/negative TSS
        self.weights = weights / weights.sum()

    def predict_proba(self, X):
        probs = np.stack(
            [mdl.predict_proba(X)[:, 1] for _, mdl in self.estimators], axis=1
        )
        p = (probs * self.weights).sum(axis=1)
        return np.column_stack([1 - p, p])


# =============================================================================
# 9. STACKING META-CLASSIFIER (BAYESIAN GPC META-LEARNER)
# =============================================================================

# ── SVGP classifier — one of two backends depending on what's installed ───
if _GP_BACKEND == 'torch':
    class _SVGPModel(ApproximateGP):
        """GPyTorch SVGP model with ARD Matern-2.5 kernel."""
        def __init__(self, inducing_points, n_feat):
            var_dist  = CholeskyVariationalDistribution(inducing_points.size(0))
            var_strat = VariationalStrategy(
                self, inducing_points, var_dist, learn_inducing_locations=True
            )
            super().__init__(var_strat)
            self.mean_module  = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=n_feat)
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    class SVGPClassifier:
        """GPyTorch SVGP — ARD Matern-2.5, BernoulliLikelihood, variational ELBO."""
        def __init__(self, n_inducing=150, n_epochs=200, lr=0.01, random_state=42):
            self.n_inducing = n_inducing; self.n_epochs = n_epochs
            self.lr = lr;                 self.random_state = random_state
            self.kernel_ = None;          self.length_scales_ = None

        def fit(self, X, y):
            torch.manual_seed(self.random_state)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_t, y_t = (torch.tensor(X, dtype=torch.float32).to(device),
                        torch.tensor(y, dtype=torch.float32).to(device))
            km = MiniBatchKMeans(n_clusters=self.n_inducing,
                                  random_state=self.random_state, n_init=10)
            km.fit(X)
            Z = torch.tensor(km.cluster_centers_, dtype=torch.float32).to(device)
            model = _SVGPModel(Z, X.shape[1]).to(device)
            likelihood = BernoulliLikelihood().to(device)
            mll = VariationalELBO(likelihood, model, num_data=len(y))
            model.train(); likelihood.train()
            opt = torch.optim.Adam(
                list(model.parameters()) + list(likelihood.parameters()), lr=self.lr)
            for epoch in range(self.n_epochs):
                opt.zero_grad()
                loss = -mll(model(X_t), y_t)
                loss.backward(); opt.step()
                if (epoch + 1) % 50 == 0:
                    print(f"    SVGP epoch {epoch+1}/{self.n_epochs}, "
                          f"ELBO={-loss.item():.4f}", flush=True)
            self.model_ = model; self.likelihood_ = likelihood; self.device_ = device
            ls  = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().squeeze()
            os_ = model.covar_module.outputscale.detach().cpu().item()
            self.length_scales_ = ls
            self.kernel_ = (f"{os_:.3f}² × Matern(length_scale={np.round(ls,3)}, "
                            f"nu=2.5) [SVGP/torch, m={self.n_inducing}]")
            return self

        def predict_proba(self, X):
            self.model_.eval(); self.likelihood_.eval()
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device_)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                probs = self.likelihood_(self.model_(X_t)).mean.cpu().numpy()
            return np.column_stack([1 - probs, probs])

elif _GP_BACKEND == 'gpy':
    class SVGPClassifier:
        """GPy sparse GP classifier — ARD Matern-5/2, FITC approximation, O(n·m²).

        Uses GPy's SparseGPClassification with k-means inducing points.
        Exposes length_scales_ for ARD diagnostic (same interface as GPC path)."""
        def __init__(self, n_inducing=150, n_epochs=200, lr=0.01, random_state=42):
            self.n_inducing = n_inducing; self.n_epochs = n_epochs
            self.random_state = random_state
            self.kernel_ = None;          self.length_scales_ = None

        def fit(self, X, y):
            np.random.seed(self.random_state)
            km = MiniBatchKMeans(n_clusters=self.n_inducing,
                                  random_state=self.random_state, n_init=10)
            km.fit(X)
            Z      = km.cluster_centers_
            kernel = GPy.kern.Matern52(input_dim=X.shape[1], ARD=True)
            model  = GPy.models.SparseGPClassification(
                X, y[:, None].astype(float), kernel=kernel, Z=Z
            )
            print(f"    Optimising GPy sparse GP ({self.n_epochs} max iters)...",
                  flush=True)
            model.optimize(messages=False, max_iters=self.n_epochs)
            self.model_ = model
            ls  = model.kern.lengthscale.values.copy()
            os_ = float(model.kern.variance.values[0])
            self.length_scales_ = ls
            self.kernel_ = (f"{os_:.3f}² × Matern52(length_scale={np.round(ls,3)}, "
                            f"ARD=True) [SVGP/GPy, m={self.n_inducing}]")
            return self

        def predict_proba(self, X):
            probs, _ = self.model_.predict(X)
            probs = np.clip(probs.ravel(), 1e-7, 1 - 1e-7)
            return np.column_stack([1 - probs, probs])


# ── Auxiliary meta-learners for the capacity-ladder comparison ───────────────
# All consume the SAME OOF base-model meta-features as LR_Meta / GPC_Meta, so
# the comparison is controlled.  Each exposes predict_proba1(X) → P(presence).
# Inputs are the logit-space base-model predictions (matching _base_probs).

class BayesianLogisticMeta:
    """Bayesian logistic regression via Laplace approximation (Tier B:
    Bayesian-linear).  MAP fit with a Gaussian prior, a Gaussian posterior
    around the MAP, and a probit-approximation posterior-predictive — giving
    calibrated, shrunk probabilities without MCMC.  Keeps the paper's
    Bayesian/uncertainty contribution in a LINEAR model that will not overfit
    the collinear meta-space the way the ARD GP does."""

    def __init__(self, prior_var=10.0, random_state=42):
        self.prior_var    = prior_var
        self.random_state = random_state

    def fit(self, X, y):
        X  = np.asarray(X, dtype=np.float64)
        Xa = np.column_stack([X, np.ones(len(X))])          # + intercept
        lr = LogisticRegression(C=self.prior_var, penalty='l2', solver='lbfgs',
                                max_iter=2000, random_state=self.random_state)
        lr.fit(X, y)
        self.w_ = np.concatenate([lr.coef_.ravel(), lr.intercept_])
        p     = expit(Xa @ self.w_)
        s     = np.clip(p * (1 - p), 1e-6, None)
        prior = np.full(Xa.shape[1], 1.0 / self.prior_var)
        prior[-1] = 1e-6                                    # ~flat prior on intercept
        H = (Xa * s[:, None]).T @ Xa + np.diag(prior)       # posterior precision
        try:
            self.cov_ = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self.cov_ = np.linalg.pinv(H)
        return self

    def predict_proba1(self, X):
        X  = np.asarray(X, dtype=np.float64)
        Xa = np.column_stack([X, np.ones(len(X))])
        mu    = Xa @ self.w_
        var   = np.einsum('ij,jk,ik->i', Xa, self.cov_, Xa)
        kappa = 1.0 / np.sqrt(1.0 + np.pi * np.clip(var, 0, None) / 8.0)
        return expit(kappa * mu)


class SuperLearnerMeta:
    """Non-negative least-squares convex blend of base-model probabilities
    (Tier A: van der Laan Super Learner).  Weights ≥ 0, renormalised to sum 1,
    so the output is a convex combination of base predictions — the classic
    robust stacking combiner.  Operates in probability space (expit of the
    logit meta-features), where a convex blend is well-defined."""

    def fit(self, X, y):
        P = expit(np.asarray(X, dtype=np.float64))
        w, _ = nnls(P, np.asarray(y, dtype=np.float64))
        s = w.sum()
        self.weights_ = (w / s) if s > 0 else np.full(P.shape[1], 1.0 / P.shape[1])
        return self

    def predict_proba1(self, X):
        P = expit(np.asarray(X, dtype=np.float64))
        return np.clip(P @ self.weights_, 0.0, 1.0)


class GAMMeta:
    """Logistic generalised additive model (Tier C: mild interpretable
    nonlinearity) — one B-spline smooth per base-model feature, via statsmodels.
    Falls back to plain logistic regression if the GAM fit fails, so a single
    bad replicate never yields NaN metrics."""

    def __init__(self, df=6, degree=3, alpha=1.0, random_state=42):
        self.df = df; self.degree = degree; self.alpha = alpha
        self.random_state = random_state
        self._ok = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        # Always keep a logistic fallback so neither a GAM fit nor a GAM predict
        # failure can ever produce NaN metrics.
        self._fallback = LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced',
            solver='lbfgs', random_state=self.random_state).fit(X, y)
        self._xmin_ = X.min(axis=0)
        self._xmax_ = X.max(axis=0)
        try:
            import statsmodels.api as sm
            from statsmodels.gam.api import GLMGam, BSplines
            p   = X.shape[1]
            bs  = BSplines(X, df=[self.df] * p, degree=[self.degree] * p)
            gam = GLMGam(y, exog=np.ones((len(X), 1)), smoother=bs,
                         alpha=[self.alpha] * p, family=sm.families.Binomial())
            self.res_ = gam.fit()
            self._ok  = True
        except Exception as e:
            print(f"    [GAM_Meta] GAM fit failed ({type(e).__name__}); "
                  f"using logistic-regression fallback")
        return self

    def predict_proba1(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self._ok:
            # statsmodels B-splines cannot extrapolate beyond the training knot
            # range; clip test points into [train_min, train_max] before transform.
            Xc = np.clip(X, self._xmin_, self._xmax_)
            try:
                pred = self.res_.predict(exog=np.ones((len(Xc), 1)), exog_smooth=Xc)
                return np.clip(np.asarray(pred, dtype=np.float64), 1e-7, 1 - 1e-7)
            except Exception:
                pass   # fall through to logistic fallback
        return self._fallback.predict_proba(X)[:, 1]


class CustomGPCStacking:
    """
    Custom stacking ensemble with Bayesian GPC meta-learner.

    Fixes four structural problems with using sklearn's StackingClassifier
    for GPC on large SDM datasets:

      1. Scale (O(n³)): sklearn's StackingClassifier passes ALL training
         OOF predictions (~7300 samples) to GPC.fit(). Exact GPC is
         reliable only below ~1500 samples. This class trains GPC on a
         stratified subsample of the OOF matrix.

      2. Class imbalance: GPC has no class_weight parameter. The 1:20
         presence:absence ratio in the OOF matrix causes the Laplace
         approximation to be dominated by absences. The subsample is drawn
         at ~1:3 ratio to give GPC a learnable signal.

      3. Kernel misspecification: the original isotropic Matern used a
         single length_scale shared across all meta-feature dimensions.
         This class builds an ARD (Automatic Relevance Determination)
         Matern kernel with one length scale per meta-feature.

      4. Passthrough violation: following Halder et al. (2025), the
         meta-learner is a pure combiner — it sees only the 6 base model
         probability outputs, not the original environmental features.
         passthrough=True was removed.
    """

    # Basin-A warm-start length-scales (raw-logit space, transform='none' only)
    _BASIN_A_LS = np.array([40.0, 4.0, 12.0, 10.0, 36.0])

    def __init__(self, base_models_dict, spatial_cv, n_restarts_optimizer=10,
                 meta_max_samples=1500, oof_n_splits=5, random_state=42,
                 use_svgp=False, n_inducing=150, n_gpc_ensemble=5,
                 meta_transform='zca', kernel_type='matern25',
                 calibrate=True, calibrate_method='sigmoid', calibrate_holdout=0.2,
                 gpc_select='holdout_auc'):
        self.base_models_dict     = base_models_dict
        self.spatial_cv           = spatial_cv
        self.n_restarts_optimizer = n_restarts_optimizer
        self.meta_max_samples     = meta_max_samples
        self.oof_n_splits         = oof_n_splits
        self.random_state         = random_state
        self.use_svgp             = use_svgp and _GPYTORCH_AVAILABLE
        self.n_inducing           = n_inducing
        self.n_gpc_ensemble       = n_gpc_ensemble  # H: GPCs on K different subsamples
        # Tier-3 stabilisation knobs:
        self.meta_transform       = meta_transform     # 'none'|'standard'|'zca'|'pca'
        self.kernel_type          = kernel_type        # 'matern25'|'matern15'|'rq'
        self.calibrate            = calibrate          # Platt/isotonic on held-out slice
        self.calibrate_method     = calibrate_method   # 'sigmoid'|'isotonic'
        self.calibrate_holdout    = calibrate_holdout  # fraction of meta-set for calibration
        self.gpc_select           = gpc_select         # 'lml'|'holdout_auc' restart selection
        self.gpc_                 = None   # raw GPC (kernel inspection / maps)
        self.gpc_predictor_       = None   # calibrated wrapper (or raw GPC) used at predict
        self.gpc_list_            = []     # all K GPCs for ensemble prediction
        self.gpc_weights_         = None   # softmax(LML) weights for ensemble
        self.lr_meta_             = None   # competing meta-learner for paper comparison
        self._tf_mean_            = None   # meta-feature transform: column means
        self._tf_W_               = None   # meta-feature transform: whitening matrix
        # Capacity-ladder meta-learners (all fitted on identical OOF features):
        self.gpc_lin_             = None   # raw GP with linear (DotProduct) kernel
        self.gpc_lin_predictor_   = None   # calibrated wrapper for the linear GP
        self.enet_meta_           = None   # ElasticNet logistic
        self.sl_meta_             = None   # Super Learner (NNLS convex blend)
        self.blr_meta_            = None   # Bayesian logistic (Laplace)
        self.gam_meta_            = None   # logistic GAM
        self.rf_meta_             = None   # random forest meta
        self.gbm_meta_            = None   # gradient boosting (XGB) meta

    # ── Logit transform ───────────────────────────────────────────────────
    @staticmethod
    def _to_logit(p):
        """Map probabilities to logit space for Euclidean kernel compatibility.
        Euclidean distance is distorted near p=0/1 in probability space;
        logit maps [0,1] → (-inf,+inf) where distances are uniform."""
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p)).astype(np.float32)

    # ── OOF prediction collection ─────────────────────────────────────────
    def _collect_oof(self, X, y, oof_cv):
        n_models = len(self.base_models_dict)
        oof      = np.zeros((len(y), n_models), dtype=np.float32)
        for fold, (tr_idx, te_idx) in enumerate(oof_cv.split(X, y)):
            print(f"    Fold {fold + 1}/{oof_cv.n_splits}...", flush=True)
            X_tr, X_te, y_tr = X[tr_idx], X[te_idx], y[tr_idx]
            for j, (name, model) in enumerate(self.base_models_dict.items()):
                m = clone(model)
                m.fit(X_tr, y_tr)
                oof[te_idx, j] = m.predict_proba(X_te)[:, 1]
        return self._to_logit(oof)

    def _stratified_subsample(self, y, max_n, seed=None):
        """1:2 balanced subsample (all presences + 2× absences), capped at max_n."""
        rng     = np.random.RandomState(seed if seed is not None else self.random_state)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_pos   = len(pos_idx)                      # take all presences
        n_neg   = min(len(neg_idx), n_pos * 2)      # 2× absences → 1:2 ratio
        if n_pos + n_neg > max_n:                   # fallback cap
            n_pos = min(len(pos_idx), max_n // 3)
            n_neg = min(len(neg_idx), max_n - n_pos)
        return np.concatenate([
            rng.choice(pos_idx, n_pos, replace=False),
            rng.choice(neg_idx, n_neg, replace=False),
        ])

    # ── Meta-feature transform (decorrelate logit space for stable ARD) ───
    def _fit_meta_transform(self, X):
        """Fit the meta-feature transform on logit OOF features and return X_t.

        ZCA-whitening decorrelates the collinear base-model logits (LR-SVM ~0.9)
        that make the ARD length-scales non-identifiable and cause the
        replicate-to-replicate basin-hopping / bound-pinning.  ZCA (symmetric
        whitening) rotates back toward the original axes, so the per-base-model
        length-scale interpretation stays approximately valid (unlike PCA)."""
        Xc = X.astype(np.float64)
        self._tf_mean_ = Xc.mean(axis=0)
        Xc = Xc - self._tf_mean_
        if self.meta_transform == 'none':
            self._tf_W_ = None
            return X
        if self.meta_transform == 'standard':
            self._tf_W_ = np.diag(1.0 / (Xc.std(axis=0) + 1e-8))
        else:  # 'zca' or 'pca'
            cov            = np.cov(Xc, rowvar=False)
            eigval, eigvec = np.linalg.eigh(cov)
            eigval         = np.clip(eigval, 1e-8, None)
            D_inv_sqrt     = np.diag(1.0 / np.sqrt(eigval))
            W_pca          = eigvec @ D_inv_sqrt           # X_t = Xc @ W
            self._tf_W_    = W_pca if self.meta_transform == 'pca' else W_pca @ eigvec.T
        return self._apply_meta_transform(X)

    def _apply_meta_transform(self, X):
        """Apply the fitted transform to a logit meta-feature matrix."""
        if self._tf_W_ is None:
            return X
        return ((X.astype(np.float64) - self._tf_mean_) @ self._tf_W_).astype(np.float32)

    def _build_kernel(self, n_feat, kernel_type=None):
        """Build a GPC kernel.  `kernel_type` overrides self.kernel_type so the
        same machinery can fit several GP variants (Matern/RQ/linear).  Init and
        bounds depend on the meta-feature space: whitened space is unit-variance/
        decorrelated → neutral ℓ=1 init and tight (0.1, 100) bounds; raw-logit
        space keeps the Basin-A warm-start."""
        kt = kernel_type or self.kernel_type
        if self.meta_transform == 'none':
            ls_init   = self._BASIN_A_LS[:n_feat] if n_feat == 5 else np.ones(n_feat)
            ls_bounds = (3.0, 1e3)
        else:
            ls_init   = np.ones(n_feat)
            ls_bounds = (0.1, 100.0)
        if kt == 'linear':
            # DotProduct = Bayesian linear classifier in GP form (Tier B): no
            # length-scales to overfit, isolates "it's the nonlinearity, not the GP".
            # Bounds are kept TIGHT: the default ConstantKernel bounds (1e-5,1e5)
            # let the held-out-AUC random restarts sample a kernel amplitude of
            # ~1e5, which overflows the Laplace-approx Newton iteration → NaN
            # predict_proba.  Whitened meta-features are ~unit scale, so a
            # well-scaled constant in (1e-1,1e1) and sigma_0 in (1e-2,1e1) suffice.
            return (ConstantKernel(1.0, constant_value_bounds=(1e-1, 1e1)) *
                    DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e1)))
        if kt == 'rq':
            # sklearn RationalQuadratic is isotropic (no ARD); pairs well with
            # whitened unit-variance axes where a single length-scale suffices.
            return 1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0,
                                           length_scale_bounds=ls_bounds,
                                           alpha_bounds=(1e-2, 1e3))
        nu = 1.5 if kt == 'matern15' else 2.5
        return 1.0 * Matern(length_scale=ls_init.copy(),
                            length_scale_bounds=ls_bounds, nu=nu)

    def _fit_gpc_holdout_auc(self, X_fit, y_fit, X_sel, y_sel, n_feat,
                             kernel_type=None, n_restarts=None):
        """Run `n_restarts` GP fits from sampled starts and keep the one with the
        best HELD-OUT AUC, not the best LML.

        Why: the base-model meta-features are collinear, so many kernels give
        near-equal marginal likelihood — and the optimizer can settle in a
        high-LML basin that overfits the OOF meta-features yet collapses on
        unseen data (observed at pa_seed 42/50: good LML, AUC ~0.83-0.90).
        LML cannot discriminate these, so we select by predictive skill on a
        held-out slice instead.  This is kernel-agnostic (works for Matern/RQ).

        Restart 0 uses the kernel's warm-start init; the rest sample a
        log-uniform start within the kernel bounds — the same scheme sklearn
        uses internally for `n_restarts_optimizer`, except we score by AUC."""
        rng        = np.random.RandomState(self.random_state)
        n_restarts = max(1, n_restarts or self.n_restarts_optimizer)
        best       = {'auc': -np.inf, 'lml': -np.inf, 'gpc': None}
        t_i        = time.time()
        for r in range(n_restarts):
            kernel = self._build_kernel(n_feat, kernel_type)
            if r > 0:
                bounds       = kernel.bounds          # log-space (n_theta, 2)
                kernel.theta = rng.uniform(bounds[:, 0], bounds[:, 1])
            gpc = GaussianProcessClassifier(
                kernel=kernel, n_restarts_optimizer=0,
                random_state=self.random_state, copy_X_train=True,
                max_iter_predict=200)
            gpc.fit(X_fit, y_fit)
            p_sel = gpc.predict_proba(X_sel)[:, 1]
            # NaN-guard: an ill-conditioned restart (e.g. a kernel amplitude
            # sampled at the bound) can diverge the Laplace-approx Newton step
            # and return non-finite probabilities.  Skip it instead of crashing
            # the whole pipeline; a later restart keeps the best valid fit.
            if not np.all(np.isfinite(p_sel)):
                continue
            auc = roc_auc_score(y_sel, p_sel)
            lml = gpc.log_marginal_likelihood_value_
            # Primary key held-out AUC; LML breaks ties (near-equal AUC).
            if (auc > best['auc'] + 1e-6 or
                    (abs(auc - best['auc']) <= 1e-6 and lml > best['lml'])):
                best = {'auc': auc, 'lml': lml, 'gpc': gpc}
        if best['gpc'] is None:
            # Every restart diverged (should not happen with the tight linear
            # bounds, but stay safe): fall back to a single warm-start fit.
            kernel = self._build_kernel(n_feat, kernel_type)
            best['gpc'] = GaussianProcessClassifier(
                kernel=kernel, n_restarts_optimizer=0,
                random_state=self.random_state, copy_X_train=True,
                max_iter_predict=200).fit(X_fit, y_fit)
            best['lml'] = best['gpc'].log_marginal_likelihood_value_
            best['auc'] = roc_auc_score(
                y_sel, np.nan_to_num(best['gpc'].predict_proba(X_sel)[:, 1],
                                     nan=0.5))
        print(f"    GPC fit ({time.time()-t_i:.1f}s, {n_restarts} restarts, "
              f"select=holdout_auc) | sel_AUC={best['auc']:.4f} | "
              f"LML={best['lml']:.3f} | kernel: {best['gpc'].kernel_}")
        return best['gpc']

    def _fit_gpc_variant(self, kernel_type, X_gpc, y_meta, n_restarts):
        """Fit one GPC on the (already transformed) meta-space with held-out-AUC
        restart selection and optional calibration.  Reused for every GP variant
        (Matern-2.5 headline, linear/DotProduct).  Returns (predictor, raw_gpc):
        predictor is the calibrated wrapper (or the raw GPC if calibrate=False)."""
        n_feat       = X_gpc.shape[1]
        select       = (self.gpc_select == 'holdout_auc')
        need_holdout = select or self.calibrate
        if need_holdout:
            X_fit, X_hold, y_fit, y_hold = train_test_split(
                X_gpc, y_meta, test_size=self.calibrate_holdout,
                stratify=y_meta, random_state=self.random_state)
        else:
            X_fit, y_fit = X_gpc, y_meta

        if select:
            gpc = self._fit_gpc_holdout_auc(X_fit, y_fit, X_hold, y_hold, n_feat,
                                            kernel_type=kernel_type,
                                            n_restarts=n_restarts)
        else:
            kernel = self._build_kernel(n_feat, kernel_type)
            gpc = GaussianProcessClassifier(
                kernel=kernel, n_restarts_optimizer=n_restarts,
                random_state=self.random_state, copy_X_train=True,
                max_iter_predict=200)
            t_i = time.time()
            gpc.fit(X_fit, y_fit)
            print(f"    GPC fit ({time.time()-t_i:.1f}s) | "
                  f"LML={gpc.log_marginal_likelihood_value_:.3f} | "
                  f"kernel: {gpc.kernel_}")

        if self.calibrate:
            cal = CalibratedClassifierCV(FrozenEstimator(gpc),
                                         method=self.calibrate_method)
            cal.fit(X_hold, y_hold)
            print(f"    calibrated ({self.calibrate_method}, "
                  f"holdout={self.calibrate_holdout:.0%}, n_cal={len(y_hold)})")
            return cal, gpc
        return gpc, gpc

    # ── Fit: OOF collection → stratified subsample → GPC ─────────────────
    def fit(self, X, y, coords=None):
        # Build a higher-fold spatial CV for OOF if coords are provided.
        # More folds → each OOF fold model trains on more data → OOF predictions
        # are closer in quality to the final full-data models → less covariate shift.
        if coords is not None and self.oof_n_splits > self.spatial_cv.n_splits:
            print(f"  Building {self.oof_n_splits}-fold spatial CV for OOF "
                  f"(reduces covariate shift vs {self.spatial_cv.n_splits}-fold)...")
            oof_cv = SpatialBlockCV(
                n_splits       = self.oof_n_splits,
                block_size_deg = self.spatial_cv.block_size_deg,
                random_state   = self.spatial_cv.random_state,
            )
            oof_cv.precompute(coords)
        else:
            oof_cv = self.spatial_cv

        n_models = len(self.base_models_dict)
        print(f"  Collecting OOF predictions ({n_models} models × "
              f"{oof_cv.n_splits} folds)...")
        t0  = time.time()
        oof = self._collect_oof(X, y, oof_cv)
        print(f"  OOF collection complete ({time.time()-t0:.1f}s)")

        sub_idx = self._stratified_subsample(y, self.meta_max_samples)
        X_meta  = oof[sub_idx]
        y_meta  = y[sub_idx]
        n_pos   = int(y_meta.sum())
        n_neg   = int((y_meta == 0).sum())
        print(f"  GPC meta-training set: {len(y_meta)} samples "
              f"(presence={n_pos}, absence={n_neg}, ratio=1:{n_neg/max(1,n_pos):.1f})")

        t1 = time.time()
        if self.use_svgp:
            print(f"  Fitting SVGP (ARD Matern-2.5, {self.n_inducing} inducing pts, "
                  f"BernoulliLikelihood, 200 epochs)...")
            self.gpc_ = SVGPClassifier(
                n_inducing=self.n_inducing,
                n_epochs=200,
                lr=0.01,
                random_state=self.random_state,
            )
            self.gpc_.fit(X_meta, y_meta)
            print(f"  SVGP fit complete ({time.time()-t1:.1f}s)")
            print(f"  Optimised kernel: {self.gpc_.kernel_}")
        else:
            # ── Capacity-ladder meta-learner registry ─────────────────────────
            # All meta-learners train on the SAME OOF base-model features for a
            # controlled comparison.  The paper's claim — meta-learners beat
            # traditional ensembling, but only when simple & regularised — is
            # tested across three tiers: simple/regularised linear → Bayesian-
            # linear → flexible nonlinear.  GP variants use the whitened (ZCA)
            # space; all others use the raw logit meta-features (like LR_Meta).
            spw_meta = n_neg / max(1, n_pos)
            X_gpc    = self._fit_meta_transform(X_meta)   # ZCA; shared by GP variants

            # Tier C (flexible Bayesian): ARD Matern — the headline GPC_Meta.
            # Restart-selection by held-out AUC (gpc_select) guards the overfit
            # basins that LML cannot detect on collinear meta-features.
            print(f"  [GPC_Meta]  kernel=matern25, transform={self.meta_transform}, "
                  f"select={self.gpc_select}")
            self.gpc_predictor_, self.gpc_ = self._fit_gpc_variant(
                'matern25', X_gpc, y_meta, self.n_restarts_optimizer)
            self.gpc_list_    = [self.gpc_]
            self.gpc_weights_ = None

            # Tier B (Bayesian-linear): GP with a DotProduct (linear) kernel —
            # same GP machinery, no length-scales; isolates nonlinearity's role.
            print(f"  [GPClin_Meta]  kernel=linear (DotProduct), "
                  f"transform={self.meta_transform}")
            self.gpc_lin_predictor_, self.gpc_lin_ = self._fit_gpc_variant(
                'linear', X_gpc, y_meta, max(8, self.n_restarts_optimizer // 4))

            # Tier A (simple/regularised linear).
            self.lr_meta_ = LogisticRegression(
                C=1.0, max_iter=1000, class_weight='balanced',
                solver='lbfgs', random_state=self.random_state).fit(X_meta, y_meta)
            self.enet_meta_ = LogisticRegression(
                penalty='elasticnet', l1_ratio=0.5, C=1.0, solver='saga',
                max_iter=5000, class_weight='balanced',
                random_state=self.random_state).fit(X_meta, y_meta)
            self.sl_meta_ = SuperLearnerMeta().fit(X_meta, y_meta)

            # Tier B (Bayesian-linear): Laplace logistic — keeps the Bayesian
            # contribution in a linear model that won't overfit the meta-space.
            self.blr_meta_ = BayesianLogisticMeta(
                prior_var=10.0, random_state=self.random_state).fit(X_meta, y_meta)

            # Tier C (flexible nonlinear): GAM, random forest, gradient boosting.
            self.gam_meta_ = GAMMeta(df=6, degree=3, alpha=1.0,
                                     random_state=self.random_state).fit(X_meta, y_meta)
            self.rf_meta_  = RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=5,
                class_weight='balanced', n_jobs=-1,
                random_state=self.random_state).fit(X_meta, y_meta)
            self.gbm_meta_ = xgb.XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                scale_pos_weight=spw_meta, eval_metric='logloss',
                verbosity=0, random_state=self.random_state).fit(X_meta, y_meta)

            print("  Meta-learner registry fitted (9): LR, ENet, SuperLearner, "
                  "BayesLR, GPClin, GPC(matern), GAM, RF, GBM "
                  "— all on identical OOF meta-features")
        return self

    # ── Prediction ────────────────────────────────────────────────────────
    def _base_probs(self, X):
        """Stack logit-transformed base model probs into (n, n_models) matrix."""
        raw = np.column_stack([
            mdl.predict_proba(X)[:, 1]
            for mdl in self.base_models_dict.values()
        ])
        return self._to_logit(raw)

    def predict_proba(self, X):
        # Same logit → meta-feature transform (e.g. ZCA whitening) used at fit.
        meta = self._apply_meta_transform(self._base_probs(X))
        if len(self.gpc_list_) > 1:
            # Legacy LML-weighted ensemble path (unused when n_gpc_ensemble=1).
            if self.gpc_weights_ is not None:
                probs = sum(
                    w * g.predict_proba(meta)
                    for w, g in zip(self.gpc_weights_, self.gpc_list_)
                )
            else:
                probs = np.mean([g.predict_proba(meta) for g in self.gpc_list_], axis=0)
            return probs
        # Calibrated wrapper when calibrate=True, else the raw GPC.
        return self.gpc_predictor_.predict_proba(meta)

    def predict_all_meta(self, X):
        """Return P(presence) for every meta-learner in the capacity ladder,
        as an ordered dict {name: probs(n,)}.  Base-model meta-features are
        computed once; GP variants use the whitened space, the rest use the
        raw logit features — matching how each was fitted."""
        return self._predict_all_meta_from_logit(self._base_probs(X))

    def _predict_all_meta_from_logit(self, L):
        """Meta-learner predictions from an already-computed logit base-feature
        matrix L (n, n_base).  Used by predict_all_meta and by the meta-feature
        contribution analysis (which permutes columns of L)."""
        W = self._apply_meta_transform(L)          # whitened (ZCA) — GP variants
        out = {}
        out['LR_Meta']          = self.lr_meta_.predict_proba(L)[:, 1]
        out['ENet_Meta']        = self.enet_meta_.predict_proba(L)[:, 1]
        out['SuperLearner_Meta'] = self.sl_meta_.predict_proba1(L)
        out['BayesLR_Meta']     = self.blr_meta_.predict_proba1(L)
        out['GPClin_Meta']      = self.gpc_lin_predictor_.predict_proba(W)[:, 1]
        out['GPC_Meta']         = self.gpc_predictor_.predict_proba(W)[:, 1]
        out['GAM_Meta']         = self.gam_meta_.predict_proba1(L)
        out['RF_Meta']          = self.rf_meta_.predict_proba(L)[:, 1]
        out['GBM_Meta']         = self.gbm_meta_.predict_proba(L)[:, 1]
        return out

    def transform(self, X):
        """Return meta-features (logit base-model probs)."""
        return self._base_probs(X)

    # ── sklearn-compatible properties ─────────────────────────────────────
    @property
    def final_estimator_(self):
        return self.gpc_

    @property
    def estimators(self):
        return list(self.base_models_dict.items())


def build_stacking_classifier(best_models, spatial_cv, seed=RANDOM_STATE):
    print("\n[STEP 7] Building Custom GPC Stacking Meta-Classifier")
    stacking = CustomGPCStacking(
        base_models_dict     = best_models,
        spatial_cv           = spatial_cv,
        n_restarts_optimizer = 50,
        meta_max_samples     = 1500,
        oof_n_splits         = 5,
        random_state         = seed,
        use_svgp             = False,
        n_inducing           = 0,
        n_gpc_ensemble       = 1,
        meta_transform       = 'zca',       # decorrelate collinear base-model logits
        kernel_type          = 'matern25',  # 'matern25'|'matern15'|'rq' (ARD restores per-base interpretability)
        calibrate            = True,         # Platt/isotonic on held-out slice (Brier)
        calibrate_method     = 'sigmoid',
        calibrate_holdout    = 0.2,
        gpc_select           = 'holdout_auc',  # 'lml'|'holdout_auc' (AUC guards overfit basins)
    )
    _whitened = stacking.meta_transform != 'none'
    # Basin-A warm-start applies only in the legacy raw-logit 5-feature case;
    # with the 6-base-learner set (or any n≠5) the 'none' path falls back to ℓ=1.
    _ls_desc  = ("ℓ=1, bounds=(0.1, 100)" if _whitened
                 else (f"ℓ={np.array2string(CustomGPCStacking._BASIN_A_LS, precision=0)} "
                       f"(Basin A)" if len(best_models) == 5 else "ℓ=1")
                      + ", bounds=(3.0, 1e3)")
    print("  Architecture:")
    print(f"    Base models   : {list(best_models.keys())}")
    print(f"    Meta-learners : capacity ladder ({len(META_ORDER)}) {META_ORDER}")
    print(f"    Headline GPC  : {stacking.kernel_type} (exact O(n³)) + GPClin (DotProduct)")
    print(f"    Meta-features : {len(best_models)} logit base probs"
          f" → {stacking.meta_transform} transform (GP variants)")
    print(f"    OOF folds     : {stacking.oof_n_splits} (internal, reduces covariate shift)")
    print(f"    Meta train    : up to 1500 OOF samples (1:2 ratio, presence-limited)")
    print(f"    Restarts      : {stacking.n_restarts_optimizer}"
          f" (select by {stacking.gpc_select})")
    print(f"    Kernel init   : {_ls_desc}")
    print(f"    Calibration   : {stacking.calibrate_method if stacking.calibrate else 'off'}"
          f"{f' ({stacking.calibrate_holdout:.0%} holdout)' if stacking.calibrate else ''}")
    print(f"    GPC seed      : {RANDOM_STATE} (fixed, decoupled from PA seed)")
    return stacking


# =============================================================================
# 10. EVALUATION
# =============================================================================
def max_tss_threshold(y_true, y_prob):
    """Find probability threshold maximising True Skill Statistic (TSS)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    tss      = tpr - fpr
    best_idx = int(np.argmax(tss))
    return float(thresholds[best_idx]), float(tss[best_idx])


def _guard_probs(name, y_prob):
    """Sanitise a probability vector before scoring.  If a model emits any
    non-finite value (e.g. a pathological GP predict), replace NaN/±inf with
    0.5 and warn LOUDLY rather than letting roc_auc_score raise and abort the
    whole replicate mid-run.  Returns the cleaned vector."""
    y_prob = np.asarray(y_prob, dtype=np.float64)
    bad = ~np.isfinite(y_prob)
    if bad.any():
        print(f"  [WARNING] {name}: {int(bad.sum())}/{len(y_prob)} non-finite "
              f"predictions → replaced with 0.5 (metrics for this model are "
              f"unreliable this replicate).")
        y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    return y_prob


def compute_metrics(y_true, y_prob, threshold=0.5, name='Model'):
    y_pred = (y_prob >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    tss          = float(np.max(tpr - fpr))
    return {
        'Model':     name,
        'Accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1':        round(f1_score(y_true, y_pred, zero_division=0), 4),
        'AUC':       round(roc_auc_score(y_true, y_prob), 4),
        'TSS':       round(tss, 4),
        'Brier':     round(brier_score_loss(y_true, y_prob), 4),  # lower = better
    }


# Capacity-ladder meta-learner order (simple/regularised → Bayesian-linear →
# flexible nonlinear).  Matches CustomGPCStacking.predict_all_meta and is used
# for hierarchy prints and box-plot ordering.
META_ORDER = ['LR_Meta', 'ENet_Meta', 'SuperLearner_Meta', 'BayesLR_Meta',
              'GPClin_Meta', 'GPC_Meta', 'GAM_Meta', 'RF_Meta', 'GBM_Meta']


def evaluate_all_models(best_models, stacking, mean_ensemble, X_test_sc, y_test):
    print("=" * 60)
    print("[STEP 9] Model Evaluation")
    print("=" * 60)

    all_metrics  = []
    all_probs    = {}
    thresholds   = {}

    # ── Base models ───────────────────────────────────────────────────────
    print("  Base models:")
    for name, model in best_models.items():
        y_prob = _guard_probs(name, model.predict_proba(X_test_sc)[:, 1])
        thr, _ = max_tss_threshold(y_test, y_prob)
        thresholds[name] = thr
        metrics = compute_metrics(y_test, y_prob, thr, name)
        all_metrics.append(metrics)
        all_probs[name] = y_prob
        print(f"  {name:12s}: AUC={metrics['AUC']:.4f} | TSS={metrics['TSS']:.4f} | "
              f"Brier={metrics['Brier']:.4f} | Threshold={thr:.3f}")

    # Pairwise base-model correlation (diversity audit) is plotted post-loop
    # from the MEDIAN across replicates — see _plot_base_model_correlation,
    # called from main() — so the figure summarises every replicate, not just
    # the last one that would otherwise overwrite it.

    # ── TSS-weighted mean ensemble (built from test-set TSS above) ─────────
    base_tss_weights = [m['TSS'] for m in all_metrics]
    weighted_ensemble = WeightedMeanEnsemble(
        estimators=list(best_models.items()),
        weights=base_tss_weights
    )
    print("\n  TSS weights for WeightedMeanEnsemble:")
    for (name, _), w in zip(best_models.items(), weighted_ensemble.weights):
        print(f"    {name}: {w:.4f}")

    # ── Ensemble / meta-learner evaluation ────────────────────────────────
    print("\n  Ensemble and meta-learner results:")

    y_prob_me  = _guard_probs('MeanEnsemble', mean_ensemble.predict_proba(X_test_sc)[:, 1])
    thr_me, _  = max_tss_threshold(y_test, y_prob_me)
    thresholds['MeanEnsemble'] = thr_me
    metrics_me = compute_metrics(y_test, y_prob_me, thr_me, 'MeanEnsemble')
    all_metrics.append(metrics_me)
    all_probs['MeanEnsemble'] = y_prob_me
    print(f"  {'MeanEnsemble':22s}: AUC={metrics_me['AUC']:.4f} | "
          f"TSS={metrics_me['TSS']:.4f} | Brier={metrics_me['Brier']:.4f} | "
          f"Threshold={thr_me:.3f}")

    y_prob_wme  = _guard_probs('WeightedMeanEnsemble',
                               weighted_ensemble.predict_proba(X_test_sc)[:, 1])
    thr_wme, _  = max_tss_threshold(y_test, y_prob_wme)
    thresholds['WeightedMeanEnsemble'] = thr_wme
    metrics_wme = compute_metrics(y_test, y_prob_wme, thr_wme, 'WeightedMeanEnsemble')
    all_metrics.append(metrics_wme)
    all_probs['WeightedMeanEnsemble'] = y_prob_wme
    print(f"  {'WeightedMeanEnsemble':22s}: AUC={metrics_wme['AUC']:.4f} | "
          f"TSS={metrics_wme['TSS']:.4f} | Brier={metrics_wme['Brier']:.4f} | "
          f"Threshold={thr_wme:.3f}")

    # Capacity-ladder meta-learners (all on the same OOF features).
    for name, y_prob in stacking.predict_all_meta(X_test_sc).items():
        y_prob = _guard_probs(name, y_prob)
        thr, _ = max_tss_threshold(y_test, y_prob)
        thresholds[name] = thr
        m = compute_metrics(y_test, y_prob, thr, name)
        all_metrics.append(m)
        all_probs[name] = y_prob
        print(f"  {name:22s}: AUC={m['AUC']:.4f} | "
              f"TSS={m['TSS']:.4f} | Brier={m['Brier']:.4f} | Threshold={thr:.3f}")

    metrics_df = pd.DataFrame(all_metrics).set_index('Model')
    # NOTE: not written to CSV here — a per-replicate write would be overwritten
    # by the next replicate.  The median across replicates is written to
    # evaluation_metrics.csv post-loop (see _plot_metrics_heatmap).
    print("\n  This replicate's evaluation metrics table:")
    print(metrics_df.to_string())

    print("\n  Capacity-ladder hierarchy (AUC | TSS | Brier):")
    hierarchy = ['MeanEnsemble', 'WeightedMeanEnsemble'] + META_ORDER
    rows = [h for h in hierarchy if h in metrics_df.index]
    print(metrics_df.loc[rows, ['AUC', 'TSS', 'Brier']].to_string())

    # ROC curves, confusion matrices and the metrics heatmap are NOT plotted
    # here — a per-replicate plot would just be overwritten by the next
    # replicate.  They are generated once, post-loop, from the MEDIAN across all
    # replicates (see _plot_roc_curves / _plot_confusion_matrices /
    # _plot_metrics_heatmap, called from main()).

    return metrics_df, all_probs, thresholds, weighted_ensemble


def _plot_base_model_correlation(roc_records):
    """
    MEDIAN pairwise Pearson correlation of base-model probability outputs across
    all replicates (diversity audit).

    roc_records: list of (y_test, all_probs) — one per replicate.  The base-model
    correlation matrix is computed per replicate, then median-aggregated cell-wise
    so the figure reflects every replicate rather than just the last.
    """
    n_reps     = len(roc_records)
    all_probs0 = roc_records[0][1]
    base_names = [k for k in all_probs0
                  if k not in ('MeanEnsemble', 'WeightedMeanEnsemble')
                  and k not in META_ORDER]

    corrs = [pd.DataFrame({n: all_probs[n] for n in base_names})
             .corr(method='pearson').values
             for _, all_probs in roc_records]
    med_corr = np.median(np.stack(corrs, axis=0), axis=0)
    corr_df  = pd.DataFrame(med_corr, index=base_names, columns=base_names)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, ax=ax,
                square=True, linewidths=0.5)
    ax.set_title(f'Base Model Prediction Correlation (median, N={n_reps} replicates)\n'
                 '(Pearson, test-set probabilities — low = diverse = informative to GPC)',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figures' / 'base_model_correlation.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/base_model_correlation.png (median across replicates)")
    print("  Median correlation matrix:")
    print(corr_df.round(3).to_string())


def _roc_line_style(name):
    """Per-model ROC line style (highlights the headline meta-learners)."""
    if name == 'GPC_Meta':              return 2.5, '-'
    if name == 'LR_Meta':               return 2.0, '-.'
    if name == 'WeightedMeanEnsemble':  return 2.0, (0, (3, 1, 1, 1))
    if name == 'MeanEnsemble':          return 2.0, ':'
    return 1.5, '--'


def _plot_roc_curves(roc_records):
    """
    MEDIAN ROC curve per model across all replicates.

    roc_records: list of (y_test, all_probs) — one tuple per replicate.  Each
    replicate's per-model ROC is interpolated onto a common FPR grid; the median
    TPR across replicates is plotted and the legend reports the median AUC.
    """
    n_reps      = len(roc_records)
    model_names = list(roc_records[0][1].keys())
    colors      = plt.cm.tab20(np.linspace(0, 1, len(model_names)))
    mean_fpr    = np.linspace(0, 1, 200)

    fig, ax = plt.subplots(figsize=(8, 7))
    for name, color in zip(model_names, colors):
        tprs, aucs = [], []
        for y_test, all_probs in roc_records:
            if name not in all_probs:
                continue
            fpr, tpr, _ = roc_curve(y_test, all_probs[name])
            interp_tpr    = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc_score(y_test, all_probs[name]))
        if not tprs:
            continue
        med_tpr     = np.median(np.vstack(tprs), axis=0)
        med_tpr[-1] = 1.0
        med_auc     = float(np.median(aucs))
        lw, ls      = _roc_line_style(name)
        ax.plot(mean_fpr, med_tpr, lw=lw, ls=ls, color=color,
                label=f"{name} (AUC = {med_auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'Median ROC Curves — All Models (N={n_reps} replicates)\n'
                 'solid=GPC · dash-dot=LR_Meta · dotted=MeanEnsemble · dashed=base',
                 fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figures' / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/roc_curves.png (median across replicates)")


def _plot_confusion_matrices(cm_records):
    """
    MEDIAN confusion matrix per model across all replicates.

    cm_records: list of (y_test, all_probs, thresholds) — one per replicate.
    Each replicate's confusion matrix is computed at that replicate's MaxTSS
    threshold; cells are then median-aggregated (and rounded to integer counts).
    The panel title shows the median threshold across replicates.
    """
    n_reps      = len(cm_records)
    model_names = list(cm_records[0][1].keys())
    cols = 4
    rows = int(np.ceil(len(model_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    for i, name in enumerate(model_names):
        cms, thrs = [], []
        for y_test, all_probs, thresholds in cm_records:
            thr    = thresholds[name]
            y_pred = (all_probs[name] >= thr).astype(int)
            cms.append(confusion_matrix(y_test, y_pred, labels=[0, 1]))
            thrs.append(thr)
        med_cm  = np.median(np.stack(cms, axis=0), axis=0).round().astype(int)
        med_thr = float(np.median(thrs))
        ConfusionMatrixDisplay(med_cm, display_labels=['Absent', 'Present']
                               ).plot(ax=axes[i], colorbar=False, cmap='Blues')
        axes[i].set_title(f"{name}\n(median thr={med_thr:.2f})", fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f'Median Confusion Matrices (MaxTSS threshold, N={n_reps} replicates)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figures' / 'confusion_matrices.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/confusion_matrices.png (median across replicates)")


def _plot_metrics_heatmap(metrics_records):
    """
    MEDIAN performance-metric heatmap across all replicates.

    metrics_records: list of per-replicate metrics_df (index=Model, 7 metric
    columns).  The median of each metric per model is taken across replicates;
    the median table is written to evaluation_metrics.csv alongside the figure.
    """
    n_reps   = len(metrics_records)
    order    = list(metrics_records[0].index)
    stacked  = pd.concat([m.astype(float) for m in metrics_records])
    median_df = stacked.groupby(level=0).median().loc[order]
    median_df.to_csv(OUT_DIR / 'evaluation_metrics.csv')

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(median_df, annot=True, fmt='.4f',
                cmap='YlOrRd', ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Score'})
    ax.set_title(f'Model Performance Metrics (median across N={n_reps} replicates)',
                 fontsize=13)
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figures' / 'metrics_heatmap.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/metrics_heatmap.png (median across replicates)")
    print("  Saved: evaluation_metrics.csv (median across replicates)")


# =============================================================================
# 10. META-FEATURE CONTRIBUTION ANALYSIS  (replaces SHAP)
# =============================================================================
def meta_feature_contributions(stacking, X_test, y_test, base_names,
                               n_repeats=8, seed=42):
    """How much each meta-feature (a base-model prediction) contributes to each
    meta-learner — a model-agnostic replacement for SHAP.

    Permutation importance in the base-model meta-feature space: each base
    model's logit column is shuffled and the resulting drop in that meta-
    learner's test AUC is recorded (mean over `n_repeats` shuffles).  Permuting
    in the raw base-feature space — then re-applying the ZCA transform inside
    the GP variants — attributes contributions to BASE MODELS consistently
    across every meta-learner, including those that consume whitened features.

    Returns {meta_model: {base_feature: mean_auc_drop}} for this replicate."""
    L   = stacking._base_probs(X_test)               # (n, n_base) logit features
    rng = np.random.RandomState(seed)
    base_auc = {m: roc_auc_score(y_test, p)
                for m, p in stacking._predict_all_meta_from_logit(L).items()}
    contrib = {m: {} for m in base_auc}
    for j, feat in enumerate(base_names):
        drops = {m: [] for m in base_auc}
        for _ in range(n_repeats):
            Lp = L.copy()
            Lp[:, j] = rng.permutation(Lp[:, j])
            preds = stacking._predict_all_meta_from_logit(Lp)
            for m, p in preds.items():
                drops[m].append(base_auc[m] - roc_auc_score(y_test, p))
        for m in base_auc:
            contrib[m][feat] = float(np.mean(drops[m]))
    return contrib


def _save_meta_contributions(avg_contrib, base_names):
    """Save the averaged meta-feature contribution matrix (base-model meta-
    features × meta-learners) as CSV + heatmap.  Values are the mean test-AUC
    drop when each base-model meta-feature is permuted (higher = more important
    to that meta-learner), averaged across all N_REPLICATES PA replicates."""
    print("\n  Saving meta-feature contribution outputs...")
    meta_models = [m for m in META_ORDER if m in avg_contrib]
    M = pd.DataFrame(
        {m: {f: avg_contrib[m].get(f, 0.0) for f in base_names} for m in meta_models}
    ).reindex(index=base_names, columns=meta_models)
    M.to_csv(OUT_DIR / 'meta_feature_contributions.csv')
    print("  Saved: meta_feature_contributions.csv "
          "(rows=base-model meta-features, cols=meta-learners; mean AUC drop)")

    fig, ax = plt.subplots(figsize=(1.1 * len(meta_models) + 3,
                                    0.6 * len(base_names) + 2))
    sns.heatmap(M.astype(float), annot=True, fmt='.3f', cmap='YlOrRd',
                ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Mean AUC drop when permuted'})
    ax.set_title(f'Meta-feature contribution to each meta-learner\n'
                 f'(permutation importance, mean across {N_REPLICATES} replicates)',
                 fontsize=12)
    ax.set_xlabel('Meta-learner')
    ax.set_ylabel('Base-model meta-feature')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figures' / 'meta_feature_contributions.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/meta_feature_contributions.png")
    print("\n  Contribution matrix (mean AUC drop):")
    print(M.round(4).to_string())


# =============================================================================
# 11. SPATIAL PREDICTION & MAP EXPORT
# =============================================================================


def all_model_order(best_models):
    """Canonical projection order for all 17 models:
    6 base learners → 2 traditional ensembles → 9 capacity-ladder meta-learners."""
    return (list(best_models.keys())
            + ['MeanEnsemble', 'WeightedMeanEnsemble']
            + list(META_ORDER))


def _write_geotiff(path, arr2d, profile):
    """Write a single-band float32 GeoTIFF (NaN nodata, LZW compressed)."""
    out_profile = profile.copy()
    out_profile.update(dtype='float32', count=1, nodata=np.nan, compress='lzw')
    with rasterio.open(path, 'w', **out_profile) as dst:
        dst.write(arr2d.astype(np.float32), 1)


def _vec_to_2d(vec, valid_mask, H, W):
    """Scatter a 1-D valid-pixel vector back into a full H×W grid (NaN elsewhere)."""
    out = np.full(H * W, np.nan, dtype=np.float32)
    out[valid_mask] = vec
    return out.reshape(H, W)


def _project_models(best_models, stacking, weighted_ensemble,
                    preprocessor, stack, H, W, sel_idx):
    """
    Batched raster projection of ALL 17 models for a single replicate.

    Base-model probabilities are computed once per batch and reused to derive
    the two traditional ensembles (mean / TSS-weighted) and all nine
    capacity-ladder meta-learners (LR/ENet/SuperLearner/BayesLR/GPClin/GPC/
    GAM/RF/GBM) — the meta-learners consume the same logit base-feature matrix
    the stacker was trained on, so no base model is predicted twice.

    Returns (valid_mask, {model_name: 1-D float32 P(presence) over valid pixels}).
    """
    pixels_all   = stack.reshape(len(PRED_NAMES), -1).T
    valid_mask   = ~np.any(np.isnan(pixels_all), axis=1)
    n_valid      = int(valid_mask.sum())
    X_valid_all  = pixels_all[valid_mask].astype(np.float32)
    X_valid_proc = preprocessor.transform(X_valid_all)
    X_valid_sc   = X_valid_proc[:, sel_idx]

    base_names = list(best_models.keys())
    w_wtd      = np.asarray(weighted_ensemble.weights, dtype=np.float64)  # aligned to base_names

    proba = {name: np.zeros(n_valid, dtype=np.float32)
             for name in all_model_order(best_models)}

    n_batches = int(np.ceil(n_valid / PRED_BATCH_SIZE))
    for b in range(n_batches):
        s = b * PRED_BATCH_SIZE
        e = min(s + PRED_BATCH_SIZE, n_valid)
        batch = X_valid_sc[s:e]

        # ── Base learners (single matrix, base_names order) ─────────────────
        P = np.column_stack(
            [best_models[name].predict_proba(batch)[:, 1] for name in base_names]
        ).astype(np.float32)
        for j, name in enumerate(base_names):
            proba[name][s:e] = P[:, j]

        # ── Traditional ensembles (from base probs directly) ────────────────
        proba['MeanEnsemble'][s:e]         = P.mean(axis=1)
        proba['WeightedMeanEnsemble'][s:e] = (P.astype(np.float64) * w_wtd).sum(axis=1)

        # ── Nine meta-learners from the shared logit base-feature matrix ─────
        L = stacking._to_logit(P)
        for name, p in stacking._predict_all_meta_from_logit(L).items():
            proba[name][s:e] = p

    return valid_mask, proba


def project_all_replicates(stack, profile, transform, H, W, n_reps, thr_records):
    """
    BIOMOD2-style projection phase — runs AFTER all replicates have finished.

    Phase A: for each replicate, load its saved model bundle and project all 17
             models over the raster, writing the per-run CONTINUOUS suitability
             surface to maps/replicates/{Model}_Run{r}.tif.
    Phase B: per model, take the MEDIAN across runs → {Model}_Suitability_Continuous.tif,
             threshold at the median MaxTSS → {Model}_Suitability_Binary.tif, and
             the between-run SD → {Model}_Suitability_SD.tif (PA-stochasticity
             uncertainty).  Plus GPC epistemic entropy, Jenks zones, summary figure.
    """
    print("\n" + "=" * 60)
    print(f"[STEP 11] Projection Phase (BIOMOD2-style, N={n_reps} replicates)")
    print("=" * 60)

    rep_map_dir = OUT_DIR / 'maps' / 'replicates'
    bundle_dir  = OUT_DIR / 'models' / 'replicates'

    # ── Phase A: per-run projections ───────────────────────────────────────
    print(f"\n  Phase A — projecting all 17 models for each of {n_reps} runs...")
    valid_mask = None
    model_names = None
    for r in range(n_reps):
        bundle_path = bundle_dir / f'rep{r:02d}.pkl'
        if not bundle_path.exists():
            print(f"  [WARNING] missing bundle {bundle_path.name}; skipping run {r+1}")
            continue
        bundle = joblib.load(bundle_path)
        t0 = time.time()
        valid_mask, proba = _project_models(
            bundle['best_models'], bundle['stacking'], bundle['weighted_ensemble'],
            bundle['preprocessor'], stack, H, W, bundle['sel_idx']
        )
        model_names = list(proba.keys())
        for name, vec in proba.items():
            _write_geotiff(rep_map_dir / f'{name}_Run{r+1}.tif',
                           _vec_to_2d(vec, valid_mask, H, W), profile)
        print(f"    Run {r+1}/{n_reps}: {len(proba)} models projected "
              f"({time.time()-t0:.1f}s)")
        del bundle, proba

    if model_names is None:
        print("  [ERROR] no replicate bundles found — projection phase aborted.")
        return

    # ── Median MaxTSS threshold per model (across replicates) ──────────────
    median_thr = {}
    for name in model_names:
        vals = [d[name] for d in thr_records if name in d]
        median_thr[name] = float(np.median(vals)) if vals else 0.5

    # ── Phase B: median / binary / uncertainty per model ───────────────────
    print(f"\n  Phase B — median, binary & uncertainty across {n_reps} runs...")
    out_profile = profile.copy()
    out_profile.update(dtype='float32', count=1, nodata=np.nan, compress='lzw')
    keep = {}   # cache a few 2-D arrays needed for the summary figure
    for name in model_names:
        run_paths = [rep_map_dir / f'{name}_Run{r+1}.tif' for r in range(n_reps)
                     if (rep_map_dir / f'{name}_Run{r+1}.tif').exists()]
        runs = np.stack([_read_band(p) for p in run_paths], axis=0)   # (n_runs, H, W)
        median_2d = np.nanmedian(runs, axis=0).astype(np.float32)
        sd_2d     = np.nanstd(runs, axis=0).astype(np.float32)
        thr       = median_thr[name]
        bin_2d    = np.where(np.isnan(median_2d), np.nan,
                             (median_2d >= thr).astype(np.float32))
        _write_geotiff(OUT_DIR / 'maps' / f'{name}_Suitability_Continuous.tif',
                       median_2d, profile)
        _write_geotiff(OUT_DIR / 'maps' / f'{name}_Suitability_Binary.tif',
                       bin_2d, profile)
        _write_geotiff(OUT_DIR / 'maps' / f'{name}_Suitability_SD.tif',
                       sd_2d, profile)
        print(f"    {name:22s}: median + binary (thr={thr:.3f}) + SD written")
        if name in ('GPC_Meta', 'MeanEnsemble', 'WeightedMeanEnsemble'):
            keep[name] = {'cont': median_2d, 'bin': bin_2d, 'sd': sd_2d, 'thr': thr}
        del runs

    # ── GPC epistemic entropy (Shannon entropy of the median GPC surface) ──
    gpc_cont = keep['GPC_Meta']['cont']
    p_clip   = np.clip(gpc_cont, 1e-7, 1 - 1e-7)
    entropy_2d = np.where(np.isnan(gpc_cont), np.nan,
                          -(p_clip * np.log(p_clip) +
                            (1 - p_clip) * np.log(1 - p_clip))).astype(np.float32)
    _write_geotiff(OUT_DIR / 'maps' / 'GPC_Meta_Suitability_Entropy.tif',
                   entropy_2d, profile)

    # ── Jenks natural-breaks suitability zones on the median GPC surface ───
    print("  Computing Jenks natural breaks (5 zones) on median GPC map...")
    valid_probs  = gpc_cont[~np.isnan(gpc_cont)].ravel()
    sample_probs = (np.random.choice(valid_probs, size=min(500_000, len(valid_probs)),
                                     replace=False)
                    if len(valid_probs) > 500_000 else valid_probs)
    breaks  = jenkspy.jenks_breaks(sample_probs.tolist(), n_classes=N_ZONES)
    zone_2d = np.full((H, W), np.nan, dtype=np.float32)
    for z, (lo, hi) in enumerate(zip(breaks[:-1], breaks[1:]), start=1):
        mask = ~np.isnan(gpc_cont) & (gpc_cont >= lo) & (gpc_cont <= hi)
        zone_2d[mask] = z
    zone_labels = {1: 'Very Low', 2: 'Low', 3: 'Moderate', 4: 'High', 5: 'Very High'}
    n_valid = int(valid_mask.sum())
    print(f"  Jenks breaks (median GPC): {[round(b, 3) for b in breaks]}")
    for z, lbl in zone_labels.items():
        pct = float(np.nansum(zone_2d == z)) / n_valid * 100
        print(f"    Zone {z} ({lbl}): {pct:.1f}%")
    _write_geotiff(OUT_DIR / 'maps' / 'Suitability_Zones.tif', zone_2d, profile)

    print(f"\n  Saved per-model maps to maps/  ({len(model_names)} models × "
          f"continuous + binary + SD)")
    print(f"  Saved {len(model_names)} × {n_reps} per-run continuous maps to maps/replicates/")

    # ── Summary figure (median surfaces) ───────────────────────────────────
    _plot_projection_summary(keep, entropy_2d, zone_2d, zone_labels, n_reps)


def _read_band(path):
    """Read band 1 of a GeoTIFF as float32 (nodata → NaN)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
    return arr


def _plot_projection_summary(keep, entropy_2d, zone_2d, zone_labels, n_reps):
    """3×3 summary of the headline median surfaces, binaries and uncertainty."""
    gpc, men, wtd = keep['GPC_Meta'], keep['MeanEnsemble'], keep['WeightedMeanEnsemble']
    cmap_bin = matplotlib.colors.ListedColormap(['#d62728', '#2ca02c'])
    absent_patch  = mpatches.Patch(color='#d62728', label='Absent')
    present_patch = mpatches.Patch(color='#2ca02c', label='Present')

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    for ax, d, title in [
        (axes[0], gpc, f'GPC Meta-Ensemble\n(median continuous, N={n_reps})'),
        (axes[1], men, f'Unweighted Mean Ensemble\n(median continuous, N={n_reps})'),
        (axes[2], wtd, f'TSS-Weighted Mean Ensemble\n(median continuous, N={n_reps})'),
    ]:
        im = ax.imshow(d['cont'], cmap='RdYlGn', origin='upper', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.04)
        ax.set_title(title, fontsize=11)

    for ax, d, title in [
        (axes[3], gpc, f"GPC Binary\n(median MaxTSS thr={gpc['thr']:.3f})"),
        (axes[4], men, f"Unweighted Mean Binary\n(median MaxTSS thr={men['thr']:.3f})"),
        (axes[5], wtd, f"Weighted Mean Binary\n(median MaxTSS thr={wtd['thr']:.3f})"),
    ]:
        ax.imshow(d['bin'], cmap=cmap_bin, origin='upper', vmin=0, vmax=1)
        ax.legend(handles=[absent_patch, present_patch], loc='lower right', fontsize=8)
        ax.set_title(title, fontsize=11)

    zone_colors = ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']
    cmap_z = matplotlib.colors.ListedColormap(zone_colors)
    im6 = axes[6].imshow(zone_2d, cmap=cmap_z, origin='upper', vmin=0.5, vmax=5.5)
    cbar_z = plt.colorbar(im6, ax=axes[6], ticks=[1, 2, 3, 4, 5], fraction=0.04)
    cbar_z.set_ticklabels(list(zone_labels.values()), fontsize=7)
    axes[6].set_title('Suitability Zones\n(Jenks 5-class, median GPC)', fontsize=11)

    im7 = axes[7].imshow(entropy_2d, cmap='plasma', origin='upper')
    plt.colorbar(im7, ax=axes[7], fraction=0.04)
    axes[7].set_title(f'GPC Epistemic Uncertainty\n(Shannon entropy of median, N={n_reps})',
                      fontsize=11)

    im8 = axes[8].imshow(gpc['sd'], cmap='YlOrRd', origin='upper')
    plt.colorbar(im8, ax=axes[8], fraction=0.04)
    axes[8].set_title(f'GPC PA-Stochasticity Uncertainty\n(between-run SD, N={n_reps})',
                      fontsize=11)

    for ax in axes:
        ax.axis('off')
    plt.suptitle(
        f'Bambusa oldhamii — Median Habitat Suitability Maps (N={n_reps} PA replicates)',
        fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figures' / 'suitability_maps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/suitability_maps.png")


# =============================================================================
# 12. STATISTICAL COMPARISON UTILITIES
# =============================================================================

def delong_roc_test(y_true, y_score_a, y_score_b):
    """
    DeLong et al. (1988) paired AUC test using structural components.
    Returns (auc_a, auc_b, z_stat, p_value).
    """
    pos = y_true == 1
    neg = y_true == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m < 2 or n < 2:
        return float('nan'), float('nan'), float('nan'), float('nan')

    def _V10(scores):
        ps, ns = scores[pos], scores[neg]
        return np.array(
            [(ns < s).sum() + 0.5 * (ns == s).sum() for s in ps], float
        ) / n

    def _V01(scores):
        ps, ns = scores[pos], scores[neg]
        return np.array(
            [(ps > s).sum() + 0.5 * (ps == s).sum() for s in ns], float
        ) / m

    Va = _V10(y_score_a)
    Vb = _V10(y_score_b)
    Wa = _V01(y_score_a)
    Wb = _V01(y_score_b)
    auc_a = float(Va.mean())
    auc_b = float(Vb.mean())

    S10 = np.cov(np.vstack([Va, Vb])) / m
    S01 = np.cov(np.vstack([Wa, Wb])) / n
    S   = S10 + S01
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_diff <= 1e-15:
        return auc_a, auc_b, float('nan'), float('nan')

    z = (auc_a - auc_b) / float(np.sqrt(var_diff))
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return auc_a, auc_b, float(z), float(p)


def bootstrap_metrics(y_true, proba_dict, thresholds, n_bootstrap=1000, seed=42):
    """
    Resample test set n_bootstrap times → 95% CI for AUC, TSS, Brier per model.
    Returns dict: {model_name: {AUC_CI_lo, AUC_CI_hi, Brier_CI_lo, ..., TSS_CI_lo, ...}}
    """
    rng = np.random.RandomState(seed)
    n   = len(y_true)
    result = {}
    for name, proba in proba_dict.items():
        aucs, briers, tsss = [], [], []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            yt, yp = y_true[idx], proba[idx]
            if yt.sum() == 0 or yt.sum() == len(yt):
                continue
            aucs.append(roc_auc_score(yt, yp))
            briers.append(brier_score_loss(yt, yp))
            # max-TSS over the resample's ROC curve — matches compute_metrics so
            # the CI and the reported point estimate measure the SAME quantity
            # (previously the CI used a fixed threshold → a different, lower TSS).
            fpr_c, tpr_c, _ = roc_curve(yt, yp)
            tsss.append(float(np.max(tpr_c - fpr_c)))

        def _ci(vals):
            if len(vals) < 10:
                return float('nan'), float('nan')
            return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

        result[name] = {
            'AUC_CI_lo':   _ci(aucs)[0],   'AUC_CI_hi':   _ci(aucs)[1],
            'Brier_CI_lo': _ci(briers)[0], 'Brier_CI_hi': _ci(briers)[1],
            'TSS_CI_lo':   _ci(tsss)[0],   'TSS_CI_hi':   _ci(tsss)[1],
        }
    return result


def _run_delong_tests(y_test, all_probs, rep):
    """Run DeLong pairwise tests for all model pairs; return list of row dicts."""
    model_names = list(all_probs.keys())
    rows = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a, b = model_names[i], model_names[j]
            auc_a, auc_b, z, p = delong_roc_test(y_test, all_probs[a], all_probs[b])
            rows.append({
                'replicate': rep,
                'model_A':   a,
                'model_B':   b,
                'AUC_A':     round(auc_a, 4),
                'AUC_B':     round(auc_b, 4),
                'delta_AUC': round(auc_a - auc_b, 4),
                'z_stat':    round(z, 4) if not np.isnan(z) else float('nan'),
                'p_value':   round(p, 4) if not np.isnan(p) else float('nan'),
                'significant_0.05': (p < 0.05) if not np.isnan(p) else False,
            })
    return rows


def _run_wilcoxon_tests(replicate_records):
    """
    Wilcoxon signed-rank test across N_REPLICATES for every model pair.
    Returns list of row dicts (one per metric × model pair).
    """
    rep_df = pd.DataFrame(replicate_records)
    metric_cols = ['AUC', 'TSS', 'Brier']
    rows = []
    for metric in metric_cols:
        # Discover model names from columns ending with _{metric}
        metric_suffix = f'_{metric}'
        cols_for_metric = [c for c in rep_df.columns if c.endswith(metric_suffix)]
        models = [c[: -len(metric_suffix)] for c in cols_for_metric]
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                vals_a = rep_df[f"{a}_{metric}"].dropna().values
                vals_b = rep_df[f"{b}_{metric}"].dropna().values
                n = min(len(vals_a), len(vals_b))
                if n < 3:
                    stat, p = float('nan'), float('nan')
                else:
                    try:
                        res = stats.wilcoxon(vals_a[:n], vals_b[:n], alternative='two-sided')
                        stat, p = float(res.statistic), float(res.pvalue)
                    except Exception:
                        stat, p = float('nan'), float('nan')
                rows.append({
                    'metric':           metric,
                    'model_A':          a,
                    'model_B':          b,
                    'n_replicates':     n,
                    'mean_A':           round(float(vals_a.mean()), 4) if len(vals_a) else float('nan'),
                    'mean_B':           round(float(vals_b.mean()), 4) if len(vals_b) else float('nan'),
                    'delta_mean':       round(float(vals_a.mean() - vals_b.mean()), 4) if len(vals_a) and len(vals_b) else float('nan'),
                    'W_stat':           round(stat, 4) if not np.isnan(stat) else float('nan'),
                    'p_value':          round(p, 4) if not np.isnan(p) else float('nan'),
                    'significant_0.05': (p < 0.05) if not np.isnan(p) else False,
                })
    return rows


def _export_statistics(replicate_records, delong_records, boot_records, wilcoxon_rows):
    """Write all statistical outputs to CSV."""
    # Per-replicate metrics (with bootstrap CIs merged in)
    rep_df = pd.DataFrame(replicate_records)
    boot_rows = []
    for row in boot_records:
        flat = {'replicate': row['replicate']}
        for model, cis in row['boot_cis'].items():
            for k, v in cis.items():
                flat[f"{model}_{k}"] = v
        boot_rows.append(flat)
    if boot_rows:
        boot_df = pd.DataFrame(boot_rows)
        rep_df  = rep_df.merge(boot_df, on='replicate', how='left')
    rep_df.to_csv(OUT_DIR / 'replicate_metrics.csv', index=False)
    print("  Saved: replicate_metrics.csv")

    # Summary: mean ± SD across replicates
    numeric_cols = rep_df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ('replicate', 'pa_seed')]
    summary = pd.DataFrame({
        'mean': rep_df[numeric_cols].mean(),
        'std':  rep_df[numeric_cols].std(ddof=1),
        'min':  rep_df[numeric_cols].min(),
        'max':  rep_df[numeric_cols].max(),
    })
    summary.to_csv(OUT_DIR / 'summary_metrics.csv')
    print("  Saved: summary_metrics.csv")

    # DeLong tests per replicate
    pd.DataFrame(delong_records).to_csv(OUT_DIR / 'delong_tests.csv', index=False)
    print("  Saved: delong_tests.csv")

    # Wilcoxon across replicates
    pd.DataFrame(wilcoxon_rows).to_csv(OUT_DIR / 'wilcoxon_tests.csv', index=False)
    print("  Saved: wilcoxon_tests.csv")

    return rep_df, summary


def _plot_replicate_boxplots(replicate_records, model_order=None):
    """3-panel box plot: AUC / TSS / Brier across N_REPLICATES for all models."""
    rep_df = pd.DataFrame(replicate_records)
    if model_order is None:
        model_order = (['LR', 'SVM', 'ET', 'XGB', 'MLP', 'NB',
                        'MeanEnsemble', 'WeightedMeanEnsemble'] + META_ORDER)

    metrics = [('AUC', 'AUC', True), ('TSS', 'TSS', True), ('Brier', 'Brier Score', False)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for ax, (suffix, label, higher_better) in zip(axes, metrics):
        cols = [f"{m}_{suffix}" for m in model_order if f"{m}_{suffix}" in rep_df.columns]
        present_models = [c.replace(f"_{suffix}", '') for c in cols]
        data = [rep_df[c].dropna().values for c in cols]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=2))
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(present_models)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(present_models) + 1))
        ax.set_xticklabels(present_models, rotation=45, ha='right', fontsize=9)
        ax.set_title(f"{label} across {len(rep_df)} replicates", fontsize=11)
        ax.set_ylabel(label)
        direction = '↑ better' if higher_better else '↓ better'
        ax.set_xlabel(direction, fontsize=9, color='grey')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.suptitle('Model performance across PA replicates\n'
                 '(Bambusa oldhamii SDM)', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figures' / 'replicate_boxplots.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/replicate_boxplots.png")


# Tabulated Nemenyi critical values q_α for α=0.05 (Demšar 2006, Table 5)
_NEMENYI_Q_05 = {
    2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850,
    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
    12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
    16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
}


def _run_friedman_analysis(replicate_records, alpha=0.05):
    """
    Friedman test + Nemenyi post-hoc for AUC, TSS, Brier across N replicates.
    Saves friedman_tests.csv, nemenyi_tests.csv, and one CD diagram per metric.
    Returns (friedman_rows, nemenyi_rows).
    """
    rep_df = pd.DataFrame(replicate_records)
    N = len(rep_df)
    metrics = [('AUC', True), ('TSS', True), ('Brier', False)]

    friedman_rows = []
    nemenyi_rows  = []

    for metric, higher_better in metrics:
        suffix = f'_{metric}'
        cols   = [c for c in rep_df.columns if c.endswith(suffix)]
        models = [c[: -len(suffix)] for c in cols]
        k      = len(models)

        if N < 3 or k < 3:
            print(f"  Friedman ({metric}): skipped (N={N}, k={k} — need N≥3, k≥3)")
            continue

        data = rep_df[cols].values.astype(float)  # shape (N, k)

        # Rank within each replicate; rank 1 = best
        ranks = np.zeros_like(data)
        for i in range(N):
            ranks[i] = (stats.rankdata(-data[i]) if higher_better
                        else stats.rankdata(data[i]))

        avg_ranks = ranks.mean(axis=0)
        rank_map  = dict(zip(models, avg_ranks))

        # Friedman test
        fstat, fp = stats.friedmanchisquare(*[ranks[:, j] for j in range(k)])

        # Nemenyi CD (Demšar 2006 formula)
        q_alpha = _NEMENYI_Q_05.get(k, 3.5)
        cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))

        p_str = f'{fp:.4f}' if fp >= 0.001 else '<0.001'
        print(f"  Friedman ({metric}): chi2={fstat:.3f}, p={p_str}, "
              f"CD={cd:.3f}, {'[sig]' if fp < alpha else '[n.s.]'}")

        friedman_rows.append({
            'metric':           metric,
            'n_replicates':     N,
            'n_models':         k,
            'chi2_stat':        round(float(fstat), 4),
            'p_value':          round(float(fp), 4),
            'significant_0.05': fp < alpha,
            'CD_nemenyi':       round(cd, 4),
            'alpha':            alpha,
        })

        # Nemenyi pairwise
        for i in range(k):
            for j in range(i + 1, k):
                diff = abs(avg_ranks[i] - avg_ranks[j])
                nemenyi_rows.append({
                    'metric':           metric,
                    'model_A':          models[i],
                    'model_B':          models[j],
                    'avg_rank_A':       round(avg_ranks[i], 3),
                    'avg_rank_B':       round(avg_ranks[j], 3),
                    'rank_diff':        round(diff, 3),
                    'CD':               round(cd, 3),
                    'significant_0.05': diff > cd,
                })

        # CD diagram for this metric
        _plot_cd_diagram(rank_map, cd, float(fstat), float(fp),
                         metric, N, alpha)

    pd.DataFrame(friedman_rows).to_csv(OUT_DIR / 'friedman_tests.csv', index=False)
    pd.DataFrame(nemenyi_rows).to_csv(OUT_DIR / 'nemenyi_tests.csv', index=False)
    print("  Saved: friedman_tests.csv, nemenyi_tests.csv")
    return friedman_rows, nemenyi_rows


def _plot_cd_diagram(rank_map, cd, fstat, fp, metric, N, alpha=0.05):
    """
    Critical Difference diagram (Demšar 2006).
    rank_map: {model_name: avg_rank}  (rank 1 = best).
    Non-significant groups shown as thick horizontal bars below the rank axis.
    """
    # Sort by average rank (best=1 on left)
    sorted_items  = sorted(rank_map.items(), key=lambda x: x[1])
    sorted_models = [m for m, _ in sorted_items]
    sorted_ranks  = np.array([r for _, r in sorted_items])
    k             = len(sorted_models)

    # Maximal non-significant groups: contiguous spans where span ≤ CD
    # A group (i,j) is maximal when ranks[j]-ranks[i]≤CD and it can't extend left
    groups = []
    for i in range(k):
        j = i
        while j + 1 < k and sorted_ranks[j + 1] - sorted_ranks[i] <= cd:
            j += 1
        if j > i:
            if i == 0 or sorted_ranks[j] - sorted_ranks[i - 1] > cd:
                groups.append((i, j))

    # ── Figure layout ─────────────────────────────────────────────────────
    # Each model label gets its OWN vertical row, fanning out to the left
    # (best ranks) and right (worst ranks) — the canonical Demšar style.
    # This is overlap-proof regardless of how tightly ranks cluster.
    mid      = (k + 1) / 2.0
    left     = [(m, r) for m, r in zip(sorted_models, sorted_ranks) if r <= mid]
    right    = [(m, r) for m, r in zip(sorted_models, sorted_ranks) if r >  mid]
    per_side = max(len(left), len(right))

    axis_y     = 0.0
    margin     = 0.65
    row_step   = 0.055           # vertical gap between adjacent label rows
    label_base = 0.13            # first (lowest) label row above the axis
    top_y      = label_base + max(0, per_side - 1) * row_step + 0.05
    bar_base   = -0.05           # non-sig group bars hang below the axis
    bar_gap    = 0.040
    bottom_y   = bar_base - max(0, len(groups) - 1) * bar_gap - 0.06

    # Height scales with whichever stack (labels above / bars below) is taller.
    fig_h = max(4.5, 1.0 + per_side * 0.42 + len(groups) * 0.22)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.set_axis_off()
    ax.set_xlim(1 - margin, k + margin)
    ax.set_ylim(bottom_y, top_y)

    # Main horizontal rank axis
    ax.hlines(axis_y, 1, k, colors='black', linewidth=2.0, zorder=3)

    # Tick marks + rank numbers just above axis
    for r in range(1, k + 1):
        ax.vlines(r, axis_y - 0.012, axis_y + 0.012,
                  colors='black', linewidth=1.5, zorder=4)
        ax.text(r, axis_y + 0.020, str(r),
                ha='center', va='bottom', fontsize=8, color='#333333')

    # Helper: draw a model label on one side at its own row level.
    def _draw_label(name, x, row_y, side):
        if side == 'left':
            text_x = 1 - margin
            ax.plot([x, x],      [axis_y, row_y], color='#777777', lw=0.9, zorder=2)
            ax.plot([x, text_x], [row_y,  row_y], color='#777777', lw=0.9, zorder=2)
            ax.text(text_x - 0.05, row_y, name,
                    ha='right', va='center', fontsize=9.5)
        else:
            text_x = k + margin
            ax.plot([x, x],      [axis_y, row_y], color='#777777', lw=0.9, zorder=2)
            ax.plot([x, text_x], [row_y,  row_y], color='#777777', lw=0.9, zorder=2)
            ax.text(text_x + 0.05, row_y, name,
                    ha='left', va='center', fontsize=9.5)

    # Left side: best rank (smallest x) → highest row, so connectors fan
    # outward without crossing. Right side mirrors (worst rank → highest row).
    for t, (name, rank) in enumerate(left):                    # ascending rank
        row_y = label_base + (len(left) - 1 - t) * row_step
        _draw_label(name, rank, row_y, 'left')
    for t, (name, rank) in enumerate(reversed(right)):         # descending rank
        row_y = label_base + (len(right) - 1 - t) * row_step
        _draw_label(name, rank, row_y, 'right')

    # Non-significant group bars (thick, just below axis)
    for b_idx, (i, j) in enumerate(groups):
        bar_y = bar_base - b_idx * bar_gap
        lc = ax.hlines(bar_y, sorted_ranks[i], sorted_ranks[j],
                       colors='black', linewidth=5.5, zorder=5)
        lc.set_capstyle('round')

    # CD marker bracket (above axis, top-right)
    cd_y     = top_y - 0.03
    cd_right = float(k)
    cd_left  = float(k) - cd
    ax.annotate('', xy=(cd_right, cd_y), xytext=(cd_left, cd_y),
                arrowprops=dict(arrowstyle='<->', color='black',
                                lw=1.5, mutation_scale=12))
    ax.text((cd_left + cd_right) / 2.0, cd_y + 0.012,
            f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=9)

    # Axis direction hints (just below axis)
    ax.text(1.0, axis_y - 0.020, '← best',  ha='left',  va='top',
            fontsize=8, color='grey', style='italic')
    ax.text(k,   axis_y - 0.020, 'worst →', ha='right', va='top',
            fontsize=8, color='grey', style='italic')

    # Title
    p_str  = f'{fp:.4f}' if fp >= 0.001 else '< 0.001'
    sig_lbl = '✓ significant' if fp < alpha else '✗ not significant'
    ax.set_title(
        f'Critical Difference Diagram — {metric}   (N = {N} replicates)\n'
        f'Friedman χ² = {fstat:.2f},  p = {p_str}  [{sig_lbl}] | '
        f'Nemenyi α = {alpha}\n'
        f'Rank 1 = best   |   connected bars = not significantly different',
        fontsize=10, pad=10, loc='left',
    )

    plt.tight_layout()
    fname = OUT_DIR / 'figures' / f'cd_diagram_{metric.lower()}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/cd_diagram_{metric.lower()}.png")


# =============================================================================
# 13. REPLICATE RUNNER
# =============================================================================

def _run_replicate(rep, pa_seed, stack, profile, transform, H, W,
                   save_outputs=False):
    """
    Run a full single SDM pipeline replicate.
    pa_seed controls: PA generation, train/test split, spatial CV, tuning, GPC.
    save_outputs=True → also save the named rep-0 model pkls (back-compat).

    Every replicate's fitted models are persisted to
    models/replicates/rep{NN}.pkl so the BIOMOD2-style projection phase can
    project all 17 models AFTER the whole replicate loop has finished.
    No raster projection happens inside the loop.
    Returns: (metrics_df, all_probs, thresholds, y_test, meta_contrib)
    """
    print(f"\n{'='*60}")
    print(f"  REPLICATE {rep+1}/{N_REPLICATES}  (pa_seed={pa_seed})")
    print(f"{'='*60}\n")

    # ── Steps 2-3 ─────────────────────────────────────────────────────────
    full_df = prepare_dataset(stack, transform, H, W, seed=pa_seed)
    coords  = full_df[['lon', 'lat']].values
    X_raw   = full_df[PRED_NAMES].values
    y       = full_df['presence'].values
    spw     = scale_pos_weight(full_df)

    # ── Step 4: Train/test split ───────────────────────────────────────────
    print("[STEP 4] Train/Test Split (70/30 stratified)")
    (X_tr_raw, X_te_raw,
     y_train,  y_test,
     coords_tr, _) = train_test_split(
        X_raw, y, coords,
        test_size=0.30, stratify=y, random_state=pa_seed
    )
    print(f"  Train: {len(y_train):,} | Test: {len(y_test):,}")
    print(f"  Train presence: {y_train.sum():,} | Test presence: {y_test.sum():,}\n")

    # ── Step 4b-4c ────────────────────────────────────────────────────────
    print("[STEP 4b] Feature Preprocessing")
    X_train_sc, X_test_sc, preprocessor, proc_names = build_preprocessor(
        list(PRED_NAMES), X_tr_raw, X_te_raw
    )

    print("[STEP 4c] Fitting Spatial Block CV on training set")
    spatial_cv = SpatialBlockCV(n_splits=N_FOLDS, block_size_deg=BLOCK_SIZE_DEG,
                                random_state=pa_seed)
    spatial_cv.precompute(coords_tr)
    print()

    # ── Step 5: Ensemble RFECV ─────────────────────────────────────────────
    vote_df, selected_features = ensemble_rfecv(
        X_train_sc, y_train, coords_tr, spatial_cv, spw, proc_names
    )
    if not selected_features:
        print("  WARNING: no features passed threshold; using all features.")
        selected_features = list(proc_names)

    sel_idx     = [proc_names.index(f) for f in selected_features]
    X_train_sel = X_train_sc[:, sel_idx]
    X_test_sel  = X_test_sc[:, sel_idx]

    raw_sel_names = set()
    for feat in selected_features:
        if feat in PRED_NAMES:
            raw_sel_names.add(feat)
        else:
            parent = feat.rsplit('_', 1)[0]
            if parent in CATEGORICAL_FEATURES:
                raw_sel_names.add(parent)
    raw_nan_idx = sorted([list(PRED_NAMES).index(f) for f in raw_sel_names])

    # ── Step 6: Hyperparameter tuning ─────────────────────────────────────
    best_models, best_params, best_scores = tune_base_models(
        X_train_sel, y_train, spatial_cv, spw, seed=pa_seed
    )

    # ── Step 7: Stacking ───────────────────────────────────────────────────
    stacking = build_stacking_classifier(best_models, spatial_cv, seed=pa_seed)
    print("\n  Fitting custom GPC stacking (OOF collection + GPC training)...")
    t0 = time.time()
    stacking.fit(X_train_sel, y_train, coords=coords_tr)
    print(f"  Total stacking pipeline: {time.time()-t0:.1f}s")

    # ── Step 8: Re-fit base models ─────────────────────────────────────────
    print("\n[STEP 8] Re-fitting base models on full training set...")
    for name, model in best_models.items():
        model.fit(X_train_sel, y_train)
        print(f"  {name}: re-fitted")

    mean_ensemble = MeanEnsemble(list(best_models.items()))

    # ── Save models (rep 0 only) ───────────────────────────────────────────
    if save_outputs:
        print("\n  Saving models (rep 0)...")
        joblib.dump(best_models,   OUT_DIR / 'models' / 'base_models.pkl')
        joblib.dump(stacking,      OUT_DIR / 'models' / 'stacking_gpc.pkl')
        joblib.dump(mean_ensemble, OUT_DIR / 'models' / 'mean_ensemble.pkl')
        joblib.dump(preprocessor,  OUT_DIR / 'models' / 'preprocessor.pkl')
        joblib.dump({'selected_features': selected_features,
                     'pred_names':        list(PRED_NAMES),
                     'proc_names':        proc_names,
                     'sel_idx':           sel_idx,
                     'raw_nan_idx':       raw_nan_idx},
                    OUT_DIR / 'models' / 'feature_info.pkl')

    # ── Step 9: Evaluation ─────────────────────────────────────────────────
    metrics_df, all_probs, thresholds, weighted_ensemble = evaluate_all_models(
        best_models, stacking, mean_ensemble, X_test_sel, y_test
    )

    if save_outputs:
        joblib.dump(weighted_ensemble, OUT_DIR / 'models' / 'weighted_ensemble.pkl')

    # ── Persist this replicate's full model bundle for the projection phase ─
    # One pickle per replicate (shared object refs preserved within the dump,
    # so base models are not duplicated across stacking/ensemble). The post-loop
    # projection phase reloads these to project all 17 models — BIOMOD2-style.
    print("\n  Saving replicate model bundle for projection...")
    joblib.dump({
        'best_models':       best_models,
        'stacking':          stacking,
        'weighted_ensemble': weighted_ensemble,
        'preprocessor':      preprocessor,
        'sel_idx':           sel_idx,
        'thresholds':        thresholds,
        'pa_seed':           pa_seed,
    }, OUT_DIR / 'models' / 'replicates' / f'rep{rep:02d}.pkl')

    # ── Step 10: Meta-feature contributions (every rep — averaged in main) ──
    # Permutation importance of each base-model meta-feature on each meta-
    # learner's AUC; model-agnostic replacement for SHAP.
    print("\n[STEP 10] Meta-feature contribution analysis (permutation importance)...")
    meta_contrib = {}
    try:
        meta_contrib = meta_feature_contributions(
            stacking, X_test_sel, y_test, list(best_models.keys()), seed=42
        )
    except Exception as e:
        print(f"\n  [WARNING] meta-contribution analysis failed ({e}); skipping this replicate.")

    # Raster projection is deferred to the post-loop projection phase
    # (project_all_replicates) so all 17 models are projected only after every
    # replicate has finished — mirroring BIOMOD2's modelling → projection split.
    return metrics_df, all_probs, thresholds, y_test, meta_contrib


# =============================================================================
# 14. MAIN EXECUTION  (replicate-aware)
# =============================================================================
def main():
    t_start = time.time()
    print("\n" + "=" * 60)
    print("  BAMBOO SDM — Ensemble Meta-Learning + RFE")
    print(f"  Species: Bambusa oldhamii  |  N_REPLICATES={N_REPLICATES}")
    print("=" * 60 + "\n")

    # ── Step 1: Load raster stack once (shared across all replicates) ──────
    stack, profile, transform, H, W = load_raster_stack()

    # ── Replicate loop ─────────────────────────────────────────────────────
    replicate_records = []   # one row per replicate, all model metrics
    delong_records    = []   # one row per model pair per replicate
    boot_records      = []   # bootstrap CIs per replicate
    thr_records       = []   # per-replicate MaxTSS thresholds (median → binary maps)
    roc_records       = []   # per-replicate (y_test, all_probs) for median ROC + confusion
    metrics_records   = []   # per-replicate full metrics_df for the median heatmap
    contrib_acc       = {}   # {meta_model: {base_feature: sum_auc_drop}} across reps
    contrib_n         = 0    # replicates with a valid contribution result

    for rep in range(N_REPLICATES):
        pa_seed = RANDOM_STATE + rep   # reproducible but distinct PA draws
        t_rep   = time.time()

        metrics_df, all_probs, thresholds, y_test, meta_contrib = \
            _run_replicate(rep, pa_seed, stack, profile, transform, H, W,
                           save_outputs=(rep == 0))

        # Accumulate meta-feature contributions across replicates
        if meta_contrib:
            contrib_n += 1
            for meta_model, feat_dict in meta_contrib.items():
                contrib_acc.setdefault(meta_model, {})
                for feat, val in feat_dict.items():
                    contrib_acc[meta_model][feat] = \
                        contrib_acc[meta_model].get(feat, 0.0) + val

        # Keep this replicate's thresholds for the post-loop binary maps
        thr_records.append({k: float(v) for k, v in thresholds.items()})

        # Keep raw probs / metrics for the post-loop MEDIAN evaluation figures
        # (ROC, confusion matrices, metrics heatmap) — these must summarise ALL
        # replicates, not just the last one that would otherwise overwrite them.
        roc_records.append((np.asarray(y_test),
                            {k: np.asarray(v) for k, v in all_probs.items()}))
        metrics_records.append(metrics_df.copy())

        # Flatten metrics into one row
        row = {'replicate': rep, 'pa_seed': pa_seed}
        for model_name, mrow in metrics_df.iterrows():
            for metric in ['AUC', 'TSS', 'Brier']:
                row[f"{model_name}_{metric}"] = round(float(mrow[metric]), 4)
        replicate_records.append(row)

        # DeLong pairwise tests for this replicate
        delong_records.extend(_run_delong_tests(y_test, all_probs, rep))

        # Bootstrap CIs for this replicate
        print(f"\n  Computing bootstrap CIs (n=1000) for replicate {rep+1}...")
        boot_cis = bootstrap_metrics(y_test, all_probs, thresholds,
                                     n_bootstrap=1000, seed=pa_seed)
        boot_records.append({'replicate': rep, 'boot_cis': boot_cis})

        print(f"\n  Replicate {rep+1} complete ({(time.time()-t_rep)/60:.1f} min)")

    # ── Averaged meta-feature contributions (post-loop) ─────────────────────
    if contrib_acc and contrib_n:
        avg_contrib = {
            model: {feat: val / contrib_n for feat, val in fd.items()}
            for model, fd in contrib_acc.items()
        }
        base_names = list(next(iter(avg_contrib.values())).keys())
        _save_meta_contributions(avg_contrib, base_names)

    # ── Projection phase (post-loop, BIOMOD2-style) ────────────────────────
    # Reloads each replicate's saved model bundle, projects all 17 models, writes
    # per-run continuous maps, then medians them into the final continuous/binary/
    # uncertainty surfaces.  Runs only after every replicate has finished.
    project_all_replicates(stack, profile, transform, H, W, N_REPLICATES, thr_records)

    # ── Statistical comparisons across replicates ──────────────────────────
    print("\n" + "=" * 60)
    print("  POST-REPLICATE STATISTICS")
    print("=" * 60)

    print("\n  Running Wilcoxon signed-rank tests across replicates...")
    wilcoxon_rows = _run_wilcoxon_tests(replicate_records)

    rep_df, summary = _export_statistics(
        replicate_records, delong_records, boot_records, wilcoxon_rows
    )
    _plot_replicate_boxplots(replicate_records)

    print("\n  Running Friedman + Nemenyi post-hoc + CD diagrams...")
    _run_friedman_analysis(replicate_records)

    # ── Median evaluation figures (post-loop, across all replicates) ───────
    print("\n  Generating median ROC / confusion / metrics / correlation figures...")
    _plot_roc_curves(roc_records)
    _plot_confusion_matrices([(yt, ap, thr)
                              for (yt, ap), thr in zip(roc_records, thr_records)])
    _plot_metrics_heatmap(metrics_records)
    _plot_base_model_correlation(roc_records)

    # ── Summary print ──────────────────────────────────────────────────────
    elapsed = (time.time() - t_start) / 60
    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE  ({N_REPLICATES} replicates, {elapsed:.1f} min total)")
    print("=" * 60)

    hierarchy = ['MeanEnsemble', 'WeightedMeanEnsemble'] + META_ORDER
    print("\n  Summary (mean ± SD across replicates):")
    for model in hierarchy:
        for metric in ['AUC', 'TSS', 'Brier']:
            col = f"{model}_{metric}"
            if col in summary.index:
                m, s = summary.loc[col, 'mean'], summary.loc[col, 'std']
                print(f"    {model:22s} {metric}: {m:.4f} ± {s:.4f}")

    print("\n  Key outputs in 'outputs/' folder:")
    print("    replicate_metrics.csv    — per-replicate metrics + bootstrap CIs")
    print("    summary_metrics.csv      — mean ± SD across replicates")
    print("    delong_tests.csv         — pairwise DeLong AUC tests per replicate")
    print("    wilcoxon_tests.csv       — Wilcoxon signed-rank across replicates")
    print("    friedman_tests.csv       — Friedman chi2 per metric")
    print("    nemenyi_tests.csv        — Nemenyi pairwise post-hoc per metric")
    print("    meta_feature_contributions.csv — base-model → meta-learner contributions")
    print("    evaluation_metrics.csv   — MEDIAN of each metric per model across replicates")
    print("    figures/replicate_boxplots.png")
    print("    figures/cd_diagram_auc.png | cd_diagram_tss.png | cd_diagram_brier.png")
    print("    figures/meta_feature_contributions.png — contribution heatmap")
    print("    figures/roc_curves.png            — MEDIAN ROC across replicates")
    print("    figures/confusion_matrices.png    — MEDIAN confusion matrices across replicates")
    print("    figures/metrics_heatmap.png       — MEDIAN metric heatmap across replicates")
    print("    figures/base_model_correlation.png — MEDIAN base-model correlation across replicates")
    print("    figures/suitability_maps.png — median map summary panel")
    print(f"    maps/{{Model}}_Suitability_Continuous.tif — median across {N_REPLICATES} runs (17 models)")
    print("    maps/{Model}_Suitability_Binary.tif     — median MaxTSS-thresholded binary")
    print("    maps/{Model}_Suitability_SD.tif         — between-run SD (PA-stochasticity uncertainty)")
    print("    maps/GPC_Meta_Suitability_Entropy.tif   — GPC epistemic (Shannon entropy)")
    print("    maps/Suitability_Zones.tif              — Jenks 5-class zones (median GPC)")
    print(f"    maps/replicates/{{Model}}_Run{{N}}.tif      — per-run continuous (17 × {N_REPLICATES})")
    print(f"    models/replicates/rep{{NN}}.pkl           — per-run model bundles ({N_REPLICATES})")
    print("    (Rep 0 only: models/*.pkl named bundles)")
    return rep_df, summary


if __name__ == '__main__':
    rep_df, summary = main()
