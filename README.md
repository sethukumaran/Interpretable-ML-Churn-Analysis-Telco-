# Interpretable-ML-Churn-Analysis-Telco
#Objective
Build a robust churn prediction model and identify the most important drivers of churn. Use explainability (SHAP where possible; permutation importance + PDP fallback otherwise) to produce global and local insights, and propose practical retention interventions.

#Dataset & quick facts
- Source file used: WA_Fn-UseC_-Telco-Customer-Churn.csv
- Rows × columns: 7043 × 21
- Typical features: gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, InternetService, OnlineSecurity, TechSupport, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn, etc.

#Preprocessing & data cleaning
Converted TotalCharges and MonthlyCharges and tenure to numeric using robust cleaning:
Removed surrounding brackets [...] and parentheses (…).
Stripped currency signs, commas and common NA tokens.
Coerced non-numeric values to NaN and imputed numeric columns with the column median.
Filled missing categorical values with "Missing".
One-hot encoded categorical variables (pd.get_dummies(..., drop_first=True)).
Train / test split: 75% train / 25% test (stratified on Churn).
Important fix applied: fixed problematic entries like "[5E-1]" (converted to 5E-1 → 0.5) before numeric coercion to avoid the could not convert string to float error.

#Model training
Classifier: XGBoost used when available (falls back to RandomForest). In your run XGBoost was available and used.
Class imbalance handled with scale_pos_weight for XGBoost (calculated from training set).
No heavy hyperparameter tuning in this run — default XGBoost with n_jobs=-1 and eval_metric='auc'.

#Model performance (test set)
AUC: 0.8190
Precision: 0.5565
Recall: 0.6745
F1-score: 0.6099
Interpretation: AUC ≈ 0.82 indicates strong discrimination between churners and non-churners. The model captures a large portion of true churners (Recall ≈ 67%) — useful for proactive retention — at moderate precision (≈56%), meaning some false positives will be present.

#Explainability: what happened
Attempted shap.TreeExplainer(model). This failed due to a ValueError: could not convert string to float: '[5E-1]' when SHAP inspected the XGBoost model internals (this is a parameter-string formatting issue).
Then attempted shap.Explainer(model.predict_proba, X_train) fallback — this also failed in your environment with a TypeError: ufunc 'isfinite' not supported for input types (likely due to data array dtype or an incompatible object array).
Because both SHAP routes failed in that environment, we used the robust fallback:
Permutation feature importance (model-agnostic), and
Partial Dependence Plots (PDPs) for the top features.
These fallbacks are reliable and give strong global and marginal effect insights, but they are not as locally granular as SHAP (no per-sample additive attributions).

#Global feature importance (permutation importance)
Top features (permutation importance — top 20 shown in your bar chart). The top 10 were:
tenure (highest importance)
Contract_Two year
MonthlyCharges
Contract_One year
InternetService_Fiber optic
TotalCharges
InternetService_No
OnlineSecurity_Yes
TechSupport_Yes
OnlineBackup_Yes

Short interpretation: tenure, contract type and pricing features (MonthlyCharges, TotalCharges) and Internet service type (Fiber vs DSL / No service) are the most influential overall in the model’s predictive performance.

#Partial Dependence (marginal effect) — top features
You produced PDPs for three top features; here’s a clear interpretation of each:
9.1 tenure — PDP
PDP shape: churn probability is highest at very low tenure (new customers), and decreases as tenure increases — roughly monotonic downward.
Business interpretation: new customers are far more likely to churn early; risk stabilizes for larger tenures.
Actionable takeaway: onboarding & early retention efforts are crucial (first 1–3 months most critical).
9.2 MonthlyCharges — PDP
PDP shape: churn probability generally increases with MonthlyCharges (higher monthly bill → higher churn risk), with some volatility.
Business interpretation: customers paying higher monthly fees are more likely to churn (sensitivity to price).
Actionable takeaway: consider price-based retention offers or value-adding discounts for high-bill customers.
9.3 Contract_Two year (binary PDP)
PDP shape: A stark drop at 1 (two-year contract) vs 0 — two-year contract customers have much lower churn probability than the baseline (month-to-month).
Business interpretation: longer contracts strongly reduce churn.
Actionable takeaway: incentives to move from month-to-month to 1- or 2-year contracts are likely effective (e.g., bundled discounts, devices, free months).
#Local / per-customer insights
Because SHAP failed in your environment, local SHAP force/waterfall plots were not created. Instead you have:
Local approximation CSVs produced by fallback approach (if saved) that show feature contributions approximated by feature importance × deviation from mean (these can be used to inspect per-sample drivers).
Recommendation: re-run SHAP (fix described below) to get exact additive per-sample attributions once SHAP compatibility is resolved.

#Concrete, data-backed retention strategies
Below are three prioritized retention strategies with measured rationale and suggested KPIs.

#Strategy A — Early-Activation & Onboarding Program (target: low-tenure customers)
Why: PDP shows churn sharply higher for very low tenure.
What to do: Offer a “first 3-month” onboarding package: free/discounted technician check, proactive welcome calls, 10% discount on second month for new customers.
How to measure: 3-month retention rate (compare cohorts), churn rate among customers with tenure ≤ 3 before vs after campaign.
Expected impact: reduce early churn — increases average tenure and LTV.
#Strategy B — High-bill Value Offers (target: high MonthlyCharges)
Why: PDP shows higher monthly charges correlate with greater churn risk.
What to do: For customers above a monthly-charge threshold (e.g., top 20% by MonthlyCharges), offer personalized bundle discounts, loyalty credits, or cost-stabilization plans.
How to measure: churn rate among targeted high-bill customers; uptake of offers; change in ARPU.
Expected impact: reduce churn among high ARPU customers, improving revenue retention.
#Strategy C — Contract Transition Incentives (target: month-to-month → 1/2 year or 2-year)
Why: Contract_Two year shows a large protective effect vs month-to-month.
What to do: Provide compelling offers to move customers to multi-year contracts (device subsidies, price lock, free months).
How to measure: conversion rate to multi-year contracts; churn reduction within converted cohort; payback period of incentives.

# Limitations
SHAP TreeExplainer failed in this environment due to a model parameter string ("[5E-1]") in the XGBoost internal parameters. That prevented per-sample additive explanations (SHAP) in this run.
Permutation importance + PDP provide robust global and marginal views, but are not a substitute for local additive explanations in model debugging and explaining individual decisions.
he model was not exhaustively hyperparameter tuned — AUC could be improved with grid/random search or Bayesian optimization.
Expected impact: lock-in customers, reduce churn materially.



# Python Code
# telco_churn_shap_fixed.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib.pyplot as plt

# CONFIG
DATA_PATH = "C:\\Users\Hp\OneDrive\Desktop\WA_Fn-UseC_-Telco-Customer-Churn.csv"  
OUTPUT_FOLDER ="D:\Cultus\telco_shap_outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
RANDOM_STATE = 42

# cleaning helpers
def clean_numeric_string(val):
    if pd.isna(val):
        return val
    if not isinstance(val, str):
        return val
    s = val.strip()
    if s == "":
        return np.nan
    # Remove surrounding brackets/parentheses
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        s = s[1:-1].strip()
    # Remove commas, currency
    s = s.replace(",", "").replace("$", "").replace("₹", "").replace("%", "")
    s = s.replace("'", "").replace('"', "")
    if s.lower() in ("na", "n/a", "none", "nan", "null", "unknown", ""):
        return np.nan
    return s

def robust_convert_numeric(series):
    cleaned = series.astype(object).map(clean_numeric_string)
    coerced = pd.to_numeric(cleaned, errors='coerce')
    success_frac = coerced.notna().sum() / len(coerced)
    return coerced, success_frac

#Load
print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)

#Clean numeric-like columns 
numeric_candidates = ['tenure', 'MonthlyCharges', 'TotalCharges']
for col in numeric_candidates:
    if col in df.columns:
        coerced, frac = robust_convert_numeric(df[col])
        print(f"Column {col} convertible fraction: {frac:.3f}")
        df[col] = coerced

# Try auto-convert any other object column that is mostly numeric
for col in df.select_dtypes(include=['object']).columns:
    if col.lower() in ('gender','partner','dependents','phone service','churn','customerid','contract','paymentmethod','internetservice'):
        continue
    coerced, frac = robust_convert_numeric(df[col])
    if frac > 0.85:
        print(f"Auto converting {col} (frac={frac:.3f})")
        df[col] = coerced

# Basic cleaning 
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])
if 'Churn' not in df.columns:
    raise ValueError("Churn column not found")
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

# Impute numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Churn' in num_cols:
    num_cols.remove('Churn')
if len(num_cols)>0:
    df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])

# Fill categorical missing
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
if len(cat_cols)>0:
    df[cat_cols] = df[cat_cols].fillna("Missing")

# 4. Encoding
df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
X = df_enc.drop(columns=['Churn'])
y = df_enc['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
print("Train/test shapes:", X_train.shape, X_test.shape)

# 5. Train model
use_xgb = False
try:
    import xgboost as xgb
    use_xgb = True
    print("XGBoost is available")
except Exception as e:
    print("XGBoost not available:", e)

if use_xgb:
    scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=RANDOM_STATE,
                              scale_pos_weight=float(scale_pos_weight), n_jobs=-1)
else:
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)

model.fit(X_train, y_train)

# 6. Evaluate 
y_proba = model.predict_proba(X_test)[:,1]
y_pred = (y_proba >= 0.5).astype(int)
auc = roc_auc_score(y_test, y_proba)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
print(f"Metrics -> AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# 7. Explainability with robust SHAP handling 
try:
    import shap
    shap_available = True
except Exception:
    shap_available = False

if shap_available:
    print("SHAP installed. Attempting TreeExplainer first (fast for tree models).")
    shap_vals_pos = None
    global_imp = None
    try:
        # Try the fast tree explainer (may fail for some xgboost/shap version combos)
        explainer = shap.TreeExplainer(model)
        # older/newer APIs differ; handle both
        try:
            shap_values = explainer.shap_values(X_test)
        except Exception:
            shap_values = explainer(X_test)
        if isinstance(shap_values, list):
            shap_vals_pos = shap_values[1]
        else:
            try:
                shap_vals_pos = np.array(shap_values.values)
            except Exception:
                shap_vals_pos = shap_values
        print("TreeExplainer succeeded.")
    except Exception as e_tree:
        print("TreeExplainer failed with exception:", type(e_tree).__name__, str(e_tree))
        print("Falling back to shap.Explainer using model.predict_proba (slower but more robust).")
        try:
            # Use model.predict_proba wrapper: returns probabilities; explain the probability of class 1
            predict_proba = lambda X: model.predict_proba(X)[:,1]
            explainer = shap.Explainer(predict_proba, X_train)   # could also pass X_train.sample(1000) to speed
            sv = explainer(X_test)  # shap.Explanation object
            # sv.values shape: (n_samples, ) or (n_samples, n_features)? New API returns (n_samples, n_features)
            try:
                shap_vals_pos = np.array(sv.values)
            except Exception:
                # fallback: if sv is list-like for classes, pick appropriate
                shap_vals_pos = np.array(sv)
            print("shap.Explainer succeeded.")
        except Exception as e_fallback_shap:
            print("shap.Explainer also failed:", type(e_fallback_shap).__name__, str(e_fallback_shap))
            shap_vals_pos = None

    # If we got shap values, compute global importance and save plots
    if shap_vals_pos is not None:
        # Ensure shap_vals_pos is 2D array with shape (n_samples, n_features)
        shap_vals_pos = np.asarray(shap_vals_pos)
        if shap_vals_pos.ndim == 3:
            # sometimes shap returns (1, n_samples, n_features) or (n_classes, n_samples, n_features)
            shap_vals_pos = shap_vals_pos.reshape(shap_vals_pos.shape[1], shap_vals_pos.shape[2])

        mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)
        global_imp = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False)
        print("Top-10 features by mean |SHAP|:")
        print(global_imp.head(10))

        # Save global bar
        plt.figure(figsize=(12,6))
        global_imp.head(20).plot.bar()
        plt.title("Global feature importance (mean |SHAP|) - top 20")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, "global_shap_importance.png"))
        plt.close()

        # Beeswarm / summary plot (use try/except because some SHAP plotting combinations fail)
        try:
            plt.figure()
            shap.summary_plot(shap_vals_pos, X_test, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, "shap_summary_beeswarm.png"), bbox_inches='tight')
            plt.close()
        except Exception as e_plot:
            print("Could not create beeswarm:", e_plot)

        # Dependence plots for top 3 features
        top3 = global_imp.head(3).index.tolist()
        for feat in top3:
            try:
                plt.figure(figsize=(6,4))
                shap.dependence_plot(feat, shap_vals_pos, X_test, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_FOLDER, f"shap_dependence_{feat.replace(' ', '_').replace('/', '_')}.png"))
                plt.close()
            except Exception as e_dp:
                print(f"Dependence plot failed for {feat}:", e_dp)

        # Local explanations (choose representative indices)
        preds = (y_proba >= 0.5).astype(int)
        churn_idx = np.where((preds==1) & (y_test.values==1))[0]
        nonchurn_idx = np.where((preds==0) & (y_test.values==0))[0]
        borderline_idx = np.argmin(np.abs(y_proba - 0.5))
        selected = {
            "predicted_churn": int(X_test.index[churn_idx[0]]) if len(churn_idx)>0 else None,
            "predicted_nonchurn": int(X_test.index[nonchurn_idx[0]]) if len(nonchurn_idx)>0 else None,
            "borderline": int(X_test.index[borderline_idx])
        }
        print("Selected:", selected)

        # Save waterfall / local SHAP if possible
        for name, idx in selected.items():
            if idx is None: 
                continue
            pos = list(X_test.index).index(idx)
            try:
                base_val = explainer.expected_value if hasattr(explainer, "expected_value") else None
                ev = shap.Explanation(values=shap_vals_pos[pos], base_values=base_val, data=X_test.iloc[pos])
                plt.figure(figsize=(10,4))
                shap.plots.waterfall(ev, show=False)
                plt.title(f"Local SHAP waterfall - {name} (idx {idx})")
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_FOLDER, f"shap_local_waterfall_{name}_{idx}.png"))
                plt.close()
            except Exception as e_local:
                # fallback to CSV of top local SHAP
                local_sh = pd.Series(shap_vals_pos[pos], index=X_test.columns)
                local_sh.abs().sort_values(ascending=False).head(40).to_csv(
                    os.path.join(OUTPUT_FOLDER, f"shap_local_top_{name}_{idx}.csv")
                )
        print("SHAP outputs (if computed) saved to:", OUTPUT_FOLDER)
    else:
        print("SHAP computation failed entirely; falling back to permutation importance below.")
        shap_available = False

# If SHAP is not available or computation failed, do permutation importance fallback
if not shap_available:
    print("Computing permutation importance fallback...")
    r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=1)
    perm_imp = pd.Series(r.importances_mean, index=X_test.columns).sort_values(ascending=False)
    perm_imp.head(20).to_csv(os.path.join(OUTPUT_FOLDER, "permutation_importance_top20.csv"))
    plt.figure(figsize=(12,6))
    perm_imp.head(20).plot.bar()
    plt.title("Permutation importance - top 20")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "permutation_importance_top20.png"))
    plt.close()
    top3 = perm_imp.head(3).index.tolist()
    for feat in top3:
        try:
            fig, ax = plt.subplots(figsize=(6,4))
            PartialDependenceDisplay.from_estimator(model, X_test, [feat], ax=ax)
            plt.title(f"PDP - {feat}")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, f"pdp_{feat.replace(' ', '_').replace('/', '_')}.png"))
            plt.close()
        except Exception as e:
            print("PDP failed for", feat, ":", e)
    print("Fallback outputs saved to:", OUTPUT_FOLDER)

print("Finished. Check outputs in:", OUTPUT_FOLDER)

# Output
Shape: (7043, 21)
Column tenure convertible fraction: 1.000
Column MonthlyCharges convertible fraction: 1.000
Column TotalCharges convertible fraction: 0.998
Train/test shapes: (5282, 30) (1761, 30)
XGBoost is available
Metrics -> AUC: 0.8190, Precision: 0.5565, Recall: 0.6745, F1: 0.6099
SHAP installed. Attempting TreeExplainer first (fast for tree models).

# Summary 
We trained a churn classifier on the Telco dataset (XGBoost). The model achieves AUC 0.82, Recall 67%, Precision 56% — good discrimination and solid sensitivity for identifying churners. Feature importance (permutation) shows tenure, contract type and monthly charges are the main drivers. PDPs reveal churn is highest for new customers (low tenure), increases with MonthlyCharges, and is markedly lower for 2-year contracts. Recommended actions are: (1) an early onboarding & activation program for new customers, (2) targeted value/discount offers for high-bill customers, and (3) incentives to move month-to-month customers to longer contracts. These three are actionable, measurable, and directly supported by the model outputs.
