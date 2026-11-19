"""AI Project v2: classify football player position (GK/DF/MF/FW) with extra engineered features."""
import os
import time, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

CSV = os.path.join(os.path.dirname(__file__), "players_data_light-2026_2027.csv")
MIN_90S = 1.0              # keep players with at least this many 90s (try 3.0 as an experiment)
USE_NEW_FEATURES = True    # False = original feature set (useful for an ablation in the report)
USE_GK_COLS = False        # False = drop goalkeeper columns
SEED = 42

# ---------- 1. Load & clean ----------
df = pd.read_csv(CSV)
print("Raw shape:", df.shape)
df = df.drop_duplicates(subset=["Player", "Squad"])
df = df[df["90s"] >= MIN_90S].copy()
df["Target"] = df["Pos"].str.split(",").str[0].str.strip()
df = df[df["Target"].isin(["GK", "DF", "MF", "FW"])]
print("After cleaning:", df.shape, df["Target"].value_counts().to_dict())

# ---------- 2. Feature engineering ----------
count_cols = ["Gls","Ast","G-PK","PK","PKatt","CrdY","CrdR","Sh","SoT","Crs","TklW","Int","Fld","Fls","OG"]
for c in count_cols:
    df[c + "_p90"] = df[c] / df["90s"]
df["Starts_ratio"] = df["Starts"] / df["MP"]
df["Min_per_MP"] = df["Min"] / df["MP"]
df["SoT%"] = df["SoT%"].fillna(0)

eps = 1e-6
df["DefAct_p90"]       = df["TklW_p90"] + df["Int_p90"]                    # defensive actions
df["Attack_p90"]       = df["Sh_p90"] + df["Crs_p90"]                      # attacking actions
df["Att_vs_Def"]       = df["Attack_p90"] - df["DefAct_p90"]
df["Att_ratio"]        = df["Attack_p90"] / (df["Attack_p90"] + df["DefAct_p90"] + eps)
df["Shots_per_DefAct"] = df["Sh_p90"] / (df["DefAct_p90"] + 1)
df["Cross_share"]      = df["Crs_p90"] / (df["Crs_p90"] + df["Sh_p90"] + 1)
df["Shot_share"]       = df["Sh_p90"] / (df["Sh_p90"] + df["Crs_p90"] + df["DefAct_p90"] + eps)
df["Tkl_share"]        = df["TklW_p90"] / (df["DefAct_p90"] + eps)
df["Int_share"]        = df["Int_p90"] / (df["DefAct_p90"] + eps)
df["Foul_balance"]     = df["Fld_p90"] - df["Fls_p90"]                     # fouled more than fouls -> attacker
df["Foul_ratio"]       = df["Fld_p90"] / (df["Fls_p90"] + 1)
df["GoalInvolv_p90"]   = df["Gls_p90"] + df["Ast_p90"]
df["Assist_share"]     = df["Ast_p90"] / (df["GoalInvolv_p90"] + eps)
df["Card_p90"]         = df["CrdY_p90"] + 2 * df["CrdR_p90"]
df["Card_per_foul"]    = df["CrdY"] / (df["Fls"] + 1)
df["Shot_accuracy"]    = df["SoT"] / (df["Sh"] + 1)
df["Shot_quality"]     = df["Gls"] / (df["Sh"] + 1)
df["Is_shooter"]       = (df["Sh_p90"] > 0).astype(int)
df["No_attack"]        = ((df["Sh"] == 0) & (df["Crs"] == 0)).astype(int)
df["Is_starter"]       = (df["Starts_ratio"] >= 0.6).astype(int)
df["Full_90_share"]    = df["Min"] / (df["MP"] * 90 + eps)
df["Age_sq"]           = df["Age"] ** 2
log_src = ["Sh_p90","Crs_p90","TklW_p90","Int_p90","Fls_p90","Fld_p90","DefAct_p90","Attack_p90"]
for c in log_src:
    df["log_" + c] = np.log1p(df[c])

base_feats = [c + "_p90" for c in count_cols] + ["Sh/90","SoT/90","SoT%","Starts_ratio","Min_per_MP","Age"]
new_feats = ["DefAct_p90","Attack_p90","Att_vs_Def","Att_ratio","Shots_per_DefAct","Cross_share","Shot_share",
             "Tkl_share","Int_share","Foul_balance","Foul_ratio","GoalInvolv_p90","Assist_share","Card_p90",
             "Card_per_foul","Shot_accuracy","Shot_quality","Is_shooter","No_attack","Is_starter",
             "Full_90_share","Age_sq"] + ["log_" + c for c in log_src]
feat = base_feats + (new_feats if USE_NEW_FEATURES else [])
gk_cols = ["GA90","Save%","CS%","SoTA"]
if USE_GK_COLS:
    feat += gk_cols
X = df[feat].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
y = df["Target"]
print("Number of features:", X.shape[1])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

# ---------- 3. Models ----------
def make(model):
    return Pipeline([("sc", StandardScaler()), ("clf", model)])

def stack():
    return StackingClassifier(
        estimators=[("svm", SVC(class_weight="balanced", probability=True, random_state=SEED)),
                    ("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
                    ("rf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED))],
        final_estimator=LogisticRegression(max_iter=3000), cv=5)

models = {
    "LogReg": make(LogisticRegression(max_iter=3000, class_weight="balanced")),
    "KNN": make(KNeighborsClassifier()),
    "DecisionTree": make(DecisionTreeClassifier(random_state=SEED, class_weight="balanced")),
    "RandomForest": make(RandomForestClassifier(random_state=SEED, class_weight="balanced")),
    "GradBoost": make(GradientBoostingClassifier(random_state=SEED)),
    "SVM": make(SVC(class_weight="balanced", random_state=SEED)),
    "MLP": make(MLPClassifier(max_iter=800, random_state=SEED)),
    "Stacking": make(stack()),
}

def evaluate(name, pipe, store):
    t0 = time.time(); pipe.fit(X_tr, y_tr); fit_t = time.time() - t0
    t0 = time.time(); pred = pipe.predict(X_te); pred_t = time.time() - t0
    store.append(dict(Model=name,
        CV_acc=cross_val_score(pipe, X_tr, y_tr, cv=cv).mean(),
        Test_acc=accuracy_score(y_te, pred),
        Macro_F1=f1_score(y_te, pred, average="macro"),
        Precision=precision_score(y_te, pred, average="macro"),
        Recall=recall_score(y_te, pred, average="macro"),
        Fit_s=fit_t, Pred_s=pred_t))

base = []
for n, m in models.items():
    evaluate(n, m, base)
base_df = pd.DataFrame(base).sort_values("Test_acc", ascending=False)
print("\n=== BASELINE ===\n", base_df.round(3).to_string(index=False))

# ---------- 4. Hyper-parameter tuning ----------
grids = {
    "LogReg": (models["LogReg"], {"clf__C": [0.01, 0.1, 1, 10, 100]}),
    "KNN": (models["KNN"], {"clf__n_neighbors": [3,5,9,15,25], "clf__weights": ["uniform","distance"]}),
    "DecisionTree": (models["DecisionTree"], {"clf__max_depth": [3,5,8,12,None], "clf__min_samples_leaf": [1,3,5,10]}),
    "RandomForest": (models["RandomForest"], {"clf__n_estimators": [100,300], "clf__max_depth": [None,10,20], "clf__min_samples_leaf": [1,3]}),
    "GradBoost": (models["GradBoost"], {"clf__n_estimators": [100,200], "clf__learning_rate": [0.05,0.1], "clf__max_depth": [2,3]}),
    "SVM": (models["SVM"], {"clf__C": [0.1,1,10,100], "clf__gamma": ["scale", 0.01, 0.1], "clf__kernel": ["rbf","linear"]}),
    "MLP": (models["MLP"], {"clf__hidden_layer_sizes": [(32,),(64,),(64,32)], "clf__alpha": [1e-4,1e-2], "clf__learning_rate_init": [1e-3,1e-2]}),
    "Stacking": (models["Stacking"], {"clf__final_estimator__C": [0.1, 1, 10]}),
}
tuned, best = [], {}
for n, (pipe, grid) in grids.items():
    t0 = time.time()
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1).fit(X_tr, y_tr)
    search_t = time.time() - t0
    best[n] = gs.best_estimator_
    evaluate(n, gs.best_estimator_, tuned)
    tuned[-1]["Search_s"] = search_t
    tuned[-1]["Best_params"] = gs.best_params_
tuned_df = pd.DataFrame(tuned).sort_values("Test_acc", ascending=False)
print("\n=== TUNED ===\n", tuned_df.drop(columns="Best_params").round(3).to_string(index=False))

evo = base_df[["Model","Test_acc"]].merge(tuned_df[["Model","Test_acc"]], on="Model", suffixes=("_baseline","_tuned"))
evo["Gain"] = evo["Test_acc_tuned"] - evo["Test_acc_baseline"]
print("\n=== EVOLUTION ===\n", evo.round(3).to_string(index=False))

# ---------- 5. Best model report & plots ----------
best_name = tuned_df.iloc[0]["Model"]; best_model = best[best_name]
pred = best_model.predict(X_te)
print(f"\nBest model: {best_name}\n")
print(classification_report(y_te, pred))

labels = ["GK","DF","MF","FW"]
fig, ax = plt.subplots(figsize=(7,6))
ConfusionMatrixDisplay(confusion_matrix(y_te, pred, labels=labels), display_labels=labels).plot(ax=ax, cmap="Blues")
ax.set_title(f"Confusion matrix - {best_name}"); plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

x = np.arange(len(evo)); w = 0.38
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(x-w/2, evo["Test_acc_baseline"], w, label="Baseline"); ax.bar(x+w/2, evo["Test_acc_tuned"], w, label="Tuned")
ax.set_xticks(x); ax.set_xticklabels(evo["Model"], rotation=30); ax.set_ylabel("Test accuracy"); ax.set_ylim(0.5, 1); ax.legend()
ax.set_title("Accuracy improvement after tuning"); plt.tight_layout(); plt.savefig("accuracy_evolution.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(tuned_df["Fit_s"], tuned_df["Test_acc"], s=70)
for _, r in tuned_df.iterrows():
    ax.annotate(r["Model"], (r["Fit_s"], r["Test_acc"]), textcoords="offset points", xytext=(5,5))
ax.set_xscale("log"); ax.set_xlabel("Fit time (s, log)"); ax.set_ylabel("Test accuracy"); ax.set_title("Accuracy vs training time")
plt.tight_layout(); plt.savefig("accuracy_vs_time.png", dpi=150); plt.close()

rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED).fit(X_tr, y_tr)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(20)
fig, ax = plt.subplots(figsize=(8,7)); imp.plot.barh(ax=ax); ax.set_title("Top 20 feature importances (Random Forest)")
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150); plt.close()

sizes, trs, vas = learning_curve(best_model, X_tr, y_tr, cv=cv, train_sizes=np.linspace(0.1,1,6), n_jobs=-1)
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(sizes, trs.mean(1), "o-", label="Train"); ax.plot(sizes, vas.mean(1), "o-", label="CV")
ax.set_xlabel("Training samples"); ax.set_ylabel("Accuracy"); ax.legend(); ax.set_title(f"Learning curve - {best_name}")
plt.tight_layout(); plt.savefig("learning_curve.png", dpi=150); plt.close()

base_df.to_csv("results_baseline.csv", index=False); tuned_df.to_csv("results_tuned.csv", index=False)
print("Saved plots and CSV results.")