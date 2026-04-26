from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io, base64, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

def is_pca_file(df):
    feature_cols = [c for c in df.columns if c not in ('Time','Amount','Class')]
    return all(c.startswith('V') and c[1:].isdigit() for c in feature_cols)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"

def preprocess(df, is_pca):
    has_label = 'Class' in df.columns
    y = df['Class'].astype(int) if has_label else None
    if is_pca:
        X = df.drop('Class', axis=1) if has_label else df.copy()
        sc = StandardScaler()
        for col in ['Time', 'Amount']:
            if col in X.columns:
                X[col] = sc.fit_transform(X[[col]]).ravel()
    else:
        X_raw = df.drop('Class', axis=1) if has_label else df.copy()
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X_raw)
        n = min(10, X_raw.shape[1])
        pca = PCA(n_components=n)
        X_pca = pca.fit_transform(X_scaled)
        X = pd.DataFrame(X_pca, columns=[f'V{i+1}' for i in range(n)])
    return X, y

def chart_class_dist(y):
    fig, ax = plt.subplots(figsize=(4, 2.5), facecolor='#0f172a')
    counts = y.value_counts().sort_index()
    bars = ax.bar(['Legitimate', 'Fraud'], counts.values,
                  color=['#0ea5e9', '#f43f5e'], width=0.5)
    ax.set_facecolor('#1e293b')
    ax.set_title('Class Distribution', color='white', fontsize=11)
    ax.set_ylabel('Count', color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:,}', ha='center', color='white', fontsize=9)
    plt.tight_layout()
    return fig_to_base64(fig)

def chart_roc(models, X_test, y_test):
    colors = ['#0ea5e9', '#10b981', '#f59e0b']
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    for (name, m), color in zip(models.items(), colors):
        y_proba = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})',
                color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate', color='#94a3b8')
    ax.set_ylabel('True Positive Rate', color='#94a3b8')
    ax.set_title('ROC Curves', color='white', fontsize=11)
    ax.tick_params(colors='#94a3b8')
    ax.legend(facecolor='#0f172a', labelcolor='white', fontsize=7)
    ax.grid(True, color='#334155', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    plt.tight_layout()
    return fig_to_base64(fig)

def chart_metrics(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), facecolor='#0f172a')
    for ax, metric, color in zip(axes,
            ['Precision', 'Recall', 'F1-Score'],
            ['#0ea5e9', '#10b981', '#f43f5e']):
        ax.bar(results_df['Model'], results_df[metric],
               color=color, alpha=0.85)
        ax.set_facecolor('#1e293b')
        ax.set_title(metric, color='white', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='x', rotation=12, colors='#94a3b8', labelsize=7)
        ax.tick_params(axis='y', colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')
    plt.tight_layout()
    return fig_to_base64(fig)

def chart_cm(cm, model_name):
    fig, ax = plt.subplots(figsize=(3.5, 3), facecolor='#0f172a')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'],
                linewidths=0.5)
    ax.set_facecolor('#1e293b')
    ax.set_title(f'Confusion Matrix\n{model_name}', color='white', fontsize=9)
    ax.set_ylabel('Actual', color='#94a3b8')
    ax.set_xlabel('Predicted', color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    plt.tight_layout()
    return fig_to_base64(fig)

def chart_feat_importance(model, feature_names):
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    ax.barh(feat_df['Feature'], feat_df['Importance'],
            color='#0ea5e9', alpha=0.85)
    ax.invert_yaxis()
    ax.set_title('Top 10 Feature Importances', color='white', fontsize=11)
    ax.set_xlabel('Importance Score', color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, axis='x', color='#334155', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    plt.tight_layout()
    return fig_to_base64(fig)

@app.get("/")
def health():
    return {"status": "Fraud Detection API is running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    if 'Class' not in df.columns:
        return {"error": "No 'Class' column found. Please include labels (0 or 1)."}

    # Limit to 5000 rows max
    if len(df) > 5000:
        fraud = df[df['Class'] == 1]
        legit = df[df['Class'] == 0].sample(
            n=min(4500, len(df[df['Class']==0])), random_state=42)
        df = pd.concat([fraud, legit]).sample(frac=1, random_state=42)

    pca_file = is_pca_file(df)
    X, y = preprocess(df, pca_file)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=500, random_state=42),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=30, max_depth=10,
            random_state=42, n_jobs=1)
    }

    results = []
    cms = {}
    for name, m in models.items():
        m.fit(X_res, y_res)
        y_pred  = m.predict(X_test)
        y_proba = m.predict_proba(X_test)[:, 1]
        results.append({
            'Model':     name,
            'Precision': round(float(precision_score(y_test, y_pred)), 4),
            'Recall':    round(float(recall_score(y_test, y_pred)),    4),
            'F1-Score':  round(float(f1_score(y_test, y_pred)),        4),
            'ROC-AUC':   round(float(roc_auc_score(y_test, y_proba)),  4),
        })
        cms[name] = confusion_matrix(y_test, y_pred)

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
    best_model = models[best]

    # Generate predictions table (first 20 from test set)
    y_pred_best  = best_model.predict(X_test)
    y_proba_best = best_model.predict_proba(X_test)[:, 1]

    predictions = []
    for i in range(min(20, len(X_test))):
        actual    = int(y_test.iloc[i])
        predicted = int(y_pred_best[i])
        prob      = round(float(y_proba_best[i]) * 100, 2)
        correct   = actual == predicted
        predictions.append({
            'transaction': i + 1,
            'actual':      'FRAUD' if actual == 1 else 'LEGITIMATE',
            'predicted':   'FRAUD' if predicted == 1 else 'LEGITIMATE',
            'fraud_prob':  prob,
            'correct':     correct
        })

    return {
        "summary": {
            "file_type":     "PCA-transformed" if pca_file else "Raw (PCA applied automatically)",
            "shape":         list(df.shape),
            "fraud_count":   int(y.sum()),
            "fraud_percent": round(float(y.sum() / len(y) * 100), 4),
            "best_model":    best,
        },
        "results":     results,
        "predictions": predictions,
        "charts": {
            "class_dist":         chart_class_dist(y),
            "roc":                chart_roc(models, X_test, y_test),
            "metrics_bar":        chart_metrics(results_df),
            "confusion_matrix":   chart_cm(cms[best], best),
            "feature_importance": chart_feat_importance(
                                    models['Random Forest'],
                                    X.columns.tolist()),
        }
    }