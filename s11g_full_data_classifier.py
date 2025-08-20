# svm_esg_linear_classifier.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay
)

output_dir = "outputs/s11g_full_data_classifier_absVal"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("processed_data/out19_all_companies_rank_fin.csv")

# target
df["ESG_Ranked"] = df["ESG Rank"].notna().astype(int)

# features 
candidate_features = [
    "Stock_Performance per", "TotalRevenueIncr per", "NetIncomeIncr per",
    "OperatingIncomeIncr per", "GrossProfitIncr per", "DilutedEPSIncr per",
    "TotalRevenue", "NetIncome", "OperatingIncome", "GrossProfit",
    "DilutedEPS", "Net Margin per", "Operating Margin per", "Gross Margin per",
    "Country", "GICS Sector", "fullTimeEmployees"
]

rename_dict = {
    "TotalRevenue": "Total Revenue",
    "GrossProfit": "Gross Profit",
    "NetIncome": "Net Income",
    "OperatingIncome": "Operating Income",
    "Gross Margin per": "Gross Margin",
    "Net Margin per": "Net Margin",
    "Operating Margin per": "Operating Margin",
    "Stock_Performance per": "Stock Performance",
    "DilutedEPS": "Diluted EPS",
    "fullTimeEmployees": "Employee Count",
    "TotalRevenueIncr per": "Revenue YoY Growth",
    "NetIncomeIncr per": "Net Income YoY Growth",
    "OperatingIncomeIncr per": "Operating Income YoY Growth",
    "GrossProfitIncr per": "Gross Profit YoY Growth",
    "DilutedEPSIncr per": "Diluted EPS YoY Growth"
}


df_model = df[["ESG_Ranked"] + candidate_features].dropna()
X = df_model[candidate_features]
y = df_model["ESG_Ranked"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

numeric_features = [f for f in candidate_features if f not in ["Country", "GICS Sector"]]
categorical_features = ["Country", "GICS Sector"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

linear_svc = LinearSVC(C=1.0, class_weight='balanced', max_iter=10000)
calibrated_clf = CalibratedClassifierCV(linear_svc, method='sigmoid', cv=5)

clf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", calibrated_clf)
])

clf_pipeline.fit(X_train, y_train)

y_pred = clf_pipeline.predict(X_test)
y_prob = clf_pipeline.predict_proba(X_test)[:, 1]
# alternative threshold 
threshold = 0.15
y_pred_thresh = (y_prob >= threshold).astype(int)

alt_report_text = classification_report(y_test, y_pred_thresh, target_names=["Unranked", "Ranked"])
alt_conf_matrix = confusion_matrix(y_test, y_pred_thresh)
alt_conf_matrix_text = str(alt_conf_matrix)

with open(os.path.join(output_dir, f"classification_report_thresh_{threshold:.2f}.txt"), "w") as f:
    f.write(f"=== Classification Report @ threshold {threshold:.2f} ===\n")
    f.write(alt_report_text + "\n")

with open(os.path.join(output_dir, f"confusion_matrix_thresh_{threshold:.2f}.txt"), "w") as f:
    f.write(f"=== Confusion Matrix @ threshold {threshold:.2f} ===\n")
    f.write(alt_conf_matrix_text + "\n")


cat_encoder = clf_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
cat_names = cat_encoder.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_names)

svc_model = clf_pipeline.named_steps["classifier"].calibrated_classifiers_[0].estimator
coef = svc_model.coef_[0]


def plot_feature_importance(coef, feature_names, top_n=20):
    imp, names = zip(*sorted(zip(coef, feature_names), key=lambda x: abs(x[0]), reverse=True))
    imp = np.array(imp[:top_n])
    names = names[:top_n]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.xlabel("SVM Coefficient")
    plt.title("Top Feature Importances (Linear SVM)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "linear_svm_feature_importance.png"), dpi=300)
    plt.show()

plot_feature_importance(coef, all_feature_names)

report_text = classification_report(y_test, y_pred, target_names=["Unranked", "Ranked"])
conf_matrix_text = str(confusion_matrix(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_prob)
roc_auc_text = f"ROC AUC: {roc_auc:.4f}\n"

with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write("=== Classification Report ===\n")
    f.write(report_text + "\n")
    f.write(roc_auc_text)

with open(os.path.join(output_dir, "confusion_matrix.txt"), "w") as f:
    f.write("=== Confusion Matrix ===\n")
    f.write(conf_matrix_text + "\n")

with open(os.path.join(output_dir, "roc_auc.txt"), "w") as f:
    f.write(roc_auc_text)

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("Linear SVM Classifier ROC Curve - ESG Ranked", fontsize=14)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/s11g_svm_roc_curve.png", dpi=300)
plt.show()

class_counts = y.value_counts(normalize=True).to_frame("Proportion")
class_counts.to_csv(os.path.join(output_dir, "class_distribution.csv"))

pred_df = X_test.copy()
pred_df["True_Label"] = y_test.values
pred_df["Predicted_Label"] = y_pred
pred_df["Predicted_Prob"] = y_prob
pred_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


import matplotlib.pyplot as plt
import numpy as np

def plot_colored_feature_importance(coef, feature_names, numeric_features, top_n=30):
    
    feature_types = []
    for name in feature_names:
        if name in numeric_features:
            feature_types.append("numeric")
        else:
            feature_types.append("categorical")

    
    zipped = list(zip(coef, feature_names, feature_types))
    sorted_feats = sorted(zipped, key=lambda x: abs(x[0]), reverse=True)[:top_n]

    
    sorted_coef, sorted_names, sorted_types = zip(*sorted_feats)
    colors = ["tab:blue" if t == "numeric" else "tab:orange" for t in sorted_types]

   
    plt.figure(figsize=(12, max(6, 0.35 * len(sorted_names))))
    bars = plt.barh(range(len(sorted_names)), sorted_coef, color=colors, edgecolor='black')
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("SVM Coefficient")
    plt.title("Top Feature Importances (Linear SVM)\nBlue = Financial, Orange = Categorical")

    
    for i, (c, bar) in enumerate(zip(sorted_coef, bars)):
        plt.text(bar.get_width() + 0.02*np.sign(c), bar.get_y() + bar.get_height()/2,
                 f"{c:.2f}", va='center', ha='left' if c > 0 else 'right', fontsize=9)

    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "linear_svm_feature_importance_colored.png"), dpi=300)
    plt.show()


plot_colored_feature_importance(coef, all_feature_names, numeric_features, top_n=30)


def plot_separate_feature_groups(
    coef,
    feature_names,
    numeric_features,
    output_path_prefix,
    top_n=20,
    use_absolute=False  
):
    financial = []
    categorical = []

    for i, name in enumerate(feature_names):
        entry = (coef[i], name)
        if name in numeric_features:
            financial.append(entry)
        else:
            categorical.append(entry)

    def make_plot(data, color, title, filename, use_abs=False):
        
        data = sorted(data, key=lambda x: abs(x[0]), reverse=True)[:top_n]
        coefs_raw, raw_names = zip(*data)
        names = [rename_dict.get(name, name) for name in raw_names]

        
        coefs = np.abs(coefs_raw) if use_abs else coefs_raw

        plt.figure(figsize=(10, max(6, 0.35 * len(names))))
        bars = plt.barh(range(len(names)), coefs, color=color, edgecolor='black')
        plt.yticks(range(len(names)), names)

       
        if use_abs:
            max_x = max(coefs)
            buffer = 0.10 * max_x
            plt.xlim(0, max_x + buffer)
        else:
            min_x = min(coefs)
            max_x = max(coefs)
            x_range = max_x - min_x
            buffer = 0.10 * x_range
            plt.xlim(min_x - buffer, max_x + buffer)
            plt.axvline(0, color='gray', linestyle='--', linewidth=1)

        
        for i, (c, bar) in enumerate(zip(coefs_raw, bars)):
            x = bar.get_width()
            label = f"{abs(c):.2f}" if use_abs else f"{c:.2f}"
            ha = 'left' if not use_abs else 'left'
            plt.text(x + 0.02, bar.get_y() + bar.get_height()/2, label,
                     va='center', ha=ha, fontsize=9, color='black')

        plt.xlabel("SVM Coefficient" + (" (Absolute)" if use_abs else ""))
        plt.title(title + (" (Abs)" if use_abs else ""))
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.show()

    
    make_plot(financial, "tab:blue",
              "Numeric Feature Contributions to ESG Classification (Linear SVM)",
              f"{output_path_prefix}_financial{'_abs' if use_absolute else ''}.png",
              use_abs=use_absolute)

    
    make_plot(categorical, "tab:orange",
              f"Top {top_n} Categorical Features (Linear SVM)",
              f"{output_path_prefix}_categorical.png",
              use_abs=False)  



plot_separate_feature_groups(
    coef=coef,
    feature_names=all_feature_names,
    numeric_features=numeric_features,
    output_path_prefix="linear_svm_feature_importance_split",
    top_n=20,
    use_absolute=True
)
