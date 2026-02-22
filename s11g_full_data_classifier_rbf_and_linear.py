import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("processed_data/out19_all_companies_rank_fin.csv")

df["ESG_Ranked"] = df["ESG Rank"].notna().astype(int)  # 1 if ranked, 0 if not

candidate_features = [
    "Stock_Performance per", "TotalRevenueIncr per", "NetIncomeIncr per",
    "OperatingIncomeIncr per", "GrossProfitIncr per", "DilutedEPSIncr per",
    "TotalRevenue", "NetIncome", "OperatingIncome", "GrossProfit",
    "DilutedEPS", "Net Margin per", "Operating Margin per", "Gross Margin per",
    "Country", "GICS Sector", "fullTimeEmployees"
]

# drop rows with missing features or target
df_model = df[["ESG_Ranked"] + candidate_features].dropna()
X = df_model[candidate_features]
y = df_model["ESG_Ranked"]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

numeric_features = [f for f in candidate_features if f not in ["Country", "GICS Sector"]]
categorical_features = ["Country", "GICS Sector"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

clf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", class_weight='balanced'))
])

# train the model
clf_pipeline.fit(X_train, y_train)

y_pred = clf_pipeline.predict(X_test)
y_prob = clf_pipeline.predict_proba(X_test)[:, 1]

os.makedirs("outputs/s11g_full_data_classifier", exist_ok=True)

report_text = classification_report(y_test, y_pred)
conf_matrix_text = str(confusion_matrix(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_prob)
roc_auc_text = f"ROC AUC: {roc_auc:.4f}\n"

print(confusion_matrix(y_test, y_pred))

with open("outputs/s11g_full_data_classifier/classification_report.txt", "w") as f:
    f.write("Classification Report\n")
    f.write(report_text + "\n")

with open("outputs/s11g_full_data_classifier/confusion_matrix.txt", "w") as f:
    f.write("Confusion Matrix\n")
    f.write(conf_matrix_text + "\n")

with open("outputs/s11g_full_data_classifier/roc_auc.txt", "w") as f:
    f.write(roc_auc_text)

# plot ROC curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("SVM Classifier ROC Curve - ESG Ranked")
plt.tight_layout()
plt.savefig("outputs/s11g_full_data_classifier/s11g_svm_roc_curve.png")
plt.show()




# Find Linear SVM feature importance

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

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

# Creating a plot
def plot_separate_feature_groups(
    coef,
    feature_names,
    numeric_features,
    output_dir,
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

        
        names = [rename_dict.get(n, n) for n in raw_names]

        coefs = np.abs(coefs_raw) if use_abs else coefs_raw

        plt.figure(figsize=(10, max(6, 0.35 * len(names))))
        bars = plt.barh(range(len(names)), coefs, color=color, edgecolor="black")
        plt.yticks(range(len(names)), names)

        if use_abs:
            max_x = float(np.max(coefs))
            buffer = 0.10 * max_x if max_x > 0 else 0.1
            plt.xlim(0, max_x + buffer)
        else:
            min_x = float(np.min(coefs))
            max_x = float(np.max(coefs))
            x_range = max_x - min_x
            buffer = 0.10 * x_range if x_range > 0 else 0.1
            plt.xlim(min_x - buffer, max_x + buffer)
            plt.axvline(0, color="gray", linestyle="--", linewidth=1)

        for c_raw, bar in zip(coefs_raw, bars):
            x = bar.get_width()
            label = f"{abs(c_raw):.2f}" if use_abs else f"{c_raw:.2f}"
            plt.text(x + 0.02, bar.get_y() + bar.get_height()/2,
                     label, va="center", ha="left", fontsize=9, color="black")

        plt.xlabel("SVM Coefficient" + (" (Absolute)" if use_abs else ""))
        plt.title(title + (" (Abs)" if use_abs else ""))
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.show()

    # Potential to use absolute values for interpretability
    make_plot(
        financial,
        "tab:blue",
        "Numeric Feature Contributions to ESG Classification (Linear SVM)",
        f"{output_path_prefix}_financial{'_abs' if use_absolute else ''}.png",
        use_abs=use_absolute
    )

    # Additional categorical plot (not used in analysis)
    make_plot(
        categorical,
        "tab:orange",
        f"Top {top_n} Categorical Features (Linear SVM)",
        f"{output_path_prefix}_categorical.png",
        use_abs=False
    )

# ensure that the linear model and RBF model see the same data for consistency
pre = clf_pipeline.named_steps["preprocessor"]
X_train_t = pre.transform(X_train)

# calibration converts raw decisions into probability estimates
# use a 5 fold model for stronger validation and results
linear_svc = LinearSVC(C=1.0, class_weight="balanced", max_iter=10000)
calibrated_clf = CalibratedClassifierCV(linear_svc, method="sigmoid", cv=5)

calibrated_clf.fit(X_train_t, y_train)

# take coef from the FIRST calibrated fold's underlying estimator
svc_model = calibrated_clf.calibrated_classifiers_[0].estimator
coef = svc_model.coef_[0]

cat_encoder = pre.named_transformers_["cat"]
cat_names = cat_encoder.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_names)

# plot the coefficient data (used absolute = true for interpretability)
output_dir = "outputs/s11g_full_data_classifier"
plot_separate_feature_groups(
    coef=coef,
    feature_names=all_feature_names,
    numeric_features=numeric_features,
    output_dir=output_dir,
    output_path_prefix="linear_svm_feature_importance_split",
    top_n=20,
    use_absolute=True
)
