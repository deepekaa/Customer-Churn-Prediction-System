
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.datasets import make_classification
import joblib

X, y = make_classification(
    n_samples=3000,
    n_features=6,
    n_informative=4,
    n_redundant=1,
    weights=[0.73, 0.27],
    random_state=42,
)

columns = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "usage",
    "contract_length",
    "payment_score",
]

df = pd.DataFrame(X, columns=columns)

# Make the features easier to interpret
df["tenure"] = ((df["tenure"] + 3) * 12).clip(0, 72)
df["monthly_charges"] = ((df["monthly_charges"] + 3) * 20).clip(10, 150)
df["support_calls"] = ((df["support_calls"] + 3) * 2).clip(0, 12)
df["usage"] = ((df["usage"] + 3) * 30).clip(0, 200)
df["contract_length"] = ((df["contract_length"] + 3) * 6).clip(1, 36)
df["payment_score"] = ((df["payment_score"] + 3) * 2).clip(0, 10)

df["churn"] = y

features = columns
X = df[features]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            features,
        )
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Model Evaluation")
print("----------------")
print(f"Accuracy:  {accuracy_score(y_test, predictions):.3f}")
print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.3f}")
print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.3f}")

joblib.dump(
    {"model": model, "features": features},
    "churn_model.joblib",
)

df.to_csv("customer_data.csv", index=False)

print("\nSaved churn_model.joblib and customer_data.csv")
print("Note: this is synthetic demonstration data, not real customer records.")
