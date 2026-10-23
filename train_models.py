# NovelMate: train regression and classification models
# Put Goodbooks-10k's books.csv inside the data folder before running this file.

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, classification_report

DATA_FILE = "data/books.csv"
MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        "Missing data/books.csv. Download Goodbooks-10k from Kaggle, "
        "then copy books.csv into the project's data folder."
    )

books = pd.read_csv(DATA_FILE)

# These columns exist in the original Goodbooks-10k books.csv.
# We use only numeric features so the project stays beginner-friendly.
feature_options = [
    "books_count",
    "original_publication_year",
    "ratings_count",
    "work_ratings_count",
    "work_text_reviews_count",
]
features = [column for column in feature_options if column in books.columns]

if "average_rating" not in books.columns:
    raise ValueError("The dataset must contain an average_rating column.")
if len(features) < 2:
    raise ValueError("Not enough expected numeric feature columns were found.")

data = books[features + ["average_rating"]].copy()
data["average_rating"] = pd.to_numeric(data["average_rating"], errors="coerce")
for column in features:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data = data.dropna(subset=["average_rating"])
data = data[data["average_rating"].between(0, 5)]
data["rating_class"] = (data["average_rating"] >= 4.0).astype(int)
data = data.dropna(subset=["rating_class"])

X = data[features]
y_regression = data["average_rating"]
y_classification = data["rating_class"]

X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)
# Use the same split positions for the classification target.
X_train_c, X_test_c, y_clf_train, y_clf_test = train_test_split(
    X, y_classification, test_size=0.2, random_state=42
)

regression_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LinearRegression())
])
regression_model.fit(X_train, y_reg_train)
reg_predictions = regression_model.predict(X_test)
reg_r2 = r2_score(y_reg_test, reg_predictions)

classification_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
classification_model.fit(X_train_c, y_clf_train)
class_predictions = classification_model.predict(X_test_c)
class_accuracy = accuracy_score(y_clf_test, class_predictions)

joblib.dump(regression_model, os.path.join(MODEL_FOLDER, "rating_regression.joblib"))
joblib.dump(classification_model, os.path.join(MODEL_FOLDER, "rating_classification.joblib"))
joblib.dump(features, os.path.join(MODEL_FOLDER, "feature_columns.joblib"))

print("Features used:", features)
print("Regression R²:", round(reg_r2, 4))
print("Classification accuracy:", round(class_accuracy, 4))
print("\nClassification report:")
print(classification_report(y_clf_test, class_predictions, zero_division=0))
print("\nModels saved in the models folder.")
