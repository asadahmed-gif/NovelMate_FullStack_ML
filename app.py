from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

app = FastAPI(title="NovelMate - Book Intelligence")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

regression_path = MODEL_DIR / "rating_regression.joblib"
classification_path = MODEL_DIR / "rating_classification.joblib"
features_path = MODEL_DIR / "feature_columns.joblib"

regression_model = None
classification_model = None
feature_columns = None

def load_models():
    global regression_model, classification_model, feature_columns
    if not (regression_path.exists() and classification_path.exists() and features_path.exists()):
        raise RuntimeError("Models not found. Run `python train_models.py` first.")
    regression_model = joblib.load(regression_path)
    classification_model = joblib.load(classification_path)
    feature_columns = joblib.load(features_path)

class BookFeatures(BaseModel):
    books_count: float = 1
    original_publication_year: float = 2000
    ratings_count: float = 1000
    work_ratings_count: float = 1000
    work_text_reviews_count: float = 100

@app.on_event("startup")
def startup():
    # Keep startup successful so the homepage can explain missing model files.
    try:
        load_models()
    except Exception:
        pass

@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")

def prepare_input(book: BookFeatures):
    if regression_model is None or classification_model is None or feature_columns is None:
        load_models()
    values = book.model_dump()
    return pd.DataFrame([[values.get(col, 0) for col in feature_columns]], columns=feature_columns)

@app.post("/predict/regression")
def predict_regression(book: BookFeatures):
    row = prepare_input(book)
    prediction = float(regression_model.predict(row)[0])
    prediction = max(0.0, min(5.0, prediction))
    return {"predicted_average_rating": round(prediction, 2), "scale": "0 to 5"}

@app.post("/predict/classification")
def predict_classification(book: BookFeatures):
    row = prepare_input(book)
    prediction = int(classification_model.predict(row)[0])
    label = "Highly rated (4.0 or above)" if prediction == 1 else "Not highly rated (below 4.0)"
    return {"prediction": prediction, "category": label}
