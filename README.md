# NovelMate — Full-Stack Machine Learning App

NovelMate is a beginner-friendly web application built for the Full-Stack Machine Learning assignment. It uses the Goodbooks-10k dataset from Kaggle.

## What it does

1. **Regression:** predicts a book's average rating on a 0–5 scale.
2. **Classification:** predicts whether a book belongs to the high-rating group (average rating >= 4.0) or the other group.
3. **FastAPI backend:** provides `/predict/regression` and `/predict/classification` POST endpoints.
4. **HTML frontend:** contains a form for each model and displays each prediction.
5. **Evaluation:** prints R² for regression and accuracy plus a classification report for classification.
6. **Model saving:** stores trained pipelines using `joblib`.

## Dataset

Download Goodbooks-10k from Kaggle:
https://www.kaggle.com/datasets/zygmunt/goodbooks-10k

Extract the archive and copy its `books.csv` file into this project's `data` folder:

```text
NovelMate_FullStack_ML/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── data/
│   └── books.csv
├── models/
└── static/
    ├── index.html
    ├── style.css
    └── script.js
```

The `books.csv` metadata file should contain `average_rating` and some of these numeric features: `books_count`, `original_publication_year`, `ratings_count`, `work_ratings_count`, `work_text_reviews_count`.

## Run on your computer

Use Python 3.10 or newer. Open a terminal in the project folder.

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Train both models

```bash
python train_models.py
```

The script prints the R² score and classification accuracy, then saves the model files in `models/`.

### 3. Start the website and API

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 in your browser. The interactive API documentation is at http://127.0.0.1:8000/docs.

## API examples

`POST /predict/regression`

```json
{
  "books_count": 1,
  "original_publication_year": 2000,
  "ratings_count": 1000,
  "work_ratings_count": 1000,
  "work_text_reviews_count": 100
}
```

`POST /predict/classification` accepts the same fields.

## Important notes

- The rating category is derived from `average_rating`: 1 means 4.0 or higher; 0 means below 4.0.
- The model metrics are printed when you train on the actual downloaded dataset. Do not report invented scores.
- The form uses example numbers; they are not an actual book's statistics.
- This is a learning project, not a production-grade recommendation system.
- Be prepared to explain each line and the model evaluation during your viva. The assignment asks students to write code themselves and understand it.
