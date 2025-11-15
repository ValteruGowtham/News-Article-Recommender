# Setup Instructions

Follow these steps to set up and run the News Article Recommender System:

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Prepare Dataset

Create a `data` folder and place your dataset:

```bash
mkdir data
```

Then move `news-article-categories.csv` into the `data` folder. The CSV should contain at least:
- `category` column (article category/label)
- `body` column (article text content)

## 3. Train the Model

```bash
python train_model.py
```

This will:
- Process the dataset
- Train the ensemble classifier
- Create recommendation matrices
- Save all models to the `models` folder

## 4. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open automatically in your browser at http://localhost:8501

## 5. Use Jupyter Notebook (Optional)

You can also run the analysis in Jupyter:

```bash
jupyter notebook code.ipynb
```

Make sure to run all cells in order. The notebook now uses relative paths and saves models automatically.

## Troubleshooting

**Issue**: Dataset not found
- **Solution**: Ensure `news-article-categories.csv` is in the `data` folder

**Issue**: Models not found when running Streamlit
- **Solution**: Run `python train_model.py` first to create the models

**Issue**: Import errors
- **Solution**: Install all dependencies with `pip install -r requirements.txt`

## Project Structure After Setup

```
News-Article-Recommender/
├── app.py
├── train_model.py
├── code.ipynb
├── requirements.txt
├── README.md
├── SETUP.md
├── data/
│   └── news-article-categories.csv  ← Your dataset
└── models/                           ← Created after training
    ├── ensemble_model.pkl
    ├── tfidf_vectorizer.pkl
    ├── tfidf_all.pkl
    ├── label_encoder.pkl
    ├── cosine_sim.pkl
    └── tfidf_matrix.pkl
```
