# News Article Recommender System

A machine learning-powered news article recommendation system that uses Natural Language Processing and ensemble learning to classify articles and provide personalized recommendations.

## 🌟 Features

- **Article Classification**: Automatically categorize news articles using an ensemble of ML models
- **Content-Based Recommendations**: Find similar articles based on content similarity
- **Text Search**: Get article recommendations from custom text input
- **Interactive Web Interface**: User-friendly Streamlit dashboard
- **High Accuracy**: Ensemble model combining Naive Bayes, Logistic Regression, and Neural Network

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ValteruGowtham/News-Article-Recommender.git
cd News-Article-Recommender
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Create a `data` folder in the project root
   - Place your `news-article-categories.csv` file inside the `data` folder
   - The CSV should have at least these columns: `category`, `body`

### Training the Model

Run the training script to train and save all models:

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train the ensemble classification model
- Create TF-IDF vectors for recommendations
- Calculate cosine similarity matrix
- Save all models in the `models` folder

### Running the Streamlit App

After training, launch the interactive web app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Project Structure

```
News-Article-Recommender/
├── app.py                          # Streamlit web application
├── train_model.py                  # Model training script
├── code.ipynb                      # Jupyter notebook for experimentation
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   └── news-article-categories.csv # Dataset (you need to add this)
└── models/                         # Saved models (created after training)
    ├── ensemble_model.pkl
    ├── tfidf_vectorizer.pkl
    ├── tfidf_all.pkl
    ├── label_encoder.pkl
    ├── cosine_sim.pkl
    └── tfidf_matrix.pkl
```

## 🔬 Technical Details

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Remove articles with missing content
   - Encode category labels
   - 80-20 train-test split with stratification

2. **Feature Extraction**
   - TF-IDF vectorization (5000 features)
   - English stop words removal
   - Captures word importance across documents

3. **Classification Models**
   - **Multinomial Naive Bayes**: Fast, effective for text classification
   - **Logistic Regression**: Robust linear classifier
   - **Multi-Layer Perceptron**: Neural network for complex patterns
   - **Ensemble**: Soft voting classifier combining all three

4. **Recommendation System**
   - Content-based filtering using cosine similarity
   - Compares TF-IDF vectors of articles
   - Returns top-N most similar articles

### Evaluation Metrics

- Accuracy Score
- Precision, Recall, F1-Score per category
- Classification Report

## 🎯 Using the App

### 1. Find Similar Articles
- Browse articles by category
- Select any article to see recommendations
- Adjust the number of recommendations (1-10)

### 2. Recommend by Text
- Enter custom text or search queries
- System finds most relevant articles
- Try sample texts for different topics

### 3. Article Classification
- Paste article content
- Get predicted category with confidence score
- View probability distribution across all categories

### 4. Dataset Statistics
- View total articles and categories
- Explore category distribution
- Sample random articles

## 📝 Example Usage

### In Python (using saved models):

```python
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load models
with open('models/tfidf_all.pkl', 'rb') as f:
    tfidf_all = pickle.load(f)

with open('models/tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

# Get recommendations for custom text
input_text = "Breaking news about technology and innovation"
input_tfidf = tfidf_all.transform([input_text])
similarity_scores = cosine_similarity(input_tfidf, tfidf_matrix)[0]
top_5_indices = similarity_scores.argsort()[-5:][::-1]

# Load data and display recommendations
news_data = pd.read_csv('data/news-article-categories.csv')
print(news_data.iloc[top_5_indices])
```

## 🛠️ Development

### Using Jupyter Notebook

Open `code.ipynb` to:
- Experiment with different models
- Tune hyperparameters
- Visualize results
- Perform exploratory data analysis

### Customization

- Adjust TF-IDF features: Modify `max_features` parameter
- Change train-test split: Modify `test_size` in train_test_split
- Add more models: Extend the ensemble with additional classifiers
- Improve neural network: Tune `hidden_layer_sizes` and `max_iter`

## 📈 Future Enhancements

- [ ] Hybrid recommendation (content + collaborative filtering)
- [ ] Real-time article updates
- [ ] User preference learning
- [ ] Multilingual support
- [ ] API endpoints for integration
- [ ] Docker containerization
- [ ] Performance optimization for large datasets

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**ValteruGowtham**
- GitHub: [@ValteruGowtham](https://github.com/ValteruGowtham)

## 🙏 Acknowledgments

- Scikit-learn for machine learning tools
- Streamlit for the web framework
- The open-source community

---

**Note**: Make sure to add your dataset (`news-article-categories.csv`) to the `data` folder before running the training script.
