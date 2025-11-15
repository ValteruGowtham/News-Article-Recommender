"""
Train and save the news article classification and recommendation models.
Run this script before using the Streamlit app.
"""

import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

def main():
    print("=" * 70)
    print("News Article Recommender - Model Training")
    print("=" * 70)
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load data
    print("\n[1/8] Loading dataset...")
    data_path = 'data/news-article-categories.csv'
    
    if not os.path.exists(data_path):
        print(f"\n⚠️  Error: Dataset not found at '{data_path}'")
        print("Please place your 'news-article-categories.csv' file in the 'data' folder")
        return
    
    news_data = pd.read_csv(data_path)
    print(f"✓ Loaded {len(news_data)} articles")
    
    # Data preprocessing
    print("\n[2/8] Preprocessing data...")
    initial_count = len(news_data)
    news_data = news_data.dropna(subset=['body'])
    print(f"✓ Removed {initial_count - len(news_data)} articles with missing content")
    print(f"✓ Final dataset: {len(news_data)} articles")
    
    # Encode labels
    print("\n[3/8] Encoding labels...")
    label_encoder = LabelEncoder()
    news_data['category_encoded'] = label_encoder.fit_transform(news_data['category'])
    print(f"✓ Encoded {len(label_encoder.classes_)} categories: {', '.join(label_encoder.classes_)}")
    
    # Split data
    print("\n[4/8] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        news_data['body'],
        news_data['category_encoded'],
        test_size=0.2,
        random_state=42,
        stratify=news_data['category_encoded']
    )
    print(f"✓ Training set: {len(X_train)} articles")
    print(f"✓ Test set: {len(X_test)} articles")
    
    # TF-IDF vectorization for classification
    print("\n[5/8] Creating TF-IDF features for classification...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"✓ Training TF-IDF shape: {X_train_tfidf.shape}")
    print(f"✓ Testing TF-IDF shape: {X_test_tfidf.shape}")
    
    # Train ensemble model
    print("\n[6/8] Training ensemble classification model...")
    print("   Models: Naive Bayes, Logistic Regression, Neural Network")
    
    nb_model = MultinomialNB()
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    nn_model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=42, verbose=False)
    
    ensemble_model = VotingClassifier(
        estimators=[
            ('naive_bayes', nb_model),
            ('logistic_regression', logistic_model),
            ('neural_network', nn_model)
        ],
        voting='soft'
    )
    
    ensemble_model.fit(X_train_tfidf, y_train)
    print("✓ Ensemble model trained successfully")
    
    # Evaluate model
    print("\n[7/8] Evaluating model...")
    y_pred = ensemble_model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ Model Accuracy: {accuracy*100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Create recommendation system components
    print("\n[8/8] Creating recommendation system components...")
    tfidf_all = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf_all.fit_transform(news_data['body'])
    print(f"✓ TF-IDF matrix for recommendations: {tfidf_matrix.shape}")
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"✓ Cosine similarity matrix: {cosine_sim.shape}")
    
    # Save all models and data
    print("\n[9/9] Saving models and data...")
    
    with open('models/ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble_model, f)
    print("✓ Saved ensemble_model.pkl")
    
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    print("✓ Saved tfidf_vectorizer.pkl")
    
    with open('models/tfidf_all.pkl', 'wb') as f:
        pickle.dump(tfidf_all, f)
    print("✓ Saved tfidf_all.pkl")
    
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("✓ Saved label_encoder.pkl")
    
    with open('models/cosine_sim.pkl', 'wb') as f:
        pickle.dump(cosine_sim, f)
    print("✓ Saved cosine_sim.pkl")
    
    with open('models/tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    print("✓ Saved tfidf_matrix.pkl")
    
    print("\n" + "=" * 70)
    print("✅ Training completed successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run the Streamlit app: streamlit run app.py")
    print("2. Open your browser and interact with the recommendation system")
    print("\n")

if __name__ == "__main__":
    main()
