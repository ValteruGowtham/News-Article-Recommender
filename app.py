import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Page configuration
st.set_page_config(
    page_title="News Article Recommender",
    page_icon="📰",
    layout="wide"
)

# Title and description
st.title("📰 News Article Recommender System")
st.markdown("### Get personalized news article recommendations based on content similarity")

# Load trained models and data
@st.cache_resource
def load_models():
    """Load pre-trained models and data"""
    try:
        with open('models/ensemble_model.pkl', 'rb') as f:
            ensemble_model = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('models/tfidf_all.pkl', 'rb') as f:
            tfidf_all = pickle.load(f)
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        with open('models/cosine_sim.pkl', 'rb') as f:
            cosine_sim = pickle.load(f)
        
        news_data = pd.read_csv('data/news-article-categories.csv')
        news_data = news_data.dropna(subset=['body'])
        
        # Load TF-IDF matrix
        with open('models/tfidf_matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
        
        return ensemble_model, tfidf, tfidf_all, label_encoder, cosine_sim, news_data, tfidf_matrix
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found. Please run train_model.py first to train and save the models.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

# Load models
ensemble_model, tfidf, tfidf_all, label_encoder, cosine_sim, news_data, tfidf_matrix = load_models()

st.success(f"✅ Successfully loaded {len(news_data)} articles from {len(label_encoder.classes_)} categories")

# Sidebar for navigation
st.sidebar.header("Navigation")
option = st.sidebar.radio(
    "Choose an option:",
    ["🏠 Home", "🔍 Find Similar Articles", "✍️ Recommend by Text", "📊 Article Classification", "📈 Dataset Statistics"]
)

# Recommendation functions
def recommend_articles(article_index, top_n=5):
    """Recommends top N similar articles based on cosine similarity."""
    similarity_scores = list(enumerate(cosine_sim[article_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_similar = similarity_scores[1:top_n+1]
    
    article_indices = [i[0] for i in top_similar]
    similarity_values = [i[1] for i in top_similar]
    
    recommendations = news_data.iloc[article_indices].copy()
    recommendations['similarity_score'] = similarity_values
    
    return recommendations

def recommend_by_text(input_text, top_n=5):
    """Recommends articles based on input text."""
    input_tfidf = tfidf_all.transform([input_text])
    similarity_scores = cosine_similarity(input_tfidf, tfidf_matrix)[0]
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    recommendations = news_data.iloc[top_indices].copy()
    recommendations['similarity_score'] = similarity_scores[top_indices]
    
    return recommendations

def classify_article(text):
    """Classify an article using the trained ensemble model."""
    text_tfidf = tfidf.transform([text])
    prediction = ensemble_model.predict(text_tfidf)[0]
    probabilities = ensemble_model.predict_proba(text_tfidf)[0]
    
    category = label_encoder.inverse_transform([prediction])[0]
    confidence = probabilities[prediction]
    
    return category, confidence, probabilities

# Main content based on navigation
if option == "🏠 Home":
    st.header("Welcome to News Article Recommender!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Features")
        st.markdown("""
        - **Find Similar Articles**: Get recommendations based on existing articles
        - **Text-Based Recommendations**: Enter custom text to find relevant articles
        - **Article Classification**: Classify new articles into categories
        - **Dataset Statistics**: Explore the news article dataset
        """)
    
    with col2:
        st.subheader("🔬 Technology Stack")
        st.markdown("""
        - **Machine Learning**: Ensemble of Naive Bayes, Logistic Regression, and Neural Network
        - **NLP**: TF-IDF vectorization for text processing
        - **Similarity**: Cosine similarity for recommendations
        - **Interface**: Streamlit for interactive web app
        """)
    
    st.info("👈 Use the sidebar to navigate through different features")

elif option == "🔍 Find Similar Articles":
    st.header("Find Similar Articles")
    st.markdown("Select an article to get recommendations based on content similarity")
    
    # Article selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        categories = ['All'] + list(news_data['category'].unique())
        selected_category = st.selectbox("Filter by category:", categories)
    
    # Filter articles
    if selected_category == 'All':
        filtered_data = news_data
    else:
        filtered_data = news_data[news_data['category'] == selected_category]
    
    with col2:
        num_recommendations = st.slider("Number of recommendations:", 1, 10, 5)
    
    # Display article selection
    article_options = [f"Article {idx}: {row['category']}" for idx, row in filtered_data.iterrows()]
    selected_article = st.selectbox("Select an article:", article_options)
    
    if selected_article:
        article_idx = int(selected_article.split(":")[0].replace("Article ", ""))
        
        # Display selected article
        st.subheader("📄 Selected Article")
        selected_row = news_data.loc[article_idx]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Category:** {selected_row['category']}")
            with st.expander("View article content"):
                st.write(selected_row['body'][:500] + "..." if len(str(selected_row['body'])) > 500 else selected_row['body'])
        
        # Get recommendations
        if st.button("🔍 Get Recommendations", type="primary"):
            with st.spinner("Finding similar articles..."):
                recommendations = recommend_articles(article_idx, top_n=num_recommendations)
                
                st.subheader(f"📚 Top {num_recommendations} Recommended Articles")
                
                for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                    with st.expander(f"#{idx} - {row['category']} (Similarity: {row['similarity_score']:.4f})"):
                        st.markdown(f"**Category:** {row['category']}")
                        st.markdown(f"**Similarity Score:** {row['similarity_score']:.4f}")
                        st.markdown(f"**Content Preview:**")
                        st.write(str(row['body'])[:300] + "...")

elif option == "✍️ Recommend by Text":
    st.header("Recommend Articles by Custom Text")
    st.markdown("Enter any text to find the most relevant news articles")
    
    num_recommendations = st.slider("Number of recommendations:", 1, 10, 5)
    
    # Text input
    input_text = st.text_area(
        "Enter your text:",
        height=150,
        placeholder="Type or paste article text, search query, or any topic you're interested in..."
    )
    
    # Sample texts
    with st.expander("💡 Try sample texts"):
        if st.button("Sample: Technology"):
            input_text = "Artificial intelligence and machine learning are transforming the tech industry"
        if st.button("Sample: Sports"):
            input_text = "The championship game was intense as both teams fought for victory"
        if st.button("Sample: Business"):
            input_text = "Stock markets showed volatility as investors reacted to economic indicators"
    
    if st.button("🔍 Find Articles", type="primary") and input_text:
        with st.spinner("Searching for relevant articles..."):
            recommendations = recommend_by_text(input_text, top_n=num_recommendations)
            
            st.subheader(f"📚 Top {num_recommendations} Relevant Articles")
            
            for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                with st.expander(f"#{idx} - {row['category']} (Similarity: {row['similarity_score']:.4f})"):
                    st.markdown(f"**Category:** {row['category']}")
                    st.markdown(f"**Similarity Score:** {row['similarity_score']:.4f}")
                    st.markdown(f"**Content Preview:**")
                    st.write(str(row['body'])[:300] + "...")

elif option == "📊 Article Classification":
    st.header("Classify News Articles")
    st.markdown("Enter article text to predict its category using the trained ensemble model")
    
    # Text input for classification
    article_text = st.text_area(
        "Enter article text:",
        height=200,
        placeholder="Paste or type the article content here..."
    )
    
    if st.button("🎯 Classify Article", type="primary") and article_text:
        with st.spinner("Classifying article..."):
            category, confidence, probabilities = classify_article(article_text)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"**Predicted Category:** {category}")
                st.metric("Confidence", f"{confidence*100:.2f}%")
            
            with col2:
                st.subheader("Category Probabilities")
                prob_df = pd.DataFrame({
                    'Category': label_encoder.classes_,
                    'Probability': probabilities
                }).sort_values('Probability', ascending=False)
                
                st.dataframe(prob_df, hide_index=True)

elif option == "📈 Dataset Statistics":
    st.header("Dataset Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Articles", len(news_data))
    with col2:
        st.metric("Categories", len(news_data['category'].unique()))
    with col3:
        avg_length = news_data['body'].str.len().mean()
        st.metric("Avg Article Length", f"{avg_length:.0f} chars")
    
    # Category distribution
    st.subheader("📊 Articles by Category")
    category_counts = news_data['category'].value_counts()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.bar_chart(category_counts)
    
    with col2:
        st.dataframe(
            pd.DataFrame({
                'Category': category_counts.index,
                'Count': category_counts.values
            }),
            hide_index=True
        )
    
    # Sample articles
    st.subheader("📰 Sample Articles")
    sample_size = st.slider("Number of samples to display:", 1, 10, 5)
    sample_articles = news_data.sample(n=sample_size)
    
    for idx, (_, row) in enumerate(sample_articles.iterrows(), 1):
        with st.expander(f"Sample {idx}: {row['category']}"):
            st.write(str(row['body'])[:400] + "...")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | News Article Recommender System</p>
    </div>
    """,
    unsafe_allow_html=True
)
