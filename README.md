
Twitter Sentiment Analysis using NLP and Machine Learning  

This project classifies Twitter sentiments as positive, negative, or neutral using Natural Language Processing (NLP) and machine learning techniques. It preprocesses text data with tokenization, stopword removal, special character cleaning, and lemmatization. A Multinomial Naive Bayes model is trained on numerical features generated via CountVectorizer for accurate sentiment prediction.  

Features  
Text Preprocessing: Tokenization, stopword removal, special character cleaning, and lemmatization.  
Vectorization: Text converted to numerical data using CountVectorizer.  
Model: Multinomial Naive Bayes for sentiment classification.  
Dataset  
Dataset: [Twitter Sentiment Dataset](https://raw.githubusercontent.com/suhasmaddali/Twitter-Sentiment-Analysis/refs/heads/main/train.csv).  
Libraries  
NLTK: Text preprocessing.  
Pandas: Data handling.  
Scikit-learn: Feature extraction, model training, and evaluation.  

Install the required dependencies:
pip install nltk pandas scikit-learn
import nltk
nltk.download('punkt')
nltk.download('stopwords')
python sentiment_analysis.py



