import os
import re
import numpy as np
import pandas as pd
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
import multiprocessing
from nltk.corpus import stopwords
from sklearn.metrics import f1_score

os.environ["OMP_NUM_THREADS"] = "1"

def load_data(file_path):
    return pd.read_excel(file_path)

def calculate_lengths(df):
    df['ReportText_length'] = df['ReportText'].apply(len)
    df['findings_length'] = df['findings'].apply(len)
    df['clinicaldata_length'] = df['clinicaldata'].apply(lambda x: len(x) if pd.notnull(x) else 0)
    df['ExamName_length'] = df['ExamName'].apply(len)
    df['impression_length'] = df['impression'].apply(len)
    return df

def summary_statistics(df):
    return df[['ReportText_length', 'findings_length', 'clinicaldata_length', 'ExamName_length', 'impression_length']].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])

def process_text(df):
    fields = ['ReportText', 'findings', 'clinicaldata', 'ExamName', 'impression']
    for field in fields:
        df[field] = df[field].str.lower().str.strip()
    return df

def generate_wordclouds(df, column):
    text = " ".join(t for t in df[column].dropna())
    generate_word_cloud(text, f'{column}_wordcloud.png')

def generate_word_cloud(text, filename, folder='wordcloud_images'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    wordcloud = WordCloud(background_color='white', max_words=100, contour_width=3, contour_color='steelblue', width=800, height=400).generate(text)
    path_to_save = os.path.join(folder, filename)
    wordcloud.to_file(path_to_save)

def perform_tfidf_tsne_analysis(df, folder='tsne_plots'):
    df_cleaned = df.dropna(subset=['clinicaldata'])
    texts = df_cleaned['findings'].tolist() + df_cleaned['ExamName'].tolist() + df_cleaned['impression'].tolist() + df_cleaned['clinicaldata'].tolist()
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = tfidf_vectorizer.fit_transform(texts)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200, n_jobs=1)
    try:
        X_reduced = tsne.fit_transform(X_tfidf.toarray())
    except Exception as e:
        print(f"Error during t-SNE: {e}")
        return
    plot_tsne(X_reduced, df_cleaned, folder)

def plot_tsne(X_reduced, df_cleaned, folder):
    categories = ['Findings', 'ExamName', 'Impressions', 'ClinicalData'] * len(df_cleaned)
    color_map = {'Findings': 'red', 'ExamName': 'yellow', 'Impressions': 'green', 'ClinicalData': 'blue'}
    colors = [color_map[cat] for cat in categories]
    df_plot = pd.DataFrame(X_reduced, columns=['Component 1', 'Component 2'])
    df_plot['Category'] = categories
    plt.figure(figsize=(10, 8))
    for category, color in color_map.items():
        subset = df_plot[df_plot['Category'] == category]
        plt.scatter(subset['Component 1'], subset['Component 2'], c=color, label=category, alpha=0.5, edgecolors='k')
    plt.title('t-SNE Visualization of Text Data with Category Color Coding')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(loc='best')
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(f'{folder}/tsne_plot.png')
    plt.close()

def prepare_text_data(df):
    df_cleaned = df.dropna(subset=['clinicaldata'])
    texts, field_types = [], []
    for _, row in df_cleaned.iterrows():
        texts.extend([row['findings'], row['ExamName'], row['impression'], row['clinicaldata']])
        field_types.extend([2, 0, 3, 1])
    return texts, field_types

def train_classifier(texts, field_types):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.5, min_df=2, sublinear_tf=True)
    X_tfidf = tfidf_vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)
    df_data = pd.DataFrame(X_reduced)
    df_data['field_type'] = field_types
    X_train, X_test, y_train, y_test = train_test_split(df_data.drop('field_type', axis=1), df_data['field_type'], test_size=0.2, random_state=42)
    classifier = LogisticRegression(random_state=42, class_weight='balanced')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Calculate F1 score
    return accuracy, report, f1

def clean_data(text):
    if not isinstance(text, str):
        text = ' '.join(text)  # Convert list to string if necessary
    stopwords_list = stopwords.words('english')
    text = text.lower()
    text = re.sub(r'[^\w\s]|(\d+)', '', text)
    return [word for word in text.split() if word not in stopwords_list and len(word) > 2]

def train_and_get_word2vec(texts):
    cleaned_texts = [clean_data(text) if isinstance(text, str) else clean_data(' '.join(text)) for text in texts]
    model = Word2Vec(sentences=cleaned_texts, vector_size=300, window=5, min_count=5, workers=multiprocessing.cpu_count())
    avg_vectors = average_word_vectors(model, cleaned_texts)
    return model, avg_vectors

def average_word_vectors(model, texts):
    cleaned_texts = [clean_data(text) for text in texts]
    vectors = []
    for words in cleaned_texts:
        vector = [model.wv[word] for word in words if word in model.wv]
        if vector:
            vectors.append(np.mean(vector, axis=0))
    return np.array(vectors)  # Ensure returning a numpy array

def visualize_word_vectors(avg_vectors):
    if avg_vectors.size == 0:
        print("No vectors to visualize.")
        return
    try:
        X_2d = TSNE(n_components=2, random_state=42).fit_transform(avg_vectors)
        plt.figure(figsize=(10, 10))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.5)
        plt.title('Word2Vec Visualization')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.savefig('word2vec_plot.png')
        plt.close()
    except Exception as e:
        print(f"Error in visualizing word vectors: {e}")

def train_word2vec_logistic_classifier(avg_vectors, categories):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(categories)
    X_train, X_test, y_train, y_test = train_test_split(avg_vectors, labels, test_size=0.2, random_state=42)
    logistic_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    return accuracy, report, f1

def compare_and_save_models(tfidf_f1, word2vec_f1, texts, field_types, word2vec_model):
    if tfidf_f1 > word2vec_f1:
        vectorizer = TfidfVectorizer(stop_words='english')
        X_tfidf = vectorizer.fit_transform(texts)
        model = LogisticRegression(random_state=42)
        model.fit(X_tfidf, field_types)
        joblib.dump(model, 'final_model_tfidf.pkl')
    else:
        joblib.dump(word2vec_model, 'final_model_word2vec.pkl')
