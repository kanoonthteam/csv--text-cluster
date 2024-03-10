"""
Comprehensive script for processing and clustering text data with visualization options.

Dependencies:
- pandas
- numpy
- nltk
- gensim
- transformers
- scikit-learn
- matplotlib
- wordcloud
"""

# --- Imports ---
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import stringเหะ
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, SpectralClustering
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

# --- Preprocessing Function ---
def preprocess_text(text):
    """Preprocess text by converting to lowercase, removing punctuation, and excluding stopwords.
    
    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    try:
        print(f"Original text: {text[:100]}")  # Log the first 100 characters of the text

        if not isinstance(text, str) or text is None:
            # If the text is not a string or is None, return an empty string
            return ""
        stop_words = set(stopwords.words('english'))
        text = text.lower()
        words = word_tokenize(text.translate(str.maketrans('', '', string.punctuation)))
        filtered_words = [word for word in words if word not in stop_words]
        preprocessed_text = " ".join(filtered_words)
        print(f"Preprocessed text: {preprocessed_text[:100]}")  # Log the first 100 characters of the processed text
        return preprocessed_text
    except Exception as e:
        print(f"Error processing text: {e}")
        return ""


# --- Vectorization Functions ---
def vectorize_with_word2vec(text, word_vectors):
    """Vectorize text using Word2Vec model."""
    words = text.split()
    vectors = [word_vectors[word] for word in words if word in word_vectors]
    return np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(300)

def vectorize_with_bert(text, tokenizer, model):
    """Vectorize text using BERT."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    print(f"Vectorized text shape: {outputs.last_hidden_state.shape}")  # Log the shape of the vectorized output
    sentence_vector = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return sentence_vector.flatten()

# --- Clustering Functions ---
def cluster_kmeans(vectors, n_clusters=5):
    """Cluster vectors using K-Means."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vectors)
    return kmeans.labels_

def cluster_hac(vectors, n_clusters=5):
    """Cluster vectors using Hierarchical Agglomerative Clustering."""
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(vectors)
    return clustering.labels_

def cluster_dbscan(vectors, eps=0.5, min_samples=5):
    """Cluster vectors using DBSCAN."""
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(vectors)
    return clustering.labels_

def cluster_mean_shift(vectors, bandwidth=None):
    """Cluster vectors using Mean Shift."""
    clustering = MeanShift(bandwidth=bandwidth).fit(vectors)
    return clustering.labels_

def cluster_spectral(vectors, n_clusters=5):
    """Cluster vectors using Spectral Clustering."""
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=0).fit(vectors)
    return clustering.labels_

# --- Visualization Functions ---
def generate_word_clouds(text_data, labels, n_clusters=5):
    """Generate and display word clouds for each cluster."""
    for i in range(n_clusters):
        cluster_text = ' '.join(text_data[labels == i])
        if not cluster_text:
            print(f"No words to generate word cloud for cluster {i}")
            continue
        print(f"Generating word cloud for cluster {i} with text length: {len(cluster_text)}")
        wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(cluster_text)
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.title(f'Word Cloud for Cluster {i}')
        plt.show()

def plot_tsne(vectors, labels):
    """Visualize vectors using t-SNE."""
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(vectors)
    plt.figure(figsize=(16,10))
    plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('t-SNE Visualization of Text Data Clusters')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

def plot_cluster_histogram(labels):
    """Plot a histogram of cluster sizes."""
    plt.hist(labels, bins=range(min(labels), max(labels) + 2), alpha=0.7, edgecolor='black')
    plt.title('Histogram of Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Text Entries')
    plt.xticks(range(len(np.unique(labels))))
    plt.show()

# --- Main Processing Function ---
def process_and_cluster(file_path, column_name, vectorization_method='bert', clustering_method='kmeans', visualization_method='wordcloud'):
    """Process, vectorize, cluster, and visualize text data based on selected methods."""
    # Load vectorization model
    if vectorization_method == 'word2vec':
        word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

    vectors = []
    texts = []
    for batch in pd.read_csv(file_path, chunksize=5000):
        batch_texts = batch[column_name].apply(preprocess_text).tolist()
        texts.extend(batch_texts)
        if vectorization_method == 'word2vec':
            batch_vectors = np.array([vectorize_with_word2vec(text, word_vectors) for text in batch_texts])
        else:
            batch_vectors = np.array([vectorize_with_bert(text, tokenizer, model) for text in batch_texts])
        vectors.append(batch_vectors)

    all_vectors = np.vstack(vectors)
    all_texts = np.array(texts)

    # Clustering
    if clustering_method == 'kmeans':
        labels = cluster_kmeans(all_vectors)
    elif clustering_method == 'hac':
        labels = cluster_hac(all_vectors)
    elif clustering_method == 'dbscan':
        labels = cluster_dbscan(all_vectors)
    elif clustering_method == 'meanshift':
        labels = cluster_mean_shift(all_vectors)
    elif clustering_method == 'spectral':
        labels = cluster_spectral(all_vectors)
    else:
        raise ValueError("Unsupported clustering method specified.")

    # Debugging: Inspect Clustering Labels
    print(f"Unique cluster labels: {np.unique(labels)}")

    # Visualization
    if visualization_method == 'wordcloud':
        generate_word_clouds(all_texts, labels)
    elif visualization_method == 'tsne':
        plot_tsne(all_vectors, labels)
    elif visualization_method == 'histogram':
        plot_cluster_histogram(labels)

# Example usage
if __name__ == "__main__":
    process_and_cluster(
        file_path="path/to/your/file.csv",
        column_name="text_column",
        vectorization_method='bert',  # or 'word2vec'
        clustering_method='kmeans',  # 'kmeans', 'hac', 'dbscan', 'meanshift', 'spectral'
        visualization_method='wordcloud'  # 'wordcloud', 'tsne', 'histogram'
    )
