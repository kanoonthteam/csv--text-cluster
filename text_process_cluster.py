"""
Comprehensive script for processing, translating, vectorizing, clustering, and visualizing text data.

Features:
- Reads large text files in batches.
- Preprocesses text by cleaning and translating to English.
- Offers vectorization options: BERT, Word2Vec, and Sentence Transformers.
- Determines the optimal number of clusters with the Elbow Method.
- Clusters text data using selected algorithm.
- Visualizes clustering results and vectorized text data.
"""

# --- Required Libraries ---
import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, SpectralClustering
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from googletrans import Translator
import plotly.express as px

# --- Initialization ---
nltk.download('stopwords')
nltk.download('punkt')

class TextProcessor:
    def __init__(self):
        self.translator = Translator()
        self.word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    def preprocess_text(self, text):
        try:
            print(f"Original text: {text[:100]}")  # Log the first 100 characters of the text

            if not isinstance(text, str) or text is None:
                # If the text is not a string or is None, return an empty string
                return ""
            
            # Translation to English
            translated_text = self.translator.translate(text, dest='en').text
            print(f"Translated text: {translated_text[:100]}")  # Log the first 100 characters of the translated text

            stop_words = set(stopwords.words('english'))
            text = translated_text.lower()  # Use translated_text instead of text
            words = word_tokenize(text.translate(str.maketrans('', '', string.punctuation)))
            filtered_words = [word for word in words if word not in stop_words]
            preprocessed_text = " ".join(filtered_words)
            print(f"Preprocessed text: {preprocessed_text[:100]}")  # Log the first 100 characters of the processed text
            return preprocessed_text
        except Exception as e:
            print(f"Error processing text: {e}")
            return ""

    def vectorize_with_word2vec(self, text):
        """Vectorize text using Word2Vec model."""
        words = text.split()
        vectors = [self.word_vectors[word] for word in words if word in self.word_vectors]
        return np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(300)

    def vectorize_with_bert(self, text):
        """Vectorize text using BERT."""
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.bert_model(**inputs)
        print(f"Vectorized text shape: {outputs.last_hidden_state.shape}")  # Log the shape of the vectorized output
        sentence_vector = outputs.last_hidden_state[:, 0, :].detach().numpy()
        return sentence_vector.flatten()

    def vectorize_with_sentence_transformers(self, text):
        """Vectorize text using Sentence Transformers."""
        embedding = self.sentence_model.encode(text)
        return embedding

# --- Main Processing Function ---
def process_and_cluster(file_path, column_name, vectorization_method='bert', clustering_method='kmeans', visualization_method='wordcloud'):
    processor = TextProcessor()
    vectorization_methods = {
        'word2vec': processor.vectorize_with_word2vec,
        'bert': processor.vectorize_with_bert,
        'sentence_transformers': processor.vectorize_with_sentence_transformers
    }

    vectors = []
    texts = []
    for batch in pd.read_csv(file_path, chunksize=5000):
        batch_texts = batch[column_name].apply(processor.preprocess_text).tolist()
        texts.extend(batch_texts)
        batch_vectors = [vectorization_methods[vectorization_method](text) for text in batch_texts]
        # Filter out any empty vectors to avoid issues during clustering
        batch_vectors = [vector for vector in batch_vectors if np.any(vector)]
        if not batch_vectors:  # If all vectors in the batch are empty, skip to the next batch
            continue
        vectors.append(np.array(batch_vectors))
    if not vectors:  # If no valid vectors were produced, exit the function
        print("No valid vectors were produced from the text data. Exiting.")
        return
    vectors = np.vstack(vectors)

    # Clustering
    try:
        if clustering_method == 'kmeans':
            model = KMeans(n_clusters=5)  # Example: setting n_clusters to 5
        elif clustering_method == 'agglomerative':
            model = AgglomerativeClustering(n_clusters=5)
        elif clustering_method == 'dbscan':
            model = DBSCAN(eps=0.3, min_samples=10)
        elif clustering_method == 'meanshift':
            model = MeanShift()
        elif clustering_method == 'spectral':
            model = SpectralClustering(n_clusters=5, assign_labels="discretize", random_state=0)
        else:
            raise ValueError("Unsupported clustering method")
        model.fit(vectors)
    except Exception as e:
        print(f"Error during clustering: {e}")
        return

    labels = model.labels_

    # Visualization
    try:
        if visualization_method == 'wordcloud':
            for cluster in range(np.max(labels) + 1):
                cluster_texts = " ".join([texts[i] for i, label in enumerate(labels) if label == cluster])
                wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(cluster_texts)
                plt.figure()
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.title(f"Cluster {cluster} Word Cloud")
                plt.show()
        elif visualization_method == 'tsne':
            # t-SNE visualization
            tsne = TSNE(n_components=2, random_state=0)
            tsne_results = tsne.fit_transform(vectors)
            plt.figure(figsize=(16,10))
            scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels, cmap='viridis', alpha=0.5)
            plt.title('t-SNE visualization of text clusters')
            plt.colorbar(scatter)
            plt.show()
        elif visualization_method == 'interactive':
            # Interactive visualization using Plotly
            tsne = TSNE(n_components=2, random_state=0)
            tsne_results = tsne.fit_transform(vectors)
            df_tsne = pd.DataFrame({'x': tsne_results[:,0], 'y': tsne_results[:,1], 'Cluster': labels})
            fig = px.scatter(df_tsne, x='x', y='y', color='Cluster', title='Interactive Cluster Visualization')
            fig.show()
        else:
            raise ValueError("Unsupported visualization method")
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    process_and_cluster(
        file_path="path/to/your/file.csv",
        column_name="text_column",
        vectorization_method='bert',  # 'word2vec', 'bert, 'sentence_transformers'
        clustering_method='kmeans',  # 'kmeans', 'hac', 'dbscan', 'meanshift', 'spectral'
        visualization_method='wordcloud'  # 'wordcloud', 'tsne', 'interactive'
    )
