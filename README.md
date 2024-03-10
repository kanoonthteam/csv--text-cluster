# Text Processing, Clustering, and Visualization Project

This project offers a comprehensive pipeline for analyzing large text datasets. It automates the process of reading text data in batches, preprocessing (cleaning and preparing text), vectorizing (converting text to numerical form using methods like BERT or Word2Vec), clustering (organizing text into meaningful groups using various algorithms), and visualizing the results to derive insights.

## Features
Batch Processing: Efficiently handles large files.
Preprocessing: Cleans text data for analysis.
Vectorization: Supports BERT and Word2Vec for converting text to vectors.
Clustering: Includes K-Means, HAC, DBSCAN, Mean Shift, and Spectral Clustering.
Visualization: Generates word clouds, t-SNE plots, and histograms to visualize clusters.

## Dependencies

This project depends on several Python libraries, including pandas, numpy, nltk, gensim, transformers, scikit-learn, matplotlib, and wordcloud. Ensure you have Python 3.6 or later installed, then install these dependencies by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Usage

To use the script, you need to specify the path to your CSV file and the column name containing the text data. The script allows you to configure the vectorization method (`bert` or `word2vec`), the clustering method (`kmeans`, `hac`, `dbscan`, `meanshift`, `spectral`), and the visualization method (`wordcloud`, `tsne`, `histogram`) directly in the code.

Example customization in the script:

```python
if __name__ == "__main__":
    process_and_cluster(
        file_path="path/to/your/file.csv",
        column_name="text_column",
        vectorization_method='bert',  # or 'word2vec'
        clustering_method='kmeans',  # 'kmeans', 'hac', 'dbscan', 'meanshift', 'spectral'
        visualization_method='wordcloud'  # 'wordcloud', 'tsne', 'histogram'
    )
```

Run the script with the following command:

```bash
python text_process_cluster.py
```

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues to discuss proposed changes or enhancements.

## License
Distributed under the MIT License. See LICENSE for more information.

