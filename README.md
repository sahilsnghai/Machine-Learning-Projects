# Machine Learning Projects

This repository contains two machine learning projects implemented in Python using Jupyter notebooks:

## 1. Movie Recommendation System

Located in the `recommendation_system` directory, this project builds a content-based movie recommendation system using the TMDB 5000 movies dataset.

- Loads movie metadata and credits data.
- Processes genres, keywords, cast, and crew information to create descriptive tags for each movie.
- Applies natural language processing techniques such as lemmatization to normalize text data.
- Uses CountVectorizer to convert text tags into feature vectors.
- Computes cosine similarity between movies based on their tags.
- Provides a function to recommend movies similar to a given movie title.

The main notebook is [`recommendation_system/movies_suggest.ipynb`](recommendation_system/movies_suggest.ipynb).

## 2. Spam Detector

Located in the `spam_detector` directory, this project builds a spam email classifier using a labeled email dataset.

- Loads and explores the email dataset.
- Performs text preprocessing including lowercasing, tokenization, removal of special characters, stop words, and lemmatization.
- Extracts features using CountVectorizer and TF-IDF vectorizer.
- Trains multiple classification models including Gaussian Naive Bayes, Multinomial Naive Bayes, Bernoulli Naive Bayes, Support Vector Machines, Logistic Regression, Decision Trees, Random Forests, and ensemble methods.
- Evaluates models based on accuracy and precision.
- Implements voting and stacking classifiers to improve performance.

The main notebook is [`spam_detector/spam_ham_model.ipynb`](spam_detector/spam_ham_model.ipynb).

## Requirements

- Python 3.x
- Jupyter Notebook
- Required Python packages (can be installed via `pip`):
  - pandas
  - numpy
  - scikit-learn
  - spacy
  - matplotlib
  - seaborn
  - nltk
  - xgboost

## Usage

1. Clone the repository.
2. Install the required packages.
3. Download the datasets and place them in the appropriate `../archive/` directories as expected by the notebooks.
4. Open the notebooks in Jupyter and run the cells to explore and use the models.

## License

This project is licensed under the MIT License.