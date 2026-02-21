# Twitter Sentiment Analysis

This repository contains an end-to-end machine learning pipeline for Twitter Sentiment Analysis. The project focuses on classifying tweets into distinct sentiment categories (Positive, Neutral, Negative) using natural language processing (NLP) techniques and evaluating multiple tree-based machine learning algorithms.

## Project Structure

The project is organized into modular Jupyter Notebooks, separating data preparation from model training and evaluation:

* **`twitter_preprocessing.ipynb`**: Handles the foundational data cleaning and preparation. This notebook covers dataset splitting (training, testing, and validation sets) and transforms the raw text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
* **`twitter_decisiontree.ipynb`**: Implements a baseline Decision Tree classifier. It includes model training, hyperparameter exploration (e.g., depth and leaf nodes), and performance evaluation using classification reports and confusion matrices.
* **`twitter_randomforest.ipynb`**: Upgrades the baseline approach using an ensemble learning method. This notebook explores how a Random Forest model can improve classification accuracy and robustness over a single decision tree.
* **`twitter_xgboost.ipynb`**: Implements a highly optimized gradient boosting framework (XGBoost). This notebook represents the most advanced modeling step in the pipeline, complete with model tuning and evaluation.

## Methodology

1.  **Data Preprocessing**: Raw tweets are cleaned and split into a standard 80/20 train/test distribution (alongside a dedicated validation set) while maintaining class stratification. 
2.  **Feature Engineering**: Textual data is converted into a structured matrix of TF-IDF features, capturing the relative importance of words across the dataset.
3.  **Model Evaluation**: Each model is evaluated on the test set using standard classification metrics:
    * Accuracy
    * Precision, Recall, and F1-Score
    * Confusion Matrices



*Each modeling notebook generates a custom confusion matrix (like the conceptual one above) to visualize the true vs. predicted sentiment classifications.*

## Requirements

To run the notebooks locally, ensure you have the following Python libraries installed:

```bash
pip install pandas matplotlib seaborn scikit-learn xgboost
```

## How to Run

1.  Clone this repository to your local machine.
2.  Place your raw Twitter dataset in the designated `datasets/` directory (or update the file paths accordingly).
3.  Run `twitter_preprocessing.ipynb` first. This will generate the `train_preprocessed.csv`, `test_preprocessed.csv`, and `validation_preprocessed.csv` files required by the models.
4.  Run the modeling notebooks (`twitter_decisiontree.ipynb`, `twitter_randomforest.ipynb`, `twitter_xgboost.ipynb`) in any order to train the algorithms and reproduce the evaluation metrics.
