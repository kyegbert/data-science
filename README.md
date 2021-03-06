# Data Science Portfolio by Kim Y. Egbert

## Description

This repo contains data science projects in Python using Jupyter Notebooks. Each of the repo subdirectories contain a stand-alone data science project.

## Projects

### Time Series Forecasting

[Time Series Forecasting with Chicago Tree Trim Requests](./notebooks/chicago-tree-trims-time-series-forecasting)  
Using the Seasonal ARIMA time series forecasting model, predict the expected volume of Chicago's 311 tree trim opened requests and closed requests between November 2018 - October 2019. This end-to-end project starts with raw, messy data and ends with forecasts. Techniques and technologies used in this project include Python, Pandas, Jupyter Notebooks, structured data, machine learning, time series forecasting, Seasonal ARIMA, StatsModels, Matplotlib, data wrangling, data visualization, exploratory data analysis, training and testing sets, and grid search.

### Binary Classification

[Don't Eat that Mushroom! Classifying Mushrooms as Poisonous or Non-Poisonous](./notebooks/mushrooms-classification)  
Using binary classification algorithms, identify the physical characteristics of mushrooms that are most predictive of the mushroom being poisonous or non-poisonous.  Techniques and technologies used in this project include Python, Pandas, Jupyter Notebooks, structured data, machine learning, Decision Trees, K-Nearest Neighbors, Random Forest, Binary Logistic Regression, scikit-learn, Matplotlib, hyperparameter tuning, grid search, cross validation, regularization, data visualization and evaluation metrics (Accuracy, Precision, Recall and ROC AUC).

### Natural Language Processing

[Detecting Literary Themes in James Joyce's Works Using Natural Language Processing](./notebooks/james-joyce-natural-language-processing)  
Using the Natural Language Processing techniques of sentiment analysis, TF-IDF weighting, topic modeling and bigram networks to accurately detect the literary themes of the early twentieth century writer James Joyce. Techniques and technologies used in this project include Python, Pandas, Jupyter Notebooks, machine learning, unstructured data, data preprocessing, data visualization, Natural Language Processing, sentiment analysis, topic modeling, TF-IDF, Latent Dirichlet Allocation, Non-negative Matrix Factorization, Markov chain, NLTK, scikit-learn, Matplotlib, and network graphs.

## Requirements

1. Docker

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. In the `data-science` directory, run the `make start_jupyter` command to start the jupyter server.
3. Copy the displayed localhost url from your terminal, including the token, and paste into a browser.
4. Navigate to your desired notebook.
5. Execute `Ctrl+C` to stop the Jupyter server and exit the Docker container.

## License

MIT License. Copyright (c) 2020 Kim Y. Egbert, except where otherwise noted in additional License files.
