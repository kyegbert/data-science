# Time Series Forecasting with Chicago Tree Trim Requests

## Project Description
Chicago's 311 system allows city residents to request city-provided services, such as patching potholes, fixing non-functioning street lights or trimming parkway trees of broken or low-hanging branches. These trees, which are located on city property which sits between the sidewalk and the curb, line the streets of Chicago. Between the years 2013-2017, the 311 system received, on average, approximately 38,000 tree trim requests per year. The volume of tree trim requests that are opened in the winter are relatively low in winter and typically peak in the summers months. Closed requests, indicating the trim has been completed, vary more across time than opened requests.

Using time series forecasting, predict the expected volume of 311 tree trim opened requests and closed requests between November 2018 - October 2019.

### Methods Used
* Data Wrangling
* Exploratory Data Analysis
* Data Visualization
* Grid Search
* Training and Test Sets
* Time Series Forecasting
* Seasonal ARIMA

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).

### Requirements

1. Python 3.6
2. Jupyter Notebooks
2. Install project requirements using `pip install -r requirements.txt`.

### Notebooks Execution Order

Run the notebooks in the following order. The data cleaning tasks were split into multiple notebooks to keep the notebook size manageable.

1. data-wrangling-optimize-verify.ipynb
2. data-wrangling-deduplicate.ipynb
3. exploratory-data-analysis.ipynb
4. forecasting-opened-requests.ipynb
5. forecasting-closed-requests.ipynb
6. summary.ipynb