# NLP-stckPredict
Stock Prediction By News Headlines
Predict next-day stock price direction (up/down) using news headlines sentiment analysis and machine learning.

# Stock Prediction by News Headlines Sentiment

A lightweight, end-to-end project that predicts whether a stock will go **up or down the next trading day** based only on daily news headlines sentiment.

### Features
- Fetches historical stock prices with `yfinance`
- Performs sentiment analysis on news headlines using VADER (compound score)
- Builds lagged sentiment features (today + past 1–3 days)
- Trains a Random Forest classifier with scikit-learn
- Evaluates accuracy, shows classification report and confusion matrix
- Generates 3 beautiful visualizations:
  1. Actual vs Predicted price direction over time
  2. Cumulative returns comparison (strategy vs buy-and-hold)
  3. Monthly accuracy heatmap
- Saves the trained model with joblib for future use

### Requirements & Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Stock-Prediction-by-News-Headlines-Sentiment.git
cd Stock-Prediction-by-News-Headlines-Sentiment

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install pandas numpy scikit-learn yfinance vaderSentiment matplotlib seaborn joblib


Quick Start

Prepare a CSV file named news_headlines.csv with at least two columns:csvdate,headline
2020-01-15,Apple reports record quarterly revenue
2020-01-15,Analysts raise AAPL price target to $400
...


Edit the parameters at the bottom of stock_prediction_by_news_headlines.py if needed:PythonTICKER = "AAPL"          # change to any stock or crypto
NEWS_CSV_PATH = "news_headlines.csv"

Run the script:
Bash
python stock_prediction_by_news_headlines.py

<img width="1601" height="890" alt="outpt" src="https://github.com/user-attachments/assets/78580dd9-2f4e-4701-b70c-9e4480b7c25c" />

