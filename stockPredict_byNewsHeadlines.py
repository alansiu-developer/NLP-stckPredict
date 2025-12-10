#Author: Alan Siu
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ========================
# 1. 取得歷史股價資料
# ========================
def get_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    使用 yfinance 取得股票收盤價，並計算次日漲跌標籤
    """
    stock = yf.download(ticker, start=start_date, end=end_date)
    if stock = stock[['Close']].copy()
    stock['Return'] = stock['Close'].pct_change()
    stock['Direction'] = np.where(stock['Return'] > 0, 1, 0)  # 1=上漲, 0=下跌或持平
    stock = stock.dropna()
    stock.reset_index(inplace=True)
    stock['Date'] = stock['Date'].dt.date
    return stock

# ========================
# 2. 讀取新聞標題資料（範例格式）
# ========================
def load_news_data(csv_path: str) -> pd.DataFrame:
    """
    預期 CSV 欄位至少包含：date, headline
    date 格式為 YYYY-MM-DD
    """
    news = pd.read_csv(csv_path)
    news['date'] = pd.to_datetime(news['date']).dt.date
    return news

# ========================
# 3. 新聞情緒分析（使用 VADER）
# ========================
def add_sentiment_scores(df_news: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    df_news['sentiment'] = df_news['headline'].apply(
        lambda x: analyzer.polarity_scores(str(x))['compound']
    )
    return df_news

# ========================
# 4. 合併股價與每日平均情緒分數
# ========================
def merge_stock_and_sentiment(stock_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    # 計算每日平均情緒
    daily_sentiment = news_df.groupby('date')['sentiment'].mean().reset_index()
    daily_sentiment.rename(columns={'sentiment': 'avg_sentiment'}, inplace=True)
    
    # 合併（以日期為鍵）
    merged = pd.merge(stock_df, daily_sentiment, left_on='Date', right_on='date', how='left')
    merged.drop(columns=['date'], inplace=True)
    
    # 填補缺失值（若某日無新聞）
    merged['avg_sentiment'].fillna(0, inplace=True)
    
    # 建立特徵：前1~3天的平均情緒（可自行擴充）
    for lag in range(1, 4):
        merged[f'avg_sentiment_lag{lag}'] = merged['avg_sentiment'].shift(lag)
    
    merged.fillna(0, inplace=True)
    
    return merged

# ========================
# 5. 訓練與評估模型
# ========================
def train_and_evaluate(df: pd.DataFrame):
    features = ['avg_sentiment', 'avg_sentiment_lag1', 'avg_sentiment_lag2', 'avg_sentiment_lag3']
    X = df[features]
    y = df['Direction']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # 時間序列不打亂
    )
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    print("=== 模型評估結果 ===")
    print(f"準確率 (Accuracy): {accuracy:.4f}")
    print("\n分類報告：")
    print(classification_report(y_test, preds, target_names=['下跌', '上漲']))
    print("混淆矩陣：")
    print(confusion_matrix(y_test, preds))
    
    # 特徵重要性
    importances = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n特徵重要性：")
    print(importances)
    
    return model, features


import matplotlib.pyplot as plt   

def plot_prediction_vs_actual(df: pd.DataFrame, model, features):
    """
    繪製整個時間範圍內的實際漲跌方向與模型預測結果
    """
    X = df[features]
    df = df.copy()
    df['Predicted_Direction'] = model.predict(X)
    
    dates = df['Date'].values
    
    # 為了讓圖更清楚，每隔 N 天顯示一個日期標籤
    sampleInterval = max(1, len(dates) // 30)  # 大約顯示 30 個日期標籤
    sampledDates = dates[::sampleInterval]
    
    plt.figure(figsize=(20, 8))
    
    # 實際方向（訓練 + 測試）
    plt.plot(dates, df['Direction'], label='實際漲跌 (1=漲, 0=跌)', color='black', alpha=0.6, linewidth=1.2)
    
    # 預測方向（全資料集預測）
    plt.plot(dates, df['Predicted_Direction'], label='模型預測漲跌', color='red', alpha=0.7, linewidth=1.8)
    
    # 可選：標記預測錯誤的地方（紅色圓點）
    errors = df[df['Direction'] != df['Predicted_Direction']]
    plt.scatter(errors['Date'], errors['Predicted_Direction'], 
                color='magenta', s=60, label=f'預測錯誤 ({len(errors)} 筆)', zorder=5)
    
    plt.title(f'{TICKER} 股票漲跌方向預測 vs 實際 (基於新聞情緒分析\n'
              f'測試集準確率: {accuracy_score(df["Direction"][len(X_train):], df["Predicted_Direction"][len(X_train):]):.1%}', 
              fontsize=16)
    plt.xlabel('日期')
    plt.ylabel('方向 (1 = 上漲, 0 = 下跌)')
    plt.yticks([0, 1], ['下跌', '上漲'])
    plt.xticks(sampledDates, rotation=45)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    print("\n開始訓練模型...")
    model, feature_cols = train_and_evaluate(final_df)
    
    # 儲存模型
    import joblib
    joblib.dump(model, f"{TICKER}_news_sentiment_model.pkl")
    print(f"\n模型已儲存為 {TICKER}_news_sentiment_model.pkl")
    
    # === 新增：繪製預測結果圖表 ===
    print("\n正在繪製預測 vs 實際漲跌圖表...")
    plot_prediction_vs_actual(final_df, model, feature_cols)
