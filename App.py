"""
Streamlit Stock Analysis Tool
- Supports multiple ticker input
- Fetches historical data with yfinance
- Computes technical indicators
- Displays summary stats and plots
- Interactive web app interface

Install dependencies:
  pip install streamlit yfinance pandas numpy matplotlib seaborn

Run:
  streamlit run stock_analysis_streamlit.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')

def fetch_stock_data(ticker, period='1y', interval='1d'):
    """
    Fetch historical data for a stock ticker from Yahoo Finance.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data fetched for {ticker}. Please check the ticker symbol and internet connection.")
    return df

def compute_technical_indicators(df):
    """
    Add technical indicators columns to dataframe.
    """
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))

    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['Bollinger_High'] = sma20 + (std20 * 2)
    df['Bollinger_Low'] = sma20 - (std20 * 2)

    return df

def calculate_return_stats(df):
    """
    Calculate daily returns, cumulative return, and annualized volatility.
    """
    df['Daily Return'] = df['Close'].pct_change()
    cumulative_return = (df['Daily Return'] + 1).prod() - 1
    trading_days = 252
    annualized_vol = df['Daily Return'].std() * np.sqrt(trading_days)
    return {
        'Cumulative Return': cumulative_return,
        'Annualized Volatility': annualized_vol
    }

def plot_stock_data(df, ticker):
    """
    Plot closing price, moving averages, RSI, and Bollinger Bands.
    Return the matplotlib figure.
    """
    fig, axs = plt.subplots(3,1, figsize=(12,10), sharex=True)
    
    ax1 = axs[0]
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax1.plot(df.index, df['SMA20'], label='SMA 20', linestyle='--', color='orange')
    ax1.plot(df.index, df['SMA50'], label='SMA 50', linestyle='--', color='red')
    ax1.plot(df.index, df['EMA20'], label='EMA 20', linestyle=':', color='green')
    ax1.plot(df.index, df['EMA50'], label='EMA 50', linestyle=':', color='purple')
    ax1.fill_between(df.index, df['Bollinger_Low'], df['Bollinger_High'], color='grey', alpha=0.2, label='Bollinger Bands')
    ax1.set_title(f"{ticker} Price and Moving Averages with Bollinger Bands")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc='upper left')

    ax2 = axs[1]
    ax2.plot(df.index, df['RSI14'], label='RSI 14', color='magenta')
    ax2.axhline(70, color='red', linestyle='--')
    ax2.axhline(30, color='green', linestyle='--')
    ax2.set_title(f"{ticker} RSI (14)")
    ax2.set_ylabel("RSI")
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')

    ax3 = axs[2]
    sns.histplot(df['Daily Return'].dropna(), bins=50, kde=True, color='cyan', ax=ax3)
    ax3.set_title(f"{ticker} Daily Returns Distribution")
    ax3.set_xlabel("Daily Return")
    ax3.set_ylabel("Frequency")

    plt.tight_layout()
    return fig

def main():
    st.title("🚀 Advanced Stock Analysis Tool (like BlackRock & Lo Keng Hong)")
    st.markdown("""
    Enter one or multiple stock ticker symbols separated by commas.
    Supported ticker formats e.g. AAPL, MSFT, GOTO.JK.
    \nSelect the data period to analyze.
    """)

    tickers_input = st.text_input("Enter stock ticker(s)", value="AAPL,MSFT")
    period = st.selectbox("Select data period", options=['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'], index=3)
    interval = st.selectbox("Select data interval", options=['1d', '1wk', '1mo'], index=0)

    if st.button("Run Analysis"):
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]
        if not tickers:
            st.error("Please enter at least one valid ticker symbol.")
            return

        for ticker in tickers:
            st.header(f"📈 Analysis for {ticker}")
            try:
                df = fetch_stock_data(ticker, period=period, interval=interval)
                df = compute_technical_indicators(df)
                stats = calculate_return_stats(df)

                st.subheader("Summary Statistics")
                st.write(f"**Cumulative Return over period:** {stats['Cumulative Return']*100:.2f}%")
                st.write(f"**Annualized Volatility:** {stats['Annualized Volatility']*100:.2f}%")

                fig = plot_stock_data(df, ticker)
                st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Error processing {ticker}: {e}")

if __name__ == "__main__":
    main()

