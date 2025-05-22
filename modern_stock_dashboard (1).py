
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

def fetch_and_prepare(ticker, period='1y', interval='1d'):
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = 100 - 100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() / df['Close'].diff().clip(upper=0).abs().rolling(14).mean())
    df['MACD'] = df['Close'].ewm(12).mean() - df['Close'].ewm(26).mean()
    df['Signal'] = df['MACD'].ewm(9).mean()
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    return df

def plot_main_chart(df, ticker):
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axs[0].plot(df.index, df['Close'], label='Close')
    axs[0].plot(df.index, df['SMA20'], label='SMA20', linestyle='--')
    axs[0].plot(df.index, df['SMA50'], label='SMA50', linestyle='--')
    axs[0].set_title(f"{ticker} - Harga dan Moving Average")
    axs[0].legend()

    axs[1].plot(df.index, df['RSI'], color='purple')
    axs[1].axhline(70, color='red', linestyle='--')
    axs[1].axhline(30, color='green', linestyle='--')
    axs[1].set_title("RSI (14)")

    axs[2].bar(df.index, df['Volume'], label='Volume', alpha=0.5)
    axs[2].plot(df.index, df['Volume_MA'], label='Volume MA(20)', color='orange')
    axs[2].legend()
    axs[2].set_title("Volume")

    plt.tight_layout()
    return fig

def app():
    st.set_page_config(layout="wide")
    st.title("📈 Dashboard Modern Analisis Saham")

    with st.sidebar:
        st.header("Pengaturan Analisis")
        tickers = st.text_input("Ticker (pisahkan koma)", "AAPL,MSFT")
        period = st.selectbox("Periode", ['6mo', '1y', '2y'], index=1)
        interval = st.selectbox("Interval", ['1d', '1wk'], index=0)
        dca_value = st.number_input("DCA per pembelian", value=1000000)
        freq = st.radio("Frekuensi DCA", ['Bulanan', 'Mingguan'])

    if tickers:
        tickers = [t.strip().upper() for t in tickers.split(",")]
        for ticker in tickers:
            try:
                df = fetch_and_prepare(ticker, period, interval)
                st.markdown(f"## {ticker}")
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.pyplot(plot_main_chart(df, ticker))

                with col2:
                    last_price = df['Close'].iloc[-1]
                    dca_df = df.resample('M' if freq == 'Bulanan' else 'W').first().dropna()
                    unit = dca_value / dca_df['Close']
                    total_unit = unit.sum()
                    total_cost = (unit * dca_df['Close']).sum()
                    nilai_akhir = total_unit * last_price
                    avg_price = total_cost / total_unit
                    dca_return = (nilai_akhir - total_cost) / total_cost * 100

                    divs = yf.Ticker(ticker).dividends
                    ydiv = divs.resample('Y').sum()
                    avg_div = ydiv[-3:].mean() if len(ydiv) >= 3 else ydiv.mean()
                    yield_rate = (avg_div / last_price * 100) if avg_div > 0 else 0
                    proj_div = avg_div * 1000

                    st.metric("Harga Terakhir", f"${last_price:.2f}")
                    st.metric("Return DCA", f"{dca_return:.2f}%")
                    st.metric("Rata-rata Yield", f"{yield_rate:.2f}%")
                    st.metric("Proyeksi Dividen (10 lot)", f"${proj_div:.2f}")

                with st.expander("Data Historis"):
                    st.dataframe(df.tail(50))

            except Exception as e:
                st.error(f"{ticker}: {e}")

if __name__ == "__main__":
    app()
