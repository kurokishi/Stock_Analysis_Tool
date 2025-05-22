import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from babel.numbers import format_currency

def fetch_and_prepare(ticker, period='1y', interval='1d'):
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = 100 - 100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() / df['Close'].diff().clip(upper=0).abs().rolling(14).mean())
    df['MACD'] = df['Close'].ewm(12).mean() - df['Close'].ewm(26).mean()
    df['Signal'] = df['MACD'].ewm(9).mean()
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    return df

def format_rp(value):
    try:
        return format_currency(value, "IDR", locale="id_ID")
    except:
        return f"Rp{value:,.0f}"

def interactive_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50'))
    fig.update_layout(title=f"{ticker} Harga Saham & MA", xaxis_title='Tanggal', yaxis_title='Harga (Rp)', template='plotly_white')
    return fig

def app():
    st.set_page_config(layout="wide")
    st.title("📊 Dashboard Saham Interaktif (Rupiah + Plotly)")

    with st.sidebar:
        st.header("Pengaturan")
        tickers = st.text_input("Ticker (pisahkan koma)", "ADRO.JK, ANTM.JK")
        period = st.selectbox("Periode", ['6mo', '1y', '2y'], index=1)
        interval = st.selectbox("Interval", ['1d', '1wk'], index=0)
        dca_value = st.number_input("Nominal DCA", value=1000000)
        freq = st.radio("Frekuensi DCA", ['Bulanan', 'Mingguan'])

    if tickers:
        tickers = [t.strip().upper() for t in tickers.split(",")]
        for ticker in tickers:
            try:
                df = fetch_and_prepare(ticker, period, interval)
                st.markdown(f"## {ticker}")
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.plotly_chart(interactive_chart(df, ticker), use_container_width=True)

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

                    st.metric("Harga Terakhir", format_rp(last_price))
                    st.metric("Return DCA", f"{dca_return:.2f}%")
                    st.metric("Yield Rata-rata", f"{yield_rate:.2f}%")
                    st.metric("Proyeksi Dividen (10 lot)", format_rp(proj_div))

                with st.expander("Data Historis"):
                    st.dataframe(df.tail(50))

            except Exception as e:
                st.error(f"{ticker}: {e}")

if __name__ == "__main__":
    app()
        
