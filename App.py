"""
Aplikasi Analisis Saham dengan Streamlit
- Mendukung input beberapa simbol ticker
- Mengambil data historis menggunakan yfinance
- Menghitung indikator teknis
- Menampilkan statistik ringkasan dan grafik
- Antarmuka web interaktif
- Memberikan rekomendasi beli/jual/tahan berdasarkan crossover SMA dan RSI

Instal dependensi:
  pip install streamlit yfinance pandas numpy matplotlib seaborn

Jalankan:
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
    Mengambil data historis untuk simbol ticker saham dari Yahoo Finance.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"Tidak ada data yang diambil untuk {ticker}. Silakan periksa simbol ticker dan koneksi internet.")
    return df

def compute_technical_indicators(df):
    """
    Menambahkan kolom indikator teknis ke dataframe.
    """
    df['SMA20'] = df['Close'].rolling(window=20).mean()  # Rata-rata bergerak sederhana 20 hari
    df['SMA50'] = df['Close'].rolling(window=50).mean()  # Rata-rata bergerak sederhana 50 hari
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()  # Rata-rata bergerak eksponensial 20 hari
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()  # Rata-rata bergerak eksponensial 50 hari

    delta = df['Close'].diff()  # Perubahan harga harian
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # Rata-rata keuntungan
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # Rata-rata kerugian
    rs = gain / loss  # Rasio keuntungan terhadap kerugian
    df['RSI14'] = 100 - (100 / (1 + rs))  # Indeks kekuatan relatif (RSI)

    sma20 = df['Close'].rolling(window=20).mean()  # Rata-rata bergerak sederhana 20 hari untuk Bollinger Bands
    std20 = df['Close'].rolling(window=20).std()  # Deviasi standar untuk Bollinger Bands
    df['Bollinger_High'] = sma20 + (std20 * 2)  # Garis atas Bollinger Bands
    df['Bollinger_Low'] = sma20 - (std20 * 2)  # Garis bawah Bollinger Bands

    return df

def calculate_return_stats(df):
    """
    Menghitung pengembalian harian, pengembalian kumulatif, dan volatilitas tahunan.
    """
    df['Daily Return'] = df['Close'].pct_change()  # Menghitung pengembalian harian
    cumulative_return = (df['Daily Return'] + 1).prod() - 1  # Menghitung pengembalian kumulatif
    trading_days = 252  # Jumlah hari perdagangan dalam setahun
    annualized_vol = df['Daily Return'].std() * np.sqrt(trading_days)  # Menghitung volatilitas tahunan
    return {
        'Cumulative Return': cumulative_return,
        'Annualized Volatility': annualized_vol
    }

def generate_recommendation(df):
    """
    Menghasilkan rekomendasi beli/jual/tahan berdasarkan:
    - Crossover SMA20 dan SMA50 (dua hari terakhir)
    - Nilai RSI saat ini

    Logika:
    - Sinyal beli jika SMA20 melintasi di atas SMA50 atau RSI < 30 (oversold)
    - Sinyal jual jika SMA20 melintasi di bawah SMA50 atau RSI > 70 (overbought)
    - Tahan jika tidak ada sinyal
    """
    rec = "Tahan"

    if df.shape[0] < 2:
        return rec  # Tidak cukup data untuk menganalisis crossover

    # Mendapatkan nilai terakhir untuk SMA20 dan SMA50
    sma20_yesterday = df['SMA20'].iloc[-2]
    sma50_yesterday = df['SMA50'].iloc[-2]
    sma20_today = df['SMA20'].iloc[-1]
    sma50_today = df['SMA50'].iloc[-1]

    rsi_today = df['RSI14'].iloc[-1]

    # Memeriksa crossover
    if pd.notna(sma20_yesterday) and pd.notna(sma50_yesterday) and pd.notna(sma20_today) and pd.notna(sma50_today):
        if (sma20_yesterday < sma50_yesterday) and (sma20_today > sma50_today):
            rec = "Beli"
        elif (sma20_yesterday > sma50_yesterday) and (sma20_today < sma50_today):
            rec = "Jual"
    # Sinyal berdasarkan RSI (baik melengkapi atau menggantikan crossover)
    if rsi_today < 30:
        rec = "Beli (RSI Oversold)"
    elif rsi_today > 70:
        rec = "Jual (RSI Overbought)"

    return rec

def plot_stock_data(df, ticker):
    """
    Menggambar harga penutupan, rata-rata bergerak, RSI, dan Bollinger Bands.
    Mengembalikan figure matplotlib.
    """
    fig, axs = plt.subplots(3,1, figsize=(12,10), sharex=True)
    
    ax1 = axs[0]
    ax1.plot(df.index, df['Close'], label='Harga Penutupan', color='blue')
    ax1.plot(df.index, df['SMA20'], label='SMA 20', linestyle='--', color='orange')
    ax1.plot(df.index, df['SMA50'], label='SMA 50', linestyle='--', color='red')
    ax1.plot(df.index, df['EMA20'], label='EMA 20', linestyle=':', color='green')
    ax1.plot(df.index, df['EMA50'], label='EMA 50', linestyle=':', color='purple')
    ax1.fill_between(df.index, df['Bollinger_Low'], df['Bollinger_High'], color='grey', alpha=0.2, label='Bollinger Bands')
    ax1.set_title(f"{ticker} Harga dan Rata-rata Bergerak dengan Bollinger Bands")
    ax1.set_ylabel("Harga (USD)")
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
    ax3.set_title(f"{ticker} Distribusi Pengembalian Harian")
    ax3.set_xlabel("Pengembalian Harian")
    ax3.set_ylabel("Frekuensi")

    plt.tight_layout()
    return fig

def main():
    st.title("🚀 Aplikasi Analisis Saham Lanjutan (seperti BlackRock & Lo Keng Hong)")
    st.markdown("""
    Masukkan satu atau beberapa simbol ticker saham yang dipisahkan dengan koma.
    Format simbol yang didukung misalnya AAPL, MSFT, GOTO.JK.
    \nPilih periode data untuk dianalisis.
    """)

    tickers_input = st.text_input("Masukkan simbol ticker saham", value="AAPL,MSFT")
    period = st.selectbox("Pilih periode data", options=['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'], index=3)
    interval = st.selectbox("Pilih interval data", options=['1d', '1wk', '1mo'], index=0)

    if st.button("Jalankan Analisis"):
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]
        if not tickers:
            st.error("Silakan masukkan setidaknya satu simbol ticker yang valid.")
            return

        for ticker in tickers:
            st.header(f"📈 Analisis untuk {ticker}")
            try:
                df = fetch_stock_data(ticker, period=period, interval=interval)
                df = compute_technical_indicators(df)
                stats = calculate_return_stats(df)
                recommendation = generate_recommendation(df)

                st.subheader("Statistik Ringkasan")
                st.write(f"**Pengembalian Kumulatif selama periode:** {stats['Cumulative Return']*100:.2f}%")
                st.write(f"**Volatilitas Tahunan:** {stats['Annualized Volatility']*100:.2f}%")

                st.subheader("Rekomendasi Beli/Jual")
                if recommendation.startswith("Beli"):
                    st.success(f"Rekomendasi: {recommendation}")
                elif recommendation.startswith("Jual"):
                    st.error(f"Rekomendasi: {recommendation}")
                else:
                    st.info(f"Rekomendasi: {recommendation}")

                fig = plot_stock_data(df, ticker)
                st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Kesalahan saat memproses {ticker}: {e}")

if __name__ == "__main__":
    main()
