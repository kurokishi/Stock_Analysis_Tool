"""
Aplikasi Analisis Saham dengan Manajemen Portofolio dan Backtesting Strategi
- Mendukung input beberapa simbol ticker
- Mengambil data historis menggunakan yfinance
- Menghitung indikator teknis
- Menampilkan statistik ringkasan dan grafik
- Manajemen portofolio dengan perhitungan return gabungan
- Backtesting strategi beli/jual berdasarkan crossover SMA dan RSI
- Antarmuka web interaktif dengan Streamlit

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

def generate_signals(df):
    """
    Menghasilkan sinyal beli/jual berdasarkan crossover SMA20/SMA50 dan nilai RSI14.
    Keluaran kolom baru di dataframe: 'Signal' dengan nilai 1=Buy, -1=Sell, 0=Hold.
    """
    df = df.copy()
    df['Signal'] = 0

    # Sinyal berdasarkan crossover SMA
    df['SMA_Cross'] = 0
    df.loc[(df['SMA20'] > df['SMA50']) & (df['SMA20'].shift(1) <= df['SMA50'].shift(1)), 'SMA_Cross'] = 1  # Buy
    df.loc[(df['SMA20'] < df['SMA50']) & (df['SMA20'].shift(1) >= df['SMA50'].shift(1)), 'SMA_Cross'] = -1  # Sell

    # Sinyal berdasarkan RSI
    df['RSI_Signal'] = 0
    df.loc[df['RSI14'] < 30, 'RSI_Signal'] = 1  # Oversold, Buy
    df.loc[df['RSI14'] > 70, 'RSI_Signal'] = -1  # Overbought, Sell

    # Gabungkan sinyal (berikan prioritas sinyal RSI jika ada, jika tidak gunakan SMA crossover)
    def combine_signal(row):
        if row['RSI_Signal'] != 0:
            return row['RSI_Signal']
        else:
            return row['SMA_Cross']

    df['Signal'] = df.apply(combine_signal, axis=1)
    return df

def backtest_strategy(df):
    """
    Backtest strategi beli/jual berdasarkan sinyal di dataframe.
    Simulasi dengan modal awal 1 (atau 100%) jual-beli saham secara penuh (all-in).
    Output hasil berupa:
    - dataframe equity curve nilai investasi sepanjang waktu
    - statistik performa: total return, jumlah transaksi, rasio kemenangan (win rate)
    """
    df = df.copy()
    df = generate_signals(df)
    df['Position'] = 0  # 1 jika posisi beli, 0 jika tidak pegang saham

    # Posisi masuk: setelah sinyal beli, posisi keluar setelah sinyal jual
    position = 0
    positions = []
    trades = []
    for i, row in df.iterrows():
        if position == 0 and row['Signal'] == 1:
            position = 1  # Masuk posisi beli
            trades.append({'Type': 'Buy', 'Date': i, 'Price': row['Close']})
        elif position == 1 and row['Signal'] == -1:
            position = 0  # Keluar posisi (jual)
            trades.append({'Type': 'Sell', 'Date': i, 'Price': row['Close']})
        positions.append(position)
    df['Position'] = positions

    # Hitung equity: ketika posisi saham, nilai = harga saham; jika tidak, nilai = kas (tetap)
    initial_capital = 1.0
    equity = []
    cash = initial_capital
    shares = 0
    last_price = None
    trade_idx = 0

    # Kita asumsikan all-in saat beli, all-out saat jual, tidak ada biaya transaksi
    for i, row in df.iterrows():
        price = row['Close']
        signal = row['Signal']

        # Eksekusi pada saat sinyal
        if trade_idx < len(trades) and trades[trade_idx]['Date'] == i:
            trade = trades[trade_idx]
            if trade['Type'] == 'Buy':
                shares = cash / price
                cash = 0
            elif trade['Type'] == 'Sell':
                cash = shares * price
                shares = 0
            trade_idx += 1

        total_value = cash + shares * price
        equity.append(total_value)
        last_price = price

    df['Equity'] = equity

    # Statistik performa
    total_return = equity[-1] - initial_capital
    num_trades = len(trades)//2  # Hitung pasangan buy-sell
    wins = 0
    for i in range(0, len(trades)-1, 2):
        buy = trades[i]
        sell = trades[i+1]
        if sell['Price'] > buy['Price']:
            wins += 1
    win_rate = (wins / num_trades) if num_trades > 0 else 0

    stats = {
        'Total Return (%)': total_return * 100,
        'Number of Trades': num_trades,
        'Win Rate (%)': win_rate * 100
    }

    return df, stats

def generate_recommendation(df):
    """
    Menghasilkan rekomendasi beli/jual/tahan berdasarkan:
    - Sinyal pada bar terakhir (kolom Signal)
    """
    if df.empty or 'Signal' not in df.columns:
        df = generate_signals(df)

    last_signal = df['Signal'].iloc[-1]
    rec = "Tahan"
    if last_signal == 1:
        rec = "Beli"
    elif last_signal == -1:
        rec = "Jual"
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

def plot_backtest_equity(df, ticker):
    """
    Menggambar kurva equity hasil backtesting.
    """
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df.index, df['Equity'], label='Equity Curve', color='blue')
    ax.set_title(f"{ticker} Kurva Equity Backtest Strategi")
    ax.set_ylabel("Nilai Investasi (relatif)")
    ax.set_xlabel("Tanggal")
    ax.legend()
    plt.tight_layout()
    return fig

def main():
    st.title("🚀 Aplikasi Analisis Saham dengan Manajemen Portofolio dan Backtesting")
    st.markdown("""
    Masukkan satu atau beberapa simbol ticker saham yang dipisahkan dengan koma.<br>
    Pilih periode dan interval data untuk dianalisis.<br>
    Anda dapat memasukkan bobot portofolio untuk memperkirakan return gabungan.<br>
    """, unsafe_allow_html=True)

    tickers_input = st.text_input("Masukkan simbol ticker saham", value="AAPL,MSFT,GOOG")
    period = st.selectbox("Pilih periode data", options=['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'], index=3)
    interval = st.selectbox("Pilih interval data", options=['1d', '1wk', '1mo'], index=0)

    if not tickers_input.strip():
        st.warning("Silakan masukkan setidaknya satu simbol ticker.")
        return

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]

    # Input bobot portofolio
    st.subheader("Bobot Portofolio (dalam persen, total harus 100%)")
    default_weights = [round(100/len(tickers),2)] * len(tickers)
    weights_input = []
    for i, ticker in enumerate(tickers):
        w = st.number_input(f"Bobot {ticker}", min_value=0.0, max_value=100.0, value=default_weights[i], step=0.1, format="%.2f")
        weights_input.append(w)

    total_weight = sum(weights_input)
    if not np.isclose(total_weight, 100.0):
        st.warning(f"Total bobot portofolio adalah {total_weight}%. Total harus 100%. Silakan sesuaikan bobot.")

    # Tombol untuk menjalankan analisis
    if st.button("Jalankan Analisis"):
        # Dictionary untuk menyimpan data setiap ticker
        data_per_ticker = {}
        stats_per_ticker = {}
        backtest_stats_per_ticker = {}

        st.subheader("Analisis Saham Per Ticker")
        for i, ticker in enumerate(tickers):
            st.markdown(f"---\n### {ticker}")
            try:
                df = fetch_stock_data(ticker, period=period, interval=interval)
                df = compute_technical_indicators(df)
                stats = calculate_return_stats(df)
                df = generate_signals(df)
                rec = generate_recommendation(df)
                df_backtest, backtest_stats = backtest_strategy(df)

                data_per_ticker[ticker] = df
                stats_per_ticker[ticker] = stats
                backtest_stats_per_ticker[ticker] = backtest_stats

                # Tampilkan statistik dasar
                st.write(f"**Pengembalian Kumulatif:** {stats['Cumulative Return']*100:.2f}%")
                st.write(f"**Volatilitas Tahunan:** {stats['Annualized Volatility']*100:.2f}%")

                # Tampilkan rekomendasi
                if rec.startswith("Beli"):
                    st.success(f"Rekomendasi Saat Ini: {rec}")
                elif rec.startswith("Jual"):
                    st.error(f"Rekomendasi Saat Ini: {rec}")
                else:
                    st.info(f"Rekomendasi Saat Ini: {rec}")

                # Tampilkan grafik harga dan indikator
                fig = plot_stock_data(df, ticker)
                st.pyplot(fig)

                # Tampilkan hasil backtesting
                st.subheader("Hasil Backtesting Strategi")
                st.write(f"Total Return Backtest: {backtest_stats['Total Return (%)']:.2f}%")
                st.write(f"Jumlah Transaksi: {backtest_stats['Number of Trades']}")
                st.write(f"Rasio Kemenangan: {backtest_stats['Win Rate (%)']:.2f}%")

                fig_bt = plot_backtest_equity(df_backtest, ticker)
                st.pyplot(fig_bt)

            except Exception as e:
                st.error(f"Kesalahan saat memproses {ticker}: {e}")

        # Manajemen Portofolio: Hitung return dan volatilitas gabungan jika bobot valid
        if np.isclose(total_weight, 100.0):
            st.markdown("---")
            st.subheader("Analisis Portofolio Gabungan")

            # Gabungkan harga Close tiap ticker ke dataframe portfolio
            price_df = pd.DataFrame()
            for ticker in tickers:
                df_tick = data_per_ticker.get(ticker)
                if df_tick is not None:
                    price_df[ticker] = df_tick['Close']

            # Isi nilai NaN dengan metode forward fill untuk evaluasi portofolio
            price_df.fillna(method='ffill', inplace=True)
            price_df.dropna(inplace=True)  # Buang baris tanpa data lengkap

            # Hitung return harian tiap ticker
            daily_returns = price_df.pct_change().dropna()

            # Konversi bobot ke proporsi desimal
            weights = np.array(weights_input) / 100.0

            # Hitung return harian portofolio sebagai weighted sum
            portfolio_daily_return = daily_returns.dot(weights)

            # Hitung pengembalian kumulatif portofolio
            portfolio_cumulative_return = (portfolio_daily_return + 1).prod() - 1

            # Hitung volatilitas portofolio tahunan
            portfolio_annual_volatility = portfolio_daily_return.std() * np.sqrt(252)

            st.write(f"**Pengembalian Kumulatif Portofolio:** {portfolio_cumulative_return * 100:.2f}%")
            st.write(f"**Volatilitas Tahunan Portofolio:** {portfolio_annual_volatility * 100:.2f}%")

            # Kurva nilai portofolio evolusi investasi
            portfolio_equity = (portfolio_daily_return + 1).cumprod()

            fig, ax = plt.subplots(figsize=(12,5))
            ax.plot(portfolio_equity.index, portfolio_equity.values, label="Nilai Portofolio (terdiskonto)")
            ax.set_title("Kurva Nilai Portofolio")
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Nilai Investasi (relatif)")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("Bobot portofolio belum valid (total tidak 100%), analisis portofolio gabungan tidak ditampilkan.")

if __name__ == "__main__":
    main()

