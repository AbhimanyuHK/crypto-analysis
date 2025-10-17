import streamlit as st
import pandas as pd
import yfinance as yf
from pathlib import Path
import sqlite3

DB_DIR = Path("db_sql")
DB_DIR.mkdir(exist_ok=True)

DB_FILE = DB_DIR / "crypto_data.db"

st.title("📥 Fetch Crypto Data")

coin_list = ["BTC-INR", "ETH-INR", "SHIB-INR", "DOGE-INR", "XRP-INR", "USDT-INR", "PEPE24478-USD", "SOL-INR",
             "POPCAT28782-USD", "BNB-INR"]
# Dropdown for crypto selection
crypto_symbol = st.selectbox(
    "Select Crypto Symbol",
    coin_list
)

if st.button("Fetch Data"):
    st.write(f"Fetching last 1 year of daily data for {crypto_symbol}...")
    data = yf.download(crypto_symbol, period="5y", interval="1d")

    # Remove multi-index columns (if they exist)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)
    data["Coin"] = crypto_symbol  # Add coin column

    # # Save CSV
    # file_path = DB_DIR / f"{crypto_symbol}_1y_daily.csv"
    # data.to_csv(file_path, index=False)

    # Save to SQLite with override logic
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crypto_prices (
            Date TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Adj_Close REAL,
            Volume REAL,
            Coin TEXT
        )
    """)

    # Delete existing rows for this coin
    cursor.execute("DELETE FROM crypto_prices WHERE Coin = ?", (crypto_symbol,))
    conn.commit()

    # Insert fresh data
    data.to_sql("crypto_prices", conn, if_exists="append", index=False)

    conn.close()

    st.success(f"✅ Data saved overridden in SQLite {DB_FILE} for {crypto_symbol}")
    st.dataframe(data.tail())
    st.line_chart(data.set_index("Date")["Close"])
    data.set_index("Date")[["Close"]].to_csv(f"db/open_ai_{crypto_symbol}.csv")

if st.button("Fetch Data for All"):

    for coin in coin_list:
        st.write(f"Fetching last 1 year of daily data for {coin}...")
        data = yf.download(coin, period="5y", interval="1d")

        # Remove multi-index columns (if they exist)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.reset_index(inplace=True)
        data["Coin"] = coin  # Add coin column

        # Save to SQLite with override logic
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Delete existing rows for this coin
        cursor.execute("DELETE FROM crypto_prices WHERE Coin = ?", (coin,))
        conn.commit()

        # Insert fresh data
        data.to_sql("crypto_prices", conn, if_exists="append", index=False)

        conn.close()

        st.success(f"✅ Data saved overridden in SQLite {DB_FILE} for {coin}")
        st.dataframe(data.tail())
        st.line_chart(data.set_index("Date")["Close"])
