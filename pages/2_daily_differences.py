import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path

# DB Path
DB_DIR = Path("db_sql")
DB_DIR.mkdir(exist_ok=True)
SQLITE_FILE = DB_DIR / "crypto_data.db"

st.title("📈 Daily Price Differences (from DB)")

# Connect to DB
conn = sqlite3.connect(SQLITE_FILE)
cursor = conn.cursor()

# Get available coins from crypto_price table
coins = pd.read_sql("SELECT DISTINCT Coin FROM crypto_prices", conn)["Coin"].tolist()

crypto_symbol = st.selectbox("Select Crypto", coins)


def after_charges(value):
    """Apply DCX fee, GST, and TDS deductions to a percentage value."""
    dcx_fee = value * (0.1 / 100)
    gst = value * (0.1 / 100) * 0.18
    tds = value * 0.01
    return value - dcx_fee - gst - tds


if st.button("Calculate Differences"):
    # Read last 1 year OHLCV data from DB
    query = f"""
        SELECT Date, Close 
        FROM crypto_prices
        WHERE Coin = '{crypto_symbol}'
        ORDER BY Date ASC
    """
    df = pd.read_sql(query, conn)
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate differences
    df['difference'] = df['Close'].diff()
    df['difference_percent'] = (df['difference'] / df['Close'].shift(1)) * 100
    df['difference_percent_after_charges'] = df['difference_percent'].apply(after_charges)
    df['Coin'] = crypto_symbol

    df = df[['Date', 'Coin', 'difference', 'difference_percent', 'difference_percent_after_charges']]
    df = df.dropna().reset_index(drop=True)

    # Create daily_differences table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_differences (
            Date TEXT,
            Coin TEXT,
            difference REAL,
            difference_percent REAL,
            difference_percent_after_charges REAL
        )
    """)

    # Remove existing entries for this coin
    cursor.execute("DELETE FROM daily_differences WHERE Coin = ?", (crypto_symbol,))
    conn.commit()

    # Insert new data
    df.to_sql("daily_differences", conn, if_exists="append", index=False)

    st.success(f"✅ Saved differences for {crypto_symbol} into DB")
    st.dataframe(df.tail())
    st.line_chart(df.set_index("Date")["difference_percent"])

if st.button("Calculate Differences for all"):
    for crypto_symbol in coins:
        # Read last 1 year OHLCV data from DB
        query = f"""
            SELECT Date, Close 
            FROM crypto_prices
            WHERE Coin = '{crypto_symbol}'
            ORDER BY Date ASC limit 365
        """
        df = pd.read_sql(query, conn)
        df['Date'] = pd.to_datetime(df['Date'])

        # Calculate differences
        df['difference'] = df['Close'].diff()
        df['difference_percent'] = (df['difference'] / df['Close'].shift(1)) * 100
        df['difference_percent_after_charges'] = df['difference_percent'].apply(after_charges)
        df['Coin'] = crypto_symbol

        df = df[['Date', 'Coin', 'difference', 'difference_percent', 'difference_percent_after_charges']]
        df = df.dropna().reset_index(drop=True)

        # Create daily_differences table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_differences (
                Date TEXT,
                Coin TEXT,
                difference REAL,
                difference_percent REAL,
                difference_percent_after_charges REAL
            )
        """)

        # Remove existing entries for this coin
        cursor.execute("DELETE FROM daily_differences WHERE Coin = ?", (crypto_symbol,))
        conn.commit()

        # Insert new data
        df.to_sql("daily_differences", conn, if_exists="append", index=False)

        st.success(f"✅ Saved differences for {crypto_symbol} into DB")
        st.dataframe(df.tail())
        st.line_chart(df.set_index("Date")["difference_percent"])
conn.close()
