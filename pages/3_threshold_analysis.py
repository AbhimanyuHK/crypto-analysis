import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path

# DB Path
DB_DIR = Path("db_sql")
DB_DIR.mkdir(exist_ok=True)
SQLITE_FILE = DB_DIR / "crypto_data.db"

st.title("📊 Threshold Analysis (from DB)")

# Connect to DB
conn = sqlite3.connect(SQLITE_FILE)
cursor = conn.cursor()

# Get available coins from daily_differences table
coins = pd.read_sql("SELECT DISTINCT Coin FROM daily_differences", conn)["Coin"].tolist()
crypto_symbol = st.selectbox("Select Crypto", coins)


def after_charges(amount):
    """Apply DCX fee, GST, and TDS deductions to a profit value."""
    dcx_fee = amount * (0.1 / 100)
    gst = amount * (0.1 / 100) * 0.18
    tds = amount * 0.01
    return amount - dcx_fee - gst - tds


def total_profit(base_price, profit_rate=0.02, trades=10):
    """Iteratively apply profits and charges for multiple trades."""
    price = base_price
    for _ in range(trades):
        price = after_charges(price * (1 + profit_rate))
    return price


def total_profit_after_charges(base_price, profit_rate, trades, charges):
    """Formula-based profit calculation with charges."""
    final_price = base_price * (1 + profit_rate) - (charges * (((1 - profit_rate) ** trades) - 1)) / profit_rate
    return final_price


if st.button("Analyze Thresholds"):
    # Read daily differences for this coin
    query = f"""
        SELECT Date, difference_percent, difference_percent_after_charges
        FROM daily_differences
        WHERE Coin = '{crypto_symbol}'
        ORDER BY Date DESC limit 365
    """
    df = pd.read_sql(query, conn)

    thresholds = [1, 2, 3, 4, 5, 7, 10, 15]
    counts = []
    for t in thresholds:
        count = (df['difference_percent'].abs() >= t).sum()
        counts.append({
            "Coin": crypto_symbol,
            "Threshold": t,
            "Count": count,
            "Profit_before_charges": round((1 + (t / 100)) ** count, 2),
            "Profit_after_charges": after_charges(
                round((1 + (t / 100) - (1.118 / 100)) ** count, 2)
            ),
            "Profit_after_charges_2": total_profit_after_charges(
                1, t / 100, (df['difference_percent_after_charges'].abs() >= t).sum(), 0.01118
            )
        })

    result_df = pd.DataFrame(counts)

    # Save to DB
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS threshold_analysis (
            Coin TEXT,
            Threshold TEXT,
            Count INTEGER,
            Profit_before_charges REAL,
            Profit_after_charges REAL,
            Profit_after_charges_2 REAL
        )
    """)

    # Remove existing analysis for this coin
    cursor.execute("DELETE FROM threshold_analysis WHERE Coin = ?", (crypto_symbol,))
    conn.commit()

    result_df.to_sql("threshold_analysis", conn, if_exists="append", index=False)

    st.success(f"✅ Threshold analysis saved to DB for {crypto_symbol}")
    st.dataframe(result_df)

    st.bar_chart(result_df.set_index("Threshold")["Count"])

if st.button("Analyze Thresholds for All"):
    for crypto_symbol in coins:
        # Read daily differences for this coin
        query = f"""
            SELECT Date, difference_percent, difference_percent_after_charges
            FROM daily_differences
            WHERE Coin = '{crypto_symbol}'
            ORDER BY Date DESC limit 365
        """
        df = pd.read_sql(query, conn)

        thresholds = [1, 2, 3, 4, 5, 7, 10, 15]
        counts = []
        for t in thresholds:
            count = (df['difference_percent'].abs() >= t).sum()
            counts.append({
                "Coin": crypto_symbol,
                "Threshold": t,
                "Count": count,
                "Profit_before_charges": round((1 + (t / 100)) ** count, 2),
                "Profit_after_charges": after_charges(
                    round((1 + (t / 100) - (1.118 / 100)) ** count, 2)
                ),
                "Profit_after_charges_2": total_profit_after_charges(
                    1, t / 100, (df['difference_percent_after_charges'].abs() >= t).sum(), 0.01118
                )
            })

        result_df = pd.DataFrame(counts)

        # Save to DB
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threshold_analysis (
                Coin TEXT,
                Threshold TEXT,
                Count INTEGER,
                Profit_before_charges REAL,
                Profit_after_charges REAL,
                Profit_after_charges_2 REAL
            )
        """)

        # Remove existing analysis for this coin
        cursor.execute("DELETE FROM threshold_analysis WHERE Coin = ?", (crypto_symbol,))
        conn.commit()

        result_df.to_sql("threshold_analysis", conn, if_exists="append", index=False)

        st.success(f"✅ Threshold analysis saved to DB for {crypto_symbol}")
        st.dataframe(result_df)

        st.bar_chart(result_df.set_index("Threshold")["Count"])

conn.close()
