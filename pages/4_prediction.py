import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
import numpy as np
import altair as alt

from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# --- DB Path ---
DB_DIR = Path("db_sql")
DB_DIR.mkdir(exist_ok=True)
SQLITE_FILE = DB_DIR / "crypto_data.db"

st.title("🔮 Crypto Price Prediction Dashboard")

# --- Connect to DB ---
conn = sqlite3.connect(SQLITE_FILE)

# --- Get coin list ---
coins = pd.read_sql("SELECT DISTINCT Coin FROM crypto_prices", conn)["Coin"].tolist()
crypto_symbol = st.selectbox("Select Crypto", coins)

# --- Algorithm selection ---
algo_choice = st.selectbox(
    "Select Prediction Algorithm",
    ["Linear Regression", "Random Forest", "SVR", "Prophet", "LSTM"]
)

# --- ML Model Trainer ---
def train_model(X, y, algo="Linear Regression"):
    if algo == "Linear Regression":
        model = LinearRegression()
    elif algo == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif algo == "SVR":
        model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    elif algo == "Prophet":
        return Prophet()
    return model.fit(X, y)


def train_lstm(df, n_input=5, n_features=1):
    """Train LSTM on closing prices"""
    data = df['Close'].values.reshape(-1, 1)
    generator = TimeseriesGenerator(data, data, length=n_input, batch_size=1)

    model = Sequential([
        LSTM(100, activation='relu', input_shape=(n_input, n_features)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(generator, epochs=50, verbose=0)
    return model


def predict_next_days(df, coin, algo, save=True):
    df = df.reset_index(drop=True)

    if algo == "Prophet":
        prophet_df = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)
        future_forecast = forecast.tail(10)
        last_close = df['Close'].iloc[-1]

        prediction_df = pd.DataFrame({
            "Date": future_forecast['ds'],
            "Coin": coin,
            "Algorithm": algo,
            "Predicted_Close": future_forecast['yhat'],
            "Difference": future_forecast['yhat'] - last_close,
            "Percentage_Diff": ((future_forecast['yhat'] - last_close) / last_close) * 100
        })

    elif algo == "LSTM":
        n_input = 5
        n_features = 1
        model = train_lstm(df, n_input, n_features)

        data = df['Close'].values.reshape(-1, 1)
        last_seq = data[-n_input:]
        preds = []

        for _ in range(10):
            X_input = last_seq.reshape((1, n_input, n_features))
            pred = model.predict(X_input, verbose=0)[0][0]
            preds.append(pred)
            last_seq = np.append(last_seq[1:], pred)
            last_seq = last_seq.reshape(n_input, 1)

        future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=10)
        last_close = df['Close'].iloc[-1]

        prediction_df = pd.DataFrame({
            "Date": future_dates,
            "Coin": coin,
            "Algorithm": algo,
            "Predicted_Close": preds,
            "Difference": np.array(preds) - last_close,
            "Percentage_Diff": ((np.array(preds) - last_close) / last_close) * 100
        })

    else:
        df['DayIndex'] = np.arange(len(df))
        X = df[['DayIndex']]
        y = df['Close']
        model = train_model(X, y, algo)
        future_days = np.arange(len(df), len(df) + 10).reshape(-1, 1)
        future_preds = model.predict(future_days)

        future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=10)
        last_close = df['Close'].iloc[-1]

        prediction_df = pd.DataFrame({
            "Date": future_dates,
            "Coin": coin,
            "Algorithm": algo,
            "Predicted_Close": future_preds,
            "Difference": future_preds - last_close,
            "Percentage_Diff": ((future_preds - last_close) / last_close) * 100
        })

    # --- Save to SQLite ---
    if save:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_prediction (
                Date TEXT,
                Coin TEXT,
                Algorithm TEXT,
                Predicted_Close REAL,
                Difference REAL,
                Percentage_Diff REAL
            )
        """)
        cursor.execute("DELETE FROM price_prediction WHERE Coin = ? AND Algorithm = ?", (coin, algo))
        conn.commit()

        prediction_df.to_sql("price_prediction", conn, if_exists="append", index=False)

    return prediction_df


# --- Predict button for single coin ---
if st.button("Predict Next 10 Days"):
    query = f"""
        SELECT Date, Close
        FROM crypto_prices
        WHERE Coin = '{crypto_symbol}'
        ORDER BY Date ASC
    """
    df = pd.read_sql(query, conn)
    df['Date'] = pd.to_datetime(df['Date'])

    if df.empty:
        st.error("⚠️ No price data found for this coin.")
    else:
        prediction_df = predict_next_days(df, crypto_symbol, algo_choice)

        st.subheader("📈 Last 5 Records")
        st.dataframe(df.tail(5))

        st.subheader(f"🔮 Next 10 Days Prediction ({algo_choice})")
        st.dataframe(prediction_df)

        st.success(f"✅ Predictions saved for {crypto_symbol} using {algo_choice}")

        # Plot
        actual_df = df[['Date', 'Close']].rename(columns={"Close": "Price"})
        actual_df["Type"] = "Actual"
        pred_df = prediction_df[['Date', 'Predicted_Close']].rename(columns={"Predicted_Close": "Price"})
        pred_df["Type"] = "Predicted"
        chart_df = pd.concat([actual_df.tail(20), pred_df])

        line_chart = (
            alt.Chart(chart_df)
            .mark_line(point=True)
            .encode(x="Date:T", y="Price:Q", color="Type:N")
            .properties(width=700, height=400)
        )
        st.altair_chart(line_chart, use_container_width=True)


# --- Predict button for ALL coins ---
if st.button("Predict Next 10 Days for ALL Coins"):
    for coin in coins:
        query = f"""
            SELECT Date, Close
            FROM crypto_prices
            WHERE Coin = '{coin}'
            ORDER BY Date ASC
        """
        df = pd.read_sql(query, conn)
        df['Date'] = pd.to_datetime(df['Date'])
        if df.empty:
            continue
        prediction_df = predict_next_days(df, coin, algo_choice, save=True)
        st.subheader(f"🔮 Next 10 Days Prediction ({coin}, {algo_choice})")
        st.dataframe(prediction_df)

conn.close()
