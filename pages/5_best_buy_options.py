import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path

# DB Path
DB_DIR = Path("db_sql")
SQLITE_FILE = DB_DIR / "crypto_data.db"

st.title("💹 Best Buy Options Dashboard")

# Connect to DB
conn = sqlite3.connect(SQLITE_FILE)

if st.button("Show Best Options"):
    # --- Top by Daily difference ---
    query_daily_differences = "SELECT * FROM daily_differences;"
    df_daily_differences = pd.read_sql(query_daily_differences, conn)

    if df_daily_differences.empty:
        st.warning("⚠️ No data in daily_differences.")
    else:
        avg_df_daily_differences = df_daily_differences.groupby("Coin")['difference_percent'].agg(
            low='min',
            mean='mean',
            median='median',
            high='max',
            std='std'
        )

        st.subheader("🏆 Daily difference Mean & Median")
        st.dataframe(avg_df_daily_differences)

    # --- Top by Profit_after_charges (threshold_analysis) ---
    query_threshold = "SELECT * FROM threshold_analysis ORDER BY Profit_after_charges DESC"
    df_threshold = pd.read_sql(query_threshold, conn)

    if df_threshold.empty:
        st.warning("⚠️ No data in threshold_analysis.")
    else:
        top_threshold = df_threshold.groupby("Coin").head(1).head(10)

        st.subheader("🏆 Top by Profit_after_charges (Max 2 per Coin)")
        st.dataframe(top_threshold)

        # st.bar_chart(
        #     top_threshold.set_index("Coin")["Profit_after_charges"]
        # )

    # --- Top by positive Percentage_Diff (price_prediction) ---
    query_prediction = """
        SELECT * FROM price_prediction 
        WHERE Percentage_Diff > 0 
        ORDER BY Percentage_Diff DESC
    """
    df_prediction = pd.read_sql(query_prediction, conn)

    if df_prediction.empty:
        st.warning("⚠️ No positive predictions in price_prediction.")
    else:
        top_prediction = df_prediction.groupby("Coin").head(1).head(10)

        st.subheader("📈 Top by Positive Percentage_Diff (Max 2 per Coin)")
        st.dataframe(top_prediction[["Date", "Coin", "Predicted_Close", "Percentage_Diff", "Algorithm"]])

        # st.bar_chart(
        #     top_prediction.set_index("Coin")["Percentage_Diff"]
        # )

    # --- Best Buy: Coins appearing in both sets ---
    if not df_threshold.empty and not df_prediction.empty:
        best_buy = pd.merge(
            top_threshold,
            top_prediction,
            on="Coin",
            suffixes=("_threshold", "_prediction")
        )
        # ✅ Filter where prediction Percentage_Diff >= threshold
        if best_buy["Threshold"].dtype == object:
            best_buy["Threshold"] = (
                best_buy["Threshold"]
                    .astype(str)
                    .str.replace(">=", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .astype(float)
            )
        best_buy = best_buy[best_buy["Percentage_Diff"] >= best_buy["Threshold"]]

        if best_buy.empty:
            st.info("ℹ️ No overlapping coins between threshold analysis and predictions.")
        else:
            st.subheader("💎 Best Buy Options (Overlap)")
            st.dataframe(best_buy[[
                "Coin",
                "Threshold",
                "Profit_after_charges",
                "Date",
                "Predicted_Close",
                "Percentage_Diff",
                "Algorithm"
            ]])

            # # Chart Profit_after_charges vs Percentage_Diff
            # st.bar_chart(
            #     best_buy.set_index("Coin")[["Profit_after_charges", "Percentage_Diff"]]
            # )
            # --- Save to DB ---
            best_buy.to_sql("best_buy", conn, if_exists="replace", index=False)
            st.success("✅ Best Buy results saved to 'best_buy' table in database.")

conn.close()
