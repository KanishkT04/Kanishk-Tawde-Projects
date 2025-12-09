# app.py
# ============================================================
# GenAI ED Ops Copilot – Streamlit Application
# ============================================================

import os
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dotenv import load_dotenv
from mistralai import Mistral

# ------------------------------------------------------------
# 1. CONFIG & LLM CLIENT
# ------------------------------------------------------------

st.set_page_config(
    page_title="GenAI ED Ops Copilot",
    layout="wide",
)

# Matplotlib defaults
plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if MISTRAL_API_KEY is None:
    st.warning("⚠️ MISTRAL_API_KEY not found in environment. LLM features will be disabled.")
    mistral_client = None
else:
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)


# ------------------------------------------------------------
# 2. DATA GENERATION & LOADING UTILITIES
# ------------------------------------------------------------

def generate_synthetic_ed_data(
    start_date="2022-01-01",
    end_date="2024-12-31",
    base_level=120,
    weekday_uplift=15,
    weekend_drop=-20,
    seasonal_amplitude=10,
    noise_std=8,
    random_seed=42,
) -> pd.DataFrame:
    """
    Generate synthetic daily ED visit data with:
    - Baseline volume
    - Weekly seasonality (weekday vs weekend)
    - Mild annual seasonality
    - Random noise
    Returns a DataFrame with columns: ['date', 'visits'].
    """
    rng = np.random.default_rng(random_seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)

    visits = np.full(shape=n, fill_value=base_level, dtype=float)

    dow = dates.dayofweek  # Monday=0, Sunday=6
    weekday_mask = dow < 5
    weekend_mask = dow >= 5

    visits[weekday_mask] += weekday_uplift
    visits[weekend_mask] += weekend_drop

    day_of_year = dates.dayofyear
    seasonal_effect = seasonal_amplitude * np.sin(2 * np.pi * day_of_year / 365.25)
    visits += seasonal_effect

    noise = rng.normal(loc=0.0, scale=noise_std, size=n)
    visits += noise

    visits = np.maximum(visits, 0)
    visits = np.round(visits).astype(int)

    df = pd.DataFrame({"date": dates, "visits": visits})
    return df


def load_ed_data_from_csv(file) -> pd.DataFrame:
    """
    Load ED daily visits data from an uploaded CSV or a file path.
    Expected columns:
      - 'date': parseable as datetime
      - 'visits': daily ED visit count (numeric)
    """
    df = pd.read_csv(file)

    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")
    if "visits" not in df.columns:
        raise ValueError("CSV must contain a 'visits' column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["visits"] = pd.to_numeric(df["visits"], errors="coerce")
    df = df.dropna(subset=["date", "visits"])
    df = df.sort_values("date").set_index("date")
    df["visits"] = df["visits"].astype(int)

    # Enforce daily frequency and fill any gaps
    df = df.asfreq("D")
    if df["visits"].isna().sum() > 0:
        df["visits"] = df["visits"].ffill().bfill()

    return df


# ------------------------------------------------------------
# 3. MODELING UTILITIES
# ------------------------------------------------------------

def fit_sarimax_model(
    train_series: pd.Series,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
):
    """
    Fit a SARIMAX model to the training series with weekly seasonality.
    """
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return fitted


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = (np.abs((y_true - y_pred) / y_true).mean()) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ------------------------------------------------------------
# 4. SUMMARIES FOR THE LLM
# ------------------------------------------------------------

def build_history_summary(df: pd.DataFrame) -> str:
    start = df.index.min().date()
    end = df.index.max().date()
    avg_visits = df["visits"].mean()
    min_visits = df["visits"].min()
    max_visits = df["visits"].max()

    dow_means = df.groupby(df.index.day_name())["visits"].mean()
    busiest_day = dow_means.idxmax()
    quietest_day = dow_means.idxmin()

    summary = (
        f"Historical ED data covers the period from {start} to {end}. "
        f"Average daily visits are approximately {avg_visits:.1f}, "
        f"with a minimum of {min_visits} and a maximum of {max_visits}. "
        f"On average, the busiest day of the week is {busiest_day}, "
        f"while the quietest day is {quietest_day}. "
        f"There appears to be a clear weekly pattern, with "
        f"weekday volumes generally higher than weekends."
    )
    return summary


def build_forecast_summary(df_future: pd.DataFrame) -> str:
    start = df_future.index.min().date()
    end = df_future.index.max().date()
    avg_forecast = df_future["forecast"].mean()
    min_forecast = df_future["forecast"].min()
    max_forecast = df_future["forecast"].max()

    summary = (
        f"The forecast covers the period from {start} to {end}. "
        f"Average predicted daily ED visits are around {avg_forecast:.1f}, "
        f"with forecasted values ranging from {min_forecast:.0f} to {max_forecast:.0f}. "
        f"Confidence intervals widen slightly further into the future, "
        f"indicating increasing uncertainty over time."
    )
    return summary


def build_metrics_summary(metrics: dict) -> str:
    return (
        f"On the held-out test period, the forecasting model achieved "
        f"a Mean Absolute Error (MAE) of {metrics['MAE']:.2f}, "
        f"a Root Mean Squared Error (RMSE) of {metrics['RMSE']:.2f}, "
        f"and a Mean Absolute Percentage Error (MAPE) of {metrics['MAPE']:.2f}%. "
        f"This indicates that, on average, the model's predictions are within "
        f"approximately {metrics['MAPE']:.1f}% of the true daily ED visit counts."
    )


def build_llm_context(history_text: str, forecast_text: str, metrics_text: str) -> str:
    context = (
        "Emergency Department (ED) Demand Forecasting Context\n"
        "----------------------------------------------------\n\n"
        "Historical Pattern Summary:\n"
        f"{history_text}\n\n"
        "Forecast Summary:\n"
        f"{forecast_text}\n\n"
        "Model Performance Summary:\n"
        f"{metrics_text}\n\n"
        "Use this information to explain the ED demand patterns and forecasts "
        "to a non-technical hospital operations manager. "
        "Focus on trends, patterns, uncertainty, and operational implications "
        "such as staffing and capacity planning."
    )
    return context


# ------------------------------------------------------------
# 5. LLM HELPERS (MISTRAL)
# ------------------------------------------------------------

def generate_ed_explanation(context_text: str, model: str = "mistral-small-latest") -> str:
    if mistral_client is None:
        return "Mistral API key missing. Cannot generate explanation."

    response = mistral_client.chat.complete(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a healthcare operations analyst explaining ED demand forecasts "
                    "to non-technical hospital managers. Avoid technical jargon."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Use the context to explain the ED forecasting results clearly, "
                    "including historical behavior, forecast, uncertainty, and operational implications:\n\n"
                    + context_text
                ),
            },
        ],
    )
    return response.choices[0].message.content


def answer_manager_question(
    question: str,
    context_text: str,
    model: str = "mistral-small-latest",
) -> str:
    if mistral_client is None:
        return "Mistral API key missing. Cannot answer the question."

    response = mistral_client.chat.complete(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Emergency Department operations consultant. "
                    "Use ONLY the provided context (historical patterns, forecasts, and metrics) "
                    "to answer questions. If the context does not support a detailed answer, "
                    'say so clearly and avoid making up facts. Speak in practical, manager-friendly language.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_text}\n\n"
                    f"Manager's question: {question}"
                ),
            },
        ],
    )
    return response.choices[0].message.content


# ------------------------------------------------------------
# 6. STREAMLIT APP LAYOUT
# ------------------------------------------------------------

def main():
    st.title("🧠 GenAI ED Ops Copilot")
    st.subheader("Generative AI–Enhanced ED Demand Forecasting and Decision Support")

    st.markdown(
        """
        This app combines **time series forecasting** (SARIMAX) with a **Generative AI explainer** (Mistral)
        to help Emergency Department leaders:
        - Understand historical ED demand patterns  
        - See short-term forecasts and uncertainty  
        - Get plain-language explanations and Q&A for operational planning  
        """
    )

    # ---------------- Sidebar: Data & Model Config ----------------
    st.sidebar.header("Configuration")

    data_source = st.sidebar.radio(
        "Data source:",
        ["Use sample synthetic ED data", "Upload my own CSV"],
    )

    test_days = st.sidebar.slider(
        "Test set size (days for evaluation)",
        min_value=14,
        max_value=90,
        value=30,
        step=7,
    )

    forecast_days = st.sidebar.slider(
        "Forecast horizon (future days)",
        min_value=7,
        max_value=60,
        value=30,
        step=7,
    )

    random_seed = st.sidebar.number_input(
        "Synthetic data random seed",
        min_value=1,
        max_value=9999,
        value=42,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Make sure your `.env` file contains a valid `MISTRAL_API_KEY`.")

    # ---------------- Main: Data Loading ----------------
    if data_source == "Upload my own CSV":
        uploaded_file = st.file_uploader(
            "Upload ED daily visits CSV (must have 'date' and 'visits' columns)",
            type=["csv"],
        )
        if uploaded_file is None:
            st.info("Upload a CSV to continue, or switch to synthetic data in the sidebar.")
            return

        try:
            df = load_ed_data_from_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return
    else:
        df = generate_synthetic_ed_data(random_seed=random_seed)
        df = df.set_index("date")

    st.success(f"Loaded ED dataset with {len(df)} days from {df.index.min().date()} to {df.index.max().date()}.")

    # Show basic info & preview
    with st.expander("📊 Data Preview & Summary"):
        st.write(df.head())
        st.write(df.describe())

    # ---------------- EDA Plot ----------------
    st.markdown("### 📈 Historical ED Demand")
    fig_hist, ax_hist = plt.subplots()
    ax_hist.plot(df.index, df["visits"], label="Daily ED visits")
    ax_hist.set_title("Daily ED Visit Volumes")
    ax_hist.set_xlabel("Date")
    ax_hist.set_ylabel("Visits")
    ax_hist.legend()
    st.pyplot(fig_hist)

    # Day-of-week analysis
    df_dow = df.copy()
    df_dow["day_of_week"] = df_dow.index.day_name()
    dow_means = df_dow.groupby("day_of_week")["visits"].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

    with st.expander("📅 Day-of-Week Pattern"):
        fig_dow, ax_dow = plt.subplots()
        dow_means.plot(kind="bar", ax=ax_dow)
        ax_dow.set_title("Average ED Visits by Day of Week")
        ax_dow.set_ylabel("Average visits")
        plt.xticks(rotation=45)
        st.pyplot(fig_dow)
        st.write(dow_means)

    # ---------------- Train / Test Split ----------------
    if test_days >= len(df):
        st.error("Test days must be smaller than the total number of days in the dataset.")
        return

    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()

    st.markdown("### 🔧 Train/Test Split")
    st.write(
        f"**Train period:** {train.index.min().date()} → {train.index.max().date()}  "
        f"(**{len(train)} days**)  \n"
        f"**Test period:** {test.index.min().date()} → {test.index.max().date()}  "
        f"(**{len(test)} days**)"
    )

    # ---------------- Model Fit ----------------
    with st.spinner("Fitting SARIMAX model..."):
        model = fit_sarimax_model(train["visits"])

    # Forecast on test period
    test_forecast_res = model.get_forecast(steps=test_days)
    test_pred_mean = test_forecast_res.predicted_mean
    test_conf_int = test_forecast_res.conf_int(alpha=0.05)
    test_pred_mean.index = test.index
    test_conf_int.index = test.index

    metrics = evaluate_forecast(test["visits"], test_pred_mean)

    st.markdown("### 📏 Model Performance on Test Set")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{metrics['MAE']:.2f}")
    col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    col3.metric("MAPE (%)", f"{metrics['MAPE']:.2f}")

    # Plot train/test + forecast
    st.markdown("### 🔮 Test Forecast vs Actuals")
    fig_test, ax_test = plt.subplots()
    ax_test.plot(train.index, train["visits"], label="Train")
    ax_test.plot(test.index, test["visits"], label="Test (Actual)")
    ax_test.plot(test_pred_mean.index, test_pred_mean.values, label="Test Forecast", linestyle="--")
    ax_test.fill_between(
        test_conf_int.index,
        test_conf_int.iloc[:, 0],
        test_conf_int.iloc[:, 1],
        color="gray",
        alpha=0.2,
        label="95% CI",
    )
    ax_test.set_title("ED Demand – Train/Test & Test Forecast")
    ax_test.set_xlabel("Date")
    ax_test.set_ylabel("Visits")
    ax_test.legend()
    st.pyplot(fig_test)

    # ---------------- Future Forecast ----------------
    st.markdown("### ⏩ Future Forecast")

    future_forecast_res = model.get_forecast(steps=forecast_days)
    future_mean = future_forecast_res.predicted_mean
    future_conf = future_forecast_res.conf_int(alpha=0.05)

    last_date = df.index.max()
    future_index = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=forecast_days, freq="D")
    future_mean.index = future_index
    future_conf.index = future_index

    df_future = pd.DataFrame({
        "forecast": future_mean,
        "lower_ci": future_conf.iloc[:, 0],
        "upper_ci": future_conf.iloc[:, 1],
    })

    fig_future, ax_future = plt.subplots()
    ax_future.plot(df.index, df["visits"], label="History")
    ax_future.plot(df_future.index, df_future["forecast"], label="Forecast", linestyle="--")
    ax_future.fill_between(
        df_future.index,
        df_future["lower_ci"],
        df_future["upper_ci"],
        color="gray",
        alpha=0.2,
        label="95% CI",
    )
    ax_future.set_title("ED Visits – Historical and Future Forecast")
    ax_future.set_xlabel("Date")
    ax_future.set_ylabel("Visits")
    ax_future.legend()
    st.pyplot(fig_future)

    # ---------------- LLM EXPLANATION ----------------
    st.markdown("## 🧾 GenAI Explanation & Q&A")

    history_summary = build_history_summary(df)
    forecast_summary = build_forecast_summary(df_future)
    metrics_summary = build_metrics_summary(metrics)
    llm_context = build_llm_context(history_summary, forecast_summary, metrics_summary)

    with st.expander("🔍 Context sent to the LLM (for transparency)"):
        st.text(llm_context)

    if mistral_client is None:
        st.info("LLM explanation disabled because MISTRAL_API_KEY is missing.")
    else:
        if st.button("✨ Generate ED Forecast Explanation"):
            with st.spinner("Calling Mistral to generate explanation..."):
                explanation = generate_ed_explanation(llm_context)
            st.markdown("### 📣 Explanation for ED Manager")
            st.markdown(explanation)

        st.markdown("---")
        st.markdown("### 💬 Ask a Question as an ED Manager")

        question = st.text_input(
            "Type your question about the forecast or demand patterns:",
            value="Which days in the forecast period should we be most worried about for crowding and why?",
        )

        if st.button("📨 Get Answer"):
            if question.strip() == "":
                st.warning("Please enter a question.")
            else:
                with st.spinner("Calling Mistral to answer your question..."):
                    answer = answer_manager_question(question, llm_context)
                st.markdown("#### Manager's Question")
                st.write(question)
                st.markdown("#### GenAI ED Ops Copilot Answer")
                st.markdown(answer)


if __name__ == "__main__":
    main()
