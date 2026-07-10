"""
ScamLens AI - UPI Fraud Analytics
Streamlit Web App
----------------------------------
Covers: Data Overview, EDA Dashboard, ML Fraud Severity Predictor,
Model Performance Insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from utils import (
    full_pipeline,
    train_model,
    save_model,
    predict_severity,
    get_recommendation,
)

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="ScamLens AI - UPI Fraud Analytics",
    page_icon="🕵️",
    layout="wide",
)

sns.set_theme(style="whitegrid")

# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================
st.sidebar.title("🕵️ ScamLens AI")
st.sidebar.caption("UPI Fraud Analytics & Severity Predictor")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 EDA Dashboard", "🤖 Fraud Predictor", "📈 Model Insights", "📌 Insights & Conclusion"],
)

# ==========================================================
# LOAD DATA (cached so it only runs once per session)
# ==========================================================
@st.cache_data
def get_data():
    return full_pipeline("scam_dataset.csv")


try:
    df = get_data()
except FileNotFoundError:
    st.error(
        "⚠ `scam_dataset.csv` nahi mila. Is file ko app.py ke saath "
        "repo ke root folder mein rakhein."
    )
    st.stop()

# ==========================================================
# TRAIN / LOAD MODEL (cached as a resource)
# ==========================================================
@st.cache_resource
def get_model(df):
    model, encoders, target_encoder, metrics = train_model(df)
    save_model(model, encoders, target_encoder)  # keeps .pkl files updated too
    return model, encoders, target_encoder, metrics

model, encoders, target_encoder, metrics = get_model(df)


# ==========================================================
# PAGE 1 : OVERVIEW
# ==========================================================
if page == "🏠 Overview":

    st.title("🕵️ ScamLens AI - Intelligent UPI Fraud Analytics & Risk Prediction System")

    st.markdown("""
### 📖 Project Overview
                
ScamLens AI is a data-driven fraud analytics platform developed using **Python, Data Science, Machine Learning, and Streamlit** to analyze fraudulent UPI transactions and identify hidden fraud patterns.
The project combines **data preprocessing, feature engineering, exploratory data analysis (EDA), business intelligence, and machine learning** to understand fraud behavior and support secure digital payment systems.
The interactive dashboard enables users to explore fraud trends, visualize transaction patterns, and predict fraud severity using an AI-based Decision Tree model.
""")

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Total Fraud Amount", f"₹{df['amount_inr'].sum():,.0f}")
    col3.metric("Average Fraud Amount", f"₹{df['amount_inr'].mean():,.0f}")
    col4.metric("Highest Fraud Amount", f"₹{df['amount_inr'].max():,.0f}")

    st.divider()

    st.subheader("🎯 Project Objectives")

    st.markdown("""
- Analyze fraudulent UPI transaction data.
- Perform data cleaning and preprocessing.
- Engineer meaningful features for better analysis.
- Discover fraud patterns using interactive visualizations.
- Build a Machine Learning model for fraud severity prediction.
- Support digital payment security through AI-driven insights.
""")

    st.divider()

    st.subheader("🛠 Technologies Used")

    st.markdown("""
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Streamlit
""")

    st.divider()

    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)


# ==========================================================
# PAGE 2 : EDA DASHBOARD
# ==========================================================
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Fraud Types & Lures", "Apps & Transactions", "Time Patterns", "Relationships"]
    )

    # ---- TAB 1 ----
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fraud_type = df["fraud_type"].value_counts().reset_index()
            fraud_type.columns = ["Fraud Type", "Count"]
            fig = px.bar(fraud_type, x="Fraud Type", y="Count", color="Fraud Type",
                         text="Count", title="Most Common Fraud Types")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            lure = df["fraud_lure"].value_counts().reset_index()
            lure.columns = ["Fraud Lure", "Count"]
            fig = px.bar(lure, x="Fraud Lure", y="Count", color="Fraud Lure",
                         text="Count", title="Fraud Lure Distribution")
            st.plotly_chart(fig, use_container_width=True)

        lure_amt = df.groupby("fraud_lure")["amount_inr"].sum().sort_values(ascending=False)
        fig = px.bar(x=lure_amt.index, y=lure_amt.values, color=lure_amt.index,
                     text=lure_amt.values, title="Top Fraud Lures by Total Amount")
        st.plotly_chart(fig, use_container_width=True)

    # ---- TAB 2 ----
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            app = df["upi_app"].value_counts().reset_index()
            app.columns = ["UPI App", "Count"]
            fig = px.pie(app, names="UPI App", values="Count", hole=0.5,
                         title="Most Used UPI App")
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig, ax = plt.subplots(figsize=(6, 6))
            counts = df["transaction_type"].value_counts()
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
                   shadow=True, explode=[0.05] * len(counts))
            ax.set_title("Transaction Type Distribution")
            st.pyplot(fig)

        upi_avg = df.groupby("upi_app")["amount_inr"].mean().sort_values()
        fig = px.bar(x=upi_avg.values, y=upi_avg.index, orientation="h",
                     text=upi_avg.values, title="Average Fraud Amount by UPI App")
        st.plotly_chart(fig, use_container_width=True)

        fraud_avg = df.groupby("fraud_type")["amount_inr"].mean().sort_values()
        fig = px.bar(x=fraud_avg.values, y=fraud_avg.index, orientation="h",
                     text=fraud_avg.values, title="Average Fraud Amount by Fraud Type")
        st.plotly_chart(fig, use_container_width=True)

    # ---- TAB 3 ----
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            day = df["Day_Name"].value_counts().reset_index()
            day.columns = ["Day", "Count"]
            fig = px.bar(day, x="Day", y="Count", color="Day", text="Count",
                         title="Day-wise Fraud")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            day_type = df["Day_Type"].value_counts().reset_index()
            day_type.columns = ["Day Type", "Count"]
            fig = px.pie(day_type, names="Day Type", values="Count", hole=0.5,
                         title="Weekend vs Weekday")
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            order = ["Low", "Medium", "High"]
            amt = df["Amount_Category"].value_counts().reindex(order).reset_index()
            amt.columns = ["Amount Category", "Count"]
            fig = px.bar(amt, x="Amount Category", y="Count", color="Amount Category",
                         text="Count", title="Amount Category")
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            order = ["Morning", "Afternoon", "Evening", "Night"]
            ts = df["Time_Slot"].value_counts().reindex(order).reset_index()
            ts.columns = ["Time Slot", "Count"]
            fig = px.bar(ts, x="Time Slot", y="Count", color="Time Slot",
                         text="Count", title="Time Slot Distribution")
            st.plotly_chart(fig, use_container_width=True)

    # ---- TAB 4 ----
    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(pd.crosstab(df["fraud_type"], df["transaction_type"]),
                        annot=True, fmt="d", cmap="YlOrRd", ax=ax)
            ax.set_title("Fraud Type vs Transaction Type")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(pd.crosstab(df["fraud_type"], df["upi_app"]),
                        annot=True, fmt="d", cmap="BuPu", linewidths=0.5, ax=ax)
            ax.set_title("Fraud Type vs UPI App")
            st.pyplot(fig)

        c3, c4 = st.columns(2)
        with c3:
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(pd.crosstab(df["fraud_lure"], df["upi_app"]),
                        annot=True, fmt="d", cmap="crest", linewidths=0.5, ax=ax)
            ax.set_title("Fraud Lure vs UPI App")
            st.pyplot(fig)

        with c4:
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.countplot(data=df, x="upi_app", hue="Fraud_Severity", ax=ax)
            ax.set_title("Fraud Severity by UPI App")
            plt.xticks(rotation=30)
            st.pyplot(fig)

        fig = px.scatter(df, x="amount_inr", y="fraud_type", color="upi_app",
                          size="amount_inr", hover_name="fraud_lure",
                          title="Bubble Chart of Fraud Transactions")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🏆 Top 10 Largest Frauds")
        top10 = df.sort_values("amount_inr", ascending=False).head(10)
        st.dataframe(top10, use_container_width=True)

# ==========================================================
# PAGE 3 : FRAUD PREDICTOR
# ==========================================================
elif page == "🤖 Fraud Predictor":
    st.title("🤖 ScamLens AI - Fraud Severity Predictor")
    st.caption("Enter transaction details to predict fraud risk severity.")

    with st.form("predict_form"):
        c1, c2 = st.columns(2)

        with c1:
            amount = st.number_input("💰 Amount (₹)", min_value=1.0, value=5000.0, step=100.0)
            time_slot = st.selectbox("🕒 Time Slot", list(encoders["Time_Slot"].classes_))
            day_type = st.selectbox("📅 Day Type", list(encoders["Day_Type"].classes_))

        with c2:
            upi_app = st.selectbox("📲 UPI App", list(encoders["upi_app"].classes_))
            transaction_type = st.selectbox("💳 Transaction Type", list(encoders["transaction_type"].classes_))
            fraud_lure = st.selectbox("🎣 Fraud Lure", list(encoders["fraud_lure"].classes_))

        submitted = st.form_submit_button("🔮 Predict Severity")

    if submitted:
        severity = predict_severity(
            model, encoders, target_encoder,
            amount, time_slot, day_type, upi_app, transaction_type, fraud_lure,
        )
        rec = get_recommendation(severity)

        st.markdown("---")
        st.markdown(f"## {rec['emoji']} Predicted Severity: **{severity}** ({rec['level']})")
        st.markdown("### 📌 AI Recommendation")
        for tip in rec["tips"]:
            st.markdown(f"- {tip}")

# ==========================================================
# PAGE 4 : MODEL INSIGHTS
# ==========================================================
elif page == "📈 Model Insights":
    st.title("📈 Model Performance - Decision Tree Classifier")

    st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")

    st.markdown("### 📋 Classification Report")
    report_df = pd.DataFrame(metrics["report"]).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.markdown("### 🔀 Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
        xticklabels=metrics["classes"], yticklabels=metrics["classes"], ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    st.pyplot(fig)


elif page == "📌 Insights & Conclusion":

    st.title("📌 Insights, Conclusion & Future Scope")

    tab1, tab2, tab3 = st.tabs(
        ["📊 Key Insights", "✅ Conclusion", "🚀 Future Scope"]
    )

    # -------------------------------------------------------
    # KEY INSIGHTS
    # -------------------------------------------------------
    with tab1:

        st.subheader("📊 Key Insights")

        st.markdown("""
### 🔍 Fraud Pattern Analysis

✅ Fraud transactions are concentrated in specific fraud categories and time periods, indicating that targeted monitoring can improve fraud detection efficiency.

✅ High-value UPI transactions have a greater risk of fraud, making real-time verification essential for secure digital payments.

✅ Certain UPI applications and fraud techniques appear more frequently in scam cases, highlighting the need for stronger platform-specific security measures.

✅ Data visualization reveals clear fraud patterns, enabling organizations to identify suspicious activities and make faster, data-driven decisions.

✅ The Decision Tree model successfully classifies fraudulent transactions, demonstrating the effectiveness of Machine Learning in fraud detection.

✅ Early fraud prediction can help financial institutions reduce financial losses, improve customer trust, and strengthen digital payment security.

✅ Fraud occurrence depends not only on transaction amount but also on fraud type, transaction timing, payment platform, and user behavior, indicating that multiple features together improve prediction accuracy.
""")

    # -------------------------------------------------------
    # CONCLUSION
    # -------------------------------------------------------
    with tab2:

        st.subheader("✅ Project Conclusion")

        st.markdown("""
This project successfully analyzed fraudulent UPI transaction data using Python, Data Science, and Machine Learning techniques.

Data preprocessing, feature engineering, exploratory data analysis, and predictive modeling helped identify meaningful fraud patterns and classify fraud severity effectively.

The Decision Tree model demonstrates that intelligent data-driven approaches can support financial institutions, banks, payment platforms, and cybersecurity teams in detecting fraudulent transactions more efficiently.

Overall, ScamLens AI showcases how Machine Learning and interactive analytics dashboards can improve fraud detection, reduce financial losses, and enhance digital payment security.
""")

    # -------------------------------------------------------
    # FUTURE SCOPE
    # -------------------------------------------------------
    with tab3:

        st.subheader("🚀 Future Scope")

        st.markdown("""
- Implement advanced Machine Learning algorithms such as Random Forest, XGBoost, and LightGBM.

- Integrate real-time UPI transaction monitoring.

- Improve prediction accuracy using larger and more diverse datasets.

- Develop an AI-based Fraud Risk Score.

- Integrate APIs for live fraud monitoring and automated alerts.

- Add SMS and Email notification systems for suspicious transactions.

- Develop an interactive dashboard for banks, investigators, and cybersecurity teams.

- Deploy the application on Streamlit Cloud for public access.

- Enhance explainable AI (XAI) features to improve model transparency.

- Extend the system to support fraud detection across multiple digital payment platforms.
""")


st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit | ScamLens AI")