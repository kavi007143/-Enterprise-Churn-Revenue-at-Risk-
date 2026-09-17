import streamlit as st
import pandas as pd
import numpy as np
import pickle


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="B2B SaaS Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📊 B2B SaaS Enterprise Churn & Revenue-at-Risk Engine")

st.write(
    "Enter the corporate account details below "
    "to predict churn probability and financial risk."
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    try:

        with open("xgb_model.pkl", "rb") as file:
            model = pickle.load(file)

        with open("model_features.pkl", "rb") as file:
            feature_names = pickle.load(file)

        return model, feature_names

    except FileNotFoundError:

        return None, None


model, feature_names = load_model()


# =========================================================
# CHECK MODEL
# =========================================================

if model is None:

    st.error(
        "Model files not found. "
        "Please keep xgb_model.pkl and model_features.pkl "
        "inside the same folder as Revenue.py."
    )

    st.stop()


# =========================================================
# INPUT SECTION
# =========================================================

col1, col2 = st.columns(2)


# =========================================================
# ACCOUNT OPERATIONAL METRICS
# =========================================================

with col1:

    st.subheader("👤 Account Operational Metrics")

    seats = st.number_input(
        "Total Seats Purchased",
        min_value=1,
        max_value=1000,
        value=50
    )

    total_subscriptions = st.number_input(
        "Total Active Subscriptions",
        min_value=1,
        max_value=10,
        value=1
    )

    total_tickets = st.number_input(
        "Support Tickets Raised",
        min_value=0,
        max_value=50,
        value=3
    )

    avg_resolution_time = st.number_input(
        "Average Resolution Time (hours)",
        min_value=0.0,
        max_value=100.0,
        value=4.0
    )

    avg_satisfaction_score = st.slider(
        "Customer Satisfaction Score (CSAT)",
        min_value=1.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )


# =========================================================
# FINANCIAL METRICS
# =========================================================

with col2:

    st.subheader("💰 Financial Metrics")

    total_arr = st.number_input(
        "Annual Recurring Revenue (ARR) - USD",
        min_value=1000,
        max_value=1000000,
        value=50000
    )

    total_mrr = total_arr / 12

    total_upgrades = st.number_input(
        "Total Upgrades",
        min_value=0,
        max_value=20,
        value=0
    )

    total_downgrades = st.number_input(
        "Total Downgrades",
        min_value=0,
        max_value=20,
        value=0
    )

    total_escalations = st.number_input(
        "Total Escalated Tickets",
        min_value=0,
        max_value=20,
        value=0
    )


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button(
    "🔍 Analyze Account Churn Risk",
    use_container_width=True
):

    # -----------------------------------------------------
    # Create input dictionary
    # -----------------------------------------------------

    input_values = {

        "seats": seats,

        "total_subscriptions":
            total_subscriptions,

        "total_mrr":
            total_mrr,

        "total_arr":
            total_arr,

        "total_seats_purchased":
            seats,

        "total_upgrades":
            total_upgrades,

        "total_downgrades":
            total_downgrades,

        "total_tickets":
            total_tickets,

        "avg_resolution_time":
            avg_resolution_time,

        "avg_satisfaction_score":
            avg_satisfaction_score,

        "total_escalations":
            total_escalations
    }


    # -----------------------------------------------------
    # Create DataFrame
    # -----------------------------------------------------

    input_data = pd.DataFrame(
        [input_values]
    )


    # -----------------------------------------------------
    # Add missing model features
    # -----------------------------------------------------

    for column in feature_names:

        if column not in input_data.columns:

            input_data[column] = 0


    # -----------------------------------------------------
    # Keep EXACT training feature order
    # -----------------------------------------------------

    input_data = input_data[
        feature_names
    ]


    # -----------------------------------------------------
    # Convert to numeric
    # -----------------------------------------------------

    input_data = input_data.astype(float)


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    churn_probability = model.predict_proba(
        input_data
    )[0][1]


    churn_prediction = model.predict(
        input_data
    )[0]


    # =====================================================
    # RESULT
    # =====================================================

    st.divider()

    st.subheader(
        "📊 Risk Assessment Result"
    )


    # -----------------------------------------------------
    # HIGH RISK
    # -----------------------------------------------------

    if churn_prediction == 1:

        st.error(
            f"🔴 HIGH RISK — "
            f"Churn Probability: "
            f"{churn_probability * 100:.2f}%"
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )

        with result_col2:

            st.metric(
                "Revenue-at-Risk",
                f"${total_arr:,.2f}"
            )


        st.warning(
            "This account has been classified as a "
            "high-risk churn account."
        )


    # -----------------------------------------------------
    # LOW RISK
    # -----------------------------------------------------

    else:

        st.success(
            f"🟢 LOW RISK — "
            f"Churn Probability: "
            f"{churn_probability * 100:.2f}%"
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )

        with result_col2:

            st.metric(
                "Annual Revenue",
                f"${total_arr:,.2f}"
            )


        st.info(
            "This account has been classified as "
            "lower churn risk by the model."
        )


    # =====================================================
    # ACCOUNT SUMMARY
    # =====================================================

    st.divider()

    st.subheader(
        "📋 Account Summary"
    )

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:

        st.metric(
            "Seats",
            seats
        )

    with summary_col2:

        st.metric(
            "Support Tickets",
            total_tickets
        )

    with summary_col3:

        st.metric(
            "CSAT Score",
            f"{avg_satisfaction_score:.1f}/5"
        )

    with summary_col4:

        st.metric(
            "ARR",
            f"${total_arr:,.0f}"
        )


    # =====================================================
    # BUSINESS INSIGHTS
    # =====================================================

    st.divider()

    st.subheader(
        "💡 Business Insights"
    )

    if total_tickets > 5:

        st.write(
            "⚠️ High number of support tickets detected."
        )

    if avg_satisfaction_score < 3:

        st.write(
            "⚠️ Customer satisfaction score is low."
        )

    if total_downgrades > total_upgrades:

        st.write(
            "⚠️ Downgrades are higher than upgrades."
        )

    if total_escalations > 0:

        st.write(
            "⚠️ Escalated support tickets detected."
        )

    if (
        total_tickets <= 5
        and avg_satisfaction_score >= 3
        and total_escalations == 0
    ):

        st.write(
            "✅ No major operational warning indicators detected."
        )