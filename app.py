import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import json
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# USER DATABASE
# -----------------------------

USER_FILE = "users.json"

if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)

def load_users():
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# -----------------------------
# LOGIN / REGISTER
# -----------------------------

if not st.session_state.logged_in:

    st.markdown("""
    <h1 style='text-align:center; color:#1f77b4;'>
    💳 AI Fraud Detection System
    </h1>

    <p style='text-align:center; font-size:18px;'>
    Secure Banking Fraud Analytics Platform
    </p>
    """, unsafe_allow_html=True)

    login_tab, register_tab = st.tabs(
        ["🔑 Login", "📝 Register"]
    )

    with login_tab:

        username = st.text_input(
            "Username",
            key="login_username"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login"):

            users = load_users()

            if (
                username in users
                and users[username] == password
            ):

                st.session_state.logged_in = True
                st.session_state.username = username

                st.rerun()

            else:

                st.error(
                    "Invalid Username or Password"
                )

    with register_tab:

        new_user = st.text_input(
            "Create Username"
        )

        new_pass = st.text_input(
            "Create Password",
            type="password"
        )

        if st.button("Register"):

            users = load_users()

            if new_user in users:

                st.warning(
                    "Username already exists"
                )

            else:

                users[new_user] = new_pass

                save_users(users)

                st.success(
                    "Registration Successful. Please Login."
                )

    st.stop()

# -----------------------------
# LOAD MODEL
# -----------------------------

model = joblib.load(
    "models/fraud_model.pkl"
)

# -----------------------------
# HEADER
# -----------------------------

col1, col2 = st.columns([8,1])

with col1:

    st.markdown(f"""
    <h1 style='color:#1f77b4;'>
    💳 AI Fraud Detection Dashboard
    </h1>

    <p>
    Welcome, <b>{st.session_state.username}</b>
    </p>
    """, unsafe_allow_html=True)

with col2:

    if st.button("🚪 Logout"):

        st.session_state.logged_in = False
        st.session_state.username = ""

        st.rerun()

st.markdown("---")

# -----------------------------
# FILE UPLOADER
# -----------------------------

uploaded_file = st.file_uploader(
    "📂 Upload Transaction CSV File",
    type=["csv"],
    key="main_uploader"
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    original_df = df.copy()

    if "Class" in df.columns:
        df = df.drop("Class", axis=1)

    predictions = model.predict(df)

    original_df["Prediction"] = predictions

    fraud_count = (predictions == -1).sum()
    normal_count = (predictions == 1).sum()

    total_transactions = len(
        original_df
    )

    fraud_percentage = (
        fraud_count /
        total_transactions
    ) * 100

    # -----------------------------
    # KPI CARDS
    # -----------------------------

    st.subheader("📊 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "💳 Total Transactions",
            f"{total_transactions:,}"
        )

    with col2:

        st.metric(
            "🚨 Fraud Cases",
            fraud_count
        )

    with col3:

        st.metric(
            "✅ Normal Cases",
            normal_count
        )

    with col4:

        st.metric(
            "📈 Fraud Rate",
            f"{fraud_percentage:.2f}%"
        )

    if fraud_count > 0:

        st.warning(
            f"⚠ {fraud_count} suspicious transactions detected"
        )

    else:

        st.success(
            "✅ No suspicious transactions found"
        )

    st.markdown("---")

    # -----------------------------
    # CHARTS
    # -----------------------------

    st.subheader("📈 Fraud Analytics")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        fig, ax = plt.subplots(
            figsize=(4,4)
        )

        ax.pie(
            [fraud_count, normal_count],
            labels=[
                "Fraud",
                "Normal"
            ],
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title(
            "Transaction Distribution"
        )

        st.pyplot(fig)

    with chart_col2:

        chart_data = pd.DataFrame(
            {
                "Category": [
                    "Fraud",
                    "Normal"
                ],
                "Count": [
                    fraud_count,
                    normal_count
                ]
            }
        )

        st.bar_chart(
            chart_data.set_index(
                "Category"
            )
        )

    st.markdown("---")

    # -----------------------------
    # FRAUD RECORDS
    # -----------------------------

    st.subheader(
        "🚨 Suspicious Transactions"
    )

    fraud_df = original_df[
        original_df["Prediction"] == -1
    ]

    search = st.text_input(
        "🔍 Search Fraud Records"
    )

    if search:

        fraud_df = fraud_df[
            fraud_df.astype(str)
            .apply(
                lambda x:
                x.str.contains(
                    search,
                    case=False
                )
            )
            .any(axis=1)
        ]

    st.dataframe(
        fraud_df,
        use_container_width=True,
        height=400
    )

    st.markdown("---")

    # -----------------------------
    # DOWNLOAD REPORT
    # -----------------------------

    csv = fraud_df.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download Fraud Report",
        data=csv,
        file_name="fraud_report.csv",
        mime="text/csv"
    )

    st.markdown("---")

    with st.expander(
        "ℹ About Project"
    ):

        st.write("""
        ### AI Fraud Detection System

        This project uses Isolation Forest
        Machine Learning Algorithm to
        detect suspicious financial
        transactions.

        ### Technologies Used

        - Python
        - Pandas
        - NumPy
        - Scikit-Learn
        - Streamlit
        - Matplotlib

        ### Features

        - Login & Registration
        - Fraud Detection
        - Analytics Dashboard
        - Fraud Record Search
        - Download Reports
        - Interactive Charts
        """)

else:

    st.info(
        "📂 Upload a CSV file to start fraud analysis."
    )