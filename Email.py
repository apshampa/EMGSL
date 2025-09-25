import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Data Lab Dashboard", layout="wide")
st.title("🧹📊 One-Stop Data  Dashboard")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("📂 Drop your Excel or CSV file", type=["xlsx", "csv"])
if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.success(f"✅ Loaded file: {uploaded_file.name}")
    st.write("### Preview of Data", df.head())

    # --------------------------------
    # Data Cleaning Section
    # --------------------------------
    st.subheader("🧹 Data Cleaning")

    # Drop Duplicates
    if st.checkbox("Remove duplicate rows"):
        df = df.drop_duplicates()
        st.info("✅ Duplicates removed")

    # Handle Missing Values
    missing_strategy = st.selectbox(
        "Handle Missing Values",
        ["Do Nothing", "Drop Rows", "Drop Columns", "Fill with Mean", "Fill with Median", "Fill with Mode"]
    )
    if missing_strategy == "Drop Rows":
        df = df.dropna()
    elif missing_strategy == "Drop Columns":
        df = df.dropna(axis=1)
    elif missing_strategy == "Fill with Mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif missing_strategy == "Fill with Median":
        df = df.fillna(df.median(numeric_only=True))
    elif missing_strategy == "Fill with Mode":
        df = df.apply(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)

    st.write("Data after cleaning:", df.head())

    # -----------------------------
    # Data Overview
    # -----------------------------
    st.subheader("🔹 Data Overview")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["Type"]))

    # -----------------------------
    # Missing Values
    # -----------------------------
    st.subheader("🔹 Missing Values Report")
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Column", "Missing Values"]
    missing_df["% Missing"] = (missing_df["Missing Values"] / df.shape[0] * 100).round(2)
    st.dataframe(missing_df)

    # -----------------------------
    # Descriptive Statistics
    # -----------------------------
    st.subheader("🔹 Descriptive Statistics")
    st.dataframe(df.describe(include="all").T)

    # -----------------------------
    # Correlation Matrix
    # -----------------------------
    st.subheader("🔹 Correlation Matrix")
    method = st.selectbox("Select correlation method", ["pearson", "spearman", "kendall"])
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 1:
        corr = df[num_cols].corr(method=method)
        st.dataframe(corr.style.background_gradient(cmap="coolwarm"))
    else:
        st.warning("Not enough numeric columns for correlation")

    # -----------------------------
    # Outlier Detection
    # -----------------------------
    st.subheader("🔹 Outlier Detection (IQR Method)")
    outlier_summary = {}
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
        outlier_summary[col] = len(outliers)
    st.dataframe(pd.DataFrame.from_dict(outlier_summary, orient="index", columns=["Outliers"]))

    # -----------------------------
    # Column Distribution
    # -----------------------------
    st.subheader("🔹 Column Distributions")
    col_choice = st.selectbox("Pick a column to explore", df.columns)
    if df[col_choice].dtype in [np.float64, np.int64]:
        st.bar_chart(df[col_choice].dropna())
    else:
        st.write(df[col_choice].value_counts().head(10))

    
    st.subheader("⬇️ Download Cleaned Data")
    cleaned_file = "cleaned_data.xlsx"
    df.to_excel(cleaned_file, index=False)
    with open(cleaned_file, "rb") as f:
        st.download_button("Download Cleaned Excel", f, file_name="cleaned_data.xlsx")
