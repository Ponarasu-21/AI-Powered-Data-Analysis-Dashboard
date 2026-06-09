import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score

st.set_page_config(
    page_title="AI Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("AI-Powered Data Analysis Dashboard")
st.markdown(
    "Upload a CSV dataset and explore preprocessing, visualizations, correlations, and basic AI insights."
)


@st.cache_data
def load_data(file) -> pd.DataFrame:
    try:
        return pd.read_csv(file)
    except Exception:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1")


def summarize_dataframe(df: pd.DataFrame):
    st.subheader("Dataset Overview")
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", df.columns.tolist())
    st.write("**Data types:**")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]))
    st.write("**First five rows:**")
    st.dataframe(df.head())

    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        st.write("**Summary statistics (numeric features):**")
        st.dataframe(numeric.describe().T)

    st.write("**Missing values:**")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.success("No missing values detected.")
    else:
        st.dataframe(missing)


def missing_value_handler(df: pd.DataFrame, strategy: str, constant_value=None):
    df = df.copy()
    if strategy == "Drop rows with missing values":
        return df.dropna()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if strategy == "Fill with mean":
        return df.fillna(df[numeric_cols].mean())
    if strategy == "Fill with median":
        return df.fillna(df[numeric_cols].median())
    if strategy == "Fill with mode":
        return df.fillna(df.mode().iloc[0].to_dict())
    if strategy == "Fill with constant":
        return df.fillna(constant_value)
    return df


def plot_correlation(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.info("Need at least two numeric features for correlation analysis.")
        return

    corr = numeric_df.corr()
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig)
    return corr


def plot_distribution(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
    st.pyplot(fig)


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f"Scatter plot: {x_col} vs {y_col}")
    st.pyplot(fig)


def generate_ai_insights(df: pd.DataFrame, corr: pd.DataFrame | None):
    st.subheader("AI-Generated Insights")
    insights = []
    if df.empty:
        st.info("No data available for insights.")
        return

    missing = df.isna().sum().sum()
    if missing > 0:
        insights.append(f"The dataset has {missing} missing values; consider imputation or row removal.")
    else:
        insights.append("The dataset has no missing values, which simplifies preprocessing.")

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        skewed = numeric_df.skew().abs().sort_values(ascending=False)
        top_skew = skewed[skewed > 1].head(3)
        if not top_skew.empty:
            insights.append(
                f"Features {list(top_skew.index)} are highly skewed and may benefit from scaling or transformation."
            )

    if corr is not None:
        pairs = []
        corr_values = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        sorted_pairs = corr_values.abs().unstack().dropna().sort_values(ascending=False)
        top_pairs = sorted_pairs.head(3)
        for (a, b), value in top_pairs.items():
            pairs.append((a, b, value))
        if pairs:
            for a, b, value in pairs:
                insights.append(f"{a} and {b} have a strong correlation of {value:.2f}.")

    if len(df.columns) > 1 and df.shape[0] > 10:
        insights.append(
            "The dataset is large enough for basic machine learning; select a target feature to evaluate predictive models."
        )

    for insight in insights:
        st.write(f"- {insight}")


def basic_ml_analysis(df: pd.DataFrame, target: str):
    st.subheader("Basic Machine Learning Analysis")
    df = df.copy()
    if target not in df.columns:
        st.warning("Selected target is not present in the dataset.")
        return

    y = df[target]
    X = df.drop(columns=[target])
    X = X.select_dtypes(include=[np.number]).copy()

    if X.empty:
        st.warning("No numeric feature columns available for a basic model.")
        return

    X = X.fillna(X.mean())
    is_numeric = pd.api.types.is_numeric_dtype(y)
    if not is_numeric or y.nunique() < 10:
        st.write("**Task:** Classification")
        y = y.fillna("missing")
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y.astype(str))
        if len(np.unique(y_encoded)) < 2:
            st.warning("The selected target does not have enough class variety for modeling.")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42
        )
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy on hold-out set: **{score:.2f}**")
        st.write("Top coefficients for numeric features:")
        coef_df = pd.DataFrame({"feature": X.columns, "coefficient": model.coef_[0]})
        st.dataframe(coef_df.sort_values(by="coefficient", key=abs, ascending=False).head(10))
    else:
        st.write("**Task:** Regression")
        y = y.fillna(y.mean())
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        st.write(f"R² score on hold-out set: **{score:.2f}**")
        st.write("Model coefficients for numeric features:")
        coef_df = pd.DataFrame({"feature": X.columns, "coefficient": model.coef_}).sort_values(
            by="coefficient", key=abs, ascending=False
        )
        st.dataframe(coef_df.head(10))


def main():
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    st.sidebar.markdown("---")
    st.sidebar.write("If you do not have a dataset yet, upload any CSV file with numeric and categorical columns.")

    if uploaded_file is None:
        st.info("Upload a CSV file to begin exploring your dataset.")
        return

    df = load_data(uploaded_file)
    if df.empty:
        st.error("The uploaded dataset is empty or could not be loaded.")
        return

    summarize_dataframe(df)

    with st.expander("Missing Value Handling"):
        strategy = st.selectbox(
            "Choose a missing value strategy",
            [
                "No changes",
                "Drop rows with missing values",
                "Fill with mean",
                "Fill with median",
                "Fill with mode",
                "Fill with constant",
            ],
        )
        constant_value = None
        if strategy == "Fill with constant":
            constant_value = st.text_input("Constant value for imputation", "0")
        if strategy != "No changes":
            df = missing_value_handler(df, strategy, constant_value)
            st.success(f"Missing value strategy applied: {strategy}")
            st.write("Updated missing values count:")
            st.dataframe(df.isna().sum().sort_values(ascending=False).head(20))

    corr = plot_correlation(df)

    with st.expander("Visualizations"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select numeric column for distribution", numeric_cols)
            plot_distribution(df, selected_col)

        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Select x-axis column", numeric_cols, index=0)
            y_col = st.selectbox("Select y-axis column", numeric_cols, index=1)
            plot_scatter(df, x_col, y_col)
        else:
            st.info("At least two numeric columns are required for scatter plots.")

    generate_ai_insights(df, corr)

    with st.expander("Basic AI/ML Analysis"):
        target_options = st.selectbox("Select a target column for prediction", [None] + df.columns.tolist())
        if target_options:
            basic_ml_analysis(df, target_options)

    st.markdown("---")
    st.write(
        "Use this dashboard to learn how missing-value handling, feature visualization, correlation analysis,"
        " and a simple predictive model work together in a data analysis pipeline."
    )


if __name__ == "__main__":
    main()
