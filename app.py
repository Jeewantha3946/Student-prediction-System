from pathlib import Path
from html import escape

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Student Performance | Academic Analytics",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
DATA_PATH = BASE_DIR / "data" / "student_performance_cleaned.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"

NUMERICAL_FEATURES = [
    "Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
    "Tutoring_Sessions", "Physical_Activity",
]
CATEGORICAL_FEATURES = [
    "Parental_Involvement", "Access_to_Resources", "Extracurricular_Activities",
    "Motivation_Level", "Internet_Access", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender",
]
MODEL_COMPARISON = pd.DataFrame(
    [
        ["Linear Regression", 0.5116, 2.0840, 0.7166],
        ["Tuned XGBoost", 0.6890, 2.1660, 0.6943],
        ["XGBoost", 0.8095, 2.2038, 0.6837],
        ["Tuned Random Forest", 1.1058, 2.3486, 0.6414],
        ["Random Forest", 1.1701, 2.3980, 0.6262],
    ], columns=["Model", "MAE", "RMSE", "R-squared"],
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    return pd.read_csv(DATA_PATH)


def display_label(name):
    return name.replace("_", " ")


def metric_card(label, value, detail=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


def navigate_to_prediction():
    st.session_state["nav_page"] = "Predict Performance"


def set_navigation(page_name):
    st.session_state["nav_page"] = page_name


st.markdown(
    """<style>
    :root {
      --teal:#197c78;
    }
    html, body, [class*="css"] { font-family:Inter,'Segoe UI',Arial,sans-serif; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
      background-color:var(--background-color, #f7f8f7) !important;
      color:var(--text-color, #172d3d);
    }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stMainBlockContainer"] { max-width:100%; padding-top:2rem; padding-bottom:3rem; }
    [data-testid="stSidebar"] {
      background-color:var(--secondary-background-color, #ffffff) !important;
      border-right:1px solid color-mix(in srgb, var(--text-color, #172d3d) 14%, transparent);
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] { color:#fff !important; }
    [data-testid="stMain"] label,
    [data-testid="stMain"] [data-testid="stWidgetLabel"] *,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3 { color:var(--text-color, #172d3d) !important; }
    [data-testid="stMain"] [data-testid="stCaptionContainer"] { color:var(--text-color, #172d3d);opacity:.68; }
    .brand { padding:12px 5px 20px; border-bottom:1px solid color-mix(in srgb, var(--text-color, #172d3d) 14%, transparent); margin-bottom:18px; }
    .brand-mark { display:inline-flex; width:34px;height:34px;align-items:center;justify-content:center;background:color-mix(in srgb, var(--teal) 14%, var(--secondary-background-color, #fff));color:var(--teal);border-radius:7px;font-size:15px;font-weight:700;margin-right:10px;vertical-align:middle; }
    .brand-name { display:inline-block;vertical-align:middle;font-weight:700;font-size:14px;line-height:1.3; }
    .brand-sub { display:block;font-size:10px;font-weight:500;color:var(--text-color, #172d3d) !important;opacity:.68;letter-spacing:.07em;text-transform:uppercase; }
    .eyebrow { text-transform:uppercase;letter-spacing:.12em;color:var(--teal);font-size:10px;font-weight:700;margin:0 0 8px; }
    h1,h2,h3 { color:var(--text-color, #172d3d) !important; letter-spacing:-.025em; }
    h1 { font-size:32px !important; } h2 { font-size:24px !important; } h3 { font-size:18px !important; }
    .hero { background-color:var(--secondary-background-color, #fff);border:1px solid color-mix(in srgb, var(--text-color, #172d3d) 14%, transparent);border-left:4px solid var(--teal);padding:26px 28px;border-radius:7px;margin:8px 0 22px; }
    .hero h1 { margin:0 0 8px; max-width:680px; font-size:31px !important; }
    .hero p { color:var(--text-color, #172d3d);opacity:.72;max-width:650px;font-size:15px;line-height:1.65;margin:0; }
    .metric-card { background-color:var(--secondary-background-color, #fff);border:1px solid color-mix(in srgb, var(--text-color, #172d3d) 14%, transparent);border-radius:7px;padding:16px 17px;min-height:104px; }
    .metric-label { color:var(--text-color, #172d3d);opacity:.72;font-size:12px;font-weight:600; }
    .metric-value { color:var(--text-color, #172d3d);font-weight:700;font-size:22px;margin:8px 0 3px; }
    .metric-detail { color:var(--text-color, #172d3d);opacity:.6;font-size:11px; }
    .section-head { margin:22px 0 10px; }
    .soft-card { background-color:var(--secondary-background-color, #fff);border:1px solid color-mix(in srgb, var(--text-color, #172d3d) 14%, transparent);border-radius:7px;padding:19px 21px; }
    .soft-card p { color:var(--text-color, #172d3d) !important;opacity:.7; }
    .comparison-table { width:100%;border-collapse:collapse;background:var(--secondary-background-color, #fff);color:var(--text-color, #172d3d);font-size:14px;border:1px solid color-mix(in srgb, var(--text-color, #172d3d) 14%, transparent);border-radius:7px;overflow:hidden; }
    .comparison-table th,.comparison-table td { padding:11px 14px;border-bottom:1px solid color-mix(in srgb, var(--text-color, #172d3d) 14%, transparent); }
    .comparison-table th { background:color-mix(in srgb, var(--teal) 9%, var(--secondary-background-color, #fff));color:var(--text-color, #172d3d);font-weight:650;text-align:left; }
    .comparison-table td:not(:first-child) { text-align:right;font-variant-numeric:tabular-nums; }
    .comparison-table tr:last-child td { border-bottom:0; }
    .result-card { background:#173f4c;border-radius:8px;padding:23px 26px;color:white; }
    .result-card * { color:white !important; }
    .result-score { font-weight:750;font-size:42px;letter-spacing:-.03em; }
    .result-caption { color:#d4e3e5 !important;font-size:13px; }
    .stButton>button[kind="primary"],button[kind="primary"] { background:var(--teal);color:#fff;border:0;border-radius:6px;padding:9px 16px;font-weight:650; }
    .stButton>button[kind="primary"]:hover { background:#146b68; }
    .stButton>button[kind="secondary"],button[kind="secondary"] { background:var(--secondary-background-color, #fff);color:var(--text-color, #172d3d);border:1px solid color-mix(in srgb, var(--text-color, #172d3d) 14%, transparent);border-radius:6px;padding:9px 16px;font-weight:600; }
    [data-testid="stFormSubmitButton"] button { background:var(--teal) !important;color:#fff !important;border:0 !important;border-radius:6px; }
    [data-testid="stBaseButton-primary"] { background:var(--teal) !important;color:#fff !important; }
    [data-testid="stBaseButton-primary"] * { color:#fff !important; }
    div[data-testid="stForm"] { background-color:var(--secondary-background-color, #fff);border:1px solid color-mix(in srgb, var(--text-color, #172d3d) 14%, transparent);border-radius:7px;padding:18px 20px; }
    div[data-testid="stExpander"] { border:1px solid color-mix(in srgb, var(--text-color, #172d3d) 14%, transparent);border-radius:7px;background-color:var(--secondary-background-color, #fff); }
    .note { font-size:13px;color:var(--text-color, #172d3d);line-height:1.6;background:color-mix(in srgb, var(--teal) 9%, var(--secondary-background-color, #fff));border-left:3px solid var(--teal);padding:11px 14px;border-radius:2px; }
    @media(max-width:700px){ .hero{padding:22px 18px}.hero h1{font-size:26px!important}.result-score{font-size:36px} }
    </style>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">SP</span>'
        '<span class="brand-name">Student Performance<span class="brand-sub">Academic insights</span></span></div>',
        unsafe_allow_html=True,
    )
    navigation_pages = ["Dashboard", "Predict Performance", "Model Insights"]
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Dashboard"
    for navigation_page in navigation_pages:
        st.button(
            navigation_page,
            key=f"nav_{navigation_page.lower().replace(' ', '_')}",
            type="primary" if st.session_state["nav_page"] == navigation_page else "secondary",
            width="stretch",
            on_click=set_navigation,
            args=(navigation_page,),
        )
    page = st.session_state["nav_page"]
    st.markdown("<div style='height:24px'></div><span class='brand-sub'>MODEL</span>", unsafe_allow_html=True)
    st.caption("Linear Regression")

try:
    reference_df = load_reference_data()
except Exception as error:
    st.error(f"Couldn't load the cleaned student dataset at `{DATA_PATH}`. Check that the project data file is present. Details: {error}")
    st.stop()

if reference_df.empty:
    st.error("The cleaned student dataset is empty. Add the existing cleaned data file to continue.")
    st.stop()

if page == "Dashboard":
    st.markdown('<div class="eyebrow">STUDENT SUCCESS · DATA INFORMED</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><h1>Student performance, in context.</h1>'
        '<p>Explore the saved model’s evaluation or prepare an individual score estimate from a student’s learning profile.</p></div>',
        unsafe_allow_html=True,
    )
    st.button(
        "Start a prediction",
        type="primary",
        key="dashboard_predict",
        on_click=navigate_to_prediction,
    )
    st.markdown("<div class='section-head'><div class='eyebrow'>AT A GLANCE</div></div>", unsafe_allow_html=True)
    cols = st.columns(4)
    with cols[0]: metric_card("Selected model", "Linear Regression", "Saved model pipeline")
    with cols[1]: metric_card("Test MAE", "0.4160", "Final test set")
    with cols[2]: metric_card("Test RMSE", "1.5213", "Final test set")
    with cols[3]: metric_card("Test R²", "0.8250", "Final test set")
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    cols = st.columns([1, 1, 2])
    with cols[0]: metric_card("Student records", f"{len(reference_df):,}", "Cleaned dataset")
    with cols[1]: metric_card("Input features", "19", "Academic and context factors")
    with cols[2]:
        st.markdown('<div class="soft-card"><div class="eyebrow">PREDICTION</div><h3>Explore a student profile</h3><p style="color:#637783">Enter academic and contextual information to see an estimate with the submitted inputs.</p></div>', unsafe_allow_html=True)

elif page == "Predict Performance":
    st.markdown('<div class="eyebrow">STUDENT PROFILE</div>', unsafe_allow_html=True)
    st.title("Predict performance")
    st.markdown("Add the student’s academic profile and learning context. The estimate is intended to support reflection and discussion.")
    try:
        model = load_model()
        feature_order = list(model.feature_names_in_)
    except Exception as error:
        st.error(f"Couldn't load the saved model at `{MODEL_PATH}`. Confirm that `models/best_model.pkl` is available and compatible. Details: {error}")
        st.stop()
    expected = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
    if len(feature_order) != 19 or set(feature_order) != expected:
        st.error("The saved model's expected features don't match the 19 fields supported by this form. No prediction was made.")
        st.stop()

    with st.form("prediction_form"):
        st.markdown("### Learning profile")
        st.caption("Study habits and prior academic experience")
        numeric_inputs = {}
        cols = st.columns(3)
        for i, feature in enumerate(NUMERICAL_FEATURES):
            with cols[i % 3]:
                values = pd.to_numeric(reference_df[feature], errors="coerce").dropna()
                upper = 100 if feature in ("Attendance", "Previous_Scores") else max(1, int(values.max()))
                default = max(0, min(upper, int(values.median())))
                numeric_inputs[feature] = st.number_input(
                    display_label(feature), min_value=0, max_value=upper,
                    value=default, step=1, help=f"Valid range: 0–{upper}.",
                )
        st.markdown("### Support and environment")
        st.caption("Student circumstances that help provide context")
        category_inputs = {}
        cols = st.columns(3)
        for i, feature in enumerate(CATEGORICAL_FEATURES):
            with cols[i % 3]:
                options = sorted(reference_df[feature].dropna().astype(str).unique().tolist())
                if not options:
                    st.error(f"No category values found for {display_label(feature)}.")
                    st.stop()
                category_inputs[feature] = st.selectbox(display_label(feature), options)
        submitted = st.form_submit_button("Predict exam score", type="primary", width="stretch")

    if submitted:
        inputs = {**numeric_inputs, **category_inputs}
        input_df = pd.DataFrame([[inputs[name] for name in feature_order]], columns=feature_order)
        try:
            raw_prediction = float(model.predict(input_df)[0])
            if not pd.notna(raw_prediction):
                st.error("The model returned an invalid prediction. Please review the profile and try again.")
            else:
                score = max(0.0, min(100.0, raw_prediction))
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                st.markdown('<div class="eyebrow">INDIVIDUAL ESTIMATE</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="result-card"><div class="result-caption">Estimated exam score</div>'
                    f'<div class="result-score">{score:.2f}<span style="font-size:20px;font-weight:500"> / 100</span></div>'
                    '<div class="result-caption">A model based estimate for this profile</div></div>',
                    unsafe_allow_html=True,
                )
                st.progress(score / 100, text=f"Score scale · {score:.1f} out of 100")
                if raw_prediction < 0 or raw_prediction > 100:
                    st.warning("The model produced a value outside the 0–100 score range; the displayed score is capped to that range.")
                st.markdown('<div class="note">This is an estimate, not a guaranteed exam result. The model has shown large prediction errors for some high scoring students. Use it as one input to a supportive conversation, never as the sole basis for an academic decision.</div>', unsafe_allow_html=True)
                with st.expander("Review submitted student profile"):
                    st.dataframe(input_df, width="stretch", hide_index=True)
                summary = pd.DataFrame([{**inputs, "Predicted_Exam_Score": round(score, 2)}])
                st.download_button("Download this estimate as CSV", summary.to_csv(index=False), "student_score_estimate.csv", "text/csv")
        except Exception as error:
            st.error(f"Prediction failed. Please check the submitted profile and try again. Details: {error}")

elif page == "Model Insights":
    st.markdown('<div class="eyebrow">MODEL TRANSPARENCY</div>', unsafe_allow_html=True)
    st.title("Model insights")
    st.markdown("Explore the model evaluation materials saved with this project.")
    st.markdown('<div class="note">The comparison below reports cross validation results. Dashboard test metrics are from a separate final test evaluation; these values describe different evaluation stages and should not be compared as if they were the same split.</div>', unsafe_allow_html=True)
    st.markdown("### Cross validation comparison")
    table_rows = []
    for row in MODEL_COMPARISON.itertuples(index=False):
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row[0]))}</td>"
            f"<td>{row[1]:.4f}</td><td>{row[2]:.4f}</td><td>{row[3]:.4f}</td>"
            "</tr>"
        )
    st.markdown(
        '<table class="comparison-table"><thead><tr><th>Model</th><th>MAE</th>'
        '<th>RMSE</th><th>R-squared</th></tr></thead><tbody>'
        + "".join(table_rows)
        + "</tbody></table>",
        unsafe_allow_html=True,
    )
    st.caption("Cross validation metrics from the model comparison results supplied with this project.")
    st.markdown("### Evaluation visualizations")
    chart_files = [
        ("Model comparison", "final_model_comparison.png"),
        ("Actual vs predicted scores", "actual_vs_predicted.png"),
        ("Residual analysis", "residual_analysis.png"),
        ("Prediction error distribution", "error_distribution.png"),
        ("Feature importance", "feature_importance.png"),
        ("Permutation importance", "permutation_importance.png"),
    ]
    present = [(title, OUTPUTS_DIR / filename) for title, filename in chart_files if (OUTPUTS_DIR / filename).is_file()]
    for start in range(0, len(present), 2):
        cols = st.columns(2)
        for col, (title, path) in zip(cols, present[start:start + 2]):
            with col:
                st.markdown(f"#### {title}")
                st.image(str(path), width="stretch")
    if not present:
        st.info("No evaluation visualizations were found in the project's outputs folder.")
    st.markdown("### Final test set summary")
    cols = st.columns(3)
    with cols[0]: metric_card("Test MAE", "0.4160", "Final test evaluation")
    with cols[1]: metric_card("Test RMSE", "1.5213", "Final test evaluation")
    with cols[2]: metric_card("Test R²", "0.8250", "Final test evaluation")

