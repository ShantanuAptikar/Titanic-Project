from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Titanic-Dataset (1).csv"
MODEL_PATHS = {
    "Logistic Regression": BASE_DIR / "lregression.pkl",
    "Support Vector Machine": BASE_DIR / "SVM.pkl",
    "K-Nearest Neighbors": BASE_DIR / "KNN.pkl",
}
FEATURE_COLUMNS = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "Sex_female",
    "Sex_male",
    "Embarked_C",
    "Embarked_Q",
    "Embarked_S",
]

st.set_page_config(
    page_title="Titanic | Passenger Observatory",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
    :root {
        --ink: #142a24;
        --muted: #687871;
        --paper: #f3f6f3;
        --panel: #fff;
        --line: #d8e1db;
        --sea: #14795e;
        --coral: #e56b4f;
        --gold: #d9ad4c;
    }
    .stApp {
        background-color: var(--paper);
        background-image: linear-gradient(rgba(20, 42, 36, .025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(20, 42, 36, .025) 1px, transparent 1px);
        background-size: 36px 36px;
        color: var(--ink);
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    h1 { font-family: 'Fraunces', Georgia, serif; letter-spacing: 0; }
    h2, h3 { font-weight: 600; }
    .stMainBlockContainer { max-width: 1380px; padding-top: 2rem; }
    [data-testid="stSidebar"] { background: #e9efeb; border-right: 1px solid var(--line); }
    [data-testid="stMetric"] {
        background: var(--panel); border: 1px solid var(--line); border-radius: 4px;
        border-top: 3px solid var(--sea); padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    .eyebrow { color: var(--sea); font-size: 0.72rem; font-weight: 700; letter-spacing: 0; text-transform: uppercase; }
    .archive-topline {
        display: flex; justify-content: space-between; gap: 12px; align-items: center;
        border-bottom: 1px solid var(--line); padding: 0 0 12px; margin-bottom: 16px;
        color: var(--muted); font-size: .72rem; font-weight: 700; text-transform: uppercase;
    }
    .masthead {
        position: relative; display: grid; grid-template-columns: minmax(0, 1fr) 180px;
        gap: 24px; align-items: center; overflow: hidden; background: var(--ink);
        color: #f6f8f4; padding: 30px 34px; border-radius: 4px;
        animation: reveal .45s ease-out both;
    }
    .masthead:before { content: ''; position: absolute; inset: 0 auto 0 0; width: 7px; background: var(--coral); }
    .masthead .eyebrow { color: #86c4a8; margin: 0 0 8px; }
    .masthead h1 { color: #fff; font-size: 48px; line-height: 1.08; margin: 0; }
    .masthead-copy { color: #c2d0c9; font-size: .96rem; margin: 12px 0 0; max-width: 650px; }
    .voyage-stamp { border-left: 1px solid #52645c; padding-left: 22px; color: #a9bbb2; font-size: .7rem; font-weight: 700; text-transform: uppercase; }
    .voyage-stamp strong { display: block; color: var(--gold); font-family: 'Fraunces', Georgia, serif; font-size: 42px; line-height: 1; margin: 7px 0; }
    .section-note { color: var(--muted); font-size: 0.88rem; }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 20px; border-bottom: 1px solid var(--line); }
    [data-testid="stTabs"] button[role="tab"] { color: var(--muted); border-bottom: 2px solid transparent; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--sea); border-bottom-color: var(--coral); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 4px; }
    [data-testid="stForm"] { background: var(--panel); border: 1px solid var(--line); border-radius: 4px; padding: 18px; }
    div.stButton > button, div.stDownloadButton > button, div.stFormSubmitButton > button {
        border-radius: 3px; font-weight: 700;
    }
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {
        background: var(--sea); border-color: var(--sea); color: white;
    }
    .section-note { color: var(--muted); font-size: 0.9rem; }
    @keyframes reveal { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    @media (max-width: 640px) {
        .stMainBlockContainer { padding: 1rem 1rem 2rem; }
        .masthead { grid-template-columns: 1fr; padding: 24px 22px; }
        .masthead h1 { font-size: 34px; }
        .voyage-stamp { border-left: 0; border-top: 1px solid #52645c; padding: 12px 0 0; }
        .archive-topline { align-items: flex-start; flex-direction: column; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data

def load_passengers() -> pd.DataFrame:
    passengers = pd.read_csv(DATA_PATH)
    passengers["Age"] = passengers["Age"].fillna(passengers["Age"].median())
    passengers["Embarked"] = passengers["Embarked"].fillna(passengers["Embarked"].mode()[0])
    passengers["FamilySize"] = passengers["SibSp"] + passengers["Parch"] + 1
    passengers["AgeGroup"] = pd.cut(
        passengers["Age"],
        bins=[0, 12, 18, 35, 60, float("inf")],
        labels=["0-12", "13-18", "19-35", "36-60", "61+"],
        right=True,
    )
    return passengers


@st.cache_resource

def load_models() -> dict[str, object]:
    return {name: joblib.load(path) for name, path in MODEL_PATHS.items()}


try:
    passengers = load_passengers()
except (FileNotFoundError, pd.errors.ParserError) as error:
    st.error(f"Could not load the Titanic dataset: {error}")
    st.stop()

st.sidebar.markdown('<p class="eyebrow">Passenger filters</p>', unsafe_allow_html=True)
st.sidebar.caption("Refine the manifest to update the overview.")
selected_classes = st.sidebar.multiselect(
    "Ticket class", options=[1, 2, 3], default=[1, 2, 3], format_func=lambda value: f"Class {value}"
)
selected_sexes = st.sidebar.multiselect(
    "Passenger sex", options=["female", "male"], default=["female", "male"],
    format_func=lambda value: value.title(),
)

filtered = passengers[
    passengers["Pclass"].isin(selected_classes) & passengers["Sex"].isin(selected_sexes)
]



overview_tab, manifest_tab, prediction_tab = st.tabs(["Overview", "Passenger manifest", "Survival model"])

with overview_tab:
    total = len(filtered)
    survivors = int(filtered["Survived"].sum()) if total else 0
    survival_rate = survivors / total if total else 0
    average_fare = filtered["Fare"].mean() if total else 0

    metric_columns = st.columns(4)
    metric_columns[0].metric("Passengers", f"{total:,}")
    metric_columns[1].metric("Survived", f"{survivors:,}")
    metric_columns[2].metric("Survival rate", f"{survival_rate:.1%}")
    metric_columns[3].metric("Average fare", f"£{average_fare:.2f}")

    st.write("")
    left_chart, right_chart = st.columns([1.2, 1])
    with left_chart:
        st.subheader("Survival by passenger class")
        if total:
            class_rates = filtered.groupby("Pclass", observed=False)["Survived"].mean().reindex([1, 2, 3])
            st.bar_chart(class_rates.rename("Survival rate"), color="#187c78")
        else:
            st.info("Choose at least one class and passenger sex to see this chart.")
        st.markdown('<p class="section-note">Share of passengers who survived, by ticket class.</p>', unsafe_allow_html=True)

    with right_chart:
        st.subheader("Passenger age groups")
        age_counts = filtered["AgeGroup"].value_counts(sort=False).rename_axis("Age group").to_frame("Passengers")
        st.bar_chart(age_counts, color="#d66d54")
        st.markdown('<p class="section-note">Passenger count across age bands.</p>', unsafe_allow_html=True)

    st.subheader("Survival by sex")
    if total:
        sex_rates = filtered.groupby("Sex")["Survived"].mean().reindex(["female", "male"]).dropna()
        st.bar_chart(sex_rates.rename("Survival rate"), color="#d4a64e")
    else:
        st.info("Choose at least one class and passenger sex to see this chart.")

with manifest_tab:
    st.subheader("Passenger manifest")
    st.caption(f"Showing {len(filtered):,} of {len(passengers):,} passengers after sidebar filters.")
    visible_columns = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    st.dataframe(
        filtered[visible_columns].rename(columns={"PassengerId": "ID", "Pclass": "Class", "SibSp": "Siblings / spouses", "Parch": "Parents / children"}),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Survived": st.column_config.CheckboxColumn("Survived", help="Whether the passenger survived."),
            "Age": st.column_config.NumberColumn("Age", format="%.1f years"),
            "Fare": st.column_config.NumberColumn("Fare", format="£%.2f"),
        },
    )
    st.download_button(
        "Download filtered manifest",
        data=filtered[visible_columns].to_csv(index=False).encode("utf-8"),
        file_name="titanic_filtered_manifest.csv",
        mime="text/csv",
    )

with prediction_tab:
    st.subheader("Estimate a passenger's survival")
    st.markdown('<p class="section-note">Three trained classifiers make independent predictions from the passenger details.</p>', unsafe_allow_html=True)

    with st.form("passenger_prediction"):
        form_left, form_right = st.columns(2)
        with form_left:
            passenger_class = st.selectbox("Ticket class", [1, 2, 3], index=2)
            passenger_age = st.slider("Age", min_value=0, max_value=80, value=29)
            passenger_sex = st.radio("Sex", ["female", "male"], horizontal=True)
            siblings_spouses = st.number_input("Siblings / spouses aboard", min_value=0, max_value=8, value=0)
        with form_right:
            parents_children = st.number_input("Parents / children aboard", min_value=0, max_value=6, value=0)
            passenger_fare = st.number_input("Fare (£)", min_value=0.0, max_value=600.0, value=32.2, step=1.0)
            embarkation = st.selectbox("Port of embarkation", ["S", "C", "Q"], format_func=lambda value: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[value])
        submitted = st.form_submit_button("Run all three models", type="primary", use_container_width=True)

    if submitted:
        family_size = int(siblings_spouses + parents_children + 1)
        feature_values = {
            "Pclass": passenger_class,
            "Age": passenger_age,
            "SibSp": siblings_spouses,
            "Parch": parents_children,
            "Fare": passenger_fare,
            "FamilySize": family_size,
            "IsAlone": int(family_size == 1),
            "Sex_female": int(passenger_sex == "female"),
            "Sex_male": int(passenger_sex == "male"),
            "Embarked_C": int(embarkation == "C"),
            "Embarked_Q": int(embarkation == "Q"),
            "Embarked_S": int(embarkation == "S"),
        }
        model_input = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)

        try:
            models = load_models()
            predictions = {name: int(model.predict(model_input)[0]) for name, model in models.items()}
        except (FileNotFoundError, ValueError, AttributeError) as error:
            st.error(f"Could not run the saved models: {error}")
        else:
            votes = sum(predictions.values())
            outcome = "Predicted to survive" if votes >= 2 else "Predicted not to survive"
            st.markdown(f"### {outcome}")
            st.caption(f"{votes} of 3 models predict survival. Model agreement is not a guarantee of an individual outcome.")
            result_columns = st.columns(3)
            for column, (model_name, prediction) in zip(result_columns, predictions.items()):
                column.metric(model_name, "Survive" if prediction else "Not survive")
            st.bar_chart(
                pd.Series(predictions, name="Prediction").map({0: "Not survive", 1: "Survive"}).value_counts(),
                color="#187c78",
            )

st.caption("Historical dataset: 891 recorded passengers. Estimates reflect the supplied trained models, not certainty.")
