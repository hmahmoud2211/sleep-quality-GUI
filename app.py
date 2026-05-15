from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

FEATURES = [
    "sleep_hours_per_night",
    "poor_sleep_under_7h",
    "snore_frequency",
    "snort_gasp_stop_breathing_frequency",
    "daytime_sleepiness_score",
    "depression_score_phq9",
    "total_exercise_days_per_week",
    "does_vigorous_recreational_activity",
    "sedentary_hours_per_day",
    "age_years",
    "avg_alcoholic_drinks_per_day",
    "is_heavy_drinker",
    "is_current_smoker",
    "smoked_at_least_100_cigarettes",
    "bmi",
    "ever_told_have_asthma",
    "ever_told_have_diabetes",
    "ever_told_high_blood_pressure",
]

PREDICTION_TARGETS = {
    "Sleep Trouble": {
        "models": {
            "LightGBM": BASE_DIR / "lgbm_trouble_model.pkl",
            "Logistic Regression": BASE_DIR / "logistic_trouble_model.pkl",
        },
        "positive_label": "Sleep trouble predicted",
        "negative_label": "No sleep trouble predicted",
    },
    "Sleep Disorder": {
        "models": {
            "LightGBM": BASE_DIR / "lgbm_disorder_model.pkl",
            "Logistic Regression": BASE_DIR / "logistic_disorder_model.pkl",
        },
        "positive_label": "Sleep disorder predicted",
        "negative_label": "No sleep disorder predicted",
    },
}


@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
    with model_path.open("rb") as file:
        return pickle.load(file)


def yes_no(label: str, value: bool = False, help_text: str | None = None) -> int:
    return int(st.checkbox(label, value=value, help=help_text))


def build_input_frame(values: dict[str, float | int]) -> pd.DataFrame:
    return pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)


def positive_probability(model, input_frame: pd.DataFrame, prediction) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(input_frame)[0]
    classes = getattr(model, "classes_", np.arange(len(probabilities)))

    if len(probabilities) == 2:
        if 1 in classes:
            positive_index = list(classes).index(1)
        else:
            positive_index = 1
        return float(probabilities[positive_index])

    predicted_class = prediction[0]
    if predicted_class in classes:
        return float(probabilities[list(classes).index(predicted_class)])

    return float(np.max(probabilities))


def predict(target_name: str, model_name: str, input_frame: pd.DataFrame) -> tuple[int | str, float | None]:
    model_path = PREDICTION_TARGETS[target_name]["models"][model_name]
    model = load_model(model_path)
    prediction = model.predict(input_frame)
    probability = positive_probability(model, input_frame, prediction)
    return prediction[0], probability


st.set_page_config(
    page_title="Sleep Prediction",
    layout="wide",
)

st.title("Sleep Prediction")
st.caption("Enter the patient values below. The app sends the features to the model in the required order.")

missing_models = [
    f"{target} / {model}"
    for target, config in PREDICTION_TARGETS.items()
    for model, path in config["models"].items()
    if not path.exists()
]
if missing_models:
    st.error("Missing model file(s): " + ", ".join(missing_models))
    st.stop()

with st.sidebar:
    st.header("Navigation")
    selected_target = st.radio("Predict", list(PREDICTION_TARGETS.keys()))
    selected_model = st.selectbox(
        "Choose model",
        list(PREDICTION_TARGETS[selected_target]["models"].keys()),
    )
    show_input = st.toggle("Show model input row", value=False)

left, middle, right = st.columns(3)

with left:
    st.subheader("Sleep")
    sleep_hours = st.number_input(
        "Sleep hours per night",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5,
    )
    poor_sleep = int(sleep_hours < 7)
    st.metric("Poor sleep under 7h", "Yes" if poor_sleep else "No")
    snore_frequency = st.number_input(
        "Snore frequency score",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=1.0,
        help="Use the same numeric scale used during training.",
    )
    breathing_frequency = st.number_input(
        "Snort/gasp/stop breathing frequency score",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=1.0,
        help="Use the same numeric scale used during training.",
    )
    daytime_sleepiness = st.number_input(
        "Daytime sleepiness score",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=1.0,
    )
    depression_score = st.number_input(
        "Depression score PHQ-9",
        min_value=0.0,
        max_value=27.0,
        value=0.0,
        step=1.0,
    )

with middle:
    st.subheader("Activity and Lifestyle")
    exercise_days = st.number_input(
        "Total exercise days per week",
        min_value=0,
        max_value=7,
        value=3,
        step=1,
    )
    vigorous_activity = yes_no("Does vigorous recreational activity")
    sedentary_hours = st.number_input(
        "Sedentary hours per day",
        min_value=0.0,
        max_value=24.0,
        value=6.0,
        step=0.5,
    )
    age_years = st.number_input(
        "Age years",
        min_value=0.0,
        max_value=120.0,
        value=35.0,
        step=1.0,
    )
    alcohol_drinks = st.number_input(
        "Average alcoholic drinks per day",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.5,
    )
    heavy_drinker = yes_no(
        "Is heavy drinker",
        help_text="Set this according to the same rule used when the model was trained.",
    )
    current_smoker = yes_no("Is current smoker")
    smoked_100 = yes_no("Smoked at least 100 cigarettes")

with right:
    st.subheader("Health")
    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=0.1,
    )
    asthma = yes_no("Ever told have asthma")
    diabetes = yes_no("Ever told have diabetes")
    high_bp = yes_no("Ever told high blood pressure")

input_values = {
    "sleep_hours_per_night": sleep_hours,
    "poor_sleep_under_7h": poor_sleep,
    "snore_frequency": snore_frequency,
    "snort_gasp_stop_breathing_frequency": breathing_frequency,
    "daytime_sleepiness_score": daytime_sleepiness,
    "depression_score_phq9": depression_score,
    "total_exercise_days_per_week": exercise_days,
    "does_vigorous_recreational_activity": vigorous_activity,
    "sedentary_hours_per_day": sedentary_hours,
    "age_years": age_years,
    "avg_alcoholic_drinks_per_day": alcohol_drinks,
    "is_heavy_drinker": heavy_drinker,
    "is_current_smoker": current_smoker,
    "smoked_at_least_100_cigarettes": smoked_100,
    "bmi": bmi,
    "ever_told_have_asthma": asthma,
    "ever_told_have_diabetes": diabetes,
    "ever_told_high_blood_pressure": high_bp,
}

input_frame = build_input_frame(input_values)

button_label = f"Predict {selected_target}"
if st.button(button_label, type="primary", use_container_width=True):
    try:
        label, probability = predict(selected_target, selected_model, input_frame)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    has_trouble = bool(int(label)) if str(label).isdigit() else bool(label)
    result_text = (
        PREDICTION_TARGETS[selected_target]["positive_label"]
        if has_trouble
        else PREDICTION_TARGETS[selected_target]["negative_label"]
    )

    st.divider()
    st.subheader("Result")
    st.metric("Prediction", result_text)

    if probability is not None:
        st.metric("Estimated probability", f"{probability:.1%}")
        st.progress(max(0.0, min(1.0, probability)))

if show_input:
    st.divider()
    st.subheader("Model input")
    st.dataframe(input_frame, use_container_width=True, hide_index=True)
