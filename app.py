from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Engine Condition Prediction",
    page_icon="⚙️",
    layout="centered"
)

st.title("Engine Condition Prediction")
st.write(
    "Enter the engine operating measurements below "
    "to obtain a prediction from the trained classification model."
)


# --------------------------------------------------
# Load the saved model
# --------------------------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "best_engine_condition_model.joblib"
)

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Unable to load the saved model: {exc}")
    st.stop()


# --------------------------------------------------
# Collect user inputs
# --------------------------------------------------

st.subheader("Engine Measurements")

with st.form("engine_measurements_form"):

    engine_rpm = st.number_input(
        "Engine rpm",
        min_value=0.0,
        value=1500.0,
        step=100.0
    )

    lub_oil_pressure = st.number_input(
        "Lub oil pressure",
        min_value=0.0,
        value=3.0,
        step=0.1
    )

    fuel_pressure = st.number_input(
        "Fuel pressure",
        min_value=0.0,
        value=5.0,
        step=0.1
    )

    coolant_pressure = st.number_input(
        "Coolant pressure",
        min_value=0.0,
        value=2.0,
        step=0.1
    )

    lub_oil_temp = st.number_input(
        "Lub oil temp",
        min_value=0.0,
        value=80.0,
        step=1.0
    )

    coolant_temp = st.number_input(
        "Coolant temp",
        min_value=0.0,
        value=85.0,
        step=1.0
    )

    submitted = st.form_submit_button("Predict Engine Condition")


# --------------------------------------------------
# Prepare input and generate prediction
# --------------------------------------------------

if submitted:

    input_data = pd.DataFrame(
        [[
            engine_rpm,
            lub_oil_pressure,
            fuel_pressure,
            coolant_pressure,
            lub_oil_temp,
            coolant_temp
        ]],
        columns=[
            "Engine rpm",
            "Lub oil pressure",
            "Fuel pressure",
            "Coolant pressure",
            "lub oil temp",
            "Coolant temp"
        ]
    )

    st.subheader("Submitted Measurements")
    st.dataframe(input_data, use_container_width=True)

    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        st.metric("Predicted Engine Condition Class", str(prediction))

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

            probability_df = pd.DataFrame({
                "Class": model.classes_,
                "Predicted Probability": probabilities
            })

            st.write("Model prediction probabilities")
            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

        st.info(
            "Class labels are displayed as recorded in the dataset. "
            "Their operational meaning must be confirmed from the "
            "dataset documentation before using predictions for "
            "maintenance decisions."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


st.divider()
st.caption(
    "Predictive Maintenance — Engine Condition Classification. "
    "This application is a prototype and is not a substitute "
    "for qualified maintenance assessment."
)
