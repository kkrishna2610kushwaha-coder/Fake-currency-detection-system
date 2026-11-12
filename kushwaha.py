
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import os
import json

# Page configuration
st.set_page_config(
    page_title="Fake Currency Detection AI",
    page_icon="💵",
    layout="wide"
)

st.title("💵 Fake Currency Detection System")
st.subheader("Indian Currency Analysis Using Deep Learning")

st.write(
    "Upload a currency image to analyze it using "
    "our CNN Deep Learning model."
)

MODEL_PATH = "currency_model.keras"

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error(
        "AI model not found. Please train the model "
        "using train.py first."
    )
    st.stop()

try:
    model = load_model()
    st.sidebar.success("AI Model Loaded Successfully")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Sidebar
st.sidebar.title("Navigation")

module = st.sidebar.selectbox(
    "Choose Module",
    [
        "Home",
        "Currency Detection",
        "Detection History",
        "About Project"
    ]
)

# History database
DATABASE = "detection_history.csv"

def save_record(result, confidence, filename):
    record = pd.DataFrame([{
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Filename": filename,
        "Prediction": result,
        "Model Score (%)": round(confidence, 2)
    }])

    if os.path.exists(DATABASE):
        record.to_csv(
            DATABASE,
            mode="a",
            header=False,
            index=False
        )
    else:
        record.to_csv(DATABASE, index=False)


# Home page
if module == "Home":

    st.header("Welcome to Fake Currency Detection AI")

    col1, col2, col3 = st.columns(3)

    col1.metric("AI Technology", "CNN")
    col2.metric("Image Size", "224 x 224")
    col3.metric("System", "Active")

    st.markdown("---")

    st.subheader("Project Features")

    st.write("""
    - Upload currency images.
    - Analyze images using a trained CNN model.
    - Display model prediction scores.
    - Maintain detection history.
    - Download detection reports.
    """)

    st.info(
        "This is an educational prototype. "
        "Its predictions are not official authentication."
    )


# Currency detection page
elif module == "Currency Detection":

    st.header("Upload Currency Image")

    uploaded_file = st.file_uploader(
        "Choose Currency Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Currency Image",
            width=400
        )

        if st.button("Analyze Currency", type="primary"):

            with st.spinner("Analyzing image..."):

                image = image.resize((224, 224))

                image_array = np.array(image)
                image_array = np.expand_dims(
                    image_array, axis=0
                )

                prediction = model.predict(
                    image_array,
                    verbose=0
                )[0][0]

                genuine_score = float(prediction) * 100
                fake_score = (1 - float(prediction)) * 100

                if prediction >= 0.5:
                    result = "Genuine-like"
                    confidence = genuine_score
                else:
                    result = "Fake-like"
                    confidence = fake_score

                st.markdown("---")
                st.subheader("AI Analysis Result")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Prediction",
                    result
                )

                col2.metric(
                    "Model Score",
                    f"{confidence:.2f}%"
                )

                st.write("Genuine-like score")
                st.progress(int(genuine_score))

                st.write("Fake-like score")
                st.progress(int(fake_score))

                st.warning(
                    "These are model scores, not verified "
                    "probabilities of authenticity. Do not "
                    "use this result to accept or reject money."
                )

                save_record(
                    result,
                    confidence,
                    uploaded_file.name
                )

                st.success("Analysis completed!")


# History page
elif module == "Detection History":

    st.header("Detection History")

    if os.path.exists(DATABASE):

        df = pd.read_csv(DATABASE)

        st.dataframe(
            df,
            use_container_width=True
        )

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Detection Report",
            csv,
            "currency_detection_report.csv",
            "text/csv"
        )

    else:
        st.info("No detection records available.")


# About page
elif module == "About Project":

    st.header("About the Project")

    st.write("""
    Fake Currency Detection System is an educational
    Deep Learning project developed using Python,
    TensorFlow, CNN, NumPy and Streamlit.

    The model learns image patterns from labeled
    genuine and counterfeit currency examples.

    Project modules:

    1. Image Upload
    2. Image Preprocessing
    3. CNN Model Prediction
    4. Result Display
    5. Detection History
    """)

    st.info(
        "Only trained and representative datasets can "
        "produce meaningful results. This software "
        "cannot replace official currency verification."
    )


st.markdown("---")
st.caption(
    "Developed using Python | TensorFlow | CNN | Streamlit"
)
