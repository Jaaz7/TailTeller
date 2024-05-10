import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import base64
from datetime import datetime
from io import BytesIO
import gc
import logging

logging.basicConfig(level=logging.DEBUG)

model = load_model("src/machine_learning/models/tailteller_model.keras")
with open("breeds.pkl", "rb") as f:
    breeds = pickle.load(f)

from src.machine_learning.load_sample_predict import (
    extract_features,
    resize_image,
    simple_fig_plot,
)


def dog_breed_detector_body():
    st.write("### Dog Breed Identification")
    st.info(
        "The client would like to be able to predict the dog breed based on an image."
    )
    st.write("---")

    uploaded_img = st.file_uploader(
        "Upload a clear dog photo.",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    if uploaded_img is not None:

        image_stream = BytesIO(uploaded_img.read())
        image_stream.seek(0)
        with Image.open(image_stream) as img_pil:
            img_pil = resize_image(img_pil, (299, 299))

        st.info(f"Image: **{uploaded_img.name}**")
        st.image(
            img_pil,
            caption="The uploaded image was resized to 299x299 pixels.",
            use_column_width=True,
        )

        # Prepare image for prediction
        img_array = np.array(img_pil)
        img_array = np.expand_dims(img_array, axis=0)

        # Clear RAM
        del img_pil
        gc.collect()

        # Feature extraction and prediction
        features = extract_features(img_array)

        predictions = model.predict(features)
        del features
        gc.collect()

        # Post-prediction processing
        predictions_percent = predictions[0] * 100
        above_5_indices = np.where(predictions_percent > 5)[0]
        df_predictions = pd.DataFrame(
            {
                "Breed": [breeds[i] for i in above_5_indices],
                "Prediction": [predictions_percent[i] for i in above_5_indices],
            }
        ).round(1)

        df_predictions["Prediction"] = df_predictions["Prediction"].astype(str) + "%"
        df_predictions = df_predictions.sort_values(
            by="Prediction", ascending=False
        ).reset_index(drop=True)
        df_predictions.index += 1

        if not df_predictions.empty:
            st.success(
                "Analysis Report: Displaying all breed predictions above 5% probability."
            )
            st.table(df_predictions)
            st.markdown(df_as_csv(df_predictions), unsafe_allow_html=True)

            fig_two = simple_fig_plot(predictions, 5)
            st.pyplot(fig_two)
            st.markdown(
                get_image_download_link("simple_fig_plot.png", "bar_chart.png"),
                unsafe_allow_html=True,
            )
        else:
            st.warning("No breed predictions were above 5% probability.")


def df_as_csv(df):
    datetime_now = datetime.now().strftime("%d%b%Y_%Hh%Mmin%Ss")
    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="Report_{datetime_now}.csv" target="_blank">Download Report</a>'


def get_image_download_link(img_path, filename):
    with open(img_path, "rb") as f:
        img_data = f.read()
    b64 = base64.b64encode(img_data).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">Download {filename}</a>'
