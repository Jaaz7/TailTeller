import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import base64
from datetime import datetime
from io import BytesIO

from src.machine_learning.load_sample_predict import (
    extract_features,
    resize_image,
    simple_fig_plot,
)

model = load_model("src/machine_learning/tailteller_model.keras")

with open("breeds.pkl", "rb") as f:
    breeds = pickle.load(f)


def dog_breed_detector_body():
    st.write("### Dog Breed Identification")

    st.info(
        "* The client would like to be able to predict the dog breed based on "
        "an image. "
    )

    st.write("---")

    uploaded_img = st.file_uploader(
        "Upload a clear dog photo.",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    if uploaded_img is not None:
        image_stream = BytesIO(uploaded_img.read())
        df_report = pd.DataFrame([])
        image_stream.seek(0)

        img_pil = Image.open(image_stream)
        img_pil = resize_image(img_pil, (299, 299))

        st.text(
            "The uploaded image was resized to 299x299 pixels. \n"
            "\n"
            "The image is ready to be fed into the model and "
            "make predictions. \n"
            "\n"
            "Please wait for the results..."
        )
        st.info(f"Image: **{uploaded_img.name}**")
        img_array = np.array(img_pil)
        original_img_shape = img_array.shape
        img_array = np.expand_dims(img_array, axis=0)
        # Get features
        features = extract_features(img_array)
        predictions = model.predict(features)
        top_breed_index = np.argmax(predictions[0])

        st.image(
            img_pil,
            caption=f"Image Size: {original_img_shape[1]}px width x "
            f"{original_img_shape[0]}px height",
            use_column_width=True,
        )

        # Convert the predictions to percentages
        predictions_percent = predictions[0] * 100

        # Get the indices of the predictions that are above 5%
        above_5_indices = np.where(predictions_percent > 5)[0]

        # Create a DataFrame for the predictions above 5%
        df_predictions = pd.DataFrame(
            {
                "Breed": [breeds[i] for i in above_5_indices],
                "Prediction": [predictions_percent[i] for i in above_5_indices],
            }
        )

        # Round the predictions to 1 decimal place and convert to string
        df_predictions["Prediction"] = df_predictions["Prediction"].round(1).astype(str)
        df_predictions["Prediction"] += "%"
        df_predictions = df_predictions.sort_values(by="Prediction", ascending=False)

        # Reset the index of the DataFrame to start from 1
        df_predictions.index = np.arange(1, len(df_predictions) + 1)

        df_report = pd.concat([df_report, df_predictions])

        def df_as_csv(df):

            datetime_now = datetime.now().strftime("%d%b%Y_%Hh%Mmin%Ss")
            csv = df.to_csv().encode()
            b64 = base64.b64encode(csv).decode()
            href = (
                f'<a href="data:file/csv;base64,{b64}" '
                f'download="Report {datetime_now}.csv" '
                f'target="_blank">Download Report</a>'
            )
            return href

        def get_image_download_link(img_path, filename):
            with open(img_path, "rb") as f:
                img_data = f.read()
            b64 = base64.b64encode(img_data).decode()
            href = f'<a href="data:image/png;base64,{b64}" download="{filename}">Download {filename}</a>'
            return href

        if not df_report.empty:
            st.success(
                "Analysis Report: all the breed predictions above 5% probability are shown.\n"
                "\n"
                "At least one breeds will always show, how to interpret: \n"
                "\n"
                "* Very low probability could mean it's not a dog picture. \n"
                "\n"
                "* Mixed dogs can have several lower percentage results. \n"
                "\n"
                "* If the dog's breed isn't  amongst our 120 breed labels, "
                "it won't show. \n"
                "\n"
                "\n"
                " Click the link at the end to download the report."
            )
            st.table(df_report)
            st.markdown(df_as_csv(df_report), unsafe_allow_html=True)
            st.write("")
            st.write("")
            fig_two = simple_fig_plot(predictions, 5)
            st.pyplot(fig_two)
            st.markdown(
                get_image_download_link("simple_fig_plot.png", "bar_chart.png"),
                unsafe_allow_html=True,
            )
