import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def page_ml_performance_body():
    st.write("### Model Performance")
    st.info(
        "This page provides a detailed analysis of the model's performance "
        "now that it's been trained and validated. \n"
        '\n'
        "Technical details about the training process itself: \n"
        "\n"
        "The model is a Convolutional Neural Network (CNN), "
        "a type of model often used for image classification tasks. "
        "CNNs are particularly good at picking up on patterns in "
        "images, like shapes and textures. \n"
        "\n"
        "Our model has several layers, including convolutional "
        "layers that apply filters to the images, "
        "pooling layers that reduce the dimensionality of the data, "
        "and dense layers that perform classification. \n"
    )
    st.write("---")

    if st.checkbox("Training and Validation Results"):

        t_v_plot = "assets/training_validation_plot.png"

        st.info(
            "The graph below is the training and validation results of the model. "
            "The model was trained for 50 epochs. \n"
            '\n'
            "The training accuracy data shows how well the model learned from the training dataset. \n"
            "\n"
            "The validation accuracy data shows how well the model generalizes to unseen data. \n"
            "\n"
            "The end results are: \n"
            "\n"
            "* The model has achieved a perfect score of 1.00, or 100%, indicating that "
            "it has perfectly learned the patterns in the training data. \n"
            "\n"
            "* The validation accuracy is 0.93, or 93%, indicating that the model is highly "
            "effective at applying what it's learned to new data."
        )

        st.image(t_v_plot, use_column_width=True)

    if st.checkbox("F1-Score"):

        f1_score = "assets/f1-score.png"

        st.info(
            "This is the F1 Score of the model. "
            "The F1 Score is the average of the precision and recall. "
            "It is a metric that considers both false positives and false negatives. "
            "This result can be reproduced by running the 5th "
            "Jupyter Notebook in the repository. \n"
            '\n'
            "The final accuracy of the model is 93%. "
        )

        st.image(f1_score, use_column_width=True)
