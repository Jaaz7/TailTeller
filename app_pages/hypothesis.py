import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def page_hypothesis_body():
    st.write("### Hypothesis Statement")

    st.success(
         "The convolutional neural network (CNN) can effectively "
        "differentiate among multiple classes within a dataset by "
        "learning distinctive features from the training images. \n"
        "\n"
        "Using layers of convolutions and pooling operations, the CNN "
        "is hypothesized to capture hierarchical patterns and features "
        "that are essential for classifying images "
        "into their respective categories accurately.\n"
        "\n"
        "Key elements for a Multi-Class Classification Model: \n"
        "\n"
        "1. Feature Engineering: the CNN is capable of detecting "
        "edges, textures to more complex shapes and patterns. \n"
        "\n"
        "2. Architecture: the CNN architecture passes through "
        "several layers of convolutions, it is suitable for this type of task. \n"
        "\n"
        "3. Generalization: the CNN is able to generalize well to "
        "unseen data Techniques like data augmentation, dropout "
        "and regularization will prevent overfitting. \n"
        "\n"
        "4. Performance Metrics: the CNN is able to elevate the accuracy "
        "to a satisfactory level, this can be checked with a F1-Score. \n"
    )
    st.write('---')
    st.warning(
        "Model Limitations Notice: \n"
        "\n"
        "1. Limited Dataset: The model was trained on a relatively small "
        "dataset of only 5111 images spread across 120 dog breeds. \n"
        '\n'
        "2. Uneven Distribution: Some breeds within the dataset are represented "
        "with as few as 27 images. This uneven distribution of samples can "
        "lead to inconsistencies in the model's ability to accurately recognize "
        " these less-represented breeds."
    )