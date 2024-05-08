import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.machine_learning.load_sample_predict import plot_predictions


def dog_breed_detector_body():
    st.write('### Dog Breed Identification')

    st.info(
        '* The client would like to be able to predict the dog breed based on '
        'an image. '
        )

    st.write('---')

