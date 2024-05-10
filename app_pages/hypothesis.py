import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import itertools
import random


def page_hypothesis_body():
    st.write("### Dog Breed Identifier Visualizer")
    st.info(
        "* The client would like to have a study to identify "
        " different dog breeds based on images.)"
    )
