import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import itertools
import random


def visualizer_body():
    st.write('### Dog Breed Identifier Visualizer')
    st.info(
        '* The client would like to have a study to identify '
         ' different dog breeds based on images.')

    if st.checkbox('Count per dog breed'):

        breed_count = 'assets/breed_countplot.jpg'

        st.info(
          '* Here we can analyze the distribution of the dog breeds. '
          'There are  120 unique breeds in the dataset. Some breeds are '
          'more common than others. The differences between breeds are'
          'we can see the highest count of breeds is 70 and the lowest '
          'count is 27. The distribution of the breeds is not uniform, '
          'this is due the random selection of the files when the '
          'train dataset was split in half. '
          'The dataset will be normalized in the next steps.')
        
        st.image(breed_count, use_column_width=True)

    if st.checkbox("Let's look at two examples of pictures"):

        two_dogs = 'assets/2_dogs.png'

        st.image(two_dogs, use_column_width=True)

    if st.checkbox('Number of images per folders: "train" and "test"'):

        nr_imgs_plot = 'assets/number_images_plot.png'

        st.info(
          '* The dataset is divided into two folders, train and test. '
          'The train folder contains 5,111 images and the test folder '
          'contains 5,111 images. The dataset is balanced, each folder '
          'contains the same number of images.')
        
        st.text('Total dataset balance report:\n '
            'Number of images in the train dataset: 5111\n '
            'Number of images in the test dataset: 5111\n ')

        st.image(nr_imgs_plot, use_column_width=True)