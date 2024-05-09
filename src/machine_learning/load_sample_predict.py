import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
import pickle
import gc
import numpy as np
from PIL import Image
from keras.layers import Input, Lambda, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from keras.applications import InceptionV3, Xception, NASNetLarge, InceptionResNetV2
from keras.applications.inception_v3 import preprocess_input as inception_preprocessor
from keras.applications.xception import preprocess_input as xception_preprocessor
from keras.applications.nasnet import preprocess_input as nasnet_preprocessor
from keras.applications.inception_resnet_v2 import (
    preprocess_input as inc_resnet_preprocessor,
)

"""
The following commented code was used to save the pre-trained models locally
I then put them inside the src/machine_learning/models folder
This is going to save a lot of time (and RAM) when predicting images live
"""

# model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
# model.save('nasnet_large.h5')

# model = InceptionV3(weights='imagenet', include_top=False)
# model.save('inception_v3.h5')

# model = Xception(weights='imagenet', include_top=False)
# model.save('xception.h5')

# model = InceptionResNetV2(weights='imagenet', include_top=False)
# model.save('inception_resnet_v2.h5')


model = load_model("src/machine_learning/models/tailteller_model.keras")

with open("breeds.pkl", "rb") as f:
    breeds = pickle.load(f)


# get_features and extract_features functions will help us to
# extract features from the image
def get_features(model_name, model_preprocessor, input_size, data, model_path):
    input_layer = Input(input_size)
    preprocessor = Lambda(model_preprocessor)(input_layer)

    base_model = model_name(weights=None, include_top=False, input_tensor=preprocessor)
    base_model.load_weights(model_path)

    avg = GlobalAveragePooling2D()(base_model.output)
    feature_extractor = Model(inputs=input_layer, outputs=avg)

    feature_maps = feature_extractor.predict(data, verbose=1)
    print("Feature maps shape: ", feature_maps.shape)
    return feature_maps


def extract_features(data, img_size=(299, 299, 3)):
    """
    Extracts features from test data using multiple pre-trained models
    and concatenate them
    Parameters:
        data: numpy array, the input data we need to extract features from
        img_size: tuple, size to which the images
        should be resized (width, height, channels)
    Returns:
        a numpy array containing concatenated features from multiple models
    """
    # Extract features using InceptionV3
    inception_features = get_features(
        InceptionV3,
        inception_preprocessor,
        img_size,
        data,
        "src/machine_learning/models/inception_v3.h5",
    )

    # Extract features using Xception
    xception_features = get_features(
        Xception,
        xception_preprocessor,
        img_size,
        data,
        "src/machine_learning/models/xception.h5",
    )

    # Extract features using NASNetLarge
    nasnet_features = get_features(
        NASNetLarge,
        nasnet_preprocessor,
        img_size,
        data,
        "src/machine_learning/models/nasnet_large.h5",
    )

    # Extract features using InceptionResNetV2
    inc_resnet_features = get_features(
        InceptionResNetV2,
        inc_resnet_preprocessor,
        img_size,
        data,
        "src/machine_learning/models/inception_resnet_v2.h5",
    )

    # Concatenate all extracted features along the last axis
    final_features = np.concatenate(
        [inception_features, xception_features, nasnet_features, inc_resnet_features],
        axis=-1,
    )

    # Free RAM
    del inception_features, xception_features, nasnet_features, inc_resnet_features
    gc.collect()

    return final_features


def resize_image(img, size):
    return img.resize(size, Image.LANCZOS)


def simple_fig_plot(predictions, n):
    """Plot bar chart of the top N predictions"""
    top_n_indices = np.argsort(predictions[0])[-n:][::-1]
    top_n_probs = predictions[0][top_n_indices]
    top_n_breeds = [breeds[i] for i in top_n_indices]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(top_n_breeds, top_n_probs, color="skyblue")
    ax.set_xlabel("Dog Breeds")
    ax.set_ylabel("Probability")
    ax.set_title("Top 5 Predictions")
    plt.xticks(rotation=45)

    ax.set_ylim(bottom=-0.03)
    # Add a text label to the right of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width(), yval, round(yval, 2), va="center")

    plt.tight_layout()
    plt.savefig("simple_fig_plot.png")

    return fig
