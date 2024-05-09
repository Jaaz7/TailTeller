import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import gc
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Input, Lambda, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import (
    InceptionV3,
    Xception,
    NASNetLarge,
    InceptionResNetV2,
)
from keras.applications.inception_v3 import preprocess_input as inception_preprocessor
from tensorflow.keras.applications.xception import (
    preprocess_input as xception_preprocessor,
)
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocessor
from tensorflow.keras.applications.inception_resnet_v2 import (
    preprocess_input as inc_resnet_preprocessor,
)
import logging

logging.basicConfig(level=logging.DEBUG)

with open("breeds.pkl", "rb") as f:
    breeds = pickle.load(f)


# Function to load weights and create a feature extractor
def load_feature_model(model_name, preprocess, img_size, model_path):
    input_layer = Input(shape=img_size)
    processed = Lambda(preprocess)(input_layer)
    model = model_name(weights=None, include_top=False, input_tensor=processed)
    model.load_weights(model_path)
    return Model(inputs=input_layer, outputs=GlobalAveragePooling2D()(model.output))


# Pre-load feature models


def extract_features(data):
    """Extracts features using pre-loaded models and concatenates them."""
    features_list = []

    inception_v3_model = load_feature_model(
        InceptionV3,
        inception_preprocessor,
        (299, 299, 3),
        "src/machine_learning/models/inception_v3.h5",
    )
    features_list.append(inception_v3_model.predict(data, verbose=0))
    del inception_v3_model
    gc.collect()

    xception_model = load_feature_model(
        Xception,
        xception_preprocessor,
        (299, 299, 3),
        "src/machine_learning/models/xception.h5",
    )
    features_list.append(xception_model.predict(data, verbose=0))
    del xception_model
    gc.collect()

    nasnet_large = load_feature_model(
        NASNetLarge,
        nasnet_preprocessor,
        (299, 299, 3),
        "src/machine_learning/models/nasnet_large.h5",
    )

    features_list.append(nasnet_large.predict(data, verbose=0))
    del nasnet_large
    gc.collect()

    # Second batch of models
    inception_resnet_v2_model = load_feature_model(
        InceptionResNetV2,
        inc_resnet_preprocessor,
        (299, 299, 3),
        "src/machine_learning/models/inception_resnet_v2.h5",
    )

    features_list.append(inception_resnet_v2_model.predict(data, verbose=0))
    del inception_resnet_v2_model
    gc.collect()

    # Concatenate all extracted features along the last axis
    final_features = np.concatenate(features_list, axis=-1)

    # RAM cleanup
    del features_list
    gc.collect()

    return final_features


def resize_image(img, size):
    return img.resize(size, Image.LANCZOS)


def simple_fig_plot(predictions, n):
    """Plot bar chart of the top N predictions."""
    top_n_indices = np.argsort(predictions[0])[-n:][::-1]
    top_n_probs = predictions[0][top_n_indices]
    top_n_breeds = [breeds[i] for i in top_n_indices]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(top_n_breeds, top_n_probs, color="skyblue")
    ax.set_xlabel("Dog Breeds")
    ax.set_ylabel("Probability")
    ax.set_title("Top 5 Predictions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("simple_fig_plot.png")
    plt.close(fig)
    return fig
