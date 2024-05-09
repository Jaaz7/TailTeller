import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
import pickle
import gc
from keras.models import load_model
from keras.preprocessing.image import load_img
import numpy as np
from keras.layers import Input, Lambda, GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import InceptionV3, Xception, NASNetLarge, InceptionResNetV2
from keras.applications.inception_v3 import preprocess_input as inception_preprocessor
from keras.applications.xception import preprocess_input as xception_preprocessor
from keras.applications.nasnet import preprocess_input as nasnet_preprocessor
from keras.applications.inception_resnet_v2 import preprocess_input as inc_resnet_preprocessor


model = load_model('tailteller_model.keras')

img_size = (299,299,3)

def get_features(model_name, model_preprocessor, input_size, data):

    input_layer = Input(input_size)
    preprocessor = Lambda(model_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs = input_layer, outputs = avg)
    
    #Extract feature
    feature_maps = feature_extractor.predict(data, verbose=1)
    print('Feature maps shape: ', feature_maps.shape)
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
    inception_features = get_features(InceptionV3, inception_preprocessor, img_size, data)

    # Extract features using Xception
    xception_features = get_features(Xception, xception_preprocessor, img_size, data)

    # Extract features using NASNetLarge
    nasnet_features = get_features(NASNetLarge, nasnet_preprocessor, img_size, data)

    # Extract features using InceptionResNetV2
    inc_resnet_features = get_features(InceptionResNetV2, inc_resnet_preprocessor, img_size, data)

    # Concatenate all extracted features along the last axis
    final_features = np.concatenate(
        [inception_features, xception_features, nasnet_features, inc_resnet_features],
        axis=-1
    )

    # Free RAM
    del inception_features, xception_features, nasnet_features, inc_resnet_features
    gc.collect()

    return final_features


def load_predict(data):
    random_dog = load_img('images/test/dab33799bceb2387a3daea652bfd8773.jpg', target_size=(299, 299, 3))
    random_dog = np.expand_dims(random_dog, axis=0)
    features = extract_features(random_dog)
    predict_dog = model.predict(features)
    print(f"Predicted label: {breeds[np.argmax(predict_dog[0])]}")
    print(f"Probability of prediction: {round(np.max(predict_dog[0])*100)}%")
    Image('images/test/dab33799bceb2387a3daea652bfd8773.jpg')


def load_and_prepare_image(image_path, target_size=(299, 299)):
    """ Load and prepare an image for prediction """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_prediction(model, img_array):
    """ Make a prediction and return probabilities """
    features = extract_features(img_array)
    predictions = model.predict(features)
    return predictions

def plot_predictions(image_path, model, breeds):
    """ Load an image, make a prediction, and plot the results """
    img_array = load_and_prepare_image(image_path)
    predictions = make_prediction(model, img_array)

    # Get the breed with the highest predicted probability
    top_breed_index = np.argmax(predictions[0])
    top_breed = breeds[top_breed_index]
    
    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(7,6), gridspec_kw={'height_ratios': [3, 1]})
    
    axs[0].imshow(img_array[0].astype('uint8'))
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    
    axs[1].bar(range(len(breeds)), predictions[0], color='skyblue')
    axs[1].set_title('Prediction Probabilities')
    axs[1].set_xlabel('Dog Breeds')
    axs[1].set_ylabel('Probability')
    
    # Show only every nth label and the top breed label
    n = 15
    labels = [breed if i % n == 0 or breed == top_breed else '' for i, breed in enumerate(breeds)]
    axs[1].set_xticks(range(len(breeds)))
    axs[1].set_xticklabels(labels, rotation=90)

    # Add the probability value next to the bar for the top breed
    axs[1].text(top_breed_index + 0.6, predictions[0][top_breed_index] - 0.06, f'{predictions[0][top_breed_index]:.2f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.show()