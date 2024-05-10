# Dog Breed Identifier
**A Data Science and Machine Learning project developed as part of a specialized pathway in Predictive Analytics.<br>
It utilizes Artificial Intelligence and its primary goal is to tell the dog breed based on image data.<br>
This application is developed using a [Streamlit Dashboard](https://tailteller-f41359586dd2.herokuapp.com/), providing users (such as veterinarians, dog breeders and groomers) with the capability to upload images of dogs and receive instant predictions regarding their breed.**
<br><br>
<img src="https://github.com/Jaaz7/TailTeller/assets/130407877/7bc4f516-4d85-4d3d-95da-2f349bad2464" width=75% height=75%>
<br>
**[Visit the live project here.](https://tailteller-f41359586dd2.herokuapp.com/)**

---
# Table of Contents
- ### [Dataset Content](https://github.com/Jaaz7/TailTeller/edit/main/README.md#dataset-content-1)
- ### [Business Requirements](https://github.com/Jaaz7/TailTeller/edit/main/README.md#business-requirements-1)
- ### [Hypothesis](https://github.com/Jaaz7/TailTeller/edit/main/README.md#hypothesis-1)
- ### [Business Case](https://github.com/Jaaz7/TailTeller/edit/main/README.md#business-case-1)
- ### [Model Development](https://github.com/Jaaz7/TailTeller/edit/main/README.md#ml-model-development)
- ### [Dashboard Design](https://github.com/Jaaz7/TailTeller/edit/main/README.md#dashboard-design-1)
- ### [Kanban Board](https://github.com/Jaaz7/TailTeller/edit/main/README.md#kanban-board-1)
- ### [Unfixed bugs](https://github.com/Jaaz7/TailTeller/edit/main/README.md#unfixed-bugs-1)
- ### [Deployment](https://github.com/Jaaz7/TailTeller/edit/main/README.md#deployment-1)
- ### [Issues](https://github.com/Jaaz7/TailTeller/edit/main/README.md#issues-1)
- ### [Testing](https://github.com/Jaaz7/TailTeller/edit/main/README.md#testing-and-validation)
  - [Manual Testing](https://github.com/Jaaz7/TailTeller/edit/main/README.md#manual-testing)
  - [Validation](https://github.com/Jaaz7/TailTeller/edit/main/README.md#validation)
- ### [References](https://github.com/Jaaz7/TailTeller/edit/main/README.md#references-1)
  - [Documentation](https://github.com/Jaaz7/TailTeller/edit/main/README.md#documentation)
  - [Inspirational Resources](https://github.com/Jaaz7/TailTeller/edit/main/README.md#inspirational-resources)
  - [Tools]([etc](https://github.com/Jaaz7/TailTeller/edit/main/README.md#tools))
  - [Content](https://github.com/Jaaz7/TailTeller/edit/main/README.md#content)
  - [Acknowledgements](https://github.com/Jaaz7/TailTeller/edit/main/README.md#acknowledgements)

---
## Dataset Content
  - ### The dataset is a [competition challenge](https://www.kaggle.com/competitions/dog-breed-identification) from Kaggle.<br>
      Its contents include:<br><br>
      - A training directory with 10,222 pictures of dogs:
      - a labels.csv metadata.<br><br>
    The training directory was split into a training and testing directories, both with 5,111 images.<br>
    The metadata has 2 columns. A column 'id' corresponding to the image files in the training directory and 'breed' corresponding to the      label of the dog breed. There are 120 unique breeds.<br>
    This dataset aids in training the machine learning model to accurately identify and differentiate between breeds.

---
## Business Requirements
  1. The client would like to have a study of the dataset collected.
  2. The client requires a machine learning model developed to accurately identify dog breeds from images.
     
  The project aims to assist people like veterinarians, breeders, groomers and dog enthusiasts by providing:

  - Accuracy: High accuracy in predicting dog breeds to aid in better breed-specific care.
  - Interpretation: Clear explanations of prediction results, helping users interpret the data.
  - Speed: Exhaustive optimization for minimal CPU and RAM usage impact to offer results as fast as possible..
  - Privacy: Ensuring that all user data is handled with strict confidentiality and security measures.

    [Back to top](#table-of-contents)
---
## Hypothesis
Initial hypotheses posit that machine learning models, particularly convolutional neural networks (CNNs), can effectively distinguish between dog breeds from images. Validation of these hypotheses will be conducted through:

Detailed analysis and performance metrics of the model.
Continuous testing to ensure model accuracy and reliability.

## Business Case
  ### Project Overview:
  - The client aims to accurately identify the breed of a dog from a given image. This business goal will be achieved through the       
  development and deployment of a TensorFlow-based deep learning pipeline. The pipeline will utilize a dataset of images classified by 
  dog breeds.

  ### Technical Approach:
  - The deep learning pipeline will utilize a convolutional neural network (CNN), a model highly effective at recognizing patterns in image 
  data.

  ### Objective:
  - The primary objective of this machine learning pipeline is to create a multi-class classification model. The desired outcome is a model 
  that can successfully categorize dog breeds.

  ### Model Output:
  - The model will output a classification label that indicates the breed of the dog, based on the probabilities calculated by the model.

  ### Application of Results:
  The images might be used by veterinary health care, training or grooming.

  ### Performance Metrics:
  - The success of this model will be evaluated based on overall accuracy and the F1 score for each breed. Given the varying sample sizes 
  per breed, particular attention will be paid to minimize the risk of misclassification.

  ### Accuracy and Reliability:
  - The client sets a high bar for accuracy given the potential for the model to influence decisions about dog care and management.
  Initial targets  might aim for an F1 score of above 0.90, accuracy and reliability are key.

---
## ML Model Development

  ### Technical Setup
  Models used in this Pipeline process:
  
  1. Inception V3: Known for its efficiency and ability to detect features.
  2. Xception: Utilizes convolutions to provide a more efficient modeling.
  3. NASNetLarge: Offers a scalable architecture for image recognition.
  4. InceptionResNet V2: Combines the connections for faster training.
     These models are put together in one to ensuring robust breed classification.

  ### Image Preprocessing
  - To standardize inputs and enhance model performance, images are resized to 299x299 pixels with three color channels (RGB). This 
  resizing matches the input requirements of the models (as seen in Keras documentation), this helps with consistency.

  ### Feature Extraction and Classification
  - Each model processes the input images to extract vital features, which are then concatenated to form a comprehensive and rich feature 
  set.

  ### Feature Concatenation:
  - Combines outputs from multiple pre-trained models to create a robust representation of the images.
  - Classification: A final set of dense layers interprets these features to classify the image into one of the dog breeds. This is 
  achieved using a softmax layer, which outputs a probability distribution over the breed classes.
  This multi-model approach improves the accuracy of classification, and enhances the model's ability to identify unseen data reducing overfitting and improving predictions.

---
## Dashboard Design
The Streamlit Dashboard serves as the user interface, allowing for easy interaction and access to the model’s capabilities. It provides:

---
## Kanban Board

---
## Unfixed Bugs
  - There are no unfixed bugs.

---
## Deployment

---
## Issues

---
## Testing and Validation
  ### Manual Testing
   - etc
  ### Validation
   - etc
---
## References
  - ### Documentation
    - [Python 3.9.19 documentation](https://docs.python.org/release/3.9.19/) - Official Python documentation, used for language syntax and library reference.
    - [Keras](https://keras.io/api/applications/) - Keras is a Python Library that runs in TensorFLow. Keras documentation 
    was crucial to develop this project.
    - [Streamlit 1.34.0 documentation](https://docs.streamlit.io/develop/quick-reference/changelog) - Comprehensive guide for the 
    Streamlit Library used to make the dashboard IDE for the end user.
    - [Scikit-learn 1.4.2 documentation](https://scikit-learn.org/stable/) - A Machine Learning python library for classification 
    algorithms.
    - [Tensorflow 2.16.1](https://www.tensorflow.org/) - A library developed by Google for Machine Learning and neural networks.
    - [Numpy 1.19.3](https://numpy.org/) - Python library that helps making multi-dimenstional arrays.
    - [Pandas 2.2.2](https://pandas.pydata.org/pandas-docs/stable/index.html) - A Python Library used to make DataFrames and other tools.
    - [Matplotlib 3.3.1](https://matplotlib.org/) -A Python Library that provides an object-oriented API for embedding plots.
    - [Seaborn 0.11.0](https://seaborn.pydata.org/) - A Python library for data science statistical visualization.
    - [Plotly 5.22.0](https://plotly.com/) - A Python Library used to represent complex data in interactive graphical visualizations.
    - [Docker](https://docs.docker.com/desktop/) - Docker is a platform that automates deployment, I couldn't have deployed a real-time 
      model prediction without it, for a Machine Learning project it can be very challenging to deal with a 500mb slug size from Heroku.
  - ### Inspirational Resources
    - I went through many inspirational projects in predictive analytics from GitHub by the search bar.
    - [freeCodeCamp mini course in Machine Learning](https://www.youtube.com/watch?v=i_LwzRVP7bg) - This video was helpful to understand         Classification using Tensorflow.
    - [Build a Deep CNN Image Classifier](https://www.youtube.com/watch?v=jztwpsIzEGc) - Great video to understand 
     the nitty gritty of passing Deep Learning (Sequential from Keras) in Jupyter Notebooks and everything that goes with it in the 
    context of a classification problem.
    - [Machine Learning on Reddit](https://www.reddit.com/r/MachineLearning/) - Engaging with this forum community has helped me answer 
      questions and issues I had with development.
    - Code Institute's "Churnometer" walkthrough project - Provided a foundational understanding for deploying an A.I system.
  - ### Tools
    - [Brave search engine](https://search.brave.com/) - Primary search engine used for research and troubleshooting.
  - ### Content
    - Code written by Jaaziel do Vale.
  - ### Acknowledgements
    - Special thanks to mentors and collaborators who provided insight and expertise.
   
    [Back to top](#table-of-contents)