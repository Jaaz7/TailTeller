# Dog Breed Identifier
**A Data Science and Machine Learning project developed as part of a specialized pathway in Predictive Analytics.<br>
It utilizes Artificial Intelligence and its primary goal is to tell the dog breed based on image data.<br>
This application is developed using a [Streamlit Dashboard](https://tailteller-f41359586dd2.herokuapp.com/), providing users (such as veterinarians, dog breeders and groomers) with the capability to upload images of dogs and receive instant predictions regarding their breed.**
<br><br>
<img src="https://github.com/Jaaz7/TailTeller/assets/130407877/7bc4f516-4d85-4d3d-95da-2f349bad2464" width=75% height=75%>
<br>
**[Visit the live project here.](etc)**

---
# Table of Contents
- ### [Dataset Content](https://github.com/Jaaz7/TailTeller/edit/main/README.md#dataset-content-1)
- ### [Business Requirements](https://github.com/Jaaz7/TailTeller/edit/main/README.md#business-requirements-1)
- ### [Hypothesis](etc)
- ### [Rationale](etc)
- ### [Business Case](etc)
- ### [Model Development](etc)
  - [Part 1](etc)
- ### [Hypothesis - Values](etc)
- ### [Project Dashboard](etc)
- ### [Unfixed bugs](etc)
- ### [Deployment](etc)
- ### [Data Analysis and Machine Learning Lbraries](etc)
- ### [Issues](etc)
- ### [Testing](etc)
  - [Manual Testing](etc)
  - [Validation](etc)
- ### [References](etc)
  - [Documentation](etc)
  - [Inspirational Resources](etc)
  - [Tools](etc)
  - [Content](etc)
  - [Acknowledgements](etc)

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

---
## ML Model Development
  - Multiple versions of our machine learning model have been developed, each iteratively improved based on testing feedback and       
    performance metrics. The final model employs advanced deep learning techniques to ensure robust breed classification.

    [Back to top](#table-of-contents)
---
## Dashboard Design
The Streamlit Dashboard serves as the user interface, allowing for easy interaction and access to the model’s capabilities. It provides:

---
## Testing and Validation
Comprehensive manual and automated testing ensures the reliability of the application. User story testing and continuous integration practices maintain the application's quality and performance.

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