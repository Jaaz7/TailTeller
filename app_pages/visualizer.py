import streamlit as st
from matplotlib.image import imread


def page_visualizer_body():
    st.write("### Data Visualizer")
    st.info(
        "We load the data to visualize and "
        "understand its distribution. "
        "This will help us to understand the data better and "
        "make better decisions when building the model. "
        "The train folder was split in half to create the train and test datasets. "
    )
    st.write("---")

    if st.checkbox("Count per dog breed"):

        breed_count = "assets/breed_countplot.jpg"

        st.info(
            "Here we can analyze the distribution of the dog breeds. "
            "There are  120 unique breeds in the dataset. Some breeds are "
            "more common than others. In the plot below, "
            "we can see the highest count of breeds is 70 and the lowest "
            "count is 27. The distribution of the breeds is not uniform, "
            "this is due the random selection of the files when the "
            "train dataset was split in half. "
            "This is important to keep in mind when training the model, "
            "It's possible that the model will have a slight better accuracy "
            "predicting the most common breeds."
        )

        st.image(breed_count, use_column_width=True)

    if st.checkbox("Let's look at two examples in our data"):

        two_dogs = "assets/2_dogs.png"

        st.info(
            '* On the left we have a dog from the breed "Kuvasz".\n'
            "\n"
            '* On the right we have a dog from the breed "Leonberg"\n'
            "\n"
            "We can observe the images have different sizes, "
            "the images will be processed in something called "
            "normalization, this will transform the images into the same size. "
            "This is important for the model to "
            "learn the image features. "
        )

        st.image(two_dogs, use_column_width=True)

    if st.checkbox('Number of images per folders: "train" and "test"'):

        nr_imgs_plot = "assets/number_images_plot.png"

        st.info(
            "The dataset is divided into two folders, train and test. "
            "The dataset is balanced, each folder "
            "contains the same number of images."
        )

        st.text(
            "Total dataset balance report:\n "
            "Number of images in the train dataset: 5111\n "
            "Number of images in the test dataset: 5111\n "
        )

        st.image(nr_imgs_plot, use_column_width=True)
