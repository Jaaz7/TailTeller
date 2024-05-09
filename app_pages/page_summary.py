import streamlit as st


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        "TailTeller is a data science and machine learning project. "
        "The business goal of this project is to identify different dog breeds "
        "based on images. This tool is realized with the Streamlit Dashboard "
        "and offers clients the ability to upload dog images to predict their "
        "breed. The dashboard provides results of the data analysis, descriptions, "
        "and an analysis of the project's hypotheses, along with details about "
        "the performance of the machine learning model."
    )

    st.write(
        "* For additional information, please visit and read the "
        "[Project's README file]"
        "(https://github.com/Jaaz7/TailTeller)."
    )

    st.success(
        "The project has 2 business requirements:\n"
        "* 1 - The client would like to have a study of the dataset collected.\n"
        "* 2 - The client requires a machine learning model developed to "
        "accurately identify dog breeds from images."
    )
