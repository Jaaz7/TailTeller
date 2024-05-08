import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.visualizer import visualizer_body
from app_pages.dog_breed_detector import dog_breed_detector_body
# from app_pages.page_prospect import page_prospect_body
# from app_pages.page_project_hypothesis import page_project_hypothesis_body
# from app_pages.page_predict_churn import page_predict_churn_body
# from app_pages.page_predict_tenure import page_predict_tenure_body
# from app_pages.page_cluster import page_cluster_body

app = MultiPage(app_name="TailTeller") # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Dog Breed Visualizer", visualizer_body)
app.add_page("Dog Breed Identifier", dog_breed_detector_body)
# app.add_page("Prospect Churnometer", page_prospect_body)
# app.add_page("Project Hypothesis and Validation", page_project_hypothesis_body)
# app.add_page("ML: Prospect Churn", page_predict_churn_body)
# app.add_page("ML: Prospect Tenure", page_predict_tenure_body)
# app.add_page("ML: Cluster Analysis", page_cluster_body)

app.run()