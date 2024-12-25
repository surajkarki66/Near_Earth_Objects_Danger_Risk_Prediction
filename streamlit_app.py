import os
import lime
import pandas as pd
import lime.lime_tabular
import streamlit as st
import numpy as np
import skops.io as sio


@st.cache_resource
def load_data():
    # Load the dataset
    base_dir = os.getcwd()
    file_path = os.path.join(base_dir, "data", "smote_data.csv")
    df = pd.read_csv(file_path)
    
    X = df.drop(['is_hazardous', 'Unnamed: 0'], axis=1)
    y = df['is_hazardous']
    return X, y

@st.cache_resource
def load_model():
    # Current working directory
    base_dir = os.getcwd()
    file_path = os.path.join(base_dir, "assets", "neo_rf.skops")
    unknown_types = sio.get_untrusted_types(file=file_path)
    # Investigate the contents of unknown_types and only load if trusted.
    clf = sio.load(file_path, trusted=unknown_types)
    return clf


# Streamlit App Configuration
st.set_page_config(page_title="NEO Hazard Prediction", layout="centered", page_icon="🌍")

# Header Section
st.title("🌍 Near Earth Object (NEO) Hazard Prediction")
st.markdown("""
**Predict whether a near-Earth object (NEO) is hazardous based on its features.**  
Powered by machine learning, this tool evaluates the danger level of celestial objects.
""")

# Input form for features
st.sidebar.header("Input Features")
absolute_magnitude = st.sidebar.number_input("Absolute Magnitude (Describes intrinsic luminosity.)", value=22.0, step=0.1)
estimated_diameter_max = st.sidebar.number_input("Maximum estimated diameter in kilometers", value=0.1, step=0.01)
relative_velocity = st.sidebar.number_input("Velocity relative to Earth in km/h.", value=20.0, step=0.1)
miss_distance = st.sidebar.number_input("Distance in kilometers missed", value=500000.0, step=1000.0)

# Load the model at app startup (cached)
model = load_model()

#Load the data at app startup (cached)
X, _ = load_data()

# Predict Button
if st.sidebar.button("Predict"):
    # Prepare input data
    input_data = [[absolute_magnitude, estimated_diameter_max, relative_velocity, np.log1p(miss_distance)]]
    input_array = np.array(input_data)
    y_pred = model.predict(input_array)[0]
    y_pred_proba = model.predict_proba(input_array)[0][y_pred]
  
    # Display the prediction result in a stylish way
    st.subheader("Prediction Result")
    if y_pred == 1:
        st.error(f"⚠️ **Hazardous!** This NEO is likely dangerous with a probability of {y_pred_proba:.2%}.")
    else:
        st.success(f"✅ **Not Hazardous.** This NEO is not dangerous with a probability of {y_pred_proba:.2%}.")
    
    st.subheader("Explanation")
    feature_names = X.columns.tolist()
    class_names = ["No Hazardous", "Hazardous"]
   
    explainer = lime.lime_tabular.LimeTabularExplainer(X.values, feature_names = feature_names, class_names = class_names, discretize_continuous=True,)
    exp = explainer.explain_instance(input_array[0], model.predict_proba, num_features=4)
    # Display explanation as HTML
    components_html = exp.as_html()
    st.components.v1.html(components_html, height=300, scrolling=True)

        
else:
    # Placeholder message when no prediction is made
    st.subheader("Awaiting Prediction")
    st.info("Use the sidebar to input the features of a near-Earth object, then click **Predict** to see the results.")

# Footer Section
st.write("---")
st.write("Developed with ❤️ using Streamlit")
