# Near_Earth_Objects_Danger_Risk_Prediction

## Overview

This project focuses on predicting whether a Near-Earth Object (NEO) poses a potential hazard based on various characteristics. Given the increasing concerns about NEOs and their potential impact, the goal is to leverage machine learning to classify them as hazardous or non-hazardous. The dataset spans observations of NEOs recorded between 1910 and 2024, with the primary target variable being is_hazardous.

The workflow encompasses several stages: data cleaning, exploratory data analysis (EDA), addressing class imbalance, model training, and performance evaluation. Among the models tested, the Random Forest classifier demonstrated the highest accuracy in identifying hazardous NEOs.

In addition to model performance, interpretability was emphasized through the use of LIME (Local Interpretable Model-agnostic Explanations). LIME provided insights into the key features influencing the model’s predictions, such as absolute magnitude, estimated diameter, relative velocity, and miss distance. These explanations helped validate the model's reliability and offered transparency in understanding how the predictions were made. All the experiments done during are available in the jupyter notebooks inside the `notebook` folder.

Dataset Link: https://drive.google.com/drive/folders/1iW295y-QzP5tQxDtVaP-rrldbIypQFzv?usp=sharing

## How to run the application?

### A. Locally

### 1. Install virtualenv package

```bash
pip install --upgrade virtualenv
```

### 2. Make and activate virtual environment

```bash
python -m venv .venv
```

or

```bash
python3 -m venv .venv
```

#### On Windows

```bash
.venv\Scripts\activate
```

#### On macOS and Linux

```bash
source .venv/bin/activate
```

### 3. Install dependencies within the virtual environment

```bash
python -m pip install --upgrade pip
pip install --upgrade -r requirements.txt
```

### 4. Download the binary model from the [this link](https://github.com/surajkarki66/Near_Earth_Objects_Danger_Risk_Prediction/releases/download/v1.0.0/neo_rf.skops) and put the model file inside the `assets` directory.

### 5. Run the streamlit app within the virtual environment

```bash
streamlit run streamlit_app.py
```
### B. Docker
### Docker commands :computer:

To build the docker image locally, run the following command:

```bash
docker build --progress=plain --tag streamlit:latest .
```

Then to run the docker container locally, run the following command:

```bash
docker run -ti -p 8501:8501 --rm streamlit:latest
```

## Credits

The dataset used in this project is sourced from Kaggle. It contains information about Near-Earth Objects (NEOs) recorded by NASA and includes various features such as absolute magnitude, estimated diameter, relative velocity, and miss distance. The dataset spans observations from 1910 to 2024, providing a comprehensive view of NEOs over time.

You can access the dataset at the following link: [NASA Nearest Earth Objects Dataset](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects/data).

Special thanks to the creator for compiling and sharing this valuable resource for analysis and machine learning projects.
