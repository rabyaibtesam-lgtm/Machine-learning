import os
import pickle
import streamlit as st
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Linear Regression Predictor")

BASE_DIR = os.path.dirname(__file__)
MODEL_FILENAME = "linear_regression_model_scaled.pkl"
SCALER_FILENAME = "scaler.pkl"
WRONG_MODEL_FILENAME = "Linear_regression_model_scled.pkl"

EXPECTED_COLUMNS = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
    'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea',
    'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
]

binary_cols = [
    'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea'
]


def get_label_encoders():
    encoders = {}
    for col in binary_cols:
        le = LabelEncoder()
        le.fit(['yes', 'no'])
        encoders[col] = le
    return encoders


@st.cache_data
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_artifacts():
    model_path = os.path.join(BASE_DIR, MODEL_FILENAME)
    scaler_path = os.path.join(BASE_DIR, SCALER_FILENAME)
    wrong_path = os.path.join(BASE_DIR, WRONG_MODEL_FILENAME)

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_pickle(model_path)
        scaler = load_pickle(scaler_path)
        return model, scaler, None

    if os.path.exists(wrong_path):
        return None, None, wrong_path

    return None, None, None


model, scaler, wrong_artifact = load_artifacts()
if wrong_artifact is not None:
    st.error(
        "Found a file named `Linear_regression_model_scled.pkl`, but it appears to contain application code instead of a serialized model. "
        "Please replace it with the actual model artifacts `linear_regression_model_scaled.pkl` and `scaler.pkl` in the same directory."
    )
    st.stop()

if model is None or scaler is None:
    st.warning(
        "Model artifacts not found. The app is using a dummy fallback predictor so the interface can run, but predictions will not reflect a trained model."
    )
    model = DummyRegressor(strategy='mean')
    model.fit([[0.0] * len(EXPECTED_COLUMNS)], [0.0])
    scaler = StandardScaler()
    scaler.fit([[0.0] * len(EXPECTED_COLUMNS)])

label_encoders = get_label_encoders()


def preprocess_input(input_data: pd.DataFrame) -> pd.DataFrame:
    processed_df = input_data.copy()
    for col, le in label_encoders.items():
        if col in processed_df.columns:
            processed_df[col] = le.transform(processed_df[col])

    processed_df = pd.get_dummies(processed_df, columns=['furnishingstatus'], drop_first=True)

    for col in EXPECTED_COLUMNS:
        if col not in processed_df.columns:
            processed_df[col] = 0

    processed_df = processed_df[EXPECTED_COLUMNS]
    processed_df_scaled = scaler.transform(processed_df)
    return pd.DataFrame(processed_df_scaled, columns=EXPECTED_COLUMNS, index=processed_df.index)


def predict_price(preprocessed_data: pd.DataFrame) -> float:
    prediction = model.predict(preprocessed_data)
    return float(prediction[0])


st.title("House Price Prediction")
st.write("Enter house features to get a price prediction.")

area = st.slider('Area (sq ft)', min_value=1000, max_value=20000, value=5000, step=100)
bedrooms = st.slider('Number of Bedrooms', min_value=1, max_value=6, value=3, step=1)
bathrooms = st.slider('Number of Bathrooms', min_value=1, max_value=4, value=2, step=1)
stories = st.slider('Number of Stories', min_value=1, max_value=4, value=2, step=1)
parking = st.slider('Number of Parking Spaces', min_value=0, max_value=3, value=1, step=1)

mainroad = st.selectbox('Main Road Access', ['yes', 'no'])
guestroom = st.selectbox('Guest Room', ['yes', 'no'])
basement = st.selectbox('Basement', ['yes', 'no'])
hotwaterheating = st.selectbox('Hot Water Heating', ['yes', 'no'])
airconditioning = st.selectbox('Air Conditioning', ['yes', 'no'])
prefarea = st.selectbox('Preferred Area', ['yes', 'no'])
furnishingstatus = st.selectbox('Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished'])

if st.button('Predict Price'):
    input_data = pd.DataFrame([{
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'hotwaterheating': hotwaterheating,
        'airconditioning': airconditioning,
        'parking': parking,
        'prefarea': prefarea,
        'furnishingstatus': furnishingstatus
    }])
    try:
        preprocessed_input = preprocess_input(input_data)
        predicted_price = predict_price(preprocessed_input)
        st.success(f'Predicted House Price: ${predicted_price:,.2f}')
    except Exception as e:
        st.error(f'An error occurred during prediction: {e}')

st.markdown('---')
st.caption(f'Looking for artifacts: `{MODEL_FILENAME}` and `{SCALER_FILENAME}` in the app folder.')
