
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Satisfaction Predictor", page_icon="✈️", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

st.title("✈️ Customer Satisfaction Predictor")
st.write("Enter passenger details to predict customer satisfaction.")

with st.sidebar:
    st.header("Passenger Information")
    gender = st.selectbox("Gender", ["Female","Male"])
    customer = st.selectbox("Customer Type", ["Loyal Customer","disloyal Customer"])
    travel = st.selectbox("Type of Travel", ["Business travel","Personal Travel"])
    travel_class = st.selectbox("Class", ["Business","Eco","Eco Plus"])
    age = st.number_input("Age",7,85,30)
    distance = st.number_input("Flight Distance",31,5000,1000)
    dep_delay = st.number_input("Departure Delay in Minutes",0,1200,0)

st.subheader("Service Ratings (0–5)")
cols = st.columns(3)
labels = [
"Inflight wifi service","Departure/Arrival time convenient",
"Ease of Online booking","Gate location","Food and drink",
"Online boarding","Seat comfort","Inflight entertainment",
"On-board service","Leg room service","Baggage handling",
"Checkin service","Inflight service","Cleanliness"
]
vals={}
for i,lbl in enumerate(labels):
    with cols[i%3]:
        vals[lbl]=st.slider(lbl,0,5,3)

gender_map={"Female":0,"Male":1}
cust_map={"Loyal Customer":0,"disloyal Customer":1}
travel_map={"Business travel":0,"Personal Travel":1}
class_map={"Business":0,"Eco":1,"Eco Plus":2}

if st.button("Predict Satisfaction", use_container_width=True):
    input_df=pd.DataFrame([{
        "Gender":gender_map[gender],
        "Customer Type":cust_map[customer],
        "Age":age,
        "Type of Travel":travel_map[travel],
        "Class":class_map[travel_class],
        "Flight Distance":distance,
        "Inflight wifi service":vals["Inflight wifi service"],
        "Departure/Arrival time convenient":vals["Departure/Arrival time convenient"],
        "Ease of Online booking":vals["Ease of Online booking"],
        "Gate location":vals["Gate location"],
        "Food and drink":vals["Food and drink"],
        "Online boarding":vals["Online boarding"],
        "Seat comfort":vals["Seat comfort"],
        "Inflight entertainment":vals["Inflight entertainment"],
        "On-board service":vals["On-board service"],
        "Leg room service":vals["Leg room service"],
        "Baggage handling":vals["Baggage handling"],
        "Checkin service":vals["Checkin service"],
        "Inflight service":vals["Inflight service"],
        "Cleanliness":vals["Cleanliness"],
        "Departure Delay in Minutes":dep_delay
    }])

    input_df=input_df[scaler.feature_names_in_]

    try:
        scaled=scaler.transform(input_df)
        pred=model.predict(scaled)[0]
        if hasattr(model,"predict_proba"):
            prob=model.predict_proba(scaled)[0]
            conf=max(prob)*100
        else:
            conf=None
    except Exception:
        pred=model.predict(input_df)[0]
        if hasattr(model,"predict_proba"):
            prob=model.predict_proba(input_df)[0]
            conf=max(prob)*100
        else:
            conf=None

    if pred==1:
        st.success("😊 Prediction: SATISFIED")
    else:
        st.error("☹️ Prediction: NEUTRAL OR DISSATISFIED")

    if conf is not None:
        st.metric("Confidence",f"{conf:.2f}%")

    with st.expander("Input Features"):
        st.dataframe(input_df)
