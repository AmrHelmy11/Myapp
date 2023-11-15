import streamlit as st
import joblib
import pandas as pd
preprocessor=joblib.load("preprocessor.h5")
pipeline=joblib.load("Car Price Prediction Model.h5")
model=joblib.load("Model.h5")
scaler=joblib.load("Scaler.h5")
inputs=joblib.load("input.h5")
Brands_List=joblib.load("Brands_List.h5")
Models_List=joblib.load("Models_List.h5")
Status_List=joblib.load("Status_List.h5")

def predict(Brand,Model,Status,Year,Mileage):
    test_df=pd.DataFrame(columns=inputs)
    test_df.at[0,"Brand"]=Brand
    test_df.at[0,"Model"]=Model
    test_df.at[0,"Status"]=Status
    test_df.at[0,"Year"]=Year
    test_df.at[0,"Mileage"]=Mileage
    xo=preprocessor.transform(test_df)
    L=pipeline.predict(xo)
    return round(L[0])
    


def main():
    st.title("Car Price Prediction Model")
    Brand=st.selectbox("Brand",Brands_List)
    Model=st.selectbox("Model",Models_List)
    Status=st.selectbox("Status",Status_List)
    Year=st.number_input('Enter Year Of Production', value=2000, min_value=1959, max_value=2024)
    Mileage=st.number_input('Enter Mileage Driven', value=200000, min_value=0, max_value=500000)
    if st.button("Predict"):
        result=predict(Brand,Model,Status,Year,Mileage)
        st.write(result)
if __name__=="__main__":
    main()
