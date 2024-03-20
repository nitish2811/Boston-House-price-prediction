import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st


print("xzy")
model=pickle.load(open("bostonmodel.pkl","rb"))

# scaler define
dataset=pd.read_csv("DATA.csv")
features=dataset.iloc[:,:-1]
scaler=StandardScaler()
features=scaler.fit_transform(features)


st.title("House Price Prediction")
st.write(" To understand input please refer to description in left navigation")
col1,col2,col3,col4,col5=st.columns(5)
Crim=col1.number_input(label="CRIME", format="%.5f")
Zn=col1.number_input("ZN",format="%.2f")
Indus=col1.number_input("INDUS",format="%.2f")
rd=col5.slider('River tract', min_value=0, max_value=1, value=0, step=1)
no=col2.number_input("NO",format="%.3f")
rm=col2.number_input("ROOMS",format="%.3f")
age=col2.number_input("AGE",format="%.1f")
dis=col3.number_input("DIS",format="%.4f")
rad=col3.number_input("RAD",format="%.0f")
tax=col3.number_input("TAX",format="%.0f")
ptratio=col4.number_input("PTRATIO",format="%.1f")
b=col4.number_input("B",format="%.2f")
Lsat=col4.number_input("LSTAT",format="%.2f")


submit=st.button(label="Predict")

testvalue=scaler.transform([[Crim,Zn,Indus,rd,no,rm,age,dis,rad,tax,ptratio,b,Lsat]])
if submit:
    prediction=model.predict(testvalue)
    st.write(" The price in 1000$ is ",prediction)



    
    
 

