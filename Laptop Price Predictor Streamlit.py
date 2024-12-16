import streamlit as st
import pickle
import numpy as np
import math
pipe = pickle.load(open('pipe (1).pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Predictor")
#brand
Company = st.selectbox('Brand',df['Company'].unique())
#type of laptop
type = st.selectbox('Type',df['TypeName'].unique())
#Ram
Ram = st.selectbox('Ram(in GB)',[2,4,6,8,12,16,24,32,64])
#Weight
Weight = st.number_input('Weight of the Laptop')
#TouchScreen
TouchScreen = st.selectbox('Touchscreen',['No','Yes'])
#IPS
IPS = st.selectbox('IPS',['No','Yes'])
#ScreenSize
screen_size = st.number_input('Enter Screen Size')
#resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900',
                                               '3840x2160','3200x1800','2880x1800'
                                               ,'3200x1800','2880x1800','2560x1600'
                                               ,'2560x1440','2304x1440'])

#CPU
CPU = st.selectbox('CPU',df['Cpu brand'].unique())
#HDD
HDD = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
#SSD
SSD = st.selectbox('SSD(in GB)',[0,8,128,512,1024])
#GPU BRAND
GPU_Brand = st.selectbox('GPU',df['Gpu brand'].unique())
#OS
OS = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
   #query
  PPI = None
  if TouchScreen == 'Yes':
    TouchScreen = 1
  else:
    TouchScreen = 0
  if IPS == 'Yes':
    IPS = 1
  else:
    IPS = 0
  X_res = int(resolution.split('x')[0])
  Y_res = int(resolution.split('x')[1])
  PPI = ((X_res**2)+(Y_res**2))**0.5/(screen_size + 0.0000001)
  query = np.array([Company,type,Ram,Weight,TouchScreen,IPS,PPI,CPU,HDD,SSD,GPU_Brand,OS])

  query = query.reshape(1,12)
  st.title(int(np.exp(pipe.predict(query)[0])))
