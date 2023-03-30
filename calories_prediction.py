import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('calories_model.sav', 'rb'))

def Calories_predicton(input_data):


# changing the input_data to numpy array
   input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
   input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)



   prediction = loaded_model.predict(input_data_reshaped)

   return "The calories burnt for the first individual in the dataset is predicted as ", prediction[0]
  
  
  
def main():
      
      st.title('Calories Burnt Prediction Model')
      Gender=st.text_input("Gender")
      Age=st.text_input("Age")
      Height=st.text_input("Height")
      Weight=st.text_input("Weight")
      Duration=st.text_input("Workout Duration")
      Heart_Rate=st.text_input("Heart Rate")
      Body_Temp=st.text_input("Body Tempature")
      

      
      
      diagnosis=''
      
      if st.button('Calories Burnt Test Result'):
          diagnosis=Calories_predicton([Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp])
          
      st.success(diagnosis)
      
      
if __name__=='__main__':
    main()
      
      
      
      
