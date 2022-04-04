#Importing Important libraries
import pandas as pd
import streamlit as st
from datetime import date
import pickle


#Importing Pickle file 
pickle_in = open("Diabeties_classifier.pkl","rb")
daibeties_classifier=pickle.load(pickle_in)


today = date.today()
#date_time = datetime.fromtimestamp(1887639468)
f = open("file_logger.txt", "a")

def main():
    
   
    # Code for Header
    html_temp = """
   <div style="background-color:tomato;padding:10px">
   <h2 style="color:white;text-align:center;">Streamlit Diabeties Prediction ML App </h2>
   </div>
   """
   
   
    st.markdown(html_temp,unsafe_allow_html=True)
    st.info("")
    
    #--------------------------------------------------------------------------
    
    
    #code for file uploader to predict.
    data = st.file_uploader("Choose a csv file for Prediction", ["csv"])
    if data is not None:
        f.write(f"{today} : Dataset uploaded")
        f.write("\n")
        
        df = pd.read_csv(data)
        st.markdown("Dataset you have uploaded:-")
        st.dataframe(df)
        
        f.write(f"{today} : Dataset Shown")
        f.write("\n")
        
        try:
            f.write(f"{today} : Prediction Started")
            f.write("\n")
            
            ans = daibeties_classifier.predict(df)
            df["predcited_value"] = ans
            
            f.write(f"{today} : Prediction Complete")
            f.write("\n")
            
            st.markdown("Dataset after Prediction:-")
            st.dataframe(df)
            
            f.write(f"{today} : Dataset Shown after prediction")
            f.write("\n")
            
            if st.button("Download"):
                df.to_csv("Result.csv")
                st.success("Download Complete")
                
                f.write(f"{today} : Dataset Downloaded after prediction")
                f.write("\n")
               

        except Exception as e:
            st.error("Inavlid Dataset, Please! Try Again....")
            
            f.write(f"{today} : {e}")
            f.write("\n")
           
        
    #-------------------------------------------------------------------------
     
    html_temp = """
    <h2 style="color:White;text-align:center;">Or</h2>
    </div>
    """


    st.markdown(html_temp,unsafe_allow_html=True)
    
    #--------------------------------------------------------------------------
    
    #Code for user input and prediction
    Pregnancies = st.text_input("Pregnancies:- Month (0-17)")
    Glucose = st.text_input("Glucose:-")
    BloodPressure = st.text_input("BloodPressure:-")
    SkinThickness = st.text_input("SkinThickness:-")
    Insulin = st.text_input("Insulin:-")
    BMI = st.text_input("BMI:-")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction:-")
    Age = st.text_input("Age:-")

    if st.button("Predict"):
        try:
            f.write(f"{today} : Prediction for individual Started")
            f.write("\n")
            
            result = daibeties_classifier.predict([[Pregnancies, Glucose, 
            BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
            st.markdown("If Output : 1 than Yes you are suffered from Diabeties")
            st.markdown("If Output : 0 than No you are not suffered from Diabeties")
            st.success('The output is {}'.format(result))
            
            f.write(f"{today} : Individual Prediction Complete")
            f.write("\n")
            
            
        except Exception as e:
            st.error("Invalid Inputs, Please! Try Again....")
            
            f.write(f"{today} : {e}")
            f.write("\n")
            
    

    
if __name__ == '__main__' :
    main()
    