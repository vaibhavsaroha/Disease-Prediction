# here we are importing libraries like streamlit pandas and PIL TO
# enter images into streamlit window
import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

# loading the saved models which we trained earlier from jupyter notebook
# loading diabetes saved model using pickle.load()
diab = pickle.load(
    open('diabetes_model.sav', 'rb'))

# loading heart disease saved model using pickle.lead()
heart = pickle.load(
    open('heart_disease.sav', 'rb'))

park = pickle.load(
    open('parkinsons_model.sav', 'rb'))

# sidebar for navigation using option_menu function from streamlit.sidebar
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Heart Disease Predictor',
                            'Parkinson Disease Predictor',
                            'Diabetes Predictor',
                           ],
                           icons=['activity', 'heart','person'],
                           default_index=0)




# Diabetes Prediction Page based on the selected option
if (selected == 'Diabetes Predictor'):

    #heading
    st.title('Diabetes Predictor Machine Learning ')
    #header
    st.sidebar.header('Diabetes Data')
    # loadinf diabetes image
    image=Image.open('diabetes.jpg')
    st.image(image,'BB')

    ## here we defined a menu for user input using streamlit sliders radio option
    ## and number input depending on different types of input
    def user_input():
        preg = st.sidebar.slider('Number of pregnancies',0,17,0)
        Glucose = st.sidebar.slider('Glucose Level',0,200,5)
        BP = st.sidebar.slider('Blood Pressure value',0,125,60)
        thick_skin = st.sidebar.slider('Skin Thickness value',0,100,5)
        insulin_level = st.sidebar.slider('insulin Level',0,900,20)
        mode = st.sidebar.radio("Select input mode to enter BMI", ["Slider", "Manual"])
        if mode == "Slider":
            BMI = st.sidebar.slider('BMI value',0.0,70.0,10.0,step=0.1)
        else:
            BMI = st.sidebar.number_input('BMI value', 0.0, 70.0, 10.0, step=0.1)
        DiabetesPedigreeFunction = st.sidebar.number_input('Diabetes Pedigree Function value',0.0,3.0,0.01)
        Age_diab_patient = st.sidebar.slider('Age of the Person',10,90,10)

        ## compiling user input into a dictionary
        user_report_data={
            'Number of pregnancies':preg,
            'Glucose Level':Glucose,
            'Blood Pressure value':BP,
            'Skin Thickness value':thick_skin,
            'insulin Level':insulin_level,
            'BMI value': BMI,
            'Diabetes Pedigree Function value':DiabetesPedigreeFunction,
            'Age of the Person':Age_diab_patient
        }

        ## converting collected data into a dataframe to pass it into our prediction model
        report_data=pd.DataFrame(user_report_data,index=[0])
        return report_data

    user_data=user_input()
    st.header('Patient Data')
    ## a box like structure for displaying user data
    st.write(user_data)

    ## prediction on our new data
    diab_prediction = diab.predict(user_data)
    st.subheader('patient status')
    result_diag = ''
    ## diabetes prediction
    if (diab_prediction[0] == 1):
        result_diag = 'The person is diabetic'
    else:
        result_diag = 'The person is not diabetic'
    st.success(result_diag)





# heart Prediction Page
## this page is using sidebars to take input from user depemding on the input and updating it at the same time at
## the data window
if (selected == 'Heart Disease Predictor'):
    st.title('Machine Learning model for Prediction Of Heart Disease')
    image = Image.open('heart.jpg')
    st.image(image, 'BB')
    age = st.sidebar.slider('Age',0,100,10)
    sex = st.sidebar.radio('Sex(1:M,0:F)',options=[1,0])
    cp = st.sidebar.selectbox('Chest Pain types',options=[0,1,2])
    trestbps = st.sidebar.slider('Resting BP',90.0,200.0,0.01)
    chol = st.sidebar.slider('Cholesterol',120,570,150)
    fbs = st.sidebar.radio('fasting blood sugar level>120mg/dl',options=[0,1])
    restecg = st.sidebar.radio('restecg',options=[0,1,2])
    thalach = st.sidebar.slider('Maximum Heart Rate achieved',60,250,120)
    exang = st.sidebar.radio('Exercise Induced Angina',options=[0,1])
    oldpeak = st.sidebar.slider('exercise induced ST depression',0.0,7.0,0.1)
    slope = st.sidebar.selectbox('Slope of the peak exercise ST segment',options=[0,1,2])
    ca = st.sidebar.slider('Major vessels colored by flourosopy',0,4,1)
    thal = st.sidebar.selectbox('Thalassemia Value',options=[0,1,2])

    ## we are storing all the data in the form of dictionary
    data = {
        "Age": age,
        "Sex": sex,
        "Chest Pain Types": cp,
        "Resting BP": trestbps,
        "Serum Cholestoral in mg/dl": chol,
        "fasting blood sugar level>120mg/dl": fbs,
        "restecg": restecg,
        "Maximum Heart Rate achieved": thalach,
        "Exercise Induced Angina": exang,
        "exercise induced ST depression": oldpeak,
        "Slope of the peak exercise ST segment": slope,
        "number of Major vessels colored by flourosopy": ca,
        "Thal": thal,
    }
    ## converting the data into a dataframe and storing it in report_data
    report_data = pd.DataFrame(data, index=[0])
    ## a window for displaying the data entered by the user
    st.write(report_data)
    
    ## model for test data
    result_diag_heart = ''

    # click to predict
    if st.button('Check Report'):
        ## passing the collected data from user into out model for testing
        heart_prediction = heart.predict(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        ## printing the results
        if (heart_prediction[0] == 1):
            result_diag_heart = 'The data provided for the patient has heart disease. Contact Doctors Immediately'
        else:
            result_diag_heart = 'The patient is safe from heart disease. Check up in regular basis for precaution'

    st.success(result_diag_heart)






## parkinson

## this page is using sidebars to take input from user depemding on the input and updating it at the same time at
## the data window
if (selected == 'Parkinson Disease Predictor'):
    st.title("Machine Learning model for Prediction Of Heart Disease")
    fo = st.sidebar.slider('MDVP:Fo(Hz)',85.0,270.0)
    fhi = st.sidebar.slider('MDVP:Fhi(Hz)',100.0,600.0)
    flo = st.sidebar.slider('MDVP:Flo(Hz)',60.0,250.0)
    Jitter_percent = st.sidebar.number_input('MDVP:Jitter(%)',0.001,0.04)
    Jitter_Abs = st.sidebar.number_input('MDVP:Jitter(Abs)',0.000001,0.009)
    RAP = st.sidebar.number_input('MDVP:RAP',0.001,0.03)
    PPQ = st.sidebar.number_input('MDVP:PPQ',0.001,0.03)
    DDP = st.sidebar.number_input('Jitter:DDP',0.001,0.09)
    Shimmer = st.sidebar.number_input('MDVP:Shimmer',0.001,0.2)
    Shimmer_dB = st.sidebar.slider ('MDVP:Shimmer(dB)',0.001,2.0)
    APQ3 = st.sidebar.slider('Shimmer:APQ3',0.001,2.0)
    APQ5 = st.sidebar.slider('Shimmer:APQ5',0.001,2.0)
    APQ = st.sidebar.slider('MDVP:APQ',0.010,2.0)
    DDA = st.sidebar.slider('Shimmer:DDA',0.001,2.0)
    NHR = st.sidebar.slider('NHR',0.001,2.0)
    HNR = st.sidebar.slider('HNR',7.0,35.0)
    RPDE = st.sidebar.slider('RPDE',0.1,1.0)
    DFA = st.sidebar.slider('DFA',0.4,0.8)
    spread1 = st.sidebar.slider('spread1',-8.0,-1.0)
    spread2 = st.sidebar.slider('spread2',0.01,0.5)
    D2 = st.sidebar.slider('D2',1.2,4.0)
    PPE = st.sidebar.slider('PPE',0.01,0.6)

    ## we are storing all the data in the form of dictionary
    data = {
        'MDVP:Fo(Hz)': fo,
        'MDVP:Fhi(Hz)': fhi,
        'MDVP:Flo(Hz)': flo,
        'MDVP:Jitter(%)': Jitter_percent,
        'MDVP:Jitter(Abs)':Jitter_Abs,
        'MDVP:RAP':RAP,
        'MDVP:PPQ':PPQ,
        'Jitter:DDP':DDP,
        'MDVP:Shimmer':Shimmer,
        'MDVP:Shimmer(dB)':Shimmer_dB,
        'Shimmer: APQ3':APQ3,
        'Shimmer:APQ5':APQ5,
        'MDVP:APQ':APQ,
        'Shimmer:DDA':DDA,
        'NHR':NHR,
        'HNR':HNR,
        'RPDE':RPDE,
        'DFA':DFA,
        'spread1':spread1,
        'spread2':spread2,
        'D2':D2,
        'PPE':PPE
    }

    ## converting the data into a dataframe and storing it in report_data
    report_data = pd.DataFrame(data, index=[0])

    ## a window for displaying the data entered by the user
    st.write(report_data)
    # code for Prediction
    result_diag_park = ''

    # creating a button for Prediction

    if st.button('parkinson Test Result'):
        ## passing the collected data from user into out model for testing
        park_prediction = park.predict(
            [[fo,fhi,flo,Jitter_percent,Jitter_Abs,RAP,PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])

        if (park_prediction[0] == 1):
            result_diag_park = 'The person is having parkinson disease'
        else:
            result_diag_park= 'The person does not have any parkinson disease'
    st.success(result_diag_park)