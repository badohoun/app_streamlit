import streamlit as st
from time import sleep
import mlflow
import re
import pandas as pd

import pycaret
from pycaret.regression import *
from pycaret.regression import load_model  , predict_model
import altair  as alt
from PIL import Image
from mlflow.tracking import MlflowClient



icon = Image.open("Inetum_logo.jpg")

st.set_page_config(
   page_title="Data Application",
   page_icon = icon,
   layout = "centered",
   initial_sidebar_state ="auto"

)
image = Image.open("MLOPS_Poste.jpeg")
#audio_file = open("Nouvel enregistrement 36.m4a", "rb")
#audio_bytes = audio_file.read()

st.image(image , width = 180)




model = None
avg_price = None
dummies = None

COMPANIES = [
  'alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda',
  'isuzu', 'jaguar', 'mazda', 'buick', 'mercury', 'mitsubishi',
  'nissan', 'peugeot', 'plymouth', 'porsche', 'renault', 'saab',
  'subaru', 'toyota', 'volkswagen', 'volvo'
]
TRUE_COLUMNS = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower',
       'fueleconomy', 'carlength', 'carwidth', 'fueltype_gas',
       'aspiration_turbo', 'carbody_hardtop', 'carbody_hatchback',
       'carbody_sedan', 'carbody_wagon', 'drivewheel_fwd',
       'drivewheel_rwd', 'enginetype_dohcv', 'enginetype_l',
       'enginetype_ohc', 'enginetype_ohcf', 'enginetype_ohcv',
       'enginetype_rotor', 'cylindernumber_five', 'cylindernumber_four',
       'cylindernumber_six', 'cylindernumber_three',
       'cylindernumber_twelve', 'cylindernumber_two',
       'company_price_medium', 'company_price_high']
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

mlflow.set_experiment("mlflow_demo_2")

mlflow_client = MlflowClient()




   
    
# Define application 



   





def main():
    page = st.sidebar.selectbox(
        "Select a Page",
        [
        "HomePage",
         "line" , 
            
        "Model Experimentation with MLflow",
        "Bonus : Pycaret"
        

        ]
    )
    if page == "HomePage":
        #st.header("Data Application")
        """
        #  MLOPS OPEN SOURCE TOOLS  
        - streamlit 
        - mlflow 
        - sqlalchemy 
        
        
        After that we  deploy this api at heroku which is a serverless platform  .
        """


         
        st.image("Reproductibility_In_ML.png" , width = 300)
        #st.audio(audio_bytes)

        st.balloons()
        st.header("Input Features")
        st.write(df.head(2))
    elif page == "line":
        st.header("DataviZ")
        visualize_line()
  
    elif page == "Model Experimentation with MLflow":
        st.header("following his experiments has become easy!")
        df2 = load_validationdata()
        track_with_mlflow = st.checkbox("Start load_artifacts ?")
        start_load_artifacts = st.button("Downloading artifacts and loading the model we choose ")
        artifacts = load_artifacts()
        st.write(predict(df.sample(10)))
        progress = st.progress(0)
        for i in range(100):
            progress.progress(i)
            sleep(0.1)
        if not artifacts:
            st.stop()
    elif page == "Bonus : Pycaret" :
        st.header("Easy MLOPS with pycaret and mlflow")
        Setup = st.button("Start loading pycaret model and predict price with new data ")
        
        pipeline = load_model('C:\\Users\\inno-demo\\Documents\\MLOPS\\Notebooks\\mlruns\\9\\f4c60a87d5f643dfa7037f4b63177274\\artifacts\\model\\model')
        
        predictions = predict_model(pipeline, data=new_data)
        st.write(predictions.head())
        progress = st.progress(0)
        for i in range(100):
            progress.progress(i)
            sleep(0.1)
        if not pipeline:
            st.stop()
        
        

       
       
        
        
  
    
    

# Load datasets 
@st.cache
def load_data():
    df = pd.read_csv("cars.csv")
    return df 
df   = load_data()


def load_new_data():
    df = pd.read_csv("primary.csv")
    return df 
new_data = load_new_data()

def visualize_line():
    df_copy = df.copy()

    
    line  = (
        alt.Chart(df_copy)
        .mark_line()
        .encode(x = "CarName" , y = "price")
        .properties(width=650, height=500)
        .interactive()
    )
    st.altair_chart(line)

def load_artifacts():
    global model
    global avg_price
    global dummies
    # On récupère le premier run (le plus récent)
    run_id = mlflow.list_run_infos("1")[0].run_id
    #avg_price = pd.read_csv("/tmp/artifacts/avg_price.csv")
    #avg_price.head(5)
    mlflow_client.download_artifacts(run_id, "process/" , '/tmp/')

    model = mlflow.sklearn.load_model("runs:/{}/model".format(run_id))
    avg_price = pd.read_csv("/tmp/process/avg_price.csv")
    with open("/tmp/process/dummies_cols.txt", "r") as f:
        dummies = f.read().split(",")
    return model 
 
def load_validationdata():

    st.sidebar.markdown("""
    [Example CSV input file](https://blent-learning-user-ressources.s3.eu-west-3.amazonaws.com/files/workshop_api_ml_cars.csv)
    """)
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file" , type = ["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
       
    
def transform(data):
    global avg_price
    global dummies
    X = data.copy()
    companies = X['CarName'].apply(lambda x : x.split(' ')[0])
    X.insert(3, "companies", companies)
    X.drop(['CarName'], axis=1, inplace=True)
  

    X.companies = X.companies.str.lower()
    
    
    def replace_name(a,b):
        X.companies.replace(a,b,inplace=True)

    # On remplace certaines occurrences identiques
    replace_name('maxda','mazda')
    replace_name('porcshce','porsche')
    replace_name('toyouta','toyota')
    replace_name('vokswagen','volkswagen')
    replace_name('vw','volkswagen')

    X['fueleconomy'] = (0.55 * X['citympg']) + (0.45 * X['highwaympg'])

    temp = X.copy()
    temp = temp.merge(avg_price.reset_index(), how='left', on='companies')

    bins = [0, 10000, 20000, 40000]
    cars_bin = ['cheap', 'medium', 'high']
    X['company_price'] = pd.cut(temp['price_y'], bins, right=False, labels=cars_bin)
    X.head()

    X = X[[
        'price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase',
        'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 
        'fueleconomy', 'carlength','carwidth', 'company_price'
    ]]

    for dummy in dummies:
        X[dummy] = 0

    dummy_cols = [
        "fueltype", "aspiration", "carbody", "drivewheel",
        "enginetype", "cylindernumber", "company_price"
    ]

    def replace_dummies(col, df):
        temp = pd.get_dummies(df[col], prefix=col, drop_first = True)
        #df = pd.concat([df, temp], axis=1)
        # lsuffix nous indique les colonnes à retirer
        df = df.join(temp, lsuffix="_toremove")
        df.drop([col], axis=1, inplace=True)
        for colname in df.columns.values:
            if re.search(r"_toremove", colname):
                df.drop([colname], axis=1, inplace=True)
        return df

    for dummy in dummy_cols:
        X = replace_dummies(dummy, X)

    return X[TRUE_COLUMNS]




def predict(X):
    global model
    if model:  
        return model.predict(transform(X))
    return None


if __name__ == "__main__":
    main()
    
   