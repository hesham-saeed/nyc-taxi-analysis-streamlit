import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@st.cache_data
def get_data(filename):
    taxi_data = pd.read_csv(filename)
    return taxi_data

# Header
st.set_page_config(page_title="Streamlit Dashboard")
st.header("Welcome to this cool streamlit app!")

st.markdown(
    """
    <style>
    .main{
    background-color: #F5F5F3;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Dataset Visualization
st.header('NYC taxi dataset')
st.text("I found this dataset on blabla.com...")
taxi_data = get_data('taxi_data.csv')
st.write(taxi_data.head())
pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts().head(50))
st.subheader('Pick-up location ID distribution on the NYC dataset')
st.bar_chart(pulocation_dist)

# Feature Selection
st.header('The features I created')
st.markdown('* **first feature:** I created this feature because of this... I calculated it using this logic....')
st.markdown('* **second feature:** I created this feature because of this... I calculated it using this logic....')

# Model Training
st.header("Time to train the model!")
sel_col, disp_col = st.columns(2)
max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10,max_value=100, value=20, step=10)
n_estimators = sel_col.selectbox('How many trees should there be?', options=[100,200,300,'No Limit'], index=0)

sel_col.text('Here is a list of features in my data:')
sel_col.write(taxi_data.columns)

input_feature = sel_col.text_input("Which feature should be used as the input feature?", 'PULocationID')

if n_estimators == 'No Limit':
    regr = RandomForestRegressor(max_depth)
else:
    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

X = taxi_data[[input_feature]]
X = X.values
y = taxi_data[['trip_distance']]
y = y.values

regr.fit(X,y)
prediction = regr.predict(y)
disp_col.markdown("Mean absolute error of the model is :")
disp_col.write(mean_absolute_error(y,prediction))
disp_col.markdown("Mean squared error of the model is :")
disp_col.write(mean_absolute_error(y,prediction))
disp_col.markdown("R2 squared score of the model is is :")
disp_col.write(r2_score(y,prediction))