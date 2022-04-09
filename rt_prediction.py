import streamlit as st
import pandas as pd
import statsmodels.api as sm
import patsy

st.set_page_config(
    page_title='Mock Risk Calculator', 
    page_icon='🖩', 
    layout='centered', 
    initial_sidebar_state='expanded',
     menu_items=None
)

st.title('Mock Risk Calculator')

# Load model
logit = sm.load('./data/logit.pkl')

with st.form('Form'):
    # Get user input
    gre = st.slider('GRE score', min_value=200, max_value=800, value=500, step=50)
    gpa = st.slider('GPA', min_value=2.0, max_value=4.0, value=3.0, step=0.5)
    rank = st.radio('Rank', options=range(1,5))

    # Submit button
    submitted = st.form_submit_button('Calculate')
    if submitted:
        # Take selection and convert to a data frame
        selection = pd.DataFrame(
            {
                'gre':[gre],
                'gpa':[gpa],
                'rank':pd.Categorical([rank], categories=[1, 2, 3, 4]),
            }
        )
        # One hot encode categorical variable
        selection = patsy.dmatrix('gre + gpa + C(rank)', selection, return_type='dataframe')

        # Predict probability
        prediction = logit.predict(selection)
        st.write(f'Predicted probability = {prediction[0]:.0%}')

        # Make recommendation based on a particular threshold probability
        threshold = 0.65
        if prediction[0] > threshold:
            refer='likely to be admitted'
        else:
            refer='not likely to be admitted'

        # Make recommendation (use a larger font)
        recommendation = f'<p style="font-size: 24px;">Based on a threshold of {threshold:.0%}, this candidate is {refer}</p>'
        st.markdown(recommendation, unsafe_allow_html=True)

with st.expander('Click here for more information'):
    st.write('Summary of the Logistic Regression Model')
    st.write(logit.summary())

with st.sidebar:
    st.markdown('# Notes')
    st.markdown('The data used from this mock up are from [UCLA](https://stats.oarc.ucla.edu/r/dae/logit-regression/)')
    st.write('*This calculator is part of a prospective research study and does not constitute medical advice.*')
    st.write('(Other useful information here)')

# https://discuss.streamlit.io/t/remove-made-with-streamlit-from-bottom-of-app/1370/2
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
