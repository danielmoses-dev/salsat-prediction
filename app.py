import streamlit as st
import pickle
import joblib
import pandas as pd
import altair as alt
import datetime

#Salary CatBoostRegressor model
salary_model = joblib.load('models/salary_prediction_model.pkl')

#Satisfaction LogisticRegressor model
satisfaction_model = joblib.load('models/satisfaction_model.pkl')

#All the maps, label encoders or cleaned data from the dataset
with open('models/dev_map.pkl', 'rb') as file:
    dev_map = pickle.load(file)

with open('models/country_map.pkl', 'rb') as file:
    country_map = pickle.load(file)

with open('models/edlevel_map.pkl', 'rb') as file:
    edlevel_map = pickle.load(file)

with open('models/remotework_map.pkl', 'rb') as file:
    remotework_map = pickle.load(file)

with open('models/orgsize_map.pkl', 'rb') as file:
    orgsize_map = pickle.load(file)

with open('models/satisfaction_labels.pkl', 'rb') as file:
    satisfaction_labels = pickle.load(file)

with open('models/convertedcompyearly_data.pkl', 'rb') as file:
    ConvertedCompYearly = pickle.load(file)

with open('models/yearscodepro_data.pkl', 'rb') as file:
    YearsCodePro = pickle.load(file)

st.title(":blue[SalSat]: Employee Salary and Satisfaction Prediction")

#Default sidebar with About and Profile outline
with st.sidebar:
    st.header(" 💸️ About :blue[SalSat]")
    st.write("Estimate tech salary and satisfaction based on your profile.")
    st.markdown("---")
    profile_container = st.container(border=True)

if "profile" not in st.session_state:
    with profile_container:
        st.header("🙍‍♂️ Your Profile")
        st.markdown("Please fill the form and click **Predict**.")

#Main form to get user data
with st.form("predict_form"):
    devtype_input = st.selectbox("What is your role??", dev_map.keys())
    devtype_encoded = dev_map[devtype_input]

    country_input = st.selectbox("Where are you located?", country_map.keys(), index=15)
    country_encoded = country_map[country_input]

    yearscodepro = st.slider("How many years of experience do you have?", min_value=0, max_value=29, value=3)
    yearscodepro = float(yearscodepro)

    edlevel_input = st.selectbox("What is your highest education qualification?", edlevel_map.keys(), index=3)
    edlevel_encoded = edlevel_map[edlevel_input]

    remotework_input = st.selectbox("What are your flexibility expectations?", remotework_map.keys())
    remotework_encoded = remotework_map[remotework_input]

    orgsize_input = st.selectbox("What is the expected size of the organization?", orgsize_map.keys())
    orgsize_encoded = orgsize_map[orgsize_input]

    currency = st.selectbox("Select Currency", ["USD", "INR", "EUR"], index=1)

    submitted = st.form_submit_button("Predict")

#After user clicks the Predict button
if submitted:
    #Creating dataframe of input features
    features = [[
        devtype_encoded, country_encoded, yearscodepro, edlevel_encoded, remotework_encoded, orgsize_encoded
    ]]
    input_df = pd.DataFrame(features, columns=['DevType', 'Country', 'YearsCodePro', 'EdLevel', 'RemoteWork', 'OrgSize'])

    #Predicting Salary using pre-trained CatBoost Regression model
    predicted_salary = float(salary_model.predict(input_df))
    st.session_state["salary_result"] = round(predicted_salary, 2)
    conversion_rates = {"USD": 1, "INR": 83.2, "EUR": 0.92}
    converted_salary = st.session_state["salary_result"] * conversion_rates[currency]

    #Predicting Satisfaction using pre-trained Logistic Regression model
    predicted_satisfaction = satisfaction_model.predict(input_df)
    satisfaction_final = satisfaction_labels.inverse_transform(predicted_satisfaction)[0]

    #The Profile in sidebar is updated with user-entered values
    with st.sidebar:
        with profile_container:
            st.markdown(f":blue-badge[Role] {devtype_input}")
            st.markdown(f":green-badge[Country] {country_input}")
            st.markdown(f":orange-badge[Experience] {yearscodepro} years")
            st.markdown(f":blue-badge[Education] {edlevel_input}")
            st.markdown(f":green-badge[Flexibility] {remotework_input}")
            st.markdown(f":orange-badge[Org Size] {orgsize_input}")

    #Predicted Salary and Satisfaction are displayed
    st.success(f"💰 Estimated Salary: {converted_salary:,.0f} {currency}")
    st.info(f"😊 Predicted Satisfaction: {satisfaction_final}")

    #Simple text report is created and made available to users for download
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_text = f"""
        SalSat Report - {now}

        🙍‍♂️ User Profile:
        -----------------
        Role: {devtype_input}
        Country: {country_input}
        Experience: {yearscodepro} years
        Education: {edlevel_input}
        Flexibility: {remotework_input}
        Org Size: {orgsize_input}

        📊 Predictions:
        -----------------
        Estimated Salary: {converted_salary:,.0f} {currency}
        Predicted Satisfaction: {satisfaction_final}

        Thank you for using SalSat
        """

    st.download_button(
        label="Download Report",
        data=report_text,
        file_name="SalSat_Report.txt",
        mime="text/plain"
    )

    st.markdown("---")

    #Graph for Salary vs Satisfaction
    st.header("Salary vs Experience Graph")
    st.write("This chart shows how annual salary (in USD) typically trends with years of professional coding experience based on real-world data.")
    st.write("The :blue[blue line] represents average salaries across different experience levels. The :red[red dot] highlights your predicted salary at your specified experience level, allowing you to visually compare your estimate with broader industry trends.")
    exp_salary_df = pd.DataFrame({
        'YearsCodePro': YearsCodePro,
        'ConvertedCompYearly': ConvertedCompYearly
    })

    user_exp = yearscodepro
    user_salary = converted_salary

    base = alt.Chart(exp_salary_df).mark_line(point=True).encode(
        x=alt.X('YearsCodePro', title='Years of Experience'),
        y=alt.Y('ConvertedCompYearly', title='Annual Salary (USD)'),
        tooltip=['YearsCodePro', 'ConvertedCompYearly']
    )

    highlight = alt.Chart(pd.DataFrame({
        'YearsCodePro': [user_exp],
        'ConvertedCompYearly': [user_salary]
    })).mark_point(size=150, color='red').encode(
        x='YearsCodePro',
        y='ConvertedCompYearly'
    )

    st.altair_chart(base + highlight, use_container_width=True)





