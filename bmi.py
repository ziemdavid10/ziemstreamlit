import streamlit as st

st.title("Welcome to BMI Calculator")

weight = st.number_input("Enter your weight in Kgs")
status = st.radio('select your height format:', ('cms', 'meters', 'feet'))

try:
    if status == 'cms':
        height = st.number_input("Enter your height in cms")
        height_in_meters = height / 100
        bmi = weight / (height_in_meters ** 2)
        st.write(f"Your BMI is {bmi:.2f}")
    elif status == 'meters':
        height_in_meters = st.number_input("Enter your height in meters")
        bmi = weight / (height_in_meters ** 2)
        st.write(f"Your BMI is {bmi:.2f}")
    else:
        height_in_feet = st.number_input("Enter your height in feet")
        height_in_meters = height_in_feet * 0.3048
        bmi = weight / (height_in_meters ** 2)
        st.write(f"Your BMI is {bmi:.2f}")
except ZeroDivisionError:
    st.error("Height cannot be zero. Please enter a valid height.")

if (st.button("Calculate BMI")):
    st.write(f"Your BMI is {bmi:.2f}")
    
    
    
    if bmi < 18.5:
        st.warning("You are underweight.")
    elif 18.5 <= bmi < 24.9:
        st.success("You have a normal weight.")
    elif 25 <= bmi < 29.9:
        st.info("You are overweight.")
    else:
        st.error("You are obese.")