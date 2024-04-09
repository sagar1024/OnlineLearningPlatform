#Name - Sagar Gurung
#Register number - 2347250
#Project - Online learning & analyzing platform

#Importing the Python libraries
import streamlit as st
import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transformers import pipeline
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Loading academic dataset for exploratory data analysis
@st.cache_data
def load_data():
    data = pd.read_csv("dataset.csv")
    return data

#Function for the about page AND exploratory data analysis
def explore_data(data):
    st.header("About Page")
    
    #About the project
    st.write("The Online Learning Platform is designed to provide users with various functionalities to enhance their learning experience. The platform consists of several key features:")
    
    #About the dashboard options
    #st.subheader("Exploratory Data Analysis -")
    st.markdown("<h4>Exploratory Data Analysis - <h4>", unsafe_allow_html=True)
    st.write("Users can explore academic data through descriptive statistics and visualizations, such as 3D scatter plots and surface plots, to gain insights into student performance and engagement.")
    
    #st.subheader("Course Recommendation System -")
    st.markdown("<h4>Course Recommendation System - <h4>", unsafe_allow_html=True)
    st.write("The system recommends courses from Udemy based on user preferences, including level, content duration, and subject. It assists users in discovering relevant courses tailored to their interests and learning objectives.")
    
    #st.subheader("Chatbot for Asking Doubts -")
    st.markdown("<h4>Chatbot for Asking Doubts - <h4>", unsafe_allow_html=True)
    st.write("The chatbot feature enables users to ask questions and receive responses related to their academic queries. Leveraging a pre-trained conversational model, the chatbot provides real-time assistance and support to students seeking clarification on various topics.")
    
    #st.subheader("Performance Prediction -")
    st.markdown("<h4>Performance Prediction - <h4>", unsafe_allow_html=True)
    st.write("Users can input information about their academic activities, such as raised hands, visited resources, announcements view, and discussion participation. Based on this data, the system predicts the student's performance level (low, medium, or high) using a RandomForestClassifier model.")
    
    st.write("The Online Learning Platform aims to create an interactive and engaging environment for users to explore educational content, seek assistance, and improve their academic performance effectively.")
    
    #Displaying basic statistics of the dataset
    st.subheader("Basic Statistics")
    st.write(data.describe())

    #Scatter plot
    st.subheader("3D Scatter Plot")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['Raisedhands'], data['VisitedResources'], data['AnnouncementsView'], c='b', marker='o')
    ax.set_xlabel('Raised Hands')
    ax.set_ylabel('Visited Resources')
    ax.set_zlabel('Announcements View')
    st.pyplot(fig)

    #Surface plot
    # st.subheader("3D Surface Plot")
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y = data['Raisedhands'], data['VisitedResources']
    # X, Y = np.meshgrid(X, Y)
    # Z = X**2 + Y**2
    # ax.plot_surface(X, Y, Z, cmap='viridis')
    # ax.set_xlabel('Raised Hands')
    # ax.set_ylabel('Visited Resources')
    # ax.set_zlabel('Z')
    # st.pyplot(fig)
    
    #Histogram for numeric columns
    st.subheader("Histogram for Numeric Columns")
    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, column in enumerate(['Raisedhands', 'VisitedResources', 'AnnouncementsView', 'Discussion']):
        sns.histplot(data[column], kde=True, ax=axes[i//2, i%2])
        axes[i//2, i%2].set_xlabel(column)
        axes[i//2, i%2].set_ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig1)
    
    #Pair plot for numeric columns
    st.subheader("Pair Plot for Numeric Columns")
    fig2 = sns.pairplot(data[['Raisedhands', 'VisitedResources', 'AnnouncementsView', 'Discussion']], diag_kind='kde')
    st.pyplot(fig2)

    #Box plot for numeric columns
    st.subheader("Box Plot for Numeric Columns")
    fig3, ax4 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data[['Raisedhands', 'VisitedResources', 'AnnouncementsView', 'Discussion']], ax=ax4)
    ax4.set_xlabel("Numeric Columns")
    ax4.set_ylabel("Values")
    st.pyplot(fig3)

#Function for course recommendation
def recommend_courses():
    st.header("Course Recommendation System")
    
    #Project description
    st.write("Welcome to our Course Recommendation System! This system suggests courses from Udemy based on your preferences such as level, content duration, and subject.")

    #Loading course data
    course_data = pd.read_csv("udemy_course.csv")

    #Displaying form for user input
    st.subheader("User Preferences")
    level = st.selectbox("Select Level", course_data['level'].unique())
    duration = st.number_input("Select Content Duration (hours)", min_value=0.0, step=0.5, value=1.0)
    subject = st.selectbox("Select Subject", course_data['subject'].unique())

    #Filtering courses based on user preferences
    recommended_courses = course_data[(course_data['level'] == level) & 
                                      (course_data['content_duration'] <= duration) &
                                      (course_data['subject'] == subject)]
    
    #Displaying recommended courses
    st.subheader("Recommended Courses")
    if recommended_courses.empty:
        st.write("No courses available based on your preferences.")
    else:
        st.write(recommended_courses)

#Function for chatbot
def chatbot():
    st.header("Chatbot for Asking Doubts")
    
    #Projecting description
    st.write("Welcome to our Chatbot for Asking Doubts! You can ask any questions related to your studies, and the chatbot will provide responses to help clarify your doubts.")

    #Loading pre-trained conversational model
    chatbot_model = pipeline("conversational")

    #Displaying chat interface
    user_input = st.text_input("You:", "")
    if user_input:
        conversation = [{'role': 'user', 'content': user_input}]
        bot_response = chatbot_model(conversation)
        st.write("Bot response:", bot_response)

# Function for chatbot
# def chatbot():
#     st.header("Chatbot for Asking Doubts")
#     # Projecting description
#     st.write("Welcome to our Chatbot for Asking Doubts! You can ask any questions related to your studies, and the chatbot will provide responses to help clarify your doubts.")
#     # Loading pre-trained conversational model
#     chatbot_model = pipeline("conversational")
#     # Displaying chat interface
#     user_input = st.text_input("You:", "")
#     if user_input:
#         # Convert the user input into the format expected by the chatbot model
#         conversation = [{"prompt": user_input, "max_length": 100}]
#         # Get response from the chatbot model
#         bot_response = chatbot_model(conversation)[0]["generated_text"]
#         # Display the bot response
#         st.text_area("Bot:", value=bot_response, height=200)
        
#Function to preprocess data and train model
def train_model():
    #Loading marks data from dataset
    marks_data = pd.read_csv("marks_dataset.csv")

    #The dataset columns are Exam1, Exam2 and Next_exam
    X = marks_data[['Exam1', 'Exam2']]
    y = marks_data['Next_exam']

    #Training linear regression model
    model = LinearRegression()
    model.fit(X, y)

    return model

#Function for performance prediction
def predict_performance(model):
    st.header("Performance Prediction")
    
    #Project description
    st.write("Welcome to our Performance Prediction feature! This feature predicts the performance level of a student based on their input marks from the previous exams.")
    
    #Input previous exam marks
    prev_marks1 = st.number_input("Enter Marks for Previous Exam 1:", min_value=0, max_value=100, value=50)
    prev_marks2 = st.number_input("Enter Marks for Previous Exam 2:", min_value=0, max_value=100, value=50)

    #Making prediction for next exam marks
    next_exam_marks = model.predict([[prev_marks1, prev_marks2]])

    #Displaying predicted marks
    st.subheader("Predicted Marks for Next Exam:")
    st.write(next_exam_marks[0])

    #Providing suggestions based on predicted performance
    if next_exam_marks[0] < 60:
        st.write("You might need to study more and seek additional help to improve your performance.")
    elif next_exam_marks[0] < 80:
        st.write("You're doing well but there's still room for improvement. Keep up the good work!")
    else:
        st.write("Congratulations! You're performing excellently. Keep it up!")

def main():
    st.title("Online Learning & Analyzing Platform")

    #Sidebar menu for navigation
    st.sidebar.title("Dashboard")
    menu = ["About", "Course Recommendation", "Chatbot - AskYourDoubts", "Performance Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    #Loading academic dataset for EDA
    data = load_data()
    
    #Menu driven webApp
    if choice == "About":
        explore_data(data)
    elif choice == "Course Recommendation":
        recommend_courses()
    elif choice == "Chatbot - AskYourDoubts":
        chatbot()
    elif choice == "Performance Prediction":
        model = train_model()
        predict_performance(model)

if __name__ == "__main__":
    main()
    
