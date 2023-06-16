import streamlit as st
import sqlite3
import subprocess

# Function to create the database connection
def create_connection():
    conn = sqlite3.connect('users.db')
    return conn

# Function to create the users table
def create_table(conn):
    query = '''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL
        )
    '''
    conn.execute(query)

# Function to insert a new user into the database
def insert_user(conn, username, password):
    query = 'INSERT INTO users (username, password) VALUES (?, ?)'
    conn.execute(query, (username, password))
    conn.commit()

# Function to check if a user exists in the database
def check_user(conn, username, password):
    query = 'SELECT * FROM users WHERE username = ? AND password = ?'
    result = conn.execute(query, (username, password))
    return result.fetchone() is not None

# Create the database connection and table
conn = create_connection()
create_table(conn)

# Streamlit app
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if check_user(conn, username, password):
            st.success("Login successful!")

            streamlit_command = ["streamlit", "run", "streamlit1.py", "--server.maxUploadSize", "2048"]

            # Run the Streamlit app using the subprocess module
            subprocess.run(streamlit_command)
        else:
            st.error("Invalid username or password.")


def signup():
    st.title("Sign Up")
    new_username = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    signup_button = st.button("Sign Up")

    if signup_button:
        if new_password == confirm_password:
            insert_user(conn, new_username, new_password)
            st.success("Sign up successful!")
        else:
            st.error("Passwords do not match.")

def main():
    col1, col2 = st.columns(2)

    with col1:
        st.title("User Authentication")

    with col2:
        page = st.radio("Go to", ("Login", "Sign Up"))

    if page == "Login":
        login()
    elif page == "Sign Up":
        signup()

if __name__ == "__main__":
    main()

# Close the database connection
conn.close()
