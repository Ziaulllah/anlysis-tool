import streamlit as st
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import seaborn as sns
import sqlite3
import re
from streamlit_option_menu import option_menu
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder  


st.set_page_config(
    page_icon="Logo.png",
    page_title="Data Analysis app",
    layout="wide"
)

internal_CSS = f"""
<style>

/* Default styles */
h1 {{ 
    font-family: "Source Sans Pro", sans-serif;
    font-weight: 700;
    font-size: 2.00rem;
    padding: 1.25rem 0px 1rem;
    margin: 0px;
    line-height: 1.2;
}}

/* Light mode styles (White background) */
body[data-theme="light"] h1 {{
    color: #000000; /* Black text */
}}

/* Dark mode styles (Black background) */
body[data-theme="dark"] h1 {{
    color: #ffffff; /* White text */
}}

</style>
"""
# Inject CSS into the app
st.markdown(internal_CSS, unsafe_allow_html=True)


# Function for the Home Page
def home_page():
    st.title("Welcome to the Data Analysis App!")
    st.write(
        """
        Unlock the power of your data with our comprehensive Data Analysis App! Whether your CSV file is ready for graphical representation or needs some polishing, we’ve got you covered.
        
        - **If your data is clean and ready**, proceed directly to the **Data Visualization** section to create insightful and interactive graphs.  
        - **If your data requires cleaning**, click on the **Data Cleaning** button to fix errors, handle missing values, and standardize your dataset before visualizing.  
        
        This app ensures that your data is prepared and presented in the most effective way possible. Let's turn your raw data into meaningful insights!
        """
    )



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# User Define Function For Data Cleaning
def data_cleaning():
    # Section selection
    section = st.radio("Select a Section", ("Data Cleaning", "Encoding and Decoding"))

    if section == "Data Cleaning":
        st.write("### Welcome to the Data Cleaning Section!")
        st.write("Upload your dataset and clean it effortlessly.")

        # File Upload
        uploaded_file = st.file_uploader("Upload a CSV file for cleaning:", type=["csv"])

        if uploaded_file is not None:
            if uploaded_file.size == 0:
                st.error("Uploaded file is empty.")
            else:
                # Load the dataset
                df = pd.read_csv(uploaded_file)
                st.success("Dataset Preview:")
                st.dataframe(df)

                st.warning(f"Shape of the Dataset: {df.shape}")
                st.write("Missing values in the dataset:")
                st.write(df.isnull().sum())

                # 1️⃣ **Delete Specific Columns**
                delete_column = st.radio("Do you want to delete specific columns?", ("Yes", "No"))
                if delete_column == "Yes":
                    columns_to_delete = st.multiselect("Select columns to delete:", df.columns)
                    if columns_to_delete:
                        df = df.drop(columns=columns_to_delete)
                        st.success(f"Deleted columns: {', '.join(columns_to_delete)}")
                        st.write("Updated Dataset:")
                        st.dataframe(df)
                    else:
                        st.warning("No columns selected for deletion.")

                # 2️⃣ **Remove Missing Values**
                remove_missing = st.radio("Do you want to remove rows with missing values?", ("Yes", "No"))
                if remove_missing == "Yes":
                    df = df.dropna()
                    st.success("Missing values removed!")
                    st.write("Updated Dataset:")
                    st.dataframe(df)

                # 3️⃣ **Fill Missing Values**
                else:  # Only offer filling missing values if user didn't delete them
                    fill_missing = st.radio("Do you want to fill missing values instead?", ("Yes", "No"))
                    if fill_missing == "Yes":
                        missing_cols = df.columns[df.isnull().any()].tolist()

                        if missing_cols:
                            selected_col = st.selectbox("Select a column to fill missing values:", missing_cols)

                            if selected_col:
                                if df[selected_col].dtype in ['int64', 'float64']:  # Numerical Columns
                                    method = st.radio(f"Choose a method to fill missing values for {selected_col}:", 
                                                      ("Top", "Median", "Mode", "Mean", "Custom"))
                                    if method == "Top":
                                        df[selected_col].fillna(df[selected_col].max(), inplace=True)
                                    elif method == "Median":
                                        df[selected_col].fillna(df[selected_col].median(), inplace=True)
                                    elif method == "Mode":
                                        df[selected_col].fillna(df[selected_col].mode()[0], inplace=True)
                                    elif method == "Mean":
                                        df[selected_col].fillna(df[selected_col].mean(), inplace=True)
                                    elif method == "Custom":
                                        custom_value = st.number_input(f"Enter a custom numeric value for {selected_col}:", value=0.0)
                                        df[selected_col].fillna(custom_value, inplace=True)

                                else:  # Categorical Columns
                                    method = st.radio(f"Choose a method to fill missing values for {selected_col}:", 
                                                      ("Top", "Custom"))
                                    if method == "Top":
                                        df[selected_col].fillna(df[selected_col].mode()[0], inplace=True)
                                    elif method == "Custom":
                                        custom_value = st.text_input(f"Enter a custom value for {selected_col}:")
                                        if custom_value:
                                            df[selected_col].fillna(custom_value, inplace=True)

                            st.success(f"Missing values in '{selected_col}' have been filled.")
                            st.write("Updated Dataset:")
                            st.dataframe(df)

                        else:
                            st.info("No missing values found in the dataset.")

                # 4️⃣ **Remove Duplicate Values**
                Duplicate_rows = df[df.duplicated()]
                st.warning(f"Duplicate values in dataset: {Duplicate_rows.shape}")

                dlt_duplicate = st.radio("Do you want to delete duplicate values?", ("Yes", "No"))
                if dlt_duplicate == "Yes":
                    df = df.drop_duplicates()
                    st.success(f"After deleting duplicates, dataset shape: {df.shape}")
                else:
                    st.info("Did not delete duplicate values.")

                # 5️⃣ **Outlier Detection**
                st.subheader("Outlier Detection")
                numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
                if not numerical_cols.empty:
                    selected_col = st.selectbox("Select a numerical column for outlier detection:", numerical_cols)
                    if selected_col:
                        # Plot before removing outliers
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.boxplot(x=df[selected_col], ax=ax)
                        ax.set_title(f"Boxplot Before Outlier Removal - {selected_col}")
                        st.pyplot(fig)

                        # Ask user if they want to remove outliers
                        remove_outliers = st.radio("Do you want to remove outliers?", ("Yes", "No"))
                        if remove_outliers == "Yes":
                            Q1 = df[selected_col].quantile(0.25)
                            Q3 = df[selected_col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            df = df[(df[selected_col] >= lower_bound) & (df[selected_col] <= upper_bound)]

                            # Plot after removing outliers
                            fig, ax = plt.subplots(figsize=(8, 5))
                            sns.boxplot(x=df[selected_col], ax=ax)
                            ax.set_title(f"Boxplot After Outlier Removal - {selected_col}")
                            st.pyplot(fig)

                            st.success(f"Outliers from '{selected_col}' have been removed.")

                        st.write("Dataset after Outlier Removal:")
                        st.dataframe(df)
                        
    elif section == "Encoding and Decoding":
        st.write("### Encoding and Decoding Section")

        uploaded_file = st.file_uploader("Upload a CSV file for encoding:", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)  # Read CSV File
                st.write("### Uploaded Dataset")
                st.dataframe(df)  # Display the whole dataset

                categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

                if categorical_columns:
                    selected_columns = st.multiselect("Select columns to encode", categorical_columns)

                    if selected_columns:
                        encoding_method = st.radio("Select Encoding Method", ["One-Hot Encoding", "Label Encoding"])

                        if encoding_method == "One-Hot Encoding":
                            df = pd.get_dummies(df, columns=selected_columns)
                        elif encoding_method == "Label Encoding":
                            for col in selected_columns:
                                le = LabelEncoder()  # Initialize LabelEncoder
                                df[col] = le.fit_transform(df[col])

                        st.success("Encoding applied successfully.")
                        st.write("### Encoded Dataset")
                        st.dataframe(df)  # Display the encoded dataset
                    else:
                        st.warning("Please select at least one column to encode.")
                else:
                    st.warning("No categorical columns found in the uploaded file.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

# Function for Data Visualization
def data_visualization_section():
    st.title("This is the Data Visualization section.")

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file is not None:
        if uploaded_file.size == 0:
            st.error("Uploaded file is empty.")
        else:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("Dataset Preview:")
                st.dataframe(df.head())

                Viz_Type = st.selectbox(
                    "Select the visualization type for this data",
                    ("Scatter Plot", "Bar Chart", "Line Chart", "Histogram", "Boxplot")
                )

                st.write("Available columns:", list(df.columns))
                x = st.text_input("Enter the column name for X-Axis")
                y = st.text_input("Enter the column name for Y-Axis (if applicable)")

                if x in df.columns and (Viz_Type != "Histogram" or y in df.columns):
                    if Viz_Type == "Scatter Plot":
                        st.title("Scatter Plot")
                        fig, ax = plt.subplots()
                        sns.scatterplot(x=x, y=y, data=df, ax=ax)
                        st.pyplot(fig)

                    elif Viz_Type == "Bar Chart":
                        st.title("Bar Chart")
                        fig, ax = plt.subplots()
                        sns.barplot(x=x, y=y, data=df, ax=ax)
                        st.pyplot(fig)

                    elif Viz_Type == "Line Chart":
                        st.title("Line Chart")
                        fig, ax = plt.subplots()
                        sns.lineplot(x=x, y=y, data=df, ax=ax)
                        st.pyplot(fig)

                    elif Viz_Type == "Histogram":
                        st.title("Histogram")
                        fig, ax = plt.subplots()
                        sns.histplot(df[x], bins=20, kde=True, ax=ax)
                        st.pyplot(fig)

                    elif Viz_Type == "Boxplot":
                        st.title("Boxplot")
                        fig, ax = plt.subplots()
                        sns.boxplot(x=x, y=y, data=df, ax=ax)
                        st.pyplot(fig)
                else:
                    st.error("Please provide valid X and Y columns for this visualization.")
            except pd.errors.ParserError:
                st.error("Uploaded file is not a valid CSV.")
    else:
        st.warning("Please upload a CSV file to proceed.")
        


# Sidebar Menu
with st.sidebar:
    submenu = option_menu(
        menu_title="Menu Options",  # Title for the submenu
        options=["Home", "Contact", "Review"],  # Submenu options
        icons=["house-fill", "telephone", "star-fill"],  # Icons for submenu
        default_index=0,  # Home will open by default now
        styles={
            "container": {"padding": "5px", "background-color": "gray"},
            "icon": {"color": "green", "font-size": "16px"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px", "--hover-color": "#ddd"},
            "nav-link-selected": {"background-color": "#ffc107", "color": "black"},
        },
    )

# Home Section
if submenu == "Home":
    selections = option_menu(
        menu_title=None,
        options=['Home', 'Data Cleaning', 'Data Visualization'],
        icons=['house-fill', 'gear-fill', 'bar-chart-fill'],
        default_index=0,
        orientation='horizontal',
        key="unique_option_menu_key",  # Unique key to avoid conflicts
        styles={
            "container": {"padding": "5px 23px", "background-color": "#0d6efd", "border-radius": "8px"},
            "icon": {"color": "#f9fafb", "font-size": "18px"},
            "nav-link": {"color": "#f9fafb", "font-size": "15px", "text-align": "center", "margin": "0 10px"},
            "nav-link-selected": {"background-color": "#ffc107", "font-size": "12px"}
        }
    )

    if selections=="Data Cleaning":
            data_cleaning()
    elif selections=="Data Visualization":
            data_visualization_section()
    else:
        home_page()

# Contact Section
elif submenu == "Contact":
    st.subheader("Contact")
    st.header(":mailbox: Get In Touch With Us!")

    contact_form = """
    <form action="https://formsubmit.co/ziaullahbj9@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here"></textarea>
        <button type="submit">Send</button>
    </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)

    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style/style.css")

# Review Section
elif submenu == "Review":
    st.header("Review")

    # Connect to Database
    conn = sqlite3.connect("reviews.db", check_same_thread=False)
    cursor = conn.cursor()

    # Create Reviews Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            review TEXT
        )
    """)
    conn.commit()

    # Function to Validate Email
    def is_valid_email(email):
        pattern = r'^[\w\.-]+@gmail\.com$'
        return re.match(pattern, email) is not None

    # Review Submission Form
    with st.form("review_form"):
        name = st.text_input("Enter your Name")
        email = st.text_input("Enter your Email")
        review = st.text_area("Write your Review")
        submitted = st.form_submit_button("Submit Review")

        if submitted:
            if not name or not email or not review:
                st.error("Please fill out all fields.")
            elif not is_valid_email(email):
                st.error("Please enter a valid email address ending with @gmail.com.")
            else:
                cursor.execute("INSERT INTO reviews (name, email, review) VALUES (?, ?, ?)", (name, email, review))
                conn.commit()
                st.success("Your review has been submitted successfully!")

    # Buttons for Admin Panel & Show All Reviews
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Admin Panel"):
            st.session_state.show_admin_login = True
    
    with col2:
        if st.button("Show Over All Reviews"):
            st.session_state.show_reviews = True

    # Show All Reviews
    if "show_reviews" in st.session_state and st.session_state.show_reviews:
        st.subheader("All User Reviews")
        cursor.execute("SELECT name, review FROM reviews")
        reviews = cursor.fetchall()

        if not reviews:
            st.info("No reviews found.")
        else:
            for review in reviews:
                st.write(f"**{review[0]}:** {review[1]}")

    # Admin Login Page
    if "show_admin_login" in st.session_state and st.session_state.show_admin_login:
        st.subheader("Admin Login")
        admin_email = st.text_input("Admin Email")
        admin_password = st.text_input("Admin Password", type="password")
        admin_login = st.button("Login as Admin")

        if admin_login:
            if admin_email == "admin@gmail.com" and admin_password == "admin@##123":
                st.session_state.admin_logged_in = True
                st.success("Admin Logged In Successfully!")

        if "admin_logged_in" in st.session_state and st.session_state.admin_logged_in:
            st.subheader("Manage Reviews")

            # Fetch and Display All Reviews for Admin
            cursor.execute("SELECT * FROM reviews")
            reviews = cursor.fetchall()

            if not reviews:
                st.info("No reviews found.")
            else:
                review_ids_to_delete = []
                for review in reviews:
                    st.write(f"**ID:** {review[0]}")
                    st.write(f"**Name:** {review[1]}")
                    st.write(f"**Email:** {review[2]}")
                    st.write(f"**Review:** {review[3]}")

                    # Use a form for delete buttons to handle state properly
                    with st.form(f"delete_form_{review[0]}"):
                        delete_button = st.form_submit_button(f"Delete Review {review[0]}")

                        if delete_button:
                            review_ids_to_delete.append(review[0])

                # Process Deletion
                if review_ids_to_delete:
                    for review_id in review_ids_to_delete:
                        cursor.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
                    conn.commit()
                    st.success("Selected review(s) deleted! Refreshing...")
                    st.experimental_rerun()



## Sidebar configuration
with st.sidebar:
    # Display logo image
    st.image("logo5.webp", width=150)  # Adjust width as needed

    # Adding a custom style with HTML and CSS for sidebar
    st.markdown("""
        <style>
            .custom-text {
                font-size: 10px;
                font-weight: bold;
                text-align: center;
                color: #ffc107;
            }
            .custom-text span {
                color: #04ECF0; /* Color for the word 'Recommendation' */
            }
        </style>
    """, unsafe_allow_html=True)



# CSS for smooth color blending with optimized arrangement of colors
sidebar_footer_css = """
<style>
    @keyframes smooth-color-blend {
        0% { color:rgb(248, 24, 17); }        /* Orange */
        10% { color:rgb(244, 141, 131); }       /* Soft Coral */
        20% { color:rgb(237, 155, 21); }       /* Amber */
        30% { color: #FFC300; }       /* Yellow */
        40% { color: #33FF57; }       /* Green */
        50% { color: #1F618D; }       /* Dark Blue */
        60% { color: #009688; }       /* Teal */
        70% { color: #3357FF; }       /* Blue */
        80% { color: #8E44AD; }       /* Purple */
        90% { color:rgb(194, 135, 218); }       /* Violet */
        100% { color:rgb(242, 185, 173); }      /* Back to Orange */
    }

    .sidebar .sidebar-content {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }

    .footer {
        text-align: center;
        font-size: 22px;
        animation: smooth-color-blend 15s infinite linear; /* 15s for smoother transition */
        padding: 10px 0;
        font-weight: bold;
    }
</style>
"""

# Footer HTML for the sidebar
sidebar_footer_html = """
<div class="footer">
    Developed by <strong>Zia Ullah</strong>  Crafting Excellence with Style!
</div>
"""

# Inject CSS into the Streamlit app
st.markdown(sidebar_footer_css, unsafe_allow_html=True)

#Display footer in the sidebar
st.sidebar.markdown(sidebar_footer_html, unsafe_allow_html=True)

