import streamlit as st
import sqlite3
import tensorflow as tf
import numpy as np
from datetime import datetime


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("./trained_model.keras")

model = load_model()

class_names = [
'Damask Rose',
'Echeveria Flower',
'Mirabilis Jalapa',
'Rain Lily',
'Zinnia Elegans'
]

plant_info = [
"Rosa × damascena (Latin for damascene rose), more commonly known as the Damask rose. The flowers are renowned for their fine fragrance and are commercially harvested for rose oil used in perfumery and to make rose water and \"rose concrete\".",
"Echeveria is a large genus of flowering plants in the family Crassulaceae, native to semi-desert areas of Central America, Mexico and northwestern South America. Echeveria plants are evergreen. Flowers on short stalks (cymes) arise from compact rosettes of succulent fleshy, often brightly coloured leaves.",
"Mirabilis jalapa, the marvel of Peru or four o'clock flower, is the most commonly grown ornamental species of Mirabilis plant, and is available in a range of colors. Mirabilis in Latin means wonderful and Jalapa (or Xalapa) is the state capital of Veracruz in Mexico. Mirabilis jalapa is believed to have been cultivated by the Aztecs for medicinal and ornamental purposes.",
"Zephyranthes is a genus of temperate and tropical bulbous plants in the Amaryllis family, subfamily Amaryllidoideae, native to the Americas and widely cultivated as ornamentals.  Common names for species in this genus include fairy lily, rainflower, zephyr lily, magic lily, Atamasco lily, and rain lily.",
"Zinnia elegans (syn. Zinnia violacea) known as youth-and-age, common zinnia or elegant zinnia, is an annual flowering plant in the family Asteraceae. It is native to Mexico but grown as an ornamental in many places and naturalised in several places, including scattered locations in South and Central America, the West Indies, the United States, Australia, and Italy."
]

def predict(image):
    img = image.resize((224, 224))  # resizes here instead
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    predictions = model.predict(arr)
    idx = np.argmax(predictions)
    return idx

# ========== PAGE CONFIG ==========

st.set_page_config(page_title="Verdant Vision", page_icon="🌿", layout="centered")

# ========== CSS ==========

def reset_css():
    st.markdown("""
    <style>
        .stApp {
            background: none;
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }
        h1, h2, h3 {color: #1abc9c !important;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        MainMenu {visibility: hidden;}
        
        /* Blurred background layer */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url('https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=1920&q=80') no-repeat center center;
            background-size: cover;
            filter: blur(2px);
            z-index: -2;
        }

        /* Green overlay */
        .stApp::after {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(20, 100, 60, 0.35); /* green tint */
            z-index: -1;
        }
                
        /* White card container */
        [data-testid="stVerticalBlock"] {
            background: rgba(205, 210, 200, 0.95);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            max-width: 550px;
            margin: auto;
        }
                
        label {
            color: #1abc9c !important;
            font-weight: 600;
        }
        
        /* Input box background */
        div[data-baseweb="input"] > div {
            width:  1000px;
            max-width: 104%;
            background-color: white !important;
            color: black !important;
        }

        /* Text inside input */
        div[data-baseweb="input"] input {
            width:  600px;
            max-width: 540px;
            border-radius: 8px;
            background-color: white !important;
            color: black !important;
        }
        
        /* Placeholder text */
        div[data-baseweb="input"] input::placeholder {
            color: #000 !important;
        }

        /* Border styling */
        div[data-baseweb="input"] > div {
            border-radius: 8px;
        }

        /* Eye icon color */
        div[data-baseweb="input"] button {
            color: #1abc9c !important;
        }

        /* Hover effect */
        div[data-baseweb="input"] button:hover {
            color: #16a085 !important;
        }

        .stButton>button {
            background: linear-gradient(135deg, #1abc9c, #16a085);
            color: white;
            border: none;
            border-radius: 8px;
            
            width:  1000px;
            max-width: 100%;
            font-weight: 600;
        }
                
        .result-box {
            background: rgba(26, 188, 156, 0.1);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #1abc9c;
            margin-top: 20px;
        }
                
        .result-item {
            background: rgba(26, 188, 156, 0.1);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #1abc9c;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)
reset_css()

# ========== DATABASE METHODS ==========

def register_user(name, email, password):
    conn = sqlite3.connect('plant_app.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                  (name, email, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Email already exists
    finally:
        conn.close()

def login_user(email, password):
    conn = sqlite3.connect('plant_app.db')
    c = conn.cursor()
    c.execute("SELECT name FROM users WHERE email=? AND password=?", (email, password))
    user = c.fetchone()
    conn.close()
    return user[0] if user else None

def save_history(user_email, label, filename):
    conn = sqlite3.connect('plant_app.db')
    c = conn.cursor()
    c.execute("INSERT INTO history (user_email, label, filename) VALUES (?, ?, ?)",
              (user_email, label, filename))
    conn.commit()
    conn.close()

def get_history(user_email):
    conn = sqlite3.connect('plant_app.db')
    c = conn.cursor()
    c.execute("SELECT label, filename, analyzed_at FROM history WHERE user_email=? ORDER BY analyzed_at DESC",
              (user_email,))
    results = c.fetchall()
    conn.close()
    return results


# ========== SESSION STATE ==========

if "page" not in st.session_state:
    st.session_state.page = "home"
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "users" not in st.session_state:
    st.session_state.users = {}
if "history" not in st.session_state:
    st.session_state.history = []

# ========== HOME PAGE ==========

def home_page():
    st.markdown("""
    <style>  
    [data-testid="stVerticalBlock"] {
        width: 1000px;
        max-width: 110%;
    } 
    </style>
    """
    , unsafe_allow_html=True)
    st.header("ORNAMENTAL PLANT IDENTIFIER")
    st.markdown("""
    
    Our mission is to help in identifying ornamental plants efficiently. Upload an image of a plant, and our system will analyze it to identify what type of plant it is.

    ### How It Works
    1. **Create Account:** Sign up or log in to access the classifier.
    2. **Upload Image:** Go to the **identify** page and upload an image of an ornamental plant.
    3. **Analysis:** Our system will process the image using advanced deep learning algorithms to identify the plant.
    4. **Results:** View the plant name and detailed information about it.

    ### Why Choose Us?
    
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds.
    - **History:** Keep track of all your previously classified plants.

    ### Supported Plants
    -  Damask Rose
    -  Echeveria Flower
    -  Mirabilis Jalapa
    -  Rain Lily
    -  Zinnia Elegans

    ### Get Started
    Click **Get Started** below to create an account and start classifying!
    """)

    if st.button("Get Started"):
        st.session_state.page = "login"
        st.rerun()

# ========== LOGIN PAGE ==========

def login_page():
    reset_css()
    st.markdown("<h2>Ornamental Plant Identifier</h2>", unsafe_allow_html=True)
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login_user(email, password)
        if user:
            st.session_state.user_name = user
            st.session_state.user_email = email
            st.session_state.page = "dashboard"
            st.rerun()
            
        else:
            st.warning("email or password wrong")

    if st.button("Create Account", key="to_signup"):
        st.session_state.page = "signup"
        st.rerun()

# ========== SIGNUP PAGE ==========

def signup_page():
    reset_css()
    st.markdown("<h2>Create Account</h2>", unsafe_allow_html=True)
    st.caption("Join Ornamental Plant Identifier")
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if name and email and password and confirm:
            if password != confirm:
                st.error("Passwords do not match")
            else:
                success = register_user(name, email, password)
                if success:
                    st.success("Account created! Please login.")
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error("Email already registered")
        else:
            st.warning("Please fill in all fields")

    if st.button("Back to Login", key="to_login"):
        st.session_state.page = "login"
        st.rerun()

# ========== DASHBOARD PAGE ==========

def dashboard_page():
    reset_css()
    st.markdown(f"<h2>Welcome {st.session_state.user_name} </h2>", unsafe_allow_html=True)

    if st.button("Identify"):
        st.session_state.page = "identifier"
        st.rerun()

    if st.button("History"):
        st.session_state.page = "history"
        st.rerun()

    if st.button("Logout"):
        st.session_state.user_name = ""
        st.session_state.page = "login"
        st.rerun()

# ========== CLASSIFIER PAGE ==========

def identifier_page():
    reset_css()
    st.markdown("<h2>Identify Plant Image</h2>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        from PIL import Image
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)

        if st.button("Identify"):
            with st.spinner("Analyzing..."):
                label = predict(image)

            st.markdown(f"""
            <div class="result-box">
                <h3>{class_names[label]}</h3>
                <p>information: {plant_info[label]}</p>
            </div>
            """, unsafe_allow_html=True)

            save_history(st.session_state.user_email, class_names[label], uploaded.name)

    if st.button("Home"):
        st.session_state.page = "dashboard"
        st.rerun()

# ========== HISTORY PAGE ==========

def history_page():
    reset_css()
    st.markdown("<h2>Your Identified Plants</h2>", unsafe_allow_html=True)

    results = get_history(st.session_state.user_email)

    if not results:
        st.info("No Identified plants yet")
    else:
        for i, result in enumerate(results):
            label, filename, analyzed_at = result
            st.markdown(f"""
            <div class="result-item">
                <h4>#{i+1} - {label}</h4>
                <p><strong>File:</strong> {filename}</p>
                <p><small>Identified: {analyzed_at}</small></p>
            </div>
            """, unsafe_allow_html=True)

    if st.button("Home"):
        st.session_state.page = "dashboard"
        st.rerun()

# ========== ROUTER ==========

if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "login":
    login_page()
elif st.session_state.page == "signup":
    signup_page()
elif st.session_state.page == "dashboard":
    dashboard_page()
elif st.session_state.page == "identifier":
    identifier_page()
elif st.session_state.page == "history":
    history_page()