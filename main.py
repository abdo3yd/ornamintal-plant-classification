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
            background: rgba(210, 200, 210, 0.85);
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
        [data-testid="stAppViewContainer"] {
        background: rgba(210, 200, 210, 0.85) !important;
        }
        [data-testid="stVerticalBlock"] {
            background: transparent !important;
            box-shadow: none !important;
            max-width: 100% !important;
            padding: 10px !important;
        }
        [data-testid="stAppViewContainer"] {
        padding-top: 0 !important;
        }
        [data-testid="block-container"] {
        padding-top: 0 !important;
        }
        /* Navbar full width */
        .navbar {
        position: relative;
        left: 50%;
        right: 50%;
        margin-left: -50vw !important;
        margin-right: -50vw !important;
        margin-top: -4rem !important;
        width: 100vw !important;
        padding: 15px 40px;
        background: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        }
        .navbar-logo {
            font-size: 22px;
            font-weight: 700;
            color: #1abc9c !important;
        }
        .navbar-links {
            display: flex;
            gap: 30px;
            list-style: none;
            margin: 0;
            padding: 0;
        }
        .navbar-links a {
            text-decoration: none;
            color: #333;
            font-weight: 500;
            font-size: 16px;
        }
        .navbar-links a:hover {
            color: #1abc9c;
        }
        .navbar-btn {
            background: linear-gradient(135deg, #1abc9c, #16a085);
            color: white !important;
            padding: 8px 20px;
            border-radius: 20px;
            text-decoration: none;
            font-weight: 600;
        }
        .hero {
        margin-top: 20px;
        padding: 30px 40px;  /* was 60px 40px */
        text-align: center;
        background: linear-gradient(135deg, rgba(26,188,156,0.2), rgba(22,160,133,0.3));
        border-radius: 20px;
        }
        .hero h1 {
        font-size: 28px;  /* was 42px */
        color: #0f4c27 !important;
        }
        .hero p {
        font-size: 14px;  /* was 18px */
        color: #333;
        }
        .features {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 40px;
            flex-wrap: nowrap;
        }
        .feature-card {
        background: white;
        padding: 8px;  /* was 15px */
        border-radius: 15px;
        width: 150px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        text-align: center;
        }
        .feature-card h3 {
        color: #1abc9c !important;
        font-size: 15px;  /* was 16px */
        }
        .feature-card p {
        color: #555;
        font-size: 11px;  /* was 13px */
        }
        .plants-section {
            margin-top: 40px;
            text-align: center;
        }
        .plants-section h2 {
        font-size: 25px;  
        }
        .plants-grid {
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
            margin-top: 20px;
        }
        .plant-tag {
        background: rgba(26,188,156,0.15);
        border: 2px solid #1abc9c;
        padding: 5px 12px;  /* smaller padding too */
        border-radius: 20px;
        color: #0f4c27;
        font-weight: 500;
        font-size: 12px;  /* smaller text */
        }
    </style>

    <!-- NAVBAR -->
    <div class="navbar">
        <div class="navbar-logo">🌿 Verdant Vision</div>
        <ul class="navbar-links">
            <li><a href="#">Home</a></li>
            <li><a href="#">About</a></li>
            <li><a href="#" class="navbar-btn">Login</a></li>
        </ul>
    </div>

    <!-- HERO -->
    <div class="hero">
        <h1>Identify Ornamental Plants for Free</h1>
        <p>Instantly identify ornamental plants from images using advanced deep learning algorithms.</p>
    </div>

    <!-- FEATURES -->
    <div class="features">
        <div class="feature-card"><h3>⚡ Fast</h3><p>Get results in seconds</p></div>
        <div class="feature-card"><h3>🎯 Accurate</h3><p>State-of-the-art CNN model</p></div>
        <div class="feature-card"><h3>📋 History</h3><p>Track your past results</p></div>
        <div class="feature-card"><h3>🔒 Secure</h3><p>Your data is safe</p></div>
    </div>

    <!-- PLANTS -->
    <div class="plants-section">
        <h2>Supported Plants</h2>
        <div class="plants-grid">
            <span class="plant-tag">🌹 Damask Rose</span>
            <span class="plant-tag">🌵 Echeveria Flower</span>
            <span class="plant-tag">🌸 Mirabilis Jalapa</span>
            <span class="plant-tag">🌼 Rain Lily</span>
            <span class="plant-tag">🌺 Zinnia Elegans</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Get Started", use_container_width=True):
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