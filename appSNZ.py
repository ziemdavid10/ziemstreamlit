import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Viral Predictor Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DESIGN CSS PERSONNALISÉ ---
def local_css():
    st.markdown("""
    <style>
    /* Import de polices */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Gradient de fond pour la sidebar */
    [data-testid="stSidebar"] {
        background-image: linear-gradient(#2e3440, #1a1c23);
        color: white;
    }

    /* Style des titres */
    .main-header {
        font-weight: 800;
        color: #1DA1F2;
        font-size: 3rem !important;
        text-shadow: 2px 2px 10px rgba(29, 161, 242, 0.3);
        margin-bottom: 0px;
    }
    
    .sub-header {
        color: #aebbc1;
        font-weight: 400;
        font-size: 1.2rem !important;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    /* Boîtes de contenu (Glassmorphism) */
    .content-box {
        background: rgba(255, 255, 255, 0.05);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    /* Badges de prédiction */
    .prediction-viral {
        background: linear-gradient(90deg, #ff4b2b, #ff416c);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.4);
    }

    .prediction-not-viral {
        background: linear-gradient(90deg, #3a6186, #89253e);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
    }

    /* Animation au survol des boutons */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: #1DA1F2;
        color: white;
        border: none;
        padding: 10px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(29, 161, 242, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data(dataset):
    return pd.read_csv(dataset)

def main():
    local_css()
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/124/124010.png", width=80) # Logo Twitter/Social
        st.markdown("# Navigation")
        menu = ['🏠 Home', '📊 Analysis', '📈 Visualization', '🤖 AI Prediction']
        choice = st.selectbox("Page", menu)
        st.markdown("---")
        st.info("Développé pour l'analyse de contenu viral.")

    # --- EN-TÊTE ---
    st.markdown("<h1 class='main-header'>Viral Predictor Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Powered by Machine Learning for Social Media Strategy</p>", unsafe_allow_html=True)

    # Chargement des données
    try:
        data = load_data('social_media_viral_content_dataset.csv')
    except:
        st.error("Dataset introuvable.")
        return

    # --- LOGIQUE DES PAGES ---
    
    if choice == '🏠 Home':
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown("""
            <div class='content-box'>
                <h3>Bienvenue dans l'outil d'analyse prédictive</h3>
                <p>Cette application utilise un modèle de <b>Random Forest</b> pour déterminer le potentiel de viralité de vos publications sur les réseaux sociaux.</p>
                <ul>
                    <li>Analyse de l'engagement</li>
                    <li>Comparaison par plateforme</li>
                    <li>Prédiction en temps réel</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.lottie = "🔥" # Placeholder pour une animation
            st.markdown(f"<div style='font-size:100px; text-align:center;'>{st.lottie}</div>", unsafe_allow_html=True)

    elif choice == '📊 Analysis':
        st.subheader("🔍 Exploratory Data Analysis")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write("Aperçu des données")
            st.metric("Total Posts", len(data))
        with col2:
            st.dataframe(data.head(10), use_container_width=True)
        
        if st.checkbox("Show Statistics Summary"):
            st.table(data.describe())

    elif choice == '📈 Visualization':
        st.subheader("📉 Data Insights")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Viralité par Classe**")
            fig2 = plt.figure(figsize=(8,6))
            sns.countplot(data=data, x='is_viral', palette="viridis")
            st.pyplot(fig2)
        with c2:
            st.markdown("**Engagement par Plateforme**")
            fig3 = plt.figure(figsize=(10,6))
            sns.boxplot(data=data, x='platform', y='engagement_rate', palette="magma")
            plt.xticks(rotation=45)
            st.pyplot(fig3)

    elif choice == '🤖 AI Prediction':
        tab1, tab2 = st.tabs(["⚡ Prediction Manuelle", "📁 Batch Prediction (CSV)"])

        with tab1:
            st.markdown("<div class='content-box'>", unsafe_allow_html=True)
            st.write("### 🎛️ Paramètres de la publication")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                followers = st.number_input('Abonnés', 0, 10000000, 50000)
                hour = st.slider('Heure de post', 0, 23, 12)
            with c2:
                likes = st.number_input('Likes attendus', 0, 1000000, 1000)
                platform = st.selectbox('Plateforme', ['Facebook', 'Instagram', 'LinkedIn', 'TikTok', 'Twitter', 'YouTube'])
            with c3:
                shares = st.number_input('Partages', 0, 500000, 100)
                content_type = st.radio('Type', ['Image', 'Video'])

            # Calcul automatique du taux d'engagement pour l'exemple
            engagement = (likes + shares) / (followers if followers > 0 else 1)
            
            if st.button("Lancer la prédiction 🚀"):
                # Simulation de l'input pour le modèle (doit matcher tes 13 features)
                # Note: Il faut reconstruire l'array exactement comme dans ton code original
                # [followers, likes, shares, comments, engagement, hour, FB, IG, LI, TT, TW, YT, Video]
                
                # ... (Logique de préparation de l'array input_data ici) ...
                
                st.markdown("---")
                # Affichage factice du résultat pour le design
                is_viral_result = engagement > 0.1 # Exemple de règle
                if is_viral_result:
                    st.markdown("<div class='prediction-viral'>🚀 POTENTIEL VIRAL DÉTECTÉ !</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='prediction-not-viral'>📉 CONTENU STANDARD</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.info("Uploadez un fichier CSV pour prédire plusieurs lignes d'un coup.")
            upload_file = st.file_uploader("Choisir un fichier", type=["csv"])

if __name__ == '__main__':
    main()