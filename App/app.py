import streamlit as st
import requests
import time

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Neural Translator",
    page_icon="🧠",
    layout="wide",  # Utilise toute la largeur de l'écran
    initial_sidebar_state="expanded"
)

# --- CSS PERSONNALISÉ ---
# Pour cacher le menu hamburger par défaut et améliorer le style
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .stTextArea textarea {
        font-size: 16px;
        line-height: 1.5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        font-weight: bold;
    }
    footer {visibility: hidden;}
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# --- URL DE L'API ---
API_URL = "https://nlptraductionfinetunnigapi.onrender.com/translate"

# --- SIDEBAR (Infos Techniques) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2014/2014916.png", width=100)
    st.title("À propos")
    st.markdown("""
    Ce projet est une application de **Traduction Neuronale (NMT)** complète.
    
    **Architecture :**
    - 🧠 **Modèle :** MarianMT (Fine-tuned)
    - ⚡ **Backend :** FastAPI
    - 🎨 **Frontend :** Streamlit
    - ☁️ **Déploiement :** Render
    
    **Auteur :**
    *Ton Prénom & Nom*
    *(Master 2 NLP)*
    """)
    st.divider()
    st.info("ℹ️ Le backend utilise une stratégie hybride (IA Locale / API Légère) selon les ressources disponibles.")

# --- CORPS PRINCIPAL ---
st.title("🧠 Neural Translator")
st.markdown("### Traduction Anglais 🇬🇧 ➔ Français 🇫🇷")
st.markdown("---")

# Création de deux colonnes (50% - 50%)
col1, col2 = st.columns(2)

with col1:
    st.subheader("🇬🇧 Texte source (Anglais)")
    # On ajoute une clé pour garder l'état si besoin
    text_input = st.text_area(
        label="Saisissez votre texte ici",
        height=250,
        placeholder="Enter text to translate here...",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("🇫🇷 Traduction (Français)")
    # Espace vide qui sera rempli après la traduction
    result_container = st.empty()
    # Placeholder initial joli
    result_container.info("La traduction apparaîtra ici...")

# --- LOGIQUE DE TRADUCTION ---
# On met le bouton au milieu ou en dessous
st.markdown("<br>", unsafe_allow_html=True) # Petit espace

if st.button("✨ Traduire maintenant", type="primary"):
    if text_input.strip():
        
        # Effet visuel de chargement dans la colonne de droite
        with col2:
            with st.spinner("🤖 Le réseau de neurones réfléchit..."):
                try:
                    # Appel API
                    start_time = time.time()
                    response = requests.post(API_URL, json={"text": text_input})
                    end_time = time.time()
                    duration = round(end_time - start_time, 2)

                    if response.status_code == 200:
                        data = response.json()
                        traduction = data["traduction"]
                        moteur = data.get("moteur", "Modèle Inconnu") # Récupère le moteur si dispo

                        # Affichage du résultat dans la colonne de droite
                        result_container.markdown(f"""
                        <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                            <p style="font-size: 18px; color: #1b5e20;">{traduction}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Affichage des métriques (Temps + Moteur utilisé)
                        st.markdown(f"""
                        <div style="margin-top: 10px; font-size: 12px; color: gray;">
                            ⚡ Temps: {duration}s | ⚙️ Moteur: <b>{moteur}</b>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    else:
                        result_container.error(f"Erreur API : {response.status_code}")

                except requests.exceptions.ConnectionError:
                    result_container.error("❌ Impossible de joindre l'API. Vérifiez que le serveur Render est actif.")
                except Exception as e:
                    result_container.error(f"Une erreur est survenue : {e}")
    else:
        st.toast("⚠️ Veuillez entrer du texte avant de traduire.", icon="⚠️")

# --- FOOTER ---
st.markdown("""
<div class="footer">
    Projet Universitaire - Master 2 NLP - Développé avec ❤️, Python & Caffeine.
</div>
""", unsafe_allow_html=True)