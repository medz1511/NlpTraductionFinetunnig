# Fichier: app.py
import streamlit as st
import requests

# Configuration de la page
st.set_page_config(page_title="Traducteur Pro", page_icon="🌍")

st.title("🌍 Traducteur Anglais-Français (Architecture Microservices)")
st.markdown("Cette interface est connectée à une **API FastAPI** qui héberge le modèle de Deep Learning.")

# Zone de texte
text_input = st.text_area("Entrez le texte en anglais :", height=150, placeholder="The AI is transforming the world...")

# Bouton d'action
if st.button("Traduire", type="primary"):
    if text_input:
        # Affichage d'un spinner pendant l'appel API
        with st.spinner('Interrogation de l\'API en cours...'):
            try:
                # Appel à l'API FastAPI (qui tourne sur le port 8000)
                response = requests.post(
                    "https://nlptraductionfinetunnigapi.onrender.com",
                    json={"text": text_input}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    traduction = data["traduction"]
                    
                    st.success("Traduction réussie !")
                    st.info(f"🇫🇷 **Français :** {traduction}")
                else:
                    st.error(f"Erreur API : {response.status_code}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Impossible de se connecter à l'API.")
                st.warning("Assurez-vous d'avoir lancé 'api.py' dans un autre terminal !")
    else:
        st.warning("Veuillez entrer du texte.")

# Footer
st.markdown("---")
st.caption("Projet réalisé avec FastAPI (Backend) et Streamlit (Frontend).")