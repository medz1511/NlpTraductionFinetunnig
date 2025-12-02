from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# --- INITIALISATION ---
app = FastAPI(title="API de Traduction NMT (Hybrid)", version="2.0")

# On vérifie si on est sur Render grâce à une variable d'environnement
IS_ON_RENDER = os.environ.get('RENDER', False)

# Variables globales pour stocker le modèle (si on peut le charger)
model = None
tokenizer = None
use_fallback = False # Si True, on utilise deep-translator

print(f"🖥️ Environnement détecté : {'CLOUD (Render)' if IS_ON_RENDER else 'LOCAL'}")

# --- TENTATIVE DE CHARGEMENT DU MODÈLE IA ---
if not IS_ON_RENDER:
    # On ne tente de charger l'IA que si on est en LOCAL (pour économiser la RAM sur Render)
    try:
        from transformers import MarianMTModel, MarianTokenizer
        import torch
        
        print("⏳ Chargement du modèle MarianMT (Local)...")
        model_name = "./modele_final_local"
        if not os.path.exists(model_name):
            model_name = "Helsinki-NLP/opus-mt-en-fr"
            
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        print("✅ Modèle IA chargé avec succès !")
        
    except Exception as e:
        print(f"⚠️ Erreur chargement IA : {e}")
        print("🔄 Bascule automatique vers le mode 'Fallback'")
        use_fallback = True
else:
    # Sur Render, on passe directement en mode léger
    print("☁️ Mode Cloud activé : Utilisation de deep-translator pour économiser la RAM.")
    use_fallback = True

# --- IMPORT DU FALLBACK (Si nécessaire) ---
if use_fallback:
    try:
        from deep_translator import GoogleTranslator
        print("✅ Module de traduction léger prêt.")
    except ImportError:
        print("❌ Erreur critique : deep-translator manquant.")

# --- ROUTE API ---
class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
def translate(request: TranslationRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Texte vide")
    
    try:
        if not use_fallback and model and tokenizer:
            # MÉTHODE 1 : TON MODÈLE IA (Local)
            inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True)
            translated = model.generate(**inputs)
            resultat = [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]
            source = "Modèle MarianMT (IA Locale)"
        else:
            # MÉTHODE 2 : MODE LÉGER (Cloud / Render)
            resultat = GoogleTranslator(source='en', target='fr').translate(request.text)
            source = "Traducteur Cloud (Optimisé RAM)"
            
        return {
            "original": request.text, 
            "traduction": resultat,
            "moteur": source
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "online", "mode": "Cloud" if use_fallback else "Local AI"}