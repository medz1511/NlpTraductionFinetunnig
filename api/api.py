from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import torch
import os

# 1. OPTIMISATION RAM : On limite PyTorch à 1 seul thread
# Cela économise environ 50-100 Mo de mémoire tampon
torch.set_num_threads(1)

app = FastAPI(title="API de Traduction NMT", version="1.0")

CHEMIN_LOCAL = "./modele_final_local"
NOM_MODELE_OFFICIEL = "Helsinki-NLP/opus-mt-en-fr"

# Choix du modèle
if os.path.exists(CHEMIN_LOCAL):
    print(f"📂 Chargement local : {CHEMIN_LOCAL}")
    model_name = CHEMIN_LOCAL
else:
    print(f"☁️ Mode Cloud : Chargement de {NOM_MODELE_OFFICIEL}")
    model_name = NOM_MODELE_OFFICIEL

# --- CHARGEMENT OPTIMISÉ ---
try:
    print("⏳ Chargement du Tokenizer...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    print("⏳ Chargement du Modèle (Mode Light)...")
    # L'astuce est ici : low_cpu_mem_usage=True évite les pics de RAM
    model = MarianMTModel.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True
    )
    
    # NOTE : On a retiré quantize_dynamic car le processus consomme trop de RAM
    # Le modèle brut (300Mo) + PyTorch CPU devrait tenir juste dans les 512Mo
    
    print("✅ Modèle chargé et prêt !")

except Exception as e:
    print(f"❌ Erreur critique : {e}")
    model = None

class TranslationRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "online", "message": "Bienvenue sur l'API de Traduction"}

@app.post("/translate")
def translate(request: TranslationRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Modèle non chargé (RAM insuffisante)")
    
    if not request.text:
        raise HTTPException(status_code=400, detail="Texte vide")
    
    # Inférence
    try:
        inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            translated = model.generate(**inputs)
        resultat = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return {"original": request.text, "traduction": resultat[0]}
    except Exception as e:
        return {"erreur": str(e)}

