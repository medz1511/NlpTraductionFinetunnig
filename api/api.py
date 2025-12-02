from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import torch
import os

app = FastAPI(title="API de Traduction NMT", version="1.0")
from torch.quantization import quantize_dynamic
# --- LOGIQUE CLOUD VS LOCAL ---
# Si le dossier local existe (sur ton PC), on l'utilise.
# Sinon (sur Render), on utilise le modèle officiel de Hugging Face.
CHEMIN_LOCAL = "./modele_final_local"
NOM_MODELE_OFFICIEL = "Helsinki-NLP/opus-mt-en-fr"

if os.path.exists(CHEMIN_LOCAL):
    print(f"📂 Chargement depuis le dossier local : {CHEMIN_LOCAL}")
    model_name = CHEMIN_LOCAL
else:
    print(f"☁️ Dossier local introuvable. Chargement depuis Hugging Face : {NOM_MODELE_OFFICIEL}")
    model_name = NOM_MODELE_OFFICIEL

# Chargement
try:
    print(f"Chargement du modèle depuis : {model_name}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model_brut = MarianMTModel.from_pretrained(model_name)
    print(" Modèle chargé avec succès !!!")
    print("Compression du modèle  en cours...")
    model = quantize_dynamic(
        model_brut, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    del model_brut  # Libérer de la mémoire
    print("✅ Modèle compressé et chargé avec succès !
except Exception as e:
    print(f"❌ Erreur critique : {e}")

class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
def translate(request: TranslationRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Texte vide")
    
    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated = model.generate(**inputs)
    resultat = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    return {"original": request.text, "traduction": resultat[0]}