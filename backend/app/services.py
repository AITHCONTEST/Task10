from fastapi import HTTPException
import requests
from config import MODEL_URL
from ml.sensim.sensim import similarity_model
import httpx


async def external_translation_service(text: str, source_language: str, target_language: str) -> str:
    print(source_language)
    print(target_language)
    print(text)
    sentences = similarity_model.find_similar(text)
    print(sentences)
    return text.upper()


async def fetchNllb(text: str, source_language: str, target_language: str) -> str:

    payload = {
        "text": text,
        "source_language": source_language,
        "target_language": target_language
    }
    
    response = requests.post(MODEL_URL, json=payload)
    
    if response.status_code == 200:
        return response.json().get("translated_text")
    else:
        raise Exception(f"Ошибка при обращении к сервису перевода: {response.text}")
