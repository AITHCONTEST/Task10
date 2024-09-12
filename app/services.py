import requests
from ml.sensim.sensim import similarity_model

async def external_translation_service(text: str, source_language: str, target_language: str) -> str:
    print(source_language)
    print(target_language)
    print(text)
    sentences = similarity_model.find_similar(text)
    print(sentences)
    return text.upper()
    # api_url = ""
    # payload = {
    #     "text": text,
    #     "source_language": source_language,
    #     "target_language": target_language
    # }
    
    # response = requests.post(api_url, json=payload)
    
    # if response.status_code == 200:
    #     return response.json().get("translated_text")
    # else:
    #     raise Exception(f"Ошибка при обращении к сервису перевода: {response.text}")
