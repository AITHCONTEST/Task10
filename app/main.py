from fastapi import FastAPI, HTTPException
from app.models import TranslationRequest, TranslationResponse
from app.services import external_translation_service
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Russian-Mansi Translation API"}


@app.post("/translate/", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    try:
        
        translated_text = await external_translation_service(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language
        )
        
        return TranslationResponse(translated_text=translated_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка перевода: {str(e)}")
