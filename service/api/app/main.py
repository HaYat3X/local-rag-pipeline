from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1.chat import router as chat_router

app = FastAPI()

# =========================
# CORS設定
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Router登録
# =========================
app.include_router(
    chat_router,
    prefix="/v1",  
    tags=["chat"]
)