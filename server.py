import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# === Pydantic Models for Request/Response ===
class Message(BaseModel):
    role: str = Field(..., description="메시지 역할 (system/user/assistant)")
    content: str = Field(..., description="메시지 내용")


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="채팅 히스토리")
    max_new_tokens: Optional[int] = Field(128, description="생성할 최대 토큰 수")
    temperature: Optional[float] = Field(1.0, description="샘플링 온도")
    do_sample: Optional[bool] = Field(False, description="샘플링 사용 여부")
    
    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "system", "content": "Mi:dm(믿:음)은 KT에서 개발한 AI 기반 어시스턴트이다."},
                    {"role": "user", "content": "트랜스포머 모델의 작동 원리를 설명해줘"}
                ],
                "max_new_tokens": 256,
                "temperature": 0.7,
                "do_sample": False
            }
        }


class ChatResponse(BaseModel):
    response: str = Field(..., description="모델의 응답")
    usage: Optional[Dict[str, int]] = Field(None, description="토큰 사용 정보")


# === Global Model Storage ===
class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.device = None
    
    def load_model(self, model_name: str):
        """모델을 메모리에 로드"""
        print(f"Loading model: {model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        # 토크나이저와 설정 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generation_config = GenerationConfig.from_pretrained(model_name)
        
        print("Model loaded successfully!")
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> tuple[str, dict]:
        """주어진 메시지로부터 응답 생성"""
        
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # 채팅 템플릿 적용
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        # 입력 토큰 수 계산
        input_token_count = input_ids.shape[1]
        
        # 생성 설정 업데이트
        gen_config = GenerationConfig.from_dict(self.generation_config.to_dict())
        gen_config.max_new_tokens = max_new_tokens
        gen_config.temperature = temperature
        gen_config.do_sample = do_sample
        
        # 응답 생성
        with torch.no_grad():
            output = self.model.generate(
                input_ids.to(self.device),
                generation_config=gen_config,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 출력 토큰 수 계산
        output_token_count = output.shape[1] - input_token_count
        
        # 전체 출력 디코딩
        full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 입력 부분 제거하여 응답만 추출
        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        response_only = full_response[len(input_text):].strip()
        
        # 사용 정보
        usage = {
            "prompt_tokens": input_token_count,
            "completion_tokens": output_token_count,
            "total_tokens": input_token_count + output_token_count
        }
        
        return response_only, usage


model_manager = ModelManager()


# === FastAPI App with Lifespan ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 모델 로드, 종료 시 정리"""
    # Startup
    model_name = "K-intelligence/Midm-2.0-Mini-Instruct"
    try:
        model_manager.load_model(model_name)
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    print("Shutting down and cleaning up resources...")
    if model_manager.model is not None:
        del model_manager.model
        del model_manager.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="Midm Mini Model API",
    description="KT Mi:dm Mini 모델 서빙 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === API Endpoints ===
@app.get("/")
async def root():
    """헬스 체크 및 API 정보"""
    return {
        "service": "Midm Mini Model API",
        "status": "running",
        "model_loaded": model_manager.model is not None,
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """서버 및 모델 상태 확인"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "device": model_manager.device,
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    채팅 완성 엔드포인트
    
    채팅 히스토리를 받아서 다음 응답을 생성합니다.
    """
    try:
        messages = [msg.model_dump() for msg in request.messages]
        
        response_text, usage = model_manager.generate_response(
            messages=messages,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=request.do_sample
        )
        
        return ChatResponse(
            response=response_text,
            usage=usage
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    스트리밍 채팅 엔드포인트 (추후 구현 가능)
    
    Server-Sent Events 또는 WebSocket을 통한 스트리밍 응답
    """
    return {"message": "Streaming endpoint not implemented yet"}


# === Main Entry Point ===
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )
