import torch
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextIteratorStreamer
from threading import Thread
from queue import Queue, Empty
from sentence_transformers import SentenceTransformer
import numpy as np


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


class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="임베딩을 생성할 텍스트 리스트")
    model: Optional[str] = Field(None, description="사용할 임베딩 모델명 (선택사항)")


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="생성된 임베딩 벡터들")
    model: str = Field(..., description="사용된 모델명")
    dimensions: int = Field(..., description="임베딩 벡터 차원수")
    usage: Optional[Dict[str, int]] = Field(None, description="토큰 사용 정보")


# === Global Model Storage ===
class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.device = None
        self.embedding_model = None
    
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
    
    def load_embedding_model(self, embedding_model_name: str = "jhgan/ko-sroberta-multitask"):
        print(f"Loading embedding model: {embedding_model_name}")
        
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
            print(f"Embedding model loaded successfully! Dimensions: {self.embedding_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> tuple[List[List[float]], dict]:
        if self.embedding_model is None:
            raise ValueError("Embedding model not loaded")
        
        if not texts:
            raise ValueError("No texts provided")
        
        embeddings = []
        total_tokens = 0
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            batch_embeddings_list = [embedding.tolist() for embedding in batch_embeddings]
            embeddings.extend(batch_embeddings_list)
            
            total_tokens += sum(len(text) // 1.5 for text in batch_texts)
        
        usage = {
            "total_tokens": int(total_tokens),
            "total_texts": len(texts)
        }
        
        return embeddings, usage
    
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
        
        usage = {
            "prompt_tokens": input_token_count,
            "completion_tokens": output_token_count,
            "total_tokens": input_token_count + output_token_count
        }
        
        return response_only, usage
    
    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> AsyncGenerator[str, None]:
        """스트리밍 응답 생성"""
        
        if self.model is None:
            raise ValueError("Model not loaded")
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        # 입력 텍스트 길이 저장 (응답만 추출하기 위해)
        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        input_length = len(input_text)
        
        # 생성 설정
        gen_config = GenerationConfig.from_dict(self.generation_config.to_dict())
        gen_config.max_new_tokens = max_new_tokens
        gen_config.temperature = temperature
        gen_config.do_sample = do_sample
        
        # TextIteratorStreamer 설정
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0
        )
        
        # 생성 스레드 시작
        generation_kwargs = dict(
            input_ids=input_ids.to(self.device),
            generation_config=gen_config,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        return streamer, thread


# 모델 매니저 인스턴스
model_manager = ModelManager()


# === FastAPI App with Lifespan ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    model_name = "K-intelligence/Midm-2.0-Mini-Instruct"
    try:
        model_manager.load_model(model_name)
        model_manager.load_embedding_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    print("Shutting down and cleaning up resources...")
    if model_manager.model is not None:
        del model_manager.model
        del model_manager.tokenizer
    if model_manager.embedding_model is not None:
        del model_manager.embedding_model
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
    return {
        "service": "Midm Mini Model API",
        "status": "running",
        "model_loaded": model_manager.model is not None,
        "embedding_model_loaded": model_manager.embedding_model is not None,
        "endpoints": {
            "chat": "/chat",
            "chat_stream": "/chat/stream",
            "embeddings": "/embeddings",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    embedding_dimensions = None
    if model_manager.embedding_model is not None:
        embedding_dimensions = model_manager.embedding_model.get_sentence_embedding_dimension()
    
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "embedding_model_loaded": model_manager.embedding_model is not None,
        "embedding_dimensions": embedding_dimensions,
        "device": model_manager.device,
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
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
    async def generate_sse():
        try:
            messages = [msg.model_dump() for msg in request.messages]
            
            # 스트리머와 스레드 가져오기
            streamer, thread = model_manager.generate_stream(
                messages=messages,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=request.do_sample
            )
            
            # SSE 시작 이벤트
            yield f"data: {json.dumps({'type': 'start', 'message': 'Stream started'})}\n\n"
            
            generated_text = ""
            token_count = 0
            
            for text_chunk in streamer:
                if text_chunk:
                    generated_text += text_chunk
                    token_count += 1
                    
                    chunk_data = {
                        'type': 'chunk',
                        'content': text_chunk,
                        'accumulated': generated_text,
                        'token_count': token_count
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                    
                    await asyncio.sleep(0)
            
            thread.join()
            
            # 완료 이벤트
            complete_data = {
                'type': 'complete',
                'full_response': generated_text,
                'total_tokens': token_count
            }
            yield f"data: {json.dumps(complete_data, ensure_ascii=False)}\n\n"
            
            # SSE 종료 신호
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_data = {'type': 'error', 'message': str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


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