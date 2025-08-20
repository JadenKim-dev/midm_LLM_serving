"""
Midm API Server 테스트 클라이언트 (with Streaming Support)
"""

import requests
import json
from typing import List, Dict, Optional, Generator
import sseclient


class MidmClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """서버 상태 확인"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> Dict:
        """채팅 요청"""
        payload = {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample
        }
        
        response = self.session.post(
            f"{self.base_url}/chat",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> Generator[Dict, None, None]:
        """스트리밍 채팅 요청"""
        payload = {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample
        }
        
        response = self.session.post(
            f"{self.base_url}/chat/stream",
            json=payload,
            stream=True
        )
        response.raise_for_status()
        
        # SSE 클라이언트 생성
        client = sseclient.SSEClient(response)
        
        for event in client.events():
            if event.data == '[DONE]':
                break
            
            try:
                data = json.loads(event.data)
                yield data
            except json.JSONDecodeError:
                continue
    
    def interactive_chat(self):
        """대화형 채팅 세션"""
        print("=== Midm 채팅 시작 (종료: 'quit' 입력) ===\n")
        
        # 시스템 메시지 초기화
        messages = [
            {
                "role": "system",
                "content": "Mi:dm(믿:음)은 KT에서 개발한 AI 기반 어시스턴트이다."
            }
        ]
        
        while True:
            # 사용자 입력 받기
            user_input = input("\n사용자: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("채팅을 종료합니다.")
                break
            
            # 사용자 메시지 추가
            messages.append({
                "role": "user",
                "content": user_input
            })
            
            try:
                # API 호출
                response = self.chat(messages, max_new_tokens=256)
                assistant_response = response["response"]
                
                # 응답 출력
                print(f"\nMi:dm: {assistant_response}")
                
                # 토큰 사용량 출력 (옵션)
                if "usage" in response and response["usage"]:
                    usage = response["usage"]
                    print(f"\n[토큰 사용: 입력 {usage['prompt_tokens']}, "
                          f"출력 {usage['completion_tokens']}, "
                          f"총 {usage['total_tokens']}]")
                
                # 어시스턴트 응답을 히스토리에 추가
                messages.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                
            except Exception as e:
                print(f"\n오류 발생: {e}")


    def interactive_streaming_chat(self):
        """스트리밍 대화형 채팅 세션"""
        print("=== Midm 스트리밍 채팅 시작 (종료: 'quit' 입력) ===\n")
        
        # 시스템 메시지 초기화
        messages = [
            {
                "role": "system",
                "content": "Mi:dm(믿:음)은 KT에서 개발한 AI 기반 어시스턴트이다."
            }
        ]
        
        while True:
            # 사용자 입력 받기
            user_input = input("\n사용자: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("채팅을 종료합니다.")
                break
            
            # 사용자 메시지 추가
            messages.append({
                "role": "user",
                "content": user_input
            })
            
            try:
                print("\nMi:dm: ", end="", flush=True)
                
                full_response = ""
                token_count = 0
                
                # 스트리밍 응답 처리
                for event in self.chat_stream(messages, max_new_tokens=512):
                    if event['type'] == 'chunk':
                        # 실시간으로 출력
                        print(event['content'], end="", flush=True)
                        full_response = event['accumulated']
                        token_count = event['token_count']
                    elif event['type'] == 'complete':
                        full_response = event['full_response']
                        token_count = event['total_tokens']
                    elif event['type'] == 'error':
                        print(f"\n오류: {event['message']}")
                        break
                
                print()  # 줄바꿈
                
                # 토큰 사용량 출력
                if token_count > 0:
                    print(f"\n[생성된 토큰: {token_count}개]")
                
                # 어시스턴트 응답을 히스토리에 추가
                if full_response:
                    messages.append({
                        "role": "assistant",
                        "content": full_response
                    })
                
            except Exception as e:
                print(f"\n오류 발생: {e}")


def test_basic_request():
    """기본 요청 테스트"""
    client = MidmClient()
    
    print("1. 서버 상태 확인...")
    health = client.health_check()
    print(f"   상태: {health}")
    
    print("\n2. 간단한 채팅 테스트...")
    messages = [
        {
            "role": "system",
            "content": "Mi:dm(믿:음)은 KT에서 개발한 AI 기반 어시스턴트이다."
        },
        {
            "role": "user",
            "content": "안녕하세요! 간단히 자기소개를 해주세요."
        }
    ]
    
    response = client.chat(messages)
    print(f"   응답: {response['response']}")
    
    print("\n3. 멀티턴 대화 테스트...")
    messages.append({
        "role": "assistant",
        "content": response['response']
    })
    messages.append({
        "role": "user",
        "content": "파이썬의 장점을 3가지만 알려주세요."
    })
    
    response = client.chat(messages, max_new_tokens=256)
    print(f"   응답: {response['response']}")


def test_performance():
    """성능 테스트"""
    import time
    
    client = MidmClient()
    
    test_queries = [
        "AI란 무엇인가요?",
        "딥러닝과 머신러닝의 차이는?",
        "트랜스포머 모델을 설명해주세요.",
    ]
    
    print("=== 성능 테스트 ===")
    for query in test_queries:
        messages = [
            {"role": "system", "content": "Mi:dm(믿:음)은 KT에서 개발한 AI 기반 어시스턴트이다."},
            {"role": "user", "content": query}
        ]
        
        start_time = time.time()
        response = client.chat(messages, max_new_tokens=128)
        elapsed_time = time.time() - start_time
        
        print(f"\n질문: {query}")
        print(f"응답 시간: {elapsed_time:.2f}초")
        print(f"응답 길이: {len(response['response'])}자")


def test_streaming():
    """스트리밍 테스트"""
    import time
    
    client = MidmClient()
    
    print("=== 스트리밍 테스트 ===\n")
    
    messages = [
        {
            "role": "system",
            "content": "Mi:dm(믿:음)은 KT에서 개발한 AI 기반 어시스턴트이다."
        },
        {
            "role": "user",
            "content": "파이썬으로 간단한 웹 서버를 만드는 방법을 단계별로 설명해주세요."
        }
    ]
    
    print("질문: 파이썬으로 간단한 웹 서버를 만드는 방법을 단계별로 설명해주세요.\n")
    print("응답 (스트리밍):\n")
    
    start_time = time.time()
    full_response = ""
    first_token_time = None
    token_count = 0
    
    try:
        for event in client.chat_stream(messages, max_new_tokens=256):
            if event['type'] == 'chunk':
                if first_token_time is None:
                    first_token_time = time.time()
                    print(f"[첫 토큰까지 시간: {first_token_time - start_time:.2f}초]\n")
                
                print(event['content'], end="", flush=True)
                token_count = event['token_count']
                
            elif event['type'] == 'complete':
                full_response = event['full_response']
                total_time = time.time() - start_time
                
                print(f"\n\n[스트리밍 완료]")
                print(f"- 총 시간: {total_time:.2f}초")
                print(f"- 생성된 토큰: {event['total_tokens']}개")
                print(f"- 토큰/초: {event['total_tokens'] / total_time:.1f}")
                
            elif event['type'] == 'error':
                print(f"\n오류 발생: {event['message']}")
                
    except Exception as e:
        print(f"\n스트리밍 오류: {e}")


def compare_streaming_vs_regular():
    """일반 응답과 스트리밍 응답 비교"""
    import time
    
    client = MidmClient()
    
    messages = [
        {
            "role": "system",
            "content": "Mi:dm(믿:음)은 KT에서 개발한 AI 기반 어시스턴트이다."
        },
        {
            "role": "user",
            "content": "인공지능의 미래에 대해 설명해주세요."
        }
    ]
    
    print("=== 일반 vs 스트리밍 비교 ===\n")
    
    # 일반 요청
    print("1. 일반 요청 테스트...")
    start_time = time.time()
    response = client.chat(messages, max_new_tokens=200)
    regular_time = time.time() - start_time
    print(f"   - 응답 시간: {regular_time:.2f}초")
    print(f"   - 응답 길이: {len(response['response'])}자")
    
    # 스트리밍 요청
    print("\n2. 스트리밍 요청 테스트...")
    start_time = time.time()
    first_token_time = None
    full_response = ""
    
    for event in client.chat_stream(messages, max_new_tokens=200):
        if event['type'] == 'chunk' and first_token_time is None:
            first_token_time = time.time() - start_time
        elif event['type'] == 'complete':
            full_response = event['full_response']
    
    streaming_total_time = time.time() - start_time
    
    print(f"   - 첫 토큰까지: {first_token_time:.2f}초")
    print(f"   - 전체 시간: {streaming_total_time:.2f}초")
    print(f"   - 응답 길이: {len(full_response)}자")
    
    print("\n3. 비교 결과:")
    print(f"   - 첫 응답 속도: 스트리밍이 {regular_time - first_token_time:.2f}초 빠름")
    print(f"   - 전체 완료 시간 차이: {abs(regular_time - streaming_total_time):.2f}초")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "test":
            print("=== 기본 테스트 실행 ===\n")
            test_basic_request()
        elif mode == "perf":
            test_performance()
        elif mode == "chat":
            client = MidmClient()
            client.interactive_chat()
        elif mode == "stream":
            client = MidmClient()
            client.interactive_streaming_chat()
        elif mode == "test-stream":
            test_streaming()
        elif mode == "compare":
            compare_streaming_vs_regular()
        else:
            print("사용법: python client.py [test|perf|chat|stream|test-stream|compare]")
    else:
        # 기본: 스트리밍 대화형 모드
        client = MidmClient()
        client.interactive_streaming_chat()