# inquiry_chain.py - 사내 문의 응답
from email import parser

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

_store: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """세션 ID별 기록 객체를 반환. 함수 시그니처만 유지하면 저장소는 무엇이든 됩니다."""
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]

llm = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1",
    temperature=0,
)

# 미션 골격 예시: 장애 신고 접수 체인 (소재를 자신의 업무로 교체)
class IncidentReport(BaseModel):
    severity: str = Field(description="장애 등급: 'P1', 'P2', 'P3' 중 하나")
    system: str = Field(description="대상 시스템 이름")
    summary: str = Field(description="장애 현상 한 줄 요약")

# system 프롬프트도 업무 톤으로 교체
# ("system", "너는 장애 접수 담당자야. 접수 확인, 예상 대응, 필요한 추가 정보 요청을 각 한 문장으로 답해.")
routing_parser = JsonOutputParser(pydantic_object=IncidentReport)
routing_prompt = PromptTemplate(
    template="다음 장애를 접수양식으로 정리.\n\n신고: {inquiry}\n\n{format_instructions}",
    input_variables=["inquiry"],
    partial_variables={"format_instructions": routing_parser.get_format_instructions()},
)
routing_chain = routing_prompt | llm | routing_parser

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "너는 장애 접수 담당자야. 접수 확인, 예상 대응, 필요한 추가 정보 요청을 각 한 문장으로 답해."),
    ("human", "{inquiry}"),
])
answer_chain = answer_prompt | llm | StrOutputParser()

full_chain = RunnableParallel(routing=routing_chain, draft=answer_chain)
with_memory = RunnableWithMessageHistory(full_chain, get_session_history)

# --- 실행 ---
inquiry = {"inquiry": "테슬라 인증 API 호출 시, 제3자는 키를 발급할 수 없습니다가 뜹니다. 원인 확인이 필요합니다. 나는 자동차 오너 입니다."}
inquiry2 = {"inquiry": "사내 메일이 갑자기 안됩니다. 메일 서버가 다운된 것 같습니다. 나는 사내 직원입니다."}
inquiry3 = {"inquiry": "사내 메일이 갑자기 안됩니다. 메일 서버가 다운된 것 같습니다. 나는 사내 직원입니다. P1 등급으로 처리해주세요."}

inquiry_c=[inquiry, inquiry2, inquiry3]


for c in inquiry_c:
        result = with_memory.invoke({"inquiry": c["inquiry"]}, config={"configurable": {"session_id": "test_session"}})
        print("---------------------------------------------------------------------------")
        print("장애 등급:", result["routing"]["severity"])
        print("대상 시스템 이름:", result["routing"]["system"])
        print("장애 요약:", result["routing"]["summary"])
        print("답변 초안:")
        print(result["draft"])

 
chain = routing_prompt | llm | routing_parser

result = chain.invoke(
    {"inquiry": "사내 메일이 갑자기 안됩니다. 메일 서버가 다운된 것 같습니다. 나는 사내 직원입니다. P1 등급으로 처리해주세요."})
print("타입:", type(result).__name__)
print("전체:", result)
print("분류만:", result["severity"])


for c in inquiry_c:
    print("\n" + "="*75)
    print(f"입력 문의: {c['inquiry']}")
    print("="*75)
    
    # stream() 메서드를 사용하여 호출합니다.
    stream_chunks = with_memory.stream(
        {"inquiry": c["inquiry"]}, 
        config={"configurable": {"session_id": "test_session"}}
    )
    
    # 각 체인(routing, draft)의 결과를 스트리밍 중에 누적하거나 바로 출력하기 위한 변수
    routing_printed = False
    
    print("답변 출력 시작...\n")
    
    for chunk in stream_chunks:
        # 1. Routing 결과 출력 (보통 첫 번째 청크에 딕셔너리 전체가 한 번에 들어오는 경우가 많습니다)
        if "routing" in chunk and not routing_printed:
            routing_data = chunk["routing"]
            # 데이터가 완성된 형태로 들어왔는지 확인 후 출력
            if all(k in routing_data for k in ["severity", "system", "summary"]):
                print(f"[장애 등급]: {routing_data['severity']}")
                print(f"[대상 시스템]: {routing_data['system']}")
                print(f"[장애 요약]: {routing_data['summary']}")
                print("-" * 50)
                print("[답변 초안]: ", end="", flush=True)
                routing_printed = True
        
        # 2. Draft(답변 초안) 결과 스트리밍 출력
        if "draft" in chunk:
            draft_chunk = chunk["draft"]
            # LangChain 버전에 따라 문자열 또는 AIMessageChunk로 들어올 수 있으므로 처리
            if hasattr(draft_chunk, "content"):
                print(draft_chunk.content, end="", flush=True)
            else:
                print(draft_chunk, end="", flush=True)
                
    print("\n") # 한 문의가 끝난 후 줄바꿈