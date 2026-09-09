# failure_agent.py - 장애 주입 도구를 Agent에 연결해 복구 확인
import json
import os
import failure_injection
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langgraph.prebuilt import create_react_agent
from failure_injection import get_asset_status, get_asset_status_backup

load_dotenv()


def get_text(message):
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모아 반환합니다."""
    content = message.content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content


llm = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1",
)
agent = create_react_agent(
    llm,
    [get_asset_status, get_asset_status_backup],
    prompt="자산 상태 질문에는 get_asset_status를 먼저 사용하고 도구 에러 메시지에 안내된 지침을 그대로 따르세요.",
)


def print_trace(result: dict):
    """2-3에서 만든 trace 출력 (실패 지점 분석용)."""
    for i, m in enumerate(result["messages"]):
        if m.type == "human":
            print(f"{i} 사용자   {get_text(m)}")
        elif m.type == "ai" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                print(f"{i} 도구요청 {tc['name']} args={json.dumps(tc['args'], ensure_ascii=False)}")
        elif m.type == "tool":
            print(f"{i} 도구결과 [{m.name}] {get_text(m)[:70]}")
        elif m.type == "ai":
            print(f"{i} 최종답변 {get_text(m)[:70]}")


def tool_names(result: dict) -> list:
    """trace에서 실제로 호출된 도구 이름만 뽑습니다."""
    return [tc["name"] for m in result["messages"]
            if getattr(m, "tool_calls", None) for tc in m.tool_calls]


def check_secret_log(path="agent.log") -> tuple:
    """로그 파일에 자격 증명 원문이 남았는지 확인합니다."""
    if not os.path.exists(path):
        return True, "agent.log 없음"
    with open(path, encoding="utf-8") as f:
        text = f.read()
    hits = [p for p in ("sk-internal-dummy-1234", "AKIA") if p in text]
    return (not hits), ("발견 0건" if not hits else f"발견: {hits}")


if __name__ == "__main__":
    question = "A-1001 자산의 현재 상태를 알려주세요."

    # 1) 정상 호출: 장애 없이 한 번 돌려 봅니다
    failure_injection.FAILURE_MODE = "none"
    normal = agent.invoke({"messages": [("user", question)]})
    normal_ok = ("get_asset_status" in tool_names(normal)
                 and "에러" not in get_text(normal["messages"][-1]))

    # 2) 실패 호출: FAILURE_MODE로 고른 장애를 주입합니다
    mode = os.environ.get("FAILURE_MODE", "timeout")
    failure_injection.FAILURE_MODE = mode
    broken = agent.invoke({"messages": [("user", question)]})
    print(f"=== 장애 주입 trace (FAILURE_MODE={mode}) ===")
    print_trace(broken)
    recovered = "get_asset_status_backup" in tool_names(broken)
    answered = bool(get_text(broken["messages"][-1]).strip())

    # 3) 로그에 Secret이 남았는지 점검합니다
    secret_ok, secret_msg = check_secret_log()

    print("\n=== DoD 판정 ===")
    print(f"정상 호출 처리     : {'통과' if normal_ok else '실패'}")
    print(f"실패 호출 처리     : {'통과' if answered else '실패'} (그래프 중단 없이 답변 생성)")
    print(f"필수 장애 1개 복구 : {'통과' if recovered else '실패'} (장애 {mode}, 백업 도구 호출 {'있음' if recovered else '없음'})")
    print(f"로그 Secret 점검   : {'통과' if secret_ok else '실패'} ({secret_msg})")
    print(f"최종               : {'DoD 충족' if all([normal_ok, answered, recovered, secret_ok]) else 'DoD 미충족'}")