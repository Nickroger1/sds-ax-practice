# verify_sources.py - 근거 문서를 함께 반환
import importlib.util
from pathlib import Path

from langchain_core.runnables import RunnableParallel

_spec = importlib.util.spec_from_file_location(
    "rag_chain", Path(__file__).parent / "rag_chain_04.py"
)
_rag_chain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rag_chain)
rag_chain = _rag_chain.rag_chain
retriever = _rag_chain.retriever

rag_with_sources = RunnableParallel(
    answer=rag_chain,
    sources=retriever | (lambda docs: sorted({d.metadata["source"] for d in docs})),
)
result = rag_with_sources.invoke("물류플랫폼팀 박도윤입니다. 해외 출장 숙박비 한도가 얼마인가요?")
print(result["answer"])
print("근거 문서:", result["sources"])