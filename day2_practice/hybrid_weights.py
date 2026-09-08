from langchain_classic.retrievers import EnsembleRetriever
from hybrid import bm25, vector

# hybrid_weights.py - 가중치 조정 실험 (hybrid.py에 이어서)
for w in [0.2, 0.5, 0.8]:
    h = EnsembleRetriever(retrievers=[bm25, vector], weights=[w, 1 - w])
    top = h.invoke("월차 며칠 받아?")[0].metadata["source"].split("/")[-1]
    print(f"BM25 {w:.1f} / 벡터 {1 - w:.1f} -> 1위: {top}")