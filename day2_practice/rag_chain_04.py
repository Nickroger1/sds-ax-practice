# rag_chain.py - 검색-생성(RAG) 체인 구성
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1",
)
db = Chroma(
    collection_name="sds_policies",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1",
    temperature=0,
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


prompt = ChatPromptTemplate.from_template(
    "너는 삼성SDS 사내 규정 안내 담당자야. 아래 문서만 근거로 질문에 답해.\n"
    "문서에 답이 없으면 모른다고 답해.\n\n"
    "문서:\n{context}\n\n질문: {question}"
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
