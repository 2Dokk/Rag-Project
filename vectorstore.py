# -*- coding: utf-8 -*-
"""
과거 진료 이력을 임베딩해서 로컬 벡터DB(Chroma)에 저장하고,
특정 환자/증상과 관련된 과거 기록을 검색(Retrieval)하는 모듈.

임베딩 모델은 HuggingFace의 로컬 sentence-transformers 모델을 사용해서
API 키 없이도 전체 파이프라인이 동작하도록 구성했다.
(LLM 응답 생성 단계에서만 선택적으로 OpenAI/Anthropic API를 사용한다.)
"""
from __future__ import annotations

import os
from langchain_chroma import Chroma
from langchain_core.documents import Document

from data.sample_notes import PAST_VISIT_HISTORY
from local_embeddings import TfidfFallbackEmbeddings

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "hanbang_visit_history"
TFIDF_VECTORIZER_PATH = os.path.join(PERSIST_DIR, "_tfidf_fallback_vectorizer.pkl")

# 다국어(한국어 포함) 성능이 준수한 경량 임베딩 모델 (인터넷 연결 가능한 환경에서 사용)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _huggingface_reachable(timeout: float = 3.0) -> bool:
    import requests
    try:
        resp = requests.head("https://huggingface.co", timeout=timeout)
        return resp.status_code < 500
    except requests.exceptions.RequestException:
        return False


def get_embeddings():
    """
    가능하면 HuggingFace 신경망 임베딩을 사용하고, 네트워크가 막혀 모델을
    내려받을 수 없는 환경(예: 이 개발 샌드박스)에서는 오프라인 TF-IDF 폴백으로
    자동 전환한다. 실제 배포 환경(인터넷 연결됨)에서는 항상 신경망 임베딩이 쓰인다.
    (연결 가능 여부를 먼저 짧게 확인해서, 막혀 있을 때 불필요한 재시도로 오래
    기다리지 않도록 했다.)
    """
    if _huggingface_reachable():
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        except Exception as e:
            print(f"[!] HuggingFace 임베딩 로드 실패({type(e).__name__}) -> 오프라인 TF-IDF 폴백 사용")
    else:
        print("[!] huggingface.co에 연결할 수 없는 환경 -> 오프라인 TF-IDF 폴백 사용")

    corpus = [
        f"[{r['visit_date']}] {r['patient_name_masked']}: {r['note']}"
        for r in PAST_VISIT_HISTORY
    ]
    return TfidfFallbackEmbeddings(TFIDF_VECTORIZER_PATH, corpus)


def build_vectorstore(rebuild: bool = False) -> Chroma:
    """과거 방문 이력을 벡터DB에 적재(ingest)한다."""
    embeddings = get_embeddings()

    if rebuild and os.path.exists(PERSIST_DIR):
        import shutil
        shutil.rmtree(PERSIST_DIR)

    docs = [
        Document(
            page_content=f"[{record['visit_date']}] {record['patient_name_masked']}: {record['note']}",
            metadata={
                "patient_id": record["patient_id"],
                "visit_date": record["visit_date"],
            },
        )
        for record in PAST_VISIT_HISTORY
    ]

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    return vectordb


def load_vectorstore() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def search_patient_history(vectordb: Chroma, patient_id: str, query_text: str, k: int = 3):
    """
    같은 환자(patient_id)의 과거 방문 기록 중, 오늘 진료 내용(query_text)과
    의미상 가장 관련 있는 기록을 top-k로 검색한다.
    """
    results = vectordb.similarity_search(
        query_text,
        k=k,
        filter={"patient_id": patient_id},
    )
    return results


if __name__ == "__main__":
    print("[*] 과거 진료 이력을 벡터DB에 적재합니다...")
    build_vectorstore(rebuild=True)
    print(f"[*] 완료. {len(PAST_VISIT_HISTORY)}건 적재됨 -> {PERSIST_DIR}")
