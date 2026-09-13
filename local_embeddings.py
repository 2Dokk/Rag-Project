# -*- coding: utf-8 -*-
"""
네트워크 연결이 막혀 HuggingFace 모델을 내려받을 수 없는 환경(예: 이 데모를 돌리고 있는
샌드박스)을 위한 오프라인 폴백 임베딩.

TF-IDF 기반이라 신경망 임베딩만큼 의미 유사도를 잘 잡아내진 못하지만,
"문서를 벡터로 바꿔서 저장하고, 질의도 벡터로 바꿔서 코사인 유사도로 가장 가까운
문서를 찾는다"는 RAG의 핵심 구조는 동일하게 보여준다.

실제 배포 환경(인터넷이 열려 있는 서버/로컬 PC)에서는 이 파일을 쓸 필요 없이
vectorstore.py가 자동으로 HuggingFaceEmbeddings(신경망 임베딩)를 사용한다.
"""
from __future__ import annotations

import os
import pickle
from typing import List

from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfFallbackEmbeddings(Embeddings):
    def __init__(self, persist_path: str, corpus: List[str]):
        self.persist_path = persist_path
        self.vectorizer = self._load_or_fit(corpus)

    def _load_or_fit(self, corpus: List[str]) -> TfidfVectorizer:
        if os.path.exists(self.persist_path):
            with open(self.persist_path, "rb") as f:
                return pickle.load(f)

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        vectorizer.fit(corpus)

        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump(vectorizer, f)
        return vectorizer

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.vectorizer.transform([text]).toarray()[0].tolist()
