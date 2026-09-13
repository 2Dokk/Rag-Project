# -*- coding: utf-8 -*-
"""
LLM 호출을 추상화하는 모듈.

- OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY 중 하나가 .env / 환경변수에 설정돼 있으면
  실제 LLM을 호출해서 답을 생성한다.
- 모두 없으면(오늘 우리 상황) generate()가 규칙 기반 폴백(fallback)으로 동작해서,
  API 키 없이도 파이프라인 전체가 끝까지 돌아가는 걸 확인할 수 있게 해준다.
  나중에 키만 넣으면 자동으로 진짜 LLM 응답으로 전환된다.
"""
from __future__ import annotations

import os
import re
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"
# Gemini는 OpenAI 호환 엔드포인트를 제공하므로 langchain-openai로 그대로 호출한다.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _get_chat_model():
    if ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    if OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if GEMINI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=GEMINI_MODEL,
            temperature=0,
            api_key=GEMINI_API_KEY,
            base_url=GEMINI_OPENAI_BASE_URL,
        )
    return None


def has_live_llm() -> bool:
    return bool(OPENAI_API_KEY or ANTHROPIC_API_KEY or GEMINI_API_KEY)


def generate(system_prompt: str, user_prompt: str) -> str:
    """
    시스템 프롬프트 + 유저 프롬프트를 받아 답변 문자열을 반환한다.
    API 키가 없으면 아주 단순한 규칙 기반 폴백으로 대체한다(데모/오프라인용).
    """
    model = _get_chat_model()
    if model is not None:
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        return _invoke_with_rate_limit_retry(model, messages).content

    return _fallback_generate(user_prompt)


def _invoke_with_rate_limit_retry(model, messages, max_attempts: int = 5):
    """
    무료 티어(예: Gemini 분당 5회)는 파이프라인 한 번(메모 N건 x 2회 호출)에도
    429(RateLimit)에 걸린다. 에러 메시지의 'retry in Xs' 안내만큼 기다렸다가 재시도한다.
    일일 한도 소진처럼 기다려도 풀리지 않는 경우를 대비해 시도 횟수는 제한한다.
    """
    import time

    for attempt in range(1, max_attempts + 1):
        try:
            return model.invoke(messages)
        except Exception as e:
            is_rate_limit = type(e).__name__ == "RateLimitError" or "429" in str(e)
            if not is_rate_limit or attempt == max_attempts:
                raise
            # 일일 한도 소진은 기다려도 당일엔 풀리지 않는데, Gemini는 이때도 'retry in Xs'를 안내한다.
            if "PerDay" in str(e):
                raise RuntimeError(
                    "LLM 일일 호출 한도를 모두 사용했습니다(무료 티어). "
                    "내일 다시 실행하거나, API 키 프로젝트에 결제를 연결하거나, "
                    ".env의 GEMINI_MODEL을 다른 모델로 바꿔 실행하세요."
                ) from e
            m = re.search(r"retry in ([\d.]+)s", str(e))
            wait = float(m.group(1)) + 1 if m else 30.0
            print(f"[!] LLM 호출 한도 초과(429) -> {wait:.0f}초 후 재시도 ({attempt}/{max_attempts - 1})")
            time.sleep(wait)


def _fallback_generate(user_prompt: str) -> str:
    """
    API 키가 없을 때 쓰는 아주 단순한 규칙 기반 요약기.
    실제 LLM만큼 자연스럽진 않지만, 파이프라인 구조(입력 -> 구조화 -> 출력)를
    끝까지 검증하는 용도로는 충분하다.
    """
    lines = [ln.strip() for ln in re.split(r"[.\n]", user_prompt) if ln.strip()]
    bullet_lines = "\n".join(f"- {ln}" for ln in lines[:6])
    return (
        "[LLM 미연결 - 폴백 모드]\n"
        "아래는 입력 텍스트를 문장 단위로 나눈 임시 요약입니다. "
        "OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY 중 하나를 설정하면 실제 LLM이 이 부분을 대체합니다.\n"
        f"{bullet_lines}"
    )
