# -*- coding: utf-8 -*-
"""
핵심 파이프라인.

문제 상황: 원장님이 하루 진료를 마친 뒤, 진료 중 급하게 적어둔 메모를
퇴근 후 다시 정리된 진료기록 형태로 옮겨 적는 데 시간을 쓰고 있다.

해결: 진료 중 남긴 '날것' 메모를 입력으로 받아
  1) 구조화(structuring) - 환자정보/증상/처방/특이사항/다음방문권장일 형태로 자동 정리
  2) 이력 검색(retrieval) - 같은 환자의 과거 방문 기록을 벡터DB에서 검색
  3) 비교 브리핑 생성(generation) - 이번 방문과 과거 이력을 비교한 한 줄 브리핑 생성
을 자동으로 수행해서, 원장님이 아침에 정리된 형태로 바로 확인만 하면 되게 만든다.

이 3단계가 합쳐지면 Retrieval-Augmented Generation(RAG) 구조가 된다:
과거 이력(지식 베이스)을 검색해서, 그 결과를 LLM 생성 단계에 근거로 활용하기 때문이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from vectorstore import load_vectorstore, search_patient_history
from llm import generate

STRUCTURE_SYSTEM_PROMPT = (
    "너는 한의원 원장의 진료 메모를 정리해주는 어시스턴트야. "
    "아래 형식의 한국어 진료기록 카드로 정리해줘:\n"
    "- 주요 증상:\n- 이번 치료/처방:\n- 특이사항:\n- 다음 방문 권장:\n"
    "메모에 없는 정보는 '언급 없음'이라고 적어. "
    "메모에 적힌 내용만 옮기고, 원인·진단·증상을 새로 추론해서 덧붙이지 마."
)

# 진료기록 요약이라 '그럴듯한 추론'이 사실처럼 섞이는 게 가장 위험하다.
# (실제로 "추나 시행 후 통증 완화"처럼 순서를 인과로 오해하거나, 메모에 없는
#  '상급 병원', '만성화' 같은 표현을 덧붙이는 오류가 반복 관찰돼서 규칙을 명시했다.)
BRIEFING_SYSTEM_PROMPT = (
    "너는 한의원 원장을 위한 브리핑 어시스턴트야. "
    "오늘 진료 메모와 같은 환자의 과거 방문 이력을 비교해서, "
    "원장이 다음 진료 때 참고하면 좋을 점을 2~3문장으로 간단히 요약해줘.\n"
    "반드시 지킬 규칙:\n"
    "1. [오늘 메모]와 [과거 방문 이력]에 명시된 사실만 써. 적혀 있지 않은 검사·병원·진단명·"
    "수치·경과(예: 재발, 만성화)를 덧붙이지 마.\n"
    "2. 메모에 적힌 순서를 인과관계로 해석하지 마. 치료와 증상 변화의 인과는 메모에 "
    "명시된 경우에만 써. 오늘 처음 시행한 치료의 효과는 아직 알 수 없다.\n"
    "3. 과거 이력을 언급할 때는 해당 방문 날짜를 (YYYY-MM-DD) 형식으로 밝혀.\n"
    "4. 기록에서 직접 확인되지 않는 해석이나 추정을 쓸 때는 문장 끝에 '(추정)'이라고 표시해.\n"
    "과거 이력이 없으면 '과거 방문 이력 없음(초진)'이라고만 답해."
)


@dataclass
class VisitSummary:
    patient_id: str
    patient_name_masked: str
    raw_note: str
    structured_record: str
    past_history_found: list = field(default_factory=list)
    briefing: str = ""


def structure_note(raw_note: str) -> str:
    return generate(STRUCTURE_SYSTEM_PROMPT, raw_note)


def build_briefing(patient_name: str, raw_note: str, past_notes: list[str]) -> str:
    if not past_notes:
        return "과거 방문 이력 없음(초진)."
    history_text = "\n".join(f"- {n}" for n in past_notes)
    user_prompt = (
        f"[오늘 메모]\n{raw_note}\n\n[{patient_name}님 과거 방문 이력]\n{history_text}"
    )
    return generate(BRIEFING_SYSTEM_PROMPT, user_prompt)


def process_today_notes(today_notes: list[dict]) -> list[VisitSummary]:
    vectordb = load_vectorstore()
    summaries = []

    for note in today_notes:
        # 1) 구조화
        structured = structure_note(note["raw_note"])

        # 2) 같은 환자의 과거 이력 검색 (Retrieval)
        past_docs = search_patient_history(
            vectordb, patient_id=note["patient_id"], query_text=note["raw_note"]
        )
        past_notes_text = [doc.page_content for doc in past_docs]

        # 3) 비교 브리핑 생성 (Generation, grounded on retrieved history)
        briefing = build_briefing(note["patient_name_masked"], note["raw_note"], past_notes_text)

        summaries.append(
            VisitSummary(
                patient_id=note["patient_id"],
                patient_name_masked=note["patient_name_masked"],
                raw_note=note["raw_note"],
                structured_record=structured,
                past_history_found=past_notes_text,
                briefing=briefing,
            )
        )

    return summaries
