# -*- coding: utf-8 -*-
"""
오늘 진료 메모를 입력받아, 원장님이 아침에 바로 확인할 수 있는
'일일 진료 정리 브리핑'을 자동 생성해서 출력(및 파일로 저장)한다.

사용법:
    python vectorstore.py   # 최초 1회: 과거 이력을 벡터DB에 적재
    python main.py          # 오늘 메모 처리 -> 정리본 출력
"""
from __future__ import annotations

import os
from datetime import date

from data.sample_notes import TODAY_RAW_NOTES
from pipeline import process_today_notes
from llm import has_live_llm

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def render_report(summaries) -> str:
    lines = [f"# {date.today().isoformat()} 진료 정리 브리핑", ""]
    if not has_live_llm():
        lines.append(
            "> ⚠ 현재 LLM API 키가 설정되지 않아 폴백(규칙 기반) 모드로 생성된 결과입니다. "
            ".env에 OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY 중 하나를 넣으면 실제 LLM 결과로 바뀝니다.\n"
        )

    for s in summaries:
        lines.append(f"## {s.patient_name_masked} ({s.patient_id})")
        lines.append(f"**오늘 원본 메모**: {s.raw_note}")
        lines.append("")
        lines.append("**자동 정리된 진료기록**")
        lines.append(s.structured_record)
        lines.append("")
        lines.append(f"**과거 이력 대비 브리핑**: {s.briefing}")
        if s.past_history_found:
            lines.append("")
            lines.append("<details><summary>검색된 과거 이력 (근거)</summary>\n")
            for h in s.past_history_found:
                lines.append(f"- {h}")
            lines.append("</details>")
        lines.append("\n---\n")

    return "\n".join(lines)


def main():
    print("[*] 오늘 진료 메모를 처리하는 중...")
    summaries = process_today_notes(TODAY_RAW_NOTES)

    report = render_report(summaries)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{date.today().isoformat()}_briefing.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\n[*] 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
