# app.py — 아랍 물통 문제 (최종·DAT1·pr 스키마 / 풀링 안정화 버전)
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json
from typing import Dict, Any, List

import streamlit as st
import mysql.connector
from mysql.connector import Error as MySQLError
from mysql.connector import pooling
from openai import OpenAI

# ─────────────────────────────────────────────────────
# 페이지/상수
# ─────────────────────────────────────────────────────
st.set_page_config(page_title="서술형 평가 — 상태 변화와 열에너지", page_icon="🧪", layout="wide")
ID_REGEX = r"^\d{5}$"
IMAGE_FILENAME = "image1.png"

QUESTION_TEXT = (
    "다음은 물의 특성을 이용한 아랍인들의 생활 속 지혜와 관련된 사례를 나타낸 것이다.\n\n"
    "아랍인들은 양가죽으로 만든 물통을 가지고 다녔는데 이 물통은 양가죽을 통해 물이 조금씩 새어 나와 항상 젖어 있었다."
    " 하지만 양가죽 물통 속의 물은 의외로 시원하여 무더운 사막에서도 시원한 물을 마실 수 있었다.\n\n"
    "무더운 사막에서도 아랍인들이 시원한 물을 마실 수 있었던 이유를 <조건>에 맞게 서술하시오.[7점]\n\n"
    "<조건>\n"
    "◦ 새어 나온 물의 상태 변화를 포함하여 서술해야 함.\n"
    "◦ 에너지 출입을 포함하여 서술해야 함.\n"
)

SCORING_RULES = {
    "max_score": 7,
    "must_include": {
        "evaporation": ["증발", "기화"],
        "heat_absorb": ["열을 흡수", "열을 빼앗", "주변의 열", "열 에너지 흡수"],
    },
    "partial_score": 2,
}

EXAMPLES_KR = (
    "- 양가죽 물통에서 새어 나온 물이 증발되면서 주변의 열을 흡수하여 물통 속의 물이 시원해진다.\n"
    "- 새어 나온 물이 증발하면서 물통 속의 물의 열을 빼앗아가기 때문이다.\n"
    "- 물이 기화하면서 주변의 열을 흡수하여 물통 속의 물이 시원해진다.\n"
    "[유의] '물이 증발/기화하면서 주변의 열을 흡수한다'를 포함하면 정답으로 본다."
)

# ─────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────
def validate_student_id(s: str) -> bool:
    import re
    return bool(s and re.match(ID_REGEX, s))

def get_model_name() -> str:
    return st.secrets.get("OPENAI_MODEL", "gpt-5")

# ─────────────────────────────────────────────────────
# DB (mysql-connector) — 커넥션 풀 사용
# ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_mysql_pool():
    cfg = st.secrets.get("connections", {}).get("mysql", {})
    # 배포 환경 안정화를 위해 풀링 사용
    return pooling.MySQLConnectionPool(
        pool_name="app_pool",
        pool_size=5,
        host=cfg.get("host"),
        port=cfg.get("port", 3306),
        database=cfg.get("database"),
        user=cfg.get("user"),
        password=cfg.get("password"),
        autocommit=True,
    )

def get_conn():
    # 매 요청마다 풀에서 연결을 '빌려오고' 작업 후 닫아서 반납
    return get_mysql_pool().get_connection()

def init_tables() -> None:
    """DAT1 테이블 보장 (pr 스키마). time=TIMESTAMP 자동기록."""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS DAT1 (
              id        VARCHAR(16) NOT NULL,
              answer1   MEDIUMTEXT,
              feedback1 MEDIUMTEXT,
              opinion1  MEDIUMTEXT,
              time      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY (id)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """
        )
        cur.close()
        conn.close()  # 풀 반납
    except MySQLError as e:
        st.error(f"[DB] 테이블 초기화 실패: {e}")

def upsert_dat1(student_id: str, answer1: str, feedback1: str, opinion1: str | None) -> None:
    """동일 학번 UPSERT. opinion1=None이면 기존 의견을 보존합니다."""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO DAT1 (id, answer1, feedback1, opinion1)
            VALUES (%s, %s, %s, %s) AS new
            ON DUPLICATE KEY UPDATE
              answer1 = new.answer1,
              feedback1 = new.feedback1,
              opinion1 = COALESCE(new.opinion1, DAT1.opinion1)
            """,
            (student_id, answer1, feedback1, opinion1),
        )
        cur.close()
        conn.close()  # 풀 반납
    except MySQLError as e:
        st.error(f"[DB] 저장 실패: {e}")
        raise

def update_opinion_only(student_id: str, opinion1: str) -> None:
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("UPDATE DAT1 SET opinion1=%s WHERE id=%s", (opinion1, student_id))
        cur.close()
        conn.close()  # 풀 반납
    except MySQLError as e:
        st.error(f"[DB] 의견 저장 실패: {e}")
        raise

# ─────────────────────────────────────────────────────
# OpenAI 채점 (gpt-5 호환 처리: temperature 미전달, max_tokens 재시도)
# ─────────────────────────────────────────────────────
def build_messages(answer_kr: str) -> List[Dict[str, str]]:
    rules = json.dumps(SCORING_RULES, ensure_ascii=False)
    system = "당신은 한국 중학교 과학 보조채점자입니다. 아래 규칙을 엄격히 적용하고, 반드시 JSON만 출력하세요."
    user = f"""
[문항]
{QUESTION_TEXT}

[학생 답안]
{answer_kr}

[채점 규칙(JSON)]
{rules}

[예시 답안/유의]
{EXAMPLES_KR}

[출력 형식(JSON only)]
{{
  "score": number,
  "reason": "채점 근거(간단)",
  "feedback": "친근한 한국어 3~4문장",
  "detected": {{
      "evaporation": true/false,
      "heat_absorb": true/false
  }}
}}
- 규칙을 반드시 따르세요. 임의 가중치/총점 변경 금지.
- 유효한 JSON만 반환(문장/코드펜스 금지).
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def grade_with_openai(student_answer: str) -> Dict[str, Any]:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .streamlit/secrets.toml 확인")
    client = OpenAI(api_key=api_key)
    model = get_model_name()
    messages = build_messages(student_answer)

    # gpt-5 계열 호환: temperature 미전달, max_tokens 미지원 시 제거
    base_kwargs = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    kwargs = dict(base_kwargs)
    kwargs["max_tokens"] = 400

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        if "max_tokens" in msg or "unsupported" in msg:
            kwargs = dict(base_kwargs)
            resp = client.chat.completions.create(**kwargs)
        elif "temperature" in msg:
            kwargs = dict(base_kwargs)
            resp = client.chat.completions.create(**kwargs)
        else:
            raise

    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(content[start:end+1])
        else:
            raise RuntimeError("모델 응답 JSON 파싱 실패")

    # 규칙 재적용(안전 가드)
    det = data.get("detected", {})
    e_ok, h_ok = bool(det.get("evaporation")), bool(det.get("heat_absorb"))
    desired = 7 if (e_ok and h_ok) else (2 if (e_ok ^ h_ok) else 0)
    data["score"] = desired
    data.setdefault("reason", ""); data.setdefault("feedback", "")
    data["detected"] = {"evaporation": e_ok, "heat_absorb": h_ok}
    return data

# ─────────────────────────────────────────────────────
# 메인 UI
# ─────────────────────────────────────────────────────
def main():
    st.title("🧪 서술형 평가 — 상태 변화와 열에너지(아랍 물통)")
    st.caption("GPT 자동 채점·피드백 → ‘한 가지 의견’까지 제출해 주세요.")

    # 테이블 보장
    init_tables()

    # 문항/이미지 가로 배치
    col_q, col_img = st.columns([2, 1])
    with col_q:
        st.subheader("문항")
        st.write(QUESTION_TEXT)
    with col_img:
        img_path = os.path.join("image", IMAGE_FILENAME)
        if os.path.exists(img_path):
            # use_container_width 폐기 예정 → width 파라미터로 교체
            st.image(img_path, caption="문항 참고 이미지", width='stretch)
        else:
            st.info(f"이미지 파일을 찾을 수 없습니다: {img_path}")

    # 학생 입력 폼
    with st.form("student_form", clear_on_submit=False):
        sid = st.text_input("학번(5자리, 예: 10130)", placeholder="10130")
        answer = st.text_area("나의 답안", height=180, placeholder="예) 답안을 입력하세요")
        submitted = st.form_submit_button("채점 받기", type="primary")

    if submitted:
        if not validate_student_id(sid):
            st.error("학번 형식이 올바르지 않습니다. (예: 10130)")
            return
        if not answer.strip():
            st.error("답안을 입력해 주세요.")
            return

        try:
            with st.spinner("채점 중입니다…"):
                result = grade_with_openai(answer.strip())
        except Exception as e:
            st.error(f"OpenAI 호출 실패: {e}")
            return

        st.success(f"점수: **{result.get('score', 0)} / {SCORING_RULES['max_score']}**")
        if result.get("reason"):
            st.write(":memo: **채점 근거**")
            st.write(result["reason"])
        st.write(":bulb: **피드백**")
        st.write(result.get("feedback", ""))

        det = result.get("detected", {})
        with st.expander("조건 충족 여부"):
            st.write(f"증발/기화 언급: {'✅' if det.get('evaporation') else '❌'}")
            st.write(f"열 흡수 언급: {'✅' if det.get('heat_absorb') else '❌'}")

        payload = {
            "score": result.get("score", 0),
            "max": SCORING_RULES["max_score"],
            "reason": result.get("reason", ""),
            "feedback": result.get("feedback", ""),
            "detected": det,
        }
        try:
            upsert_dat1(
                student_id=sid.strip(),
                answer1=answer.strip(),
                feedback1=json.dumps(payload, ensure_ascii=False),
                opinion1=None,  # 초기 제출 시 의견 없음(보존 로직 적용)
            )
            st.success("채점/피드백이 저장되었습니다. 아래에 ‘한 가지 의견’을 작성해 주세요.")
            st.session_state["last_id"] = sid.strip()
        except MySQLError:
            return

    # 의견 제출
    last_id = st.session_state.get("last_id")

    st.divider()
    st.subheader("🗣️ 한 가지 의견 제출")
    st.caption("피드백을 읽고, 무엇을 알게 되었는지/여전히 어려운 점은 무엇인지 3~5문장으로 작성하세요.")

    op = st.text_area("나의 의견", key="opinion_text", height=120)

    # 세션이 초기화되었을 때 대비: 학번 재확인 입력
    if not last_id:
        sid_fallback = st.text_input("학번(세션이 초기화된 경우 다시 입력)", key="sid_fallback", placeholder="10130")
    else:
        sid_fallback = last_id

    if st.button("의견 제출"):
        if not op.strip():
            st.warning("의견을 입력해 주세요.")
        elif not validate_student_id(sid_fallback):
            st.error("학번 형식이 올바르지 않습니다. (예: 10130)")
        else:
            try:
                update_opinion_only(sid_fallback.strip(), op.strip())
                st.success("의견이 저장되었습니다. 수고했어요! ✨")
                st.session_state.pop("last_id", None)
                st.session_state.pop("opinion_text", None)
            except MySQLError:
                pass

if __name__ == "__main__":
    main()

