# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from datetime import date

import pandas as pd
import streamlit as st

from src.config import AppConfig
from src.lotto_engine import LottoStatEngine, load_lotto_records, result_to_dict, save_result_json
from src.llm_engine import LLMAnalyzer

st.set_page_config(
    page_title="Korea Lotto AI Predictor v4.1",
    page_icon="🎰",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _load_records_from_bytes(raw: bytes):
    return load_lotto_records(raw)


@st.cache_data(show_spinner=False)
def _load_records_from_path(path: str):
    return load_lotto_records(path)


@st.cache_data(show_spinner=False)
def _run_prediction_cached(raw_or_path_key: str, raw: bytes | None, path: str | None, min_train: int, backtest_draws: int, candidate_count: int, seed: int):
    if raw is not None:
        records = _load_records_from_bytes(raw)
    else:
        records = _load_records_from_path(path or "data/korealotto.csv")
    engine = LottoStatEngine(random_seed=seed)
    result = engine.predict(records, min_train=min_train, backtest_draws=backtest_draws, candidate_count=candidate_count, today=date.today())
    return records, result


def _metric_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _render_result(result):
    st.subheader(f"🎯 {result.next_draw}회차 추천 결과")
    c1, c2, c3 = st.columns(3)
    c1.metric("예상 추첨일", result.draw_date)
    c2.metric("최종 추천 1세트", " ".join(map(str, result.main)))
    c3.metric("보너스 후보", str(result.bonus_candidate))

    st.markdown("### 🧩 상위 후보 조합")
    combo_df = pd.DataFrame({"순위": range(1, len(result.candidate_sets) + 1), "후보 조합": [" ".join(map(str, x)) for x in result.candidate_sets]})
    st.dataframe(combo_df, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### ⚖️ 전략별 최적 가중치")
        wdf = pd.DataFrame([{"전략": k, "가중치": v} for k, v in result.strategy_weights.items()]).sort_values("가중치", ascending=False)
        st.dataframe(wdf, use_container_width=True, hide_index=True)
        st.bar_chart(wdf.set_index("전략")["가중치"])

    with col_b:
        st.markdown("### 🔬 롤링 백테스트")
        bdf = pd.DataFrame([
            {
                "전략": k,
                "평균매칭": v.avg_matches,
                "3개이상": v.hit3_rate,
                "4개이상": v.hit4plus_rate,
                "종합점수": v.avg_score,
                "테스트": v.tests,
            }
            for k, v in result.backtest.items()
        ]).sort_values("종합점수", ascending=False)
        st.dataframe(
            bdf.assign(**{"3개이상": bdf["3개이상"].map(_metric_pct), "4개이상": bdf["4개이상"].map(_metric_pct)}),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### 🔢 번호별 종합 점수 TOP 15")
    top_nums = sorted(result.number_scores.items(), key=lambda kv: kv[1], reverse=True)[:15]
    ndf = pd.DataFrame([{"번호": k, "점수": v} for k, v in top_nums])
    st.dataframe(ndf, use_container_width=True, hide_index=True)
    st.bar_chart(ndf.set_index("번호")["점수"])


def main():
    cfg = AppConfig.from_env()

    st.title("🎰 한국 로또 AI 예측 시스템 v4.1")
    st.caption("VS Code 로컬 실행 + Streamlit Cloud 배포 + Hugging Face Router LLM 분석 지원")

    with st.sidebar:
        st.header("설정")
        uploaded = st.file_uploader("로또 CSV 업로드", type=["csv"])
        csv_path = st.text_input("또는 CSV 경로", value=cfg.lotto_csv_path)
        st.divider()
        min_train = st.number_input("최소 학습 회차", min_value=30, max_value=1000, value=cfg.min_train_draws, step=10)
        backtest_draws = st.number_input("백테스트 최근 회차 수", min_value=20, max_value=600, value=cfg.backtest_draws, step=10)
        candidate_count = st.number_input("후보 조합 샘플 수", min_value=1000, max_value=80000, value=cfg.ui_candidate_count, step=1000)
        seed = st.number_input("랜덤 시드", min_value=1, max_value=99999999, value=cfg.random_seed, step=1)
        st.divider()
        st.subheader("LLM 설정")
        st.text_input("LLM_ENGINE", value=cfg.llm_engine, disabled=True)
        token_state = "설정됨 ✅" if cfg.hf_token else "없음 ❌"
        st.write(f"HF_TOKEN: {token_state}")
        st.caption("로컬은 .env, Streamlit Cloud는 Secrets에 HF_TOKEN 등을 넣으세요.")

    raw = uploaded.getvalue() if uploaded is not None else None
    path = None if raw is not None else csv_path

    run = st.button("🚀 백테스트 + 예측 실행", type="primary", use_container_width=True)

    # Streamlit 버튼은 클릭할 때마다 전체 스크립트를 재실행합니다.
    # 기존 v4는 LLM 버튼을 누르는 순간 첫 번째 실행 버튼 값이 False가 되어
    # 결과 화면까지 도달하지 못했습니다. 결과를 session_state에 저장해 이 문제를 해결합니다.
    if run:
        try:
            with st.spinner("CSV 로딩 및 롤링 백테스트 실행 중..."):
                key = f"raw:{len(raw)}" if raw is not None else f"path:{path}"
                records, result = _run_prediction_cached(
                    key, raw, path, int(min_train), int(backtest_draws), int(candidate_count), int(seed)
                )
            st.session_state["last_records"] = records
            st.session_state["last_result"] = result
            st.session_state["last_source_label"] = f"업로드 CSV {len(raw)} bytes" if raw is not None else str(path)
            st.success(f"데이터 로드 완료: {len(records)}회차 / 마지막 회차: {records[-1].draw}")
        except Exception as e:
            st.error(f"실행 오류: {e}")
            st.exception(e)
            return

    if "last_result" not in st.session_state:
        st.info("왼쪽에서 CSV를 업로드하거나 경로를 지정한 뒤 실행하세요.")
        st.markdown(
            """
#### CSV 컬럼 예시
- `회차`, `추첨일`, `당첨번호1`, `당첨번호2`, `당첨번호3`, `당첨번호4`, `당첨번호5`, `당첨번호6`, `보너스`
- 또는 `draw`, `date`, `num1`~`num6`, `bonus`
"""
        )
        return

    records = st.session_state.get("last_records", [])
    result = st.session_state["last_result"]
    if records:
        st.caption(f"현재 표시 중인 결과: {st.session_state.get('last_source_label', '')} / {len(records)}회차")
    _render_result(result)

    st.markdown("### 🤖 LLM AI 추가 분석")
    analyzer = LLMAnalyzer(cfg)
    with st.expander("LLM 연결 진단", expanded=False):
        st.write(f"LLM_ENGINE: `{cfg.llm_engine}`")
        st.write(f"HF_TOKEN: {'설정됨 ✅' if cfg.hf_token else '없음 ❌'}")
        st.write("모델 후보:")
        st.code("\n".join(cfg.model_candidates()[:12]))
        if st.button("🔌 HF 연결 테스트", use_container_width=True):
            if not analyzer.available():
                st.error("HF_TOKEN이 없거나 LLM_ENGINE이 hf_api가 아닙니다.")
            else:
                with st.spinner("HF Router 연결 테스트 중..."):
                    test_resp = analyzer.test_connection()
                if test_resp.ok:
                    st.success(f"연결 성공: {test_resp.model}")
                    st.write(test_resp.content)
                else:
                    st.error("연결 실패")
                    st.code(test_resp.error)

    user_note = st.text_area(
        "AI에게 추가로 요청할 분석 조건",
        placeholder="예: 너무 낮은 번호에 몰리지 않게 해줘 / 최근 30회 흐름을 더 중시해줘",
    )
    if st.button("🤖 HF LLM으로 추가 분석", use_container_width=True):
        if not analyzer.available():
            st.error("HF_TOKEN이 없거나 LLM_ENGINE이 hf_api가 아닙니다. .env 또는 Streamlit Secrets를 확인하세요.")
        else:
            with st.spinner("Hugging Face Router API 호출 중..."):
                llm_resp = analyzer.analyze(result, user_note=user_note)
            if llm_resp.ok:
                st.success(f"LLM 분석 완료: {llm_resp.model}")
                st.markdown(llm_resp.content)
            else:
                st.error("LLM 호출 실패")
                st.code(llm_resp.error)

    st.markdown("### 💾 결과 다운로드")
    result_json = result_to_dict(result)
    st.download_button(
        "prediction_result.json 다운로드",
        data=pd.Series(result_json).to_json(force_ascii=False, indent=2),
        file_name="prediction_result.json",
        mime="application/json",
    )

    st.divider()
    st.caption("주의: 로또는 무작위 확률 게임입니다. 본 앱은 과거 데이터 기반 후보 생성 도구이며 당첨 또는 수익을 보장하지 않습니다.")


if __name__ == "__main__":
    main()
