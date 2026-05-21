# -*- coding: utf-8 -*-
"""Hugging Face Router LLM analysis layer."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

from .config import AppConfig
from .lotto_engine import PredictionResult, result_to_dict

HF_CHAT_COMPLETIONS_URL = "https://router.huggingface.co/v1/chat/completions"


@dataclass
class LLMResponse:
    ok: bool
    model: str
    content: str
    error: str = ""


class LLMAnalyzer:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig.from_env()

    def available(self) -> bool:
        return self.config.llm_engine.lower() == "hf_api" and bool(self.config.hf_token)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.hf_token}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def build_prompt(result: PredictionResult, user_note: str = "") -> str:
        data = result_to_dict(result)
        compact = {
            "next_draw": data["next_draw"],
            "draw_date": data["draw_date"],
            "recommended_main": data["main"],
            "bonus_candidate": data["bonus_candidate"],
            "candidate_sets_top10": data["candidate_sets"],
            "strategy_weights": data["strategy_weights"],
            "backtest": data["backtest"],
            "analysis": data["analysis"],
        }
        return f"""
너는 한국 로또 6/45 통계 분석 보조 AI다. 아래 백테스트 기반 예측 결과를 검토해서 사용자가 이해하기 쉽게 한국어로 분석하라.

중요 원칙:
- 로또는 독립 확률 게임이므로 당첨 보장, 확실한 예측, 고수익 보장을 말하지 마라.
- 과거 데이터 기반 통계적 후보라는 점을 명확히 하라.
- 번호를 바꿔야 한다면 근거를 제시하고, 최종 1세트는 반드시 1~45 중 중복 없는 6개 오름차순으로만 제시하라.
- 보너스는 참고 후보로만 제시하라.
- 백테스트에서 상대적으로 좋은 전략과 약한 전략을 짧게 해석하라.

사용자 추가 메모:
{user_note or "없음"}

예측 데이터 JSON:
{json.dumps(compact, ensure_ascii=False)}

출력 형식:
1. AI 최종 추천 1세트
2. 보너스 후보
3. 추천 근거
4. 백테스트/가중치 해석
5. 주의사항
""".strip()

    def analyze(self, result: PredictionResult, user_note: str = "") -> LLMResponse:
        if self.config.llm_engine.lower() != "hf_api":
            return LLMResponse(False, "", "", f"지원하지 않는 LLM_ENGINE: {self.config.llm_engine}")
        if not self.config.hf_token:
            return LLMResponse(False, "", "", "HF_TOKEN이 설정되어 있지 않습니다. .env 또는 Streamlit Secrets에 HF_TOKEN을 넣어주세요.")

        prompt = self.build_prompt(result, user_note=user_note)
        last_error = ""
        for model in self.config.model_candidates():
            for attempt in range(max(1, self.config.hf_max_retries)):
                try:
                    payload = {
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a careful statistical analysis assistant. Respond in Korean.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": self.config.hf_max_tokens,
                        "temperature": self.config.hf_temperature,
                    }
                    resp = requests.post(
                        HF_CHAT_COMPLETIONS_URL,
                        headers=self._headers(),
                        json=payload,
                        timeout=(self.config.hf_timeout_connect, self.config.hf_timeout_read),
                    )
                    if resp.status_code >= 400:
                        last_error = f"{model} HTTP {resp.status_code}: {resp.text[:500]}"
                        time.sleep(0.8 * (attempt + 1))
                        continue
                    data = resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not content:
                        content = json.dumps(data, ensure_ascii=False)[:2000]
                    return LLMResponse(True, model, content)
                except Exception as e:
                    last_error = f"{model}: {type(e).__name__}: {e}"
                    time.sleep(0.8 * (attempt + 1))
        return LLMResponse(False, "", "", last_error or "모든 HF 모델 후보 호출에 실패했습니다.")
