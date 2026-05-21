# -*- coding: utf-8 -*-
"""Configuration helpers for VS Code / Streamlit / Streamlit Cloud."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _get_streamlit_secret(name: str) -> Optional[str]:
    try:
        import streamlit as st  # type: ignore
        value = st.secrets.get(name)  # type: ignore[attr-defined]
        if value is not None:
            return str(value)
    except Exception:
        return None
    return None


def get_setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None and str(value).strip() != "":
        return str(value).strip()
    secret = _get_streamlit_secret(name)
    if secret is not None and secret.strip() != "":
        return secret.strip()
    return default


def get_float(name: str, default: float) -> float:
    try:
        return float(get_setting(name, str(default)))
    except Exception:
        return default


def get_int(name: str, default: int) -> int:
    try:
        return int(float(get_setting(name, str(default))))
    except Exception:
        return default


@dataclass
class AppConfig:
    app_name: str = "Korea Lotto AI Predictor v4"
    lotto_csv_path: str = "data/korealotto.csv"
    output_dir: str = "outputs"
    random_seed: int = 20260516
    min_train_draws: int = 200
    backtest_draws: int = 180
    candidate_count: int = 25000
    ui_candidate_count: int = 12000
    ticket_price: int = 1000

    llm_engine: str = "hf_api"
    hf_token: str = ""
    hf_router_model: str = "google/gemma-4-26B-A4B-it:deepinfra"
    hf_model_candidates: str = "google/gemma-4-26B-A4B-it:deepinfra,google/gemma-4-26B-A4B-it:novita,google/gemma-4-31B-it:deepinfra,google/gemma-4-31B-it:together,Qwen/Qwen3.5-9B:together,Qwen/Qwen2.5-7B-Instruct:together"
    hf_max_tokens: int = 1400
    hf_temperature: float = 0.2
    hf_timeout_connect: int = 10
    hf_timeout_read: int = 120
    hf_max_retries: int = 3

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            lotto_csv_path=get_setting("LOTTO_CSV_PATH", "data/korealotto.csv"),
            output_dir=get_setting("OUTPUT_DIR", "outputs"),
            random_seed=get_int("RANDOM_SEED", 20260516),
            min_train_draws=get_int("MIN_TRAIN_DRAWS", 200),
            backtest_draws=get_int("BACKTEST_DRAWS", 180),
            candidate_count=get_int("CANDIDATE_COUNT", 25000),
            ui_candidate_count=get_int("UI_CANDIDATE_COUNT", 12000),
            ticket_price=get_int("TICKET_PRICE", 1000),
            llm_engine=get_setting("LLM_ENGINE", "hf_api"),
            hf_token=get_setting("HF_TOKEN", ""),
            # HF_MODEL_ID도 허용합니다. 기존 배포본은 HF_ROUTER_MODEL만 읽었기 때문에
            # Secrets에 HF_MODEL_ID만 넣으면 사용자가 지정한 모델이 무시될 수 있었습니다.
            hf_router_model=get_setting(
                "HF_ROUTER_MODEL",
                get_setting("HF_MODEL_ID", "google/gemma-4-26B-A4B-it:deepinfra"),
            ),
            hf_model_candidates=get_setting(
                "HF_MODEL_CANDIDATES",
                "google/gemma-4-26B-A4B-it:deepinfra,google/gemma-4-26B-A4B-it:novita,google/gemma-4-31B-it:deepinfra,google/gemma-4-31B-it:together,Qwen/Qwen3.5-9B:together,Qwen/Qwen2.5-7B-Instruct:together",
            ),
            hf_max_tokens=get_int("HF_MAX_TOKENS", 1400),
            hf_temperature=get_float("HF_TEMPERATURE", 0.2),
            hf_timeout_connect=get_int("HF_TIMEOUT_CONNECT", 10),
            hf_timeout_read=get_int("HF_TIMEOUT_READ", 120),
            hf_max_retries=get_int("HF_MAX_RETRIES", 3),
        )

    def model_candidates(self) -> List[str]:
        """Return ordered HF Router model candidates.

        Supports both provider-suffixed names, e.g.
        `google/gemma-4-26B-A4B-it:deepinfra`, and bare model ids from
        HF_MODEL_ID, e.g. `google/gemma-4-26B-A4B-it`. For bare ids we add
        `:fastest` and common provider suffix variants after the original id.
        """
        ordered: List[str] = []

        def add(model: str) -> None:
            model = model.strip()
            if model and model not in ordered:
                ordered.append(model)

        raw_items = [self.hf_router_model] + self.hf_model_candidates.split(",")
        for item in raw_items:
            model = item.strip()
            if not model:
                continue
            add(model)
            # If the last path segment has no provider/policy suffix, add safe suffix variants.
            last_segment = model.rsplit("/", 1)[-1]
            if ":" not in last_segment:
                for suffix in [":fastest", ":deepinfra", ":novita", ":together", ":preferred"]:
                    add(model + suffix)
        return ordered
