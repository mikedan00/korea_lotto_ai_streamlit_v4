# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from datetime import datetime

from src.config import AppConfig
from src.lotto_engine import LottoStatEngine, load_lotto_records, save_result_json
from src.llm_engine import LLMAnalyzer


def main():
    cfg = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Korea Lotto AI Predictor v4 CLI")
    parser.add_argument("--csv", default=cfg.lotto_csv_path, help="로또 CSV 경로")
    parser.add_argument("--min-train", type=int, default=cfg.min_train_draws)
    parser.add_argument("--backtest-draws", type=int, default=cfg.backtest_draws)
    parser.add_argument("--candidates", type=int, default=cfg.candidate_count)
    parser.add_argument("--seed", type=int, default=cfg.random_seed)
    parser.add_argument("--llm", action="store_true", help="HF LLM 분석까지 실행")
    parser.add_argument("--note", default="", help="LLM에게 전달할 추가 요청")
    args = parser.parse_args()

    records = load_lotto_records(args.csv)
    engine = LottoStatEngine(random_seed=args.seed)
    result = engine.predict(records, min_train=args.min_train, backtest_draws=args.backtest_draws, candidate_count=args.candidates)

    print("=" * 70)
    print(f"🎯 {result.next_draw}회차 추천 결과")
    print(f"📅 예상 추첨일: {result.draw_date}")
    print(f"⭐ 추천 1세트: {result.main}")
    print(f"🎁 보너스 후보: {result.bonus_candidate}")
    print("=" * 70)
    print("\n상위 후보 조합:")
    for i, combo in enumerate(result.candidate_sets, 1):
        print(f"{i:02d}. {combo}")

    print("\n전략별 가중치:")
    for name, weight in sorted(result.strategy_weights.items(), key=lambda kv: kv[1], reverse=True):
        print(f"- {name:12s}: {weight:.4f}")

    print("\n백테스트:")
    for name, metric in sorted(result.backtest.items(), key=lambda kv: kv[1].avg_score, reverse=True):
        print(
            f"- {name:12s}: avg={metric.avg_matches:.3f}, "
            f"3+={metric.hit3_rate*100:.2f}%, 4+={metric.hit4plus_rate*100:.2f}%, tests={metric.tests}"
        )

    os.makedirs(cfg.output_dir, exist_ok=True)
    out = os.path.join(cfg.output_dir, f"prediction_{result.next_draw}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_result_json(result, out)
    print(f"\n💾 저장 완료: {out}")

    if args.llm:
        analyzer = LLMAnalyzer(cfg)
        resp = analyzer.analyze(result, user_note=args.note)
        print("\n" + "=" * 70)
        print("🤖 LLM 분석")
        print("=" * 70)
        if resp.ok:
            print(f"model: {resp.model}\n")
            print(resp.content)
        else:
            print("LLM 실패:", resp.error)


if __name__ == "__main__":
    main()
