# -*- coding: utf-8 -*-
"""Backtest-first Korean Lotto 6/45 statistical engine."""
from __future__ import annotations

import io
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

MAIN_COL_CANDIDATES = [
    ["당첨번호1", "당첨번호2", "당첨번호3", "당첨번호4", "당첨번호5", "당첨번호6"],
    ["번호1", "번호2", "번호3", "번호4", "번호5", "번호6"],
    ["num1", "num2", "num3", "num4", "num5", "num6"],
    ["n1", "n2", "n3", "n4", "n5", "n6"],
]
DRAW_COL_CANDIDATES = ["회차", "draw", "Draw", "round", "Round", "drwNo", "추첨회차"]
BONUS_COL_CANDIDATES = ["보너스", "보너스번호", "bonus", "Bonus", "bnusNo"]
DATE_COL_CANDIDATES = ["추첨일", "날짜", "date", "Date", "draw_date", "drwNoDate"]


@dataclass(frozen=True)
class LottoRecord:
    draw: int
    main: Tuple[int, int, int, int, int, int]
    bonus: int
    draw_date: Optional[str] = None


@dataclass
class StrategyMetrics:
    avg_matches: float
    hit3_rate: float
    hit4plus_rate: float
    avg_score: float
    tests: int


@dataclass
class PredictionResult:
    next_draw: int
    draw_date: str
    main: List[int]
    bonus_candidate: int
    candidate_sets: List[List[int]]
    number_scores: Dict[int, float]
    strategy_weights: Dict[str, float]
    backtest: Dict[str, StrategyMetrics]
    analysis: Dict[str, object]


def next_lotto_saturday(today: Optional[date] = None, include_today: bool = True) -> date:
    """Return Korean Lotto Saturday. If today is Saturday, include_today=True returns today."""
    today = today or date.today()
    saturday = 5  # Monday=0, Saturday=5
    days = (saturday - today.weekday() + 7) % 7
    if days == 0 and not include_today:
        days = 7
    return today + timedelta(days=days)


def _read_csv_any(source: Union[str, bytes, io.BytesIO]) -> pd.DataFrame:
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
        for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8", "latin1"]:
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc)
            except Exception:
                continue
        raise ValueError("CSV 인코딩을 판별하지 못했습니다.")

    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        return _read_csv_any(raw)

    if not os.path.exists(str(source)):
        raise FileNotFoundError(f"CSV 파일을 찾지 못했습니다: {source}")

    last_err = None
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8", "latin1"]:
        try:
            return pd.read_csv(str(source), encoding=enc)
        except Exception as e:
            last_err = e
    raise ValueError(f"CSV 로드 실패: {last_err}")


def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    normalized = {str(c).strip(): c for c in df.columns}
    for cand in candidates:
        if cand in normalized:
            return normalized[cand]
    lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _find_main_cols(df: pd.DataFrame) -> List[str]:
    for cols in MAIN_COL_CANDIDATES:
        found = [_find_col(df, [c]) for c in cols]
        if all(found):
            return [str(c) for c in found if c is not None]

    # Fallback: choose 6 numeric columns after draw/date columns and before bonus if possible.
    numeric_cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(5, int(len(df) * 0.6)):
            numeric_cols.append(str(c))
    draw_col = _find_col(df, DRAW_COL_CANDIDATES)
    bonus_col = _find_col(df, BONUS_COL_CANDIDATES)
    filtered = [c for c in numeric_cols if c not in {draw_col, bonus_col}]
    if len(filtered) >= 6:
        # Many Korean Lotto CSV files use: 회차, 추첨일, n1..n6, bonus
        return filtered[:6]
    raise ValueError("메인 번호 6개 컬럼을 자동 감지하지 못했습니다. 컬럼명을 당첨번호1~당첨번호6 형식으로 맞춰주세요.")


def load_lotto_records(source: Union[str, bytes, io.BytesIO]) -> List[LottoRecord]:
    df = _read_csv_any(source)
    df.columns = [str(c).strip() for c in df.columns]

    draw_col = _find_col(df, DRAW_COL_CANDIDATES)
    bonus_col = _find_col(df, BONUS_COL_CANDIDATES)
    date_col = _find_col(df, DATE_COL_CANDIDATES)
    main_cols = _find_main_cols(df)

    if draw_col is None:
        # fallback first numeric column
        draw_col = str(df.columns[0])
    if bonus_col is None:
        # fallback: the numeric col right after six main cols, else last numeric col
        numeric_cols = [str(c) for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 5]
        candidates = [c for c in numeric_cols if c not in set(main_cols + [draw_col])]
        bonus_col = candidates[-1] if candidates else str(df.columns[-1])

    records: List[LottoRecord] = []
    for _, row in df.iterrows():
        try:
            draw = int(row[draw_col])
            main = tuple(sorted(int(row[c]) for c in main_cols))
            bonus = int(row[bonus_col])
            if len(main) != 6 or len(set(main)) != 6:
                continue
            if not all(1 <= n <= 45 for n in main):
                continue
            if not (1 <= bonus <= 45) or bonus in main:
                continue
            d = str(row[date_col]) if date_col and not pd.isna(row[date_col]) else None
            records.append(LottoRecord(draw=draw, main=main, bonus=bonus, draw_date=d))
        except Exception:
            continue

    records = sorted(set(records), key=lambda x: x.draw)
    if len(records) < 30:
        raise ValueError(f"유효한 회차 데이터가 너무 적습니다: {len(records)}개")
    return records


def normalize(values: Sequence[float], default: float = 0.5) -> List[float]:
    if not values:
        return []
    mn, mx = min(values), max(values)
    if abs(mx - mn) < 1e-12:
        return [default] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def softmax(values: Sequence[float], temperature: float = 0.65) -> List[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp((v - m) / max(temperature, 1e-9)) for v in values]
    s = sum(exps)
    return [e / s for e in exps]


def weighted_sample_without_replacement(items: Sequence[int], weights: Sequence[float], k: int, rng: random.Random) -> List[int]:
    pool = list(items)
    w = [max(float(x), 1e-12) for x in weights]
    chosen: List[int] = []
    for _ in range(min(k, len(pool))):
        total = sum(w)
        r = rng.random() * total
        acc = 0.0
        idx = len(pool) - 1
        for i, wi in enumerate(w):
            acc += wi
            if acc >= r:
                idx = i
                break
        chosen.append(pool.pop(idx))
        w.pop(idx)
    return sorted(chosen)


def count_consecutive(nums: Sequence[int]) -> int:
    s = sorted(nums)
    return sum(1 for a, b in zip(s, s[1:]) if b == a + 1)


def combination_features(nums: Sequence[int]) -> Dict[str, int]:
    nums = sorted(nums)
    return {
        "sum": sum(nums),
        "odd": sum(n % 2 for n in nums),
        "high": sum(n >= 23 for n in nums),
        "consecutive": count_consecutive(nums),
        "span": max(nums) - min(nums),
    }


def historical_profile(history: Sequence[LottoRecord], window: int = 180) -> Dict[str, float]:
    recent = list(history[-window:]) if len(history) > window else list(history)
    sums = [sum(r.main) for r in recent]
    odds = [sum(n % 2 for n in r.main) for r in recent]
    highs = [sum(n >= 23 for n in r.main) for r in recent]
    cons = [count_consecutive(r.main) for r in recent]
    spans = [max(r.main) - min(r.main) for r in recent]
    return {
        "sum_mean": statistics.mean(sums),
        "sum_std": statistics.pstdev(sums) or 1.0,
        "odd_mean": statistics.mean(odds),
        "high_mean": statistics.mean(highs),
        "consecutive_mean": statistics.mean(cons),
        "span_mean": statistics.mean(spans),
    }


class LottoStatEngine:
    strategy_names = ["frequency", "gap", "momentum", "hot_cold", "pair", "distribution", "anti_crowd"]

    def __init__(self, random_seed: int = 20260516):
        self.random_seed = random_seed

    def _last_seen_gaps(self, history: Sequence[LottoRecord]) -> Dict[int, int]:
        gaps = {n: len(history) + 1 for n in range(1, 46)}
        for idx, rec in enumerate(reversed(history), start=1):
            for n in rec.main:
                if gaps[n] == len(history) + 1:
                    gaps[n] = idx
        return gaps

    def _pair_counts(self, history: Sequence[LottoRecord], window: int = 180) -> Dict[Tuple[int, int], float]:
        recent = list(history[-window:]) if len(history) > window else list(history)
        counts: Dict[Tuple[int, int], float] = defaultdict(float)
        for age, rec in enumerate(reversed(recent)):
            weight = 0.985 ** age
            for a, b in combinations(rec.main, 2):
                counts[(a, b)] += weight
        return counts

    def score_numbers(self, history: Sequence[LottoRecord]) -> Dict[str, Dict[int, float]]:
        if not history:
            return {name: {n: 1 / 45 for n in range(1, 46)} for name in self.strategy_names}

        full = list(history)
        recent_30 = full[-30:]
        recent_80 = full[-80:] if len(full) >= 80 else full
        recent_180 = full[-180:] if len(full) >= 180 else full
        scores: Dict[str, Dict[int, float]] = {name: {n: 0.0 for n in range(1, 46)} for name in self.strategy_names}

        # 1) exponentially weighted frequency: higher is more recently frequent.
        for age, rec in enumerate(reversed(recent_180)):
            w = 0.985 ** age
            for n in rec.main:
                scores["frequency"][n] += w

        # 2) gap: numbers absent longer than their recent average get moderate score.
        gaps = self._last_seen_gaps(full)
        past_positions = {n: [] for n in range(1, 46)}
        for i, rec in enumerate(full):
            for n in rec.main:
                past_positions[n].append(i)
        for n in range(1, 46):
            positions = past_positions[n]
            if len(positions) >= 2:
                intervals = [b - a for a, b in zip(positions, positions[1:])]
                avg_gap = statistics.mean(intervals)
            else:
                avg_gap = 45 / 6
            scores["gap"][n] = min(gaps[n] / max(avg_gap, 1), 3.0)

        # 3) momentum: short frequency vs medium frequency.
        c30 = Counter(n for r in recent_30 for n in r.main)
        c80 = Counter(n for r in recent_80 for n in r.main)
        for n in range(1, 46):
            short = c30[n] / max(len(recent_30), 1)
            mid = c80[n] / max(len(recent_80), 1)
            scores["momentum"][n] = short - mid

        # 4) hot_cold blend: hot recent plus overdue cold.
        for n in range(1, 46):
            scores["hot_cold"][n] = 0.65 * c30[n] + 0.35 * scores["gap"][n]

        # 5) pair score: pair-co-occurrence with top frequency/gap candidates.
        pair_counts = self._pair_counts(full)
        anchor = sorted(range(1, 46), key=lambda n: scores["frequency"][n] + scores["gap"][n], reverse=True)[:16]
        for n in range(1, 46):
            total = 0.0
            for a in anchor:
                if a == n:
                    continue
                total += pair_counts.get(tuple(sorted((a, n))), 0.0)
            scores["pair"][n] = total

        # 6) distribution: middle-range numbers are mildly favored to avoid extreme combinations.
        for n in range(1, 46):
            # bell-shaped weak prior centered around lotto mid point
            scores["distribution"][n] = 1.0 - abs(n - 23) / 23

        # 7) anti_crowd: avoid too common public picks mildly: dates 1-31 and famous Fibonacci-ish numbers.
        public_bias = set(range(1, 32)) | {1, 2, 3, 5, 8, 13, 21, 34}
        for n in range(1, 46):
            scores["anti_crowd"][n] = 1.0 if n not in public_bias else 0.35

        # Normalize each strategy to 0..1.
        for name in self.strategy_names:
            vals = [scores[name][n] for n in range(1, 46)]
            normed = normalize(vals)
            scores[name] = {n: normed[n - 1] for n in range(1, 46)}
        return scores

    def weighted_number_scores(self, history: Sequence[LottoRecord], weights: Optional[Dict[str, float]] = None) -> Dict[int, float]:
        scores_by_strategy = self.score_numbers(history)
        if weights is None:
            weights = {name: 1.0 / len(self.strategy_names) for name in self.strategy_names}
        total_w = sum(max(weights.get(name, 0.0), 0.0) for name in self.strategy_names) or 1.0
        combined = {n: 0.0 for n in range(1, 46)}
        for name in self.strategy_names:
            w = max(weights.get(name, 0.0), 0.0) / total_w
            for n in range(1, 46):
                combined[n] += scores_by_strategy[name][n] * w
        return combined

    def _combo_score(self, nums: Sequence[int], number_scores: Dict[int, float], profile: Dict[str, float], pair_counts: Dict[Tuple[int, int], float], last_main: Sequence[int]) -> float:
        nums = sorted(nums)
        feat = combination_features(nums)
        base = sum(number_scores[n] for n in nums)
        sum_penalty = abs(feat["sum"] - profile["sum_mean"]) / max(profile["sum_std"], 1.0)
        odd_penalty = abs(feat["odd"] - profile["odd_mean"]) / 2.0
        high_penalty = abs(feat["high"] - profile["high_mean"]) / 2.0
        con_penalty = max(0, feat["consecutive"] - 2) * 0.45
        span_penalty = abs(feat["span"] - profile["span_mean"]) / 35.0
        repeat_penalty = max(0, len(set(nums) & set(last_main)) - 2) * 0.35
        pair_bonus = sum(pair_counts.get(tuple(sorted((a, b))), 0.0) for a, b in combinations(nums, 2))
        pair_bonus = math.log1p(pair_bonus) * 0.08
        return base + pair_bonus - 0.17 * sum_penalty - 0.12 * odd_penalty - 0.12 * high_penalty - con_penalty - 0.06 * span_penalty - repeat_penalty

    def generate_candidate_sets(
        self,
        history: Sequence[LottoRecord],
        weights: Optional[Dict[str, float]] = None,
        candidate_count: int = 25000,
        top_k: int = 12,
    ) -> List[Tuple[List[int], float]]:
        rng = random.Random(self.random_seed + len(history))
        number_scores = self.weighted_number_scores(history, weights)
        profile = historical_profile(history)
        pair_counts = self._pair_counts(history)
        last_main = history[-1].main if history else []
        items = list(range(1, 46))
        # convert scores into sampling weights while keeping diversity.
        raw = [max(number_scores[n], 0.01) ** 1.6 + 0.025 for n in items]
        candidates: Dict[Tuple[int, ...], float] = {}

        # Add deterministic high-score combinations from top pool.
        top_pool = sorted(items, key=lambda n: number_scores[n], reverse=True)[:22]
        for combo in combinations(top_pool, 6):
            if len(candidates) > min(6000, candidate_count // 2):
                break
            score = self._combo_score(combo, number_scores, profile, pair_counts, last_main)
            candidates[tuple(sorted(combo))] = score

        # Add stochastic candidates.
        for _ in range(candidate_count):
            combo = tuple(weighted_sample_without_replacement(items, raw, 6, rng))
            # basic sanity filters based on common Lotto distribution.
            feat = combination_features(combo)
            if not (80 <= feat["sum"] <= 200):
                continue
            if feat["odd"] not in {2, 3, 4}:
                continue
            if feat["high"] not in {2, 3, 4}:
                continue
            if feat["consecutive"] > 3:
                continue
            score = self._combo_score(combo, number_scores, profile, pair_counts, last_main)
            if combo not in candidates or score > candidates[combo]:
                candidates[combo] = score

        ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[:max(top_k, 1)]
        return [(list(k), float(v)) for k, v in ranked]

    def predict_strategy(self, history: Sequence[LottoRecord], strategy: str) -> List[int]:
        """Fast single-strategy prediction used inside rolling backtest.

        The full final prediction uses a larger stochastic candidate search.
        During backtest this method intentionally evaluates a bounded number of
        combinations so Streamlit stays responsive.
        """
        scores_by_strategy = self.score_numbers(history)
        if strategy not in scores_by_strategy:
            strategy = "frequency"
        scores = scores_by_strategy[strategy]
        pool = sorted(range(1, 46), key=lambda n: scores[n], reverse=True)[:18]
        profile = historical_profile(history)
        pair_counts = self._pair_counts(history, window=120)
        best_combo: Optional[Tuple[int, ...]] = None
        best_score = -1e9

        # Deterministic bounded scan. 18C6=18564, but cap to keep backtest fast.
        for checked, combo in enumerate(combinations(pool, 6)):
            if checked >= 2500:
                break
            sc = self._combo_score(combo, scores, profile, pair_counts, history[-1].main)
            if sc > best_score:
                best_combo = combo
                best_score = sc

        # Add small random diversity from the same pool.
        rng = random.Random(self.random_seed + len(history) + hash(strategy) % 10000)
        weights = [max(scores[n], 0.01) + 0.02 for n in pool]
        for _ in range(350):
            combo = tuple(weighted_sample_without_replacement(pool, weights, 6, rng))
            sc = self._combo_score(combo, scores, profile, pair_counts, history[-1].main)
            if sc > best_score:
                best_combo = combo
                best_score = sc
        return list(best_combo or pool[:6])

    @staticmethod
    def evaluate(pred: Sequence[int], actual: Sequence[int], bonus_pred: Optional[int] = None, actual_bonus: Optional[int] = None) -> Tuple[int, int]:
        matches = len(set(pred) & set(actual))
        rank = 0
        if matches == 6:
            rank = 1
        elif matches == 5 and bonus_pred is not None and actual_bonus is not None and bonus_pred == actual_bonus:
            rank = 2
        elif matches == 5:
            rank = 3
        elif matches == 4:
            rank = 4
        elif matches == 3:
            rank = 5
        return matches, rank

    def rolling_backtest(self, records: Sequence[LottoRecord], min_train: int = 200, backtest_draws: int = 180) -> Dict[str, StrategyMetrics]:
        n = len(records)
        start = max(min_train, n - backtest_draws)
        results: Dict[str, List[int]] = {name: [] for name in self.strategy_names}
        if n <= start:
            return {name: StrategyMetrics(0, 0, 0, 0, 0) for name in self.strategy_names}
        for i in range(start, n):
            hist = records[:i]
            actual = records[i].main
            for name in self.strategy_names:
                try:
                    pred = self.predict_strategy(hist, name)
                    m, _ = self.evaluate(pred, actual)
                    results[name].append(m)
                except Exception:
                    continue
        metrics: Dict[str, StrategyMetrics] = {}
        for name, vals in results.items():
            tests = len(vals)
            avg = statistics.mean(vals) if vals else 0.0
            hit3 = sum(v >= 3 for v in vals) / tests if tests else 0.0
            hit4 = sum(v >= 4 for v in vals) / tests if tests else 0.0
            # weighted score: focus on stable 3+ and rare 4+ hits, not ROI fantasy.
            avg_score = avg + 0.55 * hit3 + 2.0 * hit4
            metrics[name] = StrategyMetrics(avg, hit3, hit4, avg_score, tests)
        return metrics

    def optimize_weights(self, metrics: Dict[str, StrategyMetrics]) -> Dict[str, float]:
        vals = [metrics.get(name, StrategyMetrics(0, 0, 0, 0, 0)).avg_score for name in self.strategy_names]
        if not any(vals):
            return {name: 1.0 / len(self.strategy_names) for name in self.strategy_names}
        probs = softmax(vals, temperature=0.45)
        # minimum weight to keep diversity.
        floor = 0.035
        adjusted = [floor + p * (1 - floor * len(probs)) for p in probs]
        return {name: adjusted[i] for i, name in enumerate(self.strategy_names)}

    def pick_bonus_candidate(self, history: Sequence[LottoRecord], main: Sequence[int], number_scores: Dict[int, float]) -> int:
        bonus_freq = Counter(r.bonus for r in history[-180:])
        candidates = [n for n in range(1, 46) if n not in set(main)]
        return max(candidates, key=lambda n: number_scores.get(n, 0.0) * 0.65 + normalize([bonus_freq.get(x, 0) for x in range(1, 46)])[n - 1] * 0.35)

    def predict(
        self,
        records: Sequence[LottoRecord],
        min_train: int = 200,
        backtest_draws: int = 180,
        candidate_count: int = 25000,
        today: Optional[date] = None,
    ) -> PredictionResult:
        if len(records) < 30:
            raise ValueError("예측을 위해 최소 30회 이상의 데이터가 필요합니다.")
        metrics = self.rolling_backtest(records, min_train=min(min_train, max(30, len(records) // 2)), backtest_draws=min(backtest_draws, max(20, len(records) - 30)))
        weights = self.optimize_weights(metrics)
        candidates_scored = self.generate_candidate_sets(records, weights, candidate_count=candidate_count, top_k=15)
        if not candidates_scored:
            scores = self.weighted_number_scores(records, weights)
            main = sorted(sorted(range(1, 46), key=lambda n: scores[n], reverse=True)[:6])
            candidates_scored = [(main, 0.0)]
        main = candidates_scored[0][0]
        number_scores = self.weighted_number_scores(records, weights)
        bonus = self.pick_bonus_candidate(records, main, number_scores)
        last_draw = max(r.draw for r in records)
        draw_date = next_lotto_saturday(today=today, include_today=True).strftime("%Y-%m-%d")
        analysis = {
            "last_draw": last_draw,
            "records": len(records),
            "last_main": list(records[-1].main),
            "last_bonus": records[-1].bonus,
            "top_numbers": sorted(number_scores.items(), key=lambda kv: kv[1], reverse=True)[:15],
            "profile": historical_profile(records),
            "candidate_scores": candidates_scored[:10],
        }
        return PredictionResult(
            next_draw=last_draw + 1,
            draw_date=draw_date,
            main=main,
            bonus_candidate=bonus,
            candidate_sets=[c for c, _ in candidates_scored[:10]],
            number_scores={int(k): float(v) for k, v in number_scores.items()},
            strategy_weights=weights,
            backtest=metrics,
            analysis=analysis,
        )


def result_to_dict(result: PredictionResult) -> Dict[str, object]:
    data = asdict(result)
    data["backtest"] = {k: asdict(v) for k, v in result.backtest.items()}
    return data


def save_result_json(result: PredictionResult, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_to_dict(result), f, ensure_ascii=False, indent=2)
