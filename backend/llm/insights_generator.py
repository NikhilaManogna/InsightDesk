from __future__ import annotations

from time import perf_counter

import pandas as pd

from backend.llm.anomaly_detector import AnomalyDetector
from backend.llm.dataframe_summarizer import DataFrameSummarizer
from backend.llm.insight_models import InsightBundle
from backend.llm.insight_prompts import INSIGHT_SYSTEM_PROMPT, build_business_insight_prompt
from backend.llm.kpi_extractor import InsightKPIExtractor
from backend.llm.providers.base import LLMRequest
from backend.llm.providers.factory import build_llm_provider
from backend.llm.trend_detector import TrendDetector
from backend.utils.config import Settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class InsightGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = build_llm_provider(settings)
        self.summarizer = DataFrameSummarizer(settings.insight_max_rows_sent)
        self.kpis = InsightKPIExtractor(settings.insight_max_kpis)
        self.trends = TrendDetector()
        self.anomalies = AnomalyDetector(settings.anomaly_sensitivity)

    def summarize(self, question: str, sql: str, frame: pd.DataFrame) -> str:
        return self.build_bundle(question, sql, frame).as_text()

    def build_bundle(self, question: str, sql: str, frame: pd.DataFrame) -> InsightBundle:
        if frame.empty:
            return InsightBundle(
                summary="No rows matched the query, so there is not enough data to infer a business pattern.",
                fallback_used=True,
            )

        started = perf_counter()
        summary = self.summarizer.summarize(frame)
        kpis = self.kpis.extract(frame)
        trends = self.trends.detect(frame)
        anomalies = self.anomalies.detect(frame)
        prompt = build_business_insight_prompt(
            question=question,
            summary=summary,
            kpis=kpis,
            trends=trends,
            anomalies=anomalies,
            verbosity=self.settings.insight_verbosity,
        )

        try:
            logger.info(
                "insight_prepared provider=%s rows=%s prompt_chars=%s anomalies=%s trends=%s",
                self.provider.name,
                len(frame),
                len(prompt),
                len(anomalies),
                len(trends),
            )
            response = self.provider.complete(
                LLMRequest(
                    system_prompt=INSIGHT_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    temperature=self.settings.insight_temperature,
                    max_tokens=self.settings.insight_max_tokens,
                )
            )
            text = self._clean_text(response)
            elapsed_ms = int((perf_counter() - started) * 1000)
            logger.info("insight_generated latency_ms=%s chars=%s", elapsed_ms, len(text))
            return InsightBundle(
                summary=text or self._fallback_summary(frame),
                kpis=kpis,
                trends=trends,
                anomalies=anomalies,
                fallback_used=not bool(text),
            )
        except Exception:
            logger.exception("insight_generation_failed using_fallback=true")
            return InsightBundle(
                summary=self._fallback_summary(frame),
                kpis=kpis,
                trends=trends,
                anomalies=anomalies,
                fallback_used=True,
            )

    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned = " ".join(text.strip().split())
        prefixes = ("Insight:", "Summary:")
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
        return cleaned

    @staticmethod
    def _fallback_summary(frame: pd.DataFrame) -> str:
        parts = [f"The query returned {len(frame):,} rows across {len(frame.columns)} fields."]
        numeric = frame.select_dtypes(include="number")
        if not numeric.empty:
            col = numeric.columns[0]
            series = numeric[col].dropna()
            if not series.empty:
                parts.append(
                    f"{col} ranges from {series.min():,.2f} to {series.max():,.2f}, with an average of {series.mean():,.2f}."
                )
        return " ".join(parts)
