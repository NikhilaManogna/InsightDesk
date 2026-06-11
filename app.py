from __future__ import annotations

from datetime import datetime
from hashlib import sha1
from html import escape
from time import perf_counter
from typing import Any
from uuid import uuid4
import re

import pandas as pd
import streamlit as st
from sqlalchemy import text

from backend.cache.query_cache import QueryCache
from backend.cache.cache_service import CacheService
from backend.db.engine_router import EngineRouter
from backend.db.query_executor import QueryExecutionError, QueryExecutor
from backend.db.schema_loader import SchemaLoader
from backend.files.upload_loader import UploadLoader
from backend.history.query_history import QueryHistoryRecord, QueryHistoryStore
from backend.llm.insights_generator import InsightGenerator
from backend.llm.ambiguity import AmbiguousQuestionError
from backend.llm.sql_generator import GeminiSQLGenerator, SQLGenerationError
from backend.security.query_sanitizer import QuerySanitizer
from backend.security.sql_validator import SQLValidator
from backend.utils.config import get_settings
from backend.utils.logger import get_logger
from backend.visualization.chart_generator import ChartGenerator
from backend.visualization.chart_models import VisualizationBundle
from backend.visualization.visualization_config import VisualizationConfig

logger = get_logger(__name__)
settings = get_settings()


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("cache", QueryCache(settings.cache_ttl_seconds))
    st.session_state.setdefault(
        "service_cache",
        CacheService(
            ttl_seconds=settings.generated_sql_cache_ttl_seconds,
            redis_url=settings.redis_url,
            enabled=settings.redis_cache_enabled,
        ),
    )


@st.cache_resource(show_spinner=False)
def get_engine(database: str):
    return EngineRouter(settings).engine_for(database)


@st.cache_data(ttl=300, show_spinner=False)
def load_schema(database: str) -> dict[str, dict[str, str]]:
    return load_schema_metadata(database).as_schema_map()


@st.cache_data(ttl=300, show_spinner=False)
def load_schema_metadata(database: str):
    dialect = dialect_for(database)
    return SchemaLoader(get_engine(database), dialect).load_metadata()


@st.cache_data(ttl=180, show_spinner=False)
def load_dataset_profile(database: str) -> list[dict[str, Any]]:
    """Return a compact schema profile for the sidebar and result cards."""
    metadata = load_schema_metadata(database)
    dialect = dialect_for(database)
    profile: list[dict[str, Any]] = []
    with get_engine(database).connect() as conn:
        for table in list(metadata.tables.values())[:8]:
            row_count: int | None = None
            try:
                quoted = quote_identifier(table.name, dialect)
                row_count = int(conn.execute(text(f"SELECT COUNT(*) FROM {quoted}")).scalar() or 0)
            except Exception:
                logger.warning("table_row_count_failed table=%s", table.name, exc_info=True)
            numeric = sum(1 for kind in table.columns.values() if looks_numeric_type(kind))
            categorical = sum(1 for kind in table.columns.values() if looks_categorical_type(kind))
            profile.append(
                {
                    "table": table.name,
                    "rows": row_count,
                    "columns": len(table.columns),
                    "numeric": numeric,
                    "categorical": categorical,
                }
            )
    return profile


def dialect_for(database: str) -> str:
    return EngineRouter(settings).dialect_for(database)


def quote_identifier(name: str, dialect: str) -> str:
    del dialect
    return f'"{name.replace(chr(34), chr(34) + chr(34))}"'


def looks_numeric_type(kind: str) -> bool:
    lowered = kind.lower()
    return any(token in lowered for token in ("int", "double", "float", "decimal", "numeric", "real"))


def looks_categorical_type(kind: str) -> bool:
    lowered = kind.lower()
    return any(token in lowered for token in ("char", "text", "string", "bool", "enum"))


def history_item(question: str, sql: str, rows: int, database: str) -> dict[str, Any]:
    return {
        "time": datetime.now().strftime("%H:%M:%S"),
        "question": question,
        "sql": sql,
        "rows": rows,
        "database": database,
    }


def result_key(payload: dict[str, Any], suffix: str) -> str:
    identity = payload.get("result_id") or str(id(payload))
    raw = f"{identity}:{payload.get('sql', '')}:{len(payload.get('frame', []))}:{suffix}"
    return sha1(raw.encode("utf-8")).hexdigest()[:12]


def markdown_safe(text: str) -> str:
    return text.replace("$", "\\$")


def format_ms(value: int | None) -> str:
    if value is None:
        return "-"
    if value < 1000:
        return f"{value} ms"
    return f"{value / 1000:.1f} s"


def compact_number(value: int | float | None) -> str:
    if value is None:
        return "-"
    number = float(value)
    return f"{number:,.0f}" if number.is_integer() else f"{number:,.2f}"


def explain_sql(sql: str) -> str:
    """Explain the query without spending another LLM call."""
    upper = f" {sql.upper()} "
    parts: list[str] = []
    if " GROUP BY " in upper:
        parts.append("Groups records by the selected dimension.")
    if any(fn in upper for fn in ("SUM(", "COUNT(", "AVG(", "MIN(", "MAX(")):
        parts.append("Calculates aggregate metrics from the matching rows.")
    if " JOIN " in upper:
        parts.append("Joins related tables before analysis.")
    if " WHERE " in upper:
        parts.append("Filters the dataset before returning results.")
    if " ORDER BY " in upper:
        parts.append("Sorts the output so the most relevant rows appear first.")
    if " LIMIT " in upper:
        parts.append("Limits result size for safety and responsiveness.")
    return " ".join(parts) or "Reads matching rows from the selected dataset without changing data."


def confidence_for(payload: dict[str, Any]) -> dict[str, Any]:
    score = 92
    reasons = ["SQL passed validation", "Tables and columns matched schema"]
    if payload.get("retries", 0):
        score -= min(18, payload["retries"] * 8)
        reasons.append(f"{payload['retries']} correction retry used")
    if payload.get("cached") or payload.get("sql_cached"):
        reasons.append("Cache assisted this response")
    if len(payload.get("frame", [])) == 0:
        score -= 10
        reasons.append("No rows returned")
    label = "High" if score >= 85 else "Medium" if score >= 70 else "Needs review"
    return {"score": max(0, min(score, 99)), "label": label, "reasons": reasons}


def metric_grid(items: list[tuple[str, str, str | None]]) -> None:
    cards = []
    for label, value, help_text in items:
        help_html = f'<div class="muted">{escape(help_text)}</div>' if help_text else ""
        cards.append(
            f"""
            <div class="metric-card">
                <div class="label">{escape(label)}</div>
                <div class="value">{escape(value)}</div>
                {help_html}
            </div>
            """
        )
    st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_panel(title: str, body: str, kicker: str | None = None, extra_class: str = "") -> None:
    kicker_html = f'<div class="section-kicker">{escape(kicker)}</div>' if kicker else ""
    st.markdown(
        f"""
        <div class="section-card {extra_class}">
            {kicker_html}
            <div class="section-title">{escape(title)}</div>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_uploaded_table_reference(
    sql: str,
    uploaded_table: str | None,
    uploaded_file: str | None,
) -> str:
    if not uploaded_table or not uploaded_file:
        return sql
    stem = uploaded_file.rsplit(".", 1)[0]
    candidates = {
        uploaded_file,
        uploaded_file.replace(".", "_"),
        stem,
        stem.replace("-", "_").replace(" ", "_"),
    }
    normalized = sql
    for candidate in sorted(candidates, key=len, reverse=True):
        if not candidate or candidate == uploaded_table:
            continue
        pattern = rf"(?<![A-Za-z0-9_])([`\"\[]?){re.escape(candidate)}([`\"\]]?)(?![A-Za-z0-9_])"
        normalized = re.sub(pattern, uploaded_table, normalized, flags=re.IGNORECASE)
    return normalized


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #0b1020;
            --panel: #111827;
            --panel-soft: #151f32;
            --panel-muted: #0f172a;
            --border: rgba(148, 163, 184, 0.22);
            --text: #e5edf7;
            --muted: #94a3b8;
            --accent: #4f8cff;
            --accent-2: #23c7a7;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(79, 140, 255, 0.14), transparent 30rem),
                linear-gradient(180deg, #0b1020 0%, #0f172a 100%);
            color: var(--text);
        }
        .block-container {
            max-width: 1240px;
            padding-top: 1.4rem;
            padding-bottom: 5rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #0a0f1d 100%);
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] * { color: #dbeafe; }
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] small { color: #94a3b8 !important; }
        [data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
        .hero {
            border: 1px solid var(--border);
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.96), rgba(21, 31, 50, 0.9));
            border-radius: 14px;
            padding: 22px 24px;
            margin-bottom: 18px;
            box-shadow: 0 22px 60px rgba(0, 0, 0, 0.25);
        }
        .hero h1 {
            font-size: 2.1rem;
            line-height: 1.1;
            margin: 0 0 8px 0;
            color: var(--text);
        }
        .hero p, .muted {
            color: var(--muted);
            margin: 0;
        }
        .section-card {
            border: 1px solid var(--border);
            background: rgba(17, 24, 39, 0.88);
            border-radius: 8px;
            padding: 16px 18px;
            margin: 14px 0;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.18);
        }
        .section-title {
            color: #f8fafc;
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .section-kicker {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: 6px;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 5px 10px;
            background: rgba(15, 23, 42, 0.78);
            color: #cbd5e1;
            font-size: 0.82rem;
        }
        .status-ok {
            color: #86efac;
            border-color: rgba(34, 197, 94, .35);
            background: rgba(22, 101, 52, .18);
        }
        .confidence {
            height: 8px;
            border-radius: 999px;
            background: rgba(148, 163, 184, .22);
            overflow: hidden;
            margin-top: 8px;
        }
        .confidence span {
            display: block;
            height: 100%;
            background: linear-gradient(90deg, var(--accent-2), var(--accent));
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin: 12px 0;
        }
        .metric-card {
            border: 1px solid var(--border);
            background: linear-gradient(180deg, rgba(21, 31, 50, .9), rgba(15, 23, 42, .95));
            border-radius: 10px;
            padding: 14px;
        }
        .metric-card .label {
            color: var(--muted);
            font-size: .78rem;
            margin-bottom: 6px;
        }
        .metric-card .value {
            color: #f8fafc;
            font-weight: 750;
            font-size: 1.3rem;
        }
        .insight-card {
            border-left: 3px solid var(--accent-2);
            background: rgba(20, 184, 166, .08);
            border-radius: 8px;
            padding: 14px 16px;
            color: #dffcf5;
        }
        .query-box {
            color: #e2e8f0;
            font-size: 1.02rem;
        }
        .sidebar-brand {
            font-size: 1.5rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 2px;
        }
        .sidebar-subtitle {
            color: #94a3b8;
            font-size: .9rem;
            margin-bottom: 18px;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 8px;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid rgba(148, 163, 184, .32);
            background: #172033;
            color: #e5edf7;
        }
        div[data-testid="stChatMessage"] {
            border-radius: 12px;
            border: 1px solid var(--border);
            background: rgba(15, 23, 42, 0.68);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.16);
        }
        h1, h2, h3 { letter-spacing: 0; color: #f8fafc; }
        .stCodeBlock pre {
            border: 1px solid var(--border);
            border-radius: 10px;
        }
        [data-testid="stFileUploader"] section {
            background: rgba(15, 23, 42, .88);
            border: 1px dashed rgba(148, 163, 184, .42);
            border-radius: 10px;
        }
        [data-testid="stFileUploader"] button { color: #e5edf7; }
        [data-testid="stChatInput"] {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: rgba(17, 24, 39, .96);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_analysis(question: str, database: str) -> dict[str, Any]:
    question = QuerySanitizer().sanitize_question(question)
    uploaded_table = st.session_state.get("uploaded_table")
    uploaded_file = st.session_state.get("uploaded_file_name")
    if database == "DuckDB" and uploaded_table and "uploaded" in question.lower():
        question = (
            f"{question}. Use only the DuckDB table `{uploaded_table}` for uploaded data. "
            "Do not reference the uploaded filename as a table."
        )
    started = perf_counter()
    dialect = dialect_for(database)
    metadata = load_schema_metadata(database)
    schema = metadata.as_schema_map()
    if not metadata.tables:
        raise RuntimeError("No tables were found. Connect a database or create DuckDB tables first.")

    cache: QueryCache = st.session_state["cache"]
    service_cache: CacheService = st.session_state["service_cache"]
    generator = GeminiSQLGenerator(settings)
    validator = SQLValidator(metadata, settings.query_limit)
    executor = QueryExecutor(get_engine(database), settings.query_timeout_seconds)

    sql_cache_key = service_cache.make_key("generated_sql", database, question, sorted(schema))
    sql = service_cache.get(sql_cache_key)
    sql_cached = sql is not None
    if sql is None:
        sql = generator.generate(question, metadata, dialect)
        sql = normalize_uploaded_table_reference(sql, uploaded_table, uploaded_file)
        service_cache.set(sql_cache_key, sql)

    validation_errors: list[str] = []
    last_error: Exception | None = None
    frame: pd.DataFrame | None = None
    cached = False
    retries = 0

    for attempt in range(settings.sql_retry_count + 1):
        validation = validator.validate(sql, dialect)
        if not validation.valid:
            validation_errors = validation.errors
            logger.warning(
                "sql_validation_failed attempt=%s errors=%s",
                attempt + 1,
                " | ".join(validation.errors),
            )
            if attempt >= settings.sql_retry_count:
                raise ValueError("SQL validation failed: " + " ".join(validation.errors))
            sql = generator.repair(question, metadata, dialect, sql, validation.errors)
            sql = normalize_uploaded_table_reference(sql, uploaded_table, uploaded_file)
            retries += 1
            continue

        sql = validation.sql
        cache_key = cache.key(database, sql)
        result_cache_key = service_cache.make_key("query_result", database, sql)
        frame = service_cache.get(result_cache_key)
        if frame is not None:
            cached = True
            break
        frame = cache.get(cache_key)
        cached = frame is not None
        if frame is not None:
            break

        try:
            frame = executor.run(sql)
            cache.set(cache_key, frame)
            service_cache.set(result_cache_key, frame)
            break
        except QueryExecutionError as exc:
            last_error = exc
            logger.warning("sql_execution_retry attempt=%s error=%s", attempt + 1, exc)
            if attempt >= settings.sql_retry_count:
                raise
            sql = generator.repair(
                question,
                metadata,
                dialect,
                sql,
                [f"Database execution error: {exc}"],
            )
            sql = normalize_uploaded_table_reference(sql, uploaded_table, uploaded_file)
            retries += 1
    else:
        if last_error:
            raise last_error
        raise ValueError("SQL validation failed: " + " ".join(validation_errors))

    if frame is None:
        raise RuntimeError("Query did not return a result frame.")

    viz_config = VisualizationConfig(
        max_chart_rows=settings.max_visualization_rows,
        sample_threshold=settings.visualization_sample_threshold,
        category_limit=settings.chart_category_limit,
        histogram_bins=settings.histogram_bins,
        max_kpis=settings.max_kpis,
    )
    visualization = ChartGenerator(viz_config).build_bundle(frame)
    insight_cache_key = service_cache.make_key("insights", database, question, sql, len(frame))
    insight_bundle = service_cache.get(insight_cache_key)
    insight_cached = insight_bundle is not None
    if insight_bundle is None:
        insight_bundle = InsightGenerator(settings).build_bundle(question, sql, frame)
        service_cache.set(insight_cache_key, insight_bundle)
    execution_ms = int((perf_counter() - started) * 1000)
    return {
        "question": question,
        "sql": sql,
        "frame": frame,
        "chart": visualization.primary,
        "visualization": visualization,
        "insights": insight_bundle.as_text(),
        "insight_bundle": insight_bundle,
        "result_id": uuid4().hex,
        "cached": cached,
        "sql_cached": sql_cached,
        "insight_cached": insight_cached,
        "execution_ms": execution_ms,
        "retries": retries,
        "database": database,
        "provider": settings.llm_provider,
        "validation_status": "Passed",
    }


def render_sidebar(database: str) -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">InsightDesk</div>
            <div class="sidebar-subtitle">AI analytics workspace</div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.subheader("Connection")
        st.caption(f"Active engine: {database} | LLM: {settings.llm_provider}")

        if database == "DuckDB":
            upload = st.file_uploader(
                "Upload dataset",
                type=["csv", "xlsx", "xls", "parquet"],
                help="Loaded into DuckDB as a queryable table.",
            )
            uploaded_table = st.session_state.get("uploaded_table")
            known_tables = set(load_schema_metadata("DuckDB").tables) if upload is not None else set()
            needs_load = (
                upload is not None
                and (
                    st.session_state.get("uploaded_file_name") != upload.name
                    or not uploaded_table
                    or uploaded_table not in known_tables
                )
            )
            if needs_load:
                with st.spinner("Loading file into DuckDB..."):
                    table = UploadLoader().save_and_load(upload, get_engine("DuckDB"))
                    st.session_state["uploaded_table"] = table
                    st.session_state["uploaded_file_name"] = upload.name
                    st.session_state["service_cache"].invalidate_prefix("generated_sql:")
                    st.session_state["service_cache"].invalidate_prefix("query_result:")
                    load_schema.clear()
                    load_schema_metadata.clear()
                    load_dataset_profile.clear()
                st.success(f"Loaded `{table}` into DuckDB.")
            if st.session_state.get("uploaded_table"):
                st.caption(f"Uploaded table: `{st.session_state['uploaded_table']}`")

        if st.button("Refresh schema", use_container_width=True):
            load_schema.clear()
            load_schema_metadata.clear()
            load_dataset_profile.clear()
            st.rerun()

        with st.expander("Dataset intelligence", expanded=True):
            try:
                profile = load_dataset_profile(database)
                if not profile:
                    st.caption("No tables detected yet.")
                for item in profile:
                    rows = compact_number(item["rows"]) if item["rows"] is not None else "unknown"
                    st.markdown(f"**{item['table']}**")
                    st.caption(
                        f"{rows} rows | {item['columns']} columns | "
                        f"{item['numeric']} numeric | {item['categorical']} categorical"
                    )
            except Exception:
                logger.warning("dataset_profile_render_failed database=%s", database, exc_info=True)
                st.caption("Dataset profile unavailable.")

        st.divider()
        st.subheader("Query history")
        stored_history = QueryHistoryStore(settings.query_history_path, settings.query_history_limit).recent(8)
        if not st.session_state["history"] and not stored_history:
            st.caption("No queries yet.")
        for item in reversed(st.session_state["history"][-12:]):
            with st.expander(f"{item['time']} - {item['rows']} rows - {item['database']}"):
                st.write(item["question"])
                st.code(item["sql"], language="sql")
        if stored_history:
            with st.expander("Saved history", expanded=False):
                for item in reversed(stored_history[-8:]):
                    st.caption(f"{item.get('created_at', '')[:19]} - {item.get('rows', 0)} rows")
                    st.write(item.get("question", ""))


def render_result(payload: dict[str, Any]) -> None:
    question = payload.get("question", "")
    if question:
        render_panel(
            "User Question",
            f'<div class="query-box">{escape(question)}</div>',
            "Request",
        )

    frame: pd.DataFrame = payload["frame"]
    confidence = confidence_for(payload)
    cache_bits: list[str] = []
    if payload.get("sql_cached"):
        cache_bits.append("SQL cache hit")
    if payload.get("insight_cached"):
        cache_bits.append("Insights cache hit")
    if payload.get("cached"):
        cache_bits.append("Result cache hit")

    metric_grid(
        [
            ("Rows", compact_number(len(frame)), "Returned by the safe query"),
            ("Execution", format_ms(payload.get("execution_ms")), "Generation, validation, execution, and insight time"),
            ("Database", payload.get("database", "-"), "Selected analytics engine"),
            ("LLM", str(payload.get("provider", "-")).title(), "Active provider"),
        ]
    )

    status_pills = [
        '<span class="pill status-ok">Validation passed</span>',
        f'<span class="pill">Retries: {int(payload.get("retries", 0))}</span>',
        f'<span class="pill">Confidence: {confidence["label"]} ({confidence["score"]}%)</span>',
    ]
    status_pills.extend(f'<span class="pill">{escape(bit)}</span>' for bit in cache_bits)
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-kicker">Safety</div>
            <div class="section-title">Validation Status</div>
            <div class="pill-row">{''.join(status_pills)}</div>
            <div class="confidence"><span style="width: {confidence['score']}%"></span></div>
            <p class="muted" style="margin-top: 10px;">{escape("; ".join(confidence["reasons"]))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">SQL</div>
            <div class="section-title">Generated SQL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(payload["sql"], language="sql")
    with st.expander("Plain-English SQL explanation", expanded=False):
        st.write(explain_sql(payload["sql"]))

    visualization: VisualizationBundle | None = payload.get("visualization")
    if visualization and visualization.kpis:
        metric_grid([(kpi.label, str(kpi.value), kpi.help_text) for kpi in visualization.kpis])

    st.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Data</div>
            <div class="section-title">Query Results</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button(
        "Download results as CSV",
        frame.to_csv(index=False).encode("utf-8"),
        file_name="insightdesk_results.csv",
        mime="text/csv",
        use_container_width=False,
        key=result_key(payload, "csv"),
    )

    if visualization and visualization.primary is not None:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-kicker">Visual analysis</div>
                <div class="section-title">Visualizations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            visualization.primary,
            use_container_width=True,
            key=result_key(payload, "primary_chart"),
        )
        for index, figure in enumerate(visualization.supporting, start=1):
            with st.expander(f"Supporting chart {index}", expanded=False):
                st.plotly_chart(
                    figure,
                    use_container_width=True,
                    key=result_key(payload, f"supporting_chart_{index}"),
                )
        for note in visualization.notes:
            st.caption(note)
    elif payload.get("chart") is not None:
        st.markdown("#### Visualizations")
        st.plotly_chart(
            payload["chart"],
            use_container_width=True,
            key=result_key(payload, "legacy_chart"),
        )
    else:
        st.caption("No chart was recommended for this result shape.")

    st.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Narrative</div>
            <div class="section-title">AI Insights</div>
        """,
        unsafe_allow_html=True,
    )
    insight_bundle = payload.get("insight_bundle")
    if insight_bundle:
        st.markdown(
            f'<div class="insight-card">{escape(insight_bundle.summary)}</div>',
            unsafe_allow_html=True,
        )
        detail_items = []
        detail_items.extend(getattr(insight_bundle, "trends", []) or [])
        detail_items.extend(getattr(insight_bundle, "anomalies", []) or [])
        if detail_items:
            with st.expander("Analysis details", expanded=False):
                for item in detail_items:
                    st.write(getattr(item, "description", str(item)))
    else:
        st.markdown(
            f'<div class="insight-card">{escape(payload["insights"])}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="InsightDesk", page_icon="ID", layout="wide")
    inject_theme()
    init_state()

    database = st.sidebar.radio("Database", ["DuckDB", "PostgreSQL"], horizontal=True)
    render_sidebar(database)

    st.markdown(
        """
        <div class="hero">
            <h1>InsightDesk</h1>
            <p>Ask in plain English, review safe SQL, and turn database results into charts and business insight.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "payload" in message:
                render_result(message["payload"])
            else:
                st.write(message["content"])

    question = st.chat_input("Ask a business question about your data")
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Generating SQL, validating it, and querying the database..."):
                payload = run_analysis(question, database)
            render_result(payload)
            st.session_state["messages"].append({"role": "assistant", "payload": payload})
            st.session_state["history"].append(
                history_item(question, payload["sql"], len(payload["frame"]), database)
            )
            QueryHistoryStore(settings.query_history_path, settings.query_history_limit).append(
                QueryHistoryRecord(
                    question=question,
                    sql=payload["sql"],
                    database=database,
                    rows=len(payload["frame"]),
                    execution_ms=payload.get("execution_ms"),
                    retries=payload.get("retries", 0),
                    chart=(
                        payload.get("visualization").selected.kind
                        if payload.get("visualization") and payload.get("visualization").selected
                        else None
                    ),
                )
            )
        except (AmbiguousQuestionError, SQLGenerationError, QueryExecutionError, ValueError, RuntimeError) as exc:
            logger.warning("Analysis request failed: %s", exc)
            st.error(str(exc))
            st.session_state["messages"].append({"role": "assistant", "content": str(exc)})
        except Exception as exc:
            logger.exception("Unexpected app failure")
            st.error("Something went wrong while handling the request.")
            st.session_state["messages"].append({"role": "assistant", "content": str(exc)})


if __name__ == "__main__":
    main()
