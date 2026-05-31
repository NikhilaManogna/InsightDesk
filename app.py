from __future__ import annotations

from datetime import datetime
from hashlib import sha1
from time import perf_counter
from typing import Any
from uuid import uuid4
import re

import pandas as pd
import streamlit as st

from backend.cache.query_cache import QueryCache
from backend.cache.cache_service import CacheService
from backend.db.duckdb_engine import create_duckdb_engine
from backend.db.engine_router import EngineRouter
from backend.db.postgres import create_postgres_engine
from backend.db.query_executor import QueryExecutionError, QueryExecutor
from backend.db.schema_loader import SchemaLoader
from backend.files.upload_loader import UploadLoader
from backend.history.query_history import QueryHistoryRecord, QueryHistoryStore
from backend.llm.insights_generator import InsightGenerator
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


def dialect_for(database: str) -> str:
    return EngineRouter(settings).dialect_for(database)


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
        .stApp {
            background: #f4f6f8;
            color: #182230;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 5rem;
        }
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #d9dee7;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dfe4ea;
            border-radius: 8px;
            padding: 12px 14px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #dfe4ea;
            border-radius: 8px;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 6px;
            border: 1px solid #b8c2cc;
            background: #ffffff;
        }
        div[data-testid="stChatMessage"] {
            border-radius: 8px;
            border: 1px solid #e3e8ef;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
        }
        h1, h2, h3 {
            letter-spacing: 0;
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
    }


def render_sidebar(database: str) -> None:
    with st.sidebar:
        st.title("InsightDesk")
        st.caption("AI analytics workspace")
        st.divider()
        st.subheader("Connection")
        st.write(f"Active engine: **{database}**")

        if database == "DuckDB":
            upload = st.file_uploader(
                "Upload dataset",
                type=["csv", "xlsx", "xls", "parquet"],
                help="Loaded into DuckDB as a queryable table.",
            )
            if upload is not None and st.session_state.get("uploaded_file_name") != upload.name:
                with st.spinner("Loading file into DuckDB..."):
                    table = UploadLoader().save_and_load(upload, get_engine("DuckDB"))
                    st.session_state["uploaded_table"] = table
                    st.session_state["uploaded_file_name"] = upload.name
                    st.session_state["service_cache"].invalidate_prefix("generated_sql:")
                    st.session_state["service_cache"].invalidate_prefix("query_result:")
                    load_schema.clear()
                    load_schema_metadata.clear()
                st.success(f"Loaded `{table}` into DuckDB.")
            if st.session_state.get("uploaded_table"):
                st.info(f"Uploaded table: {st.session_state['uploaded_table']}")

        if st.button("Refresh schema", use_container_width=True):
            load_schema.clear()
            load_schema_metadata.clear()
            st.rerun()

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
    st.markdown("#### Generated SQL")
    st.code(payload["sql"], language="sql")
    if payload["cached"]:
        st.caption("Served from local query cache.")
    cache_bits = []
    if payload.get("sql_cached"):
        cache_bits.append("SQL cache hit")
    if payload.get("insight_cached"):
        cache_bits.append("Insights cache hit")
    if cache_bits:
        st.caption(" | ".join(cache_bits))

    frame: pd.DataFrame = payload["frame"]
    visualization: VisualizationBundle | None = payload.get("visualization")
    if visualization and visualization.kpis:
        st.markdown("#### Summary")
        columns = st.columns(len(visualization.kpis))
        for column, kpi in zip(columns, visualization.kpis):
            column.metric(kpi.label, kpi.value, help=kpi.help_text)

    st.markdown("#### Results")
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
        st.markdown("#### Chart")
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
        st.markdown("#### Chart")
        st.plotly_chart(
            payload["chart"],
            use_container_width=True,
            key=result_key(payload, "legacy_chart"),
        )
    else:
        st.caption("No chart was recommended for this result shape.")

    st.markdown("#### Insights")
    insight_bundle = payload.get("insight_bundle")
    if insight_bundle:
        st.info(markdown_safe(insight_bundle.summary))
        detail_items = []
        detail_items.extend(getattr(insight_bundle, "trends", []) or [])
        detail_items.extend(getattr(insight_bundle, "anomalies", []) or [])
        if detail_items:
            with st.expander("Analysis details", expanded=False):
                for item in detail_items:
                    st.write(getattr(item, "description", str(item)))
    else:
        st.info(markdown_safe(payload["insights"]))


def main() -> None:
    st.set_page_config(page_title="InsightDesk", page_icon="ID", layout="wide")
    inject_theme()
    init_state()

    database = st.sidebar.radio("Database", ["DuckDB", "PostgreSQL"], horizontal=True)
    render_sidebar(database)

    st.title("InsightDesk")
    st.caption("Natural language analytics for PostgreSQL and DuckDB.")

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
        except (SQLGenerationError, QueryExecutionError, ValueError, RuntimeError) as exc:
            logger.warning("Analysis request failed: %s", exc)
            st.error(str(exc))
            st.session_state["messages"].append({"role": "assistant", "content": str(exc)})
        except Exception as exc:
            logger.exception("Unexpected app failure")
            st.error("Something went wrong while handling the request.")
            st.session_state["messages"].append({"role": "assistant", "content": str(exc)})


if __name__ == "__main__":
    main()
