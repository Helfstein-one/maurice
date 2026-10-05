import json
import os
import re
import subprocess
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/v1/chat/completions")

DEFAULT_BENCHMARK_RESULTS: list[dict[str, Any]] = [
    {
        "variant": "mau-llm-1.0-r",
        "hardware_acceleration": "x86_64 CPU (AVX2)",
        "metrics": {
            "tokens_per_second": 142.5,
            "time_to_first_token_ms": 18.2,
            "peak_rss_mb": 512.0,
            "evaluation_scores": {
                "eval_benchmark": "Reasoning & Logic Suite",
                "gsm8k_accuracy": 0.84,
                "math_pass_rate": 0.79,
            },
        },
    },
    {
        "variant": "mau-llm-1.0-c",
        "hardware_acceleration": "x86_64 CPU (AVX2)",
        "metrics": {
            "tokens_per_second": 168.0,
            "time_to_first_token_ms": 14.5,
            "peak_rss_mb": 480.0,
            "evaluation_scores": {
                "eval_benchmark": "Code Generation Suite",
                "humaneval_pass_at_1": 0.76,
                "mbpp_pass_at_1": 0.72,
            },
        },
    },
    {
        "variant": "mau-llm-1.0-g",
        "hardware_acceleration": "x86_64 CPU (AVX2)",
        "metrics": {
            "tokens_per_second": 155.2,
            "time_to_first_token_ms": 16.0,
            "peak_rss_mb": 500.0,
            "evaluation_scores": {
                "eval_benchmark": "General Knowledge Suite",
                "mmlu_accuracy": 0.68,
                "arc_challenge": 0.71,
            },
        },
    },
]


def render_assistant_content(content: str) -> None:
    """Renders assistant message content, parsing <think>...</think> tags into st.expander."""
    if not content:
        return

    if "<think>" in content:
        parts = re.split(r"(<think>.*?</think>)", content, flags=re.DOTALL)
        for part in parts:
            if part.startswith("<think>") and part.endswith("</think>"):
                think_content = part[7:-8].strip()
                if think_content:
                    with st.expander("Reasoning Process"):
                        st.markdown(think_content)
            elif "<think>" in part:
                idx = part.find("<think>")
                before = part[:idx].strip()
                think_content = part[idx + 7 :].strip()
                if before:
                    st.markdown(before)
                if think_content:
                    with st.expander("Reasoning Process"):
                        st.markdown(think_content)
            elif part.strip():
                st.markdown(part.strip())
    else:
        st.markdown(content)


def parse_benchmark_kpis(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculates key benchmark metrics summary for UI rendering."""
    if not results:
        return {
            "model_count": 0,
            "max_throughput": 0.0,
            "min_ttft_ms": 0.0,
            "avg_peak_rss_mb": 0.0,
        }

    tps_list = [item.get("metrics", {}).get("tokens_per_second", 0.0) for item in results]
    ttft_list = [item.get("metrics", {}).get("time_to_first_token_ms", 0.0) for item in results]
    rss_list = [item.get("metrics", {}).get("peak_rss_mb", 0.0) for item in results]

    return {
        "model_count": len(results),
        "max_throughput": max(tps_list) if tps_list else 0.0,
        "min_ttft_ms": min(ttft_list) if ttft_list else 0.0,
        "avg_peak_rss_mb": round(sum(rss_list) / len(rss_list), 2) if rss_list else 0.0,
    }


def format_variant_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Formats benchmark results into flat dictionary records for table/chart display."""
    table = []
    for item in results:
        v = item.get("variant", "unknown")
        hw = item.get("hardware_acceleration", "unknown")
        m = item.get("metrics", {})
        table.append(
            {
                "Variant": v,
                "Hardware": hw,
                "Throughput (t/s)": m.get("tokens_per_second", 0.0),
                "TTFT (ms)": m.get("time_to_first_token_ms", 0.0),
                "Peak RSS (MB)": m.get("peak_rss_mb", 0.0),
            }
        )
    return table


def render_app() -> None:
    """Renders the Streamlit multi-tab user interface."""
    st.set_page_config(page_title="maurice LLM Suite & Benchmark Analysis", layout="wide")

    st.title("maurice LLM Suite")

    tab1, tab2 = st.tabs(["💬 Chat & Reasoning", "📊 Benchmarks"])

    with tab1:
        st.markdown("Visual interface to showcase the reasoning process of the model.")

        variant = st.selectbox(
            "Select Model Variant",
            ["mau-llm-1.0-r", "mau-llm-1.0-c", "mau-llm-1.0-g", "maurice-final"],
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    render_assistant_content(msg["content"])
                else:
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Enter your prompt here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                st.empty()
                # Call API
                payload = {
                    "model": variant,
                    "messages": st.session_state.messages,
                    "stream": False,
                }
                try:
                    response = requests.post(API_URL, json=payload, timeout=60)
                    response.raise_for_status()
                    result = response.json()
                    assistant_content = result["choices"][0]["message"]["content"]
                except requests.RequestException as e:
                    assistant_content = f"Error connecting to inference server: {e}"

                render_assistant_content(assistant_content)
                st.session_state.messages.append({"role": "assistant", "content": assistant_content})

    with tab2:
        st.header("📊 Model Benchmark & Performance Analysis")
        st.markdown(
            "Analyze hardware throughput, Time to First Token (TTFT) latency, memory footprint, "
            "and domain-specific evaluation scores across trained maurice variants (`mau-llm-1.0-r`, "
            "`mau-llm-1.0-c`, `mau-llm-1.0-g`)."
        )

        col_source1, col_source2 = st.columns([2, 1])

        uploaded_file = None
        with col_source1:
            default_path = "build/benchmark_results.json"
            json_path = st.text_input("Path to Benchmark JSON Report:", value=default_path)
        with col_source2:
            uploaded_file = st.file_uploader("Or upload Benchmark JSON:", type=["json"])

        if st.button("🚀 Run / Refresh Benchmark Suite (Dry Run)"):
            with st.spinner("Running benchmark evaluation..."):
                try:
                    res = subprocess.run(
                        ["python3", "scripts/05_benchmark_eval.py", "--variant", "all", "--dry-run"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if res.returncode == 0:
                        st.success("Benchmark completed successfully!")
                    else:
                        st.error(f"Benchmark error: {res.stderr}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Failed to execute benchmark: {e}")

        benchmark_data = None
        if uploaded_file is not None:
            try:
                benchmark_data = json.load(uploaded_file)
            except Exception as e:  # noqa: BLE001
                st.error(f"Invalid uploaded JSON: {e}")
        elif json_path and os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    benchmark_data = json.load(f)
            except Exception as e:  # noqa: BLE001
                st.error(f"Error reading {json_path}: {e}")

        if not benchmark_data:
            st.info("No external benchmark JSON found or uploaded. Displaying default mock benchmark metrics.")
            benchmark_data = DEFAULT_BENCHMARK_RESULTS

        if isinstance(benchmark_data, dict):
            results_list = [benchmark_data]
        elif isinstance(benchmark_data, list):
            results_list = benchmark_data
        else:
            results_list = []

        if results_list:
            kpis = parse_benchmark_kpis(results_list)
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Evaluated Models", kpis["model_count"])
            kpi2.metric("Max Throughput", f"{kpis['max_throughput']:.1f} t/s")
            kpi3.metric("Min TTFT Latency", f"{kpis['min_ttft_ms']:.1f} ms")
            kpi4.metric("Avg Peak Memory", f"{kpis['avg_peak_rss_mb']:.1f} MB")

            st.subheader("🚀 Hardware Performance Comparison")
            table_data = format_variant_table(results_list)
            st.dataframe(table_data, use_container_width=True)

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("#### Token Throughput (tokens/sec)")
                tps_chart_data = {row["Variant"]: row["Throughput (t/s)"] for row in table_data}
                st.bar_chart(tps_chart_data)
            with chart_col2:
                st.markdown("#### TTFT Latency (ms)")
                ttft_chart_data = {row["Variant"]: row["TTFT (ms)"] for row in table_data}
                st.bar_chart(ttft_chart_data)

            st.subheader("🎯 Specialized Domain Evaluation Scores")
            for item in results_list:
                v = item.get("variant", "unknown")
                eval_scores = item.get("metrics", {}).get("evaluation_scores", {})
                bench_name = eval_scores.get("eval_benchmark", "Domain Benchmark")
                with st.expander(f"Variant '{v}' — {bench_name}", expanded=True):
                    cols = st.columns(max(len(eval_scores) - 1, 1))
                    col_idx = 0
                    for key, val in eval_scores.items():
                        if key != "eval_benchmark":
                            label = key.replace("_", " ").title()
                            val_str = f"{val:.1%}" if isinstance(val, float) and val <= 1.0 else str(val)
                            cols[col_idx % len(cols)].metric(label, val_str)
                            col_idx += 1

            with st.expander("📄 Raw Benchmark JSON Data"):
                st.json(benchmark_data)


if __name__ == "__main__":
    render_app()
