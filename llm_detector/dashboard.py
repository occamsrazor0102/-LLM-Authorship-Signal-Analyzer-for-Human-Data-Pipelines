"""Streamlit-based React-style dashboard for the LLM Detection Pipeline.

Launch with:
    streamlit run llm_detector/dashboard.py
    # or
    llm-detector-dashboard
"""

import os
import io
import sys
import json
import tempfile
from collections import Counter

import streamlit as st
import pandas as pd

from llm_detector.pipeline import analyze_prompt
from llm_detector.io import load_xlsx, load_csv

# ── Theme constants ──────────────────────────────────────────────────────────

_DET_COLORS = {
    "RED": "#d32f2f",
    "AMBER": "#f57c00",
    "MIXED": "#7b1fa2",
    "YELLOW": "#fbc02d",
    "REVIEW": "#0288d1",
    "GREEN": "#388e3c",
}

_DET_EMOJI = {
    "RED": "\U0001f534",
    "AMBER": "\U0001f7e0",
    "MIXED": "\U0001f7e3",
    "YELLOW": "\U0001f7e1",
    "REVIEW": "\U0001f535",
    "GREEN": "\U0001f7e2",
}

_CHANNELS = ["prompt_structure", "stylometry", "continuation", "windowing"]


def _det_badge(det: str) -> str:
    """Return a colored markdown badge for a determination."""
    color = _DET_COLORS.get(det, "#6b7280")
    emoji = _DET_EMOJI.get(det, "")
    return f":{color[1:]}[{emoji} **{det}**]"


# ── Page configuration ───────────────────────────────────────────────────────

def _configure_page():
    st.set_page_config(
        page_title="LLM Detector Dashboard",
        page_icon="\u2728",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Inject custom CSS for React-style look
    st.markdown("""
    <style>
    /* Card-like containers */
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 0.75rem;
    }
    /* KPI metric cards */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #1e293b;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f1f5f9;
    }
    section[data-testid="stSidebar"] .stRadio label span {
        color: #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #334155;
    }
    /* Button styling */
    .stButton > button[kind="primary"] {
        background: #2563eb;
        border: none;
        border-radius: 6px;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8;
    }
    /* Top header area */
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Session state initialization ─────────────────────────────────────────────

def _init_state():
    defaults = {
        "results": [],
        "text_map": {},
        "memory_store": None,
        "cal_table": None,
        "run_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Sidebar ──────────────────────────────────────────────────────────────────

def _render_sidebar():
    with st.sidebar:
        st.markdown("## \u2728 LLM Detector")
        st.caption("Authorship Signal Analyzer")
        st.divider()

        page = st.radio(
            "Navigation",
            [
                "\u25b6\ufe0f  Analysis",
                "\u2699\ufe0f  Configuration",
                "\U0001f9e0  Memory & Learning",
                "\u2696\ufe0f  Calibration",
                "\U0001f4ca  Reports",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        # Quick status
        n = len(st.session_state.get("results", []))
        if n > 0:
            st.success(f"{n} results in session")
        else:
            st.info("No results yet")

        st.caption("v0.67.0")

    return page


# ── Page: Analysis ───────────────────────────────────────────────────────────

def _page_analysis():
    st.markdown("### \U0001f50d Analysis")
    st.caption("Run detection on individual texts or batch files")

    # KPI Metrics Row
    results = st.session_state.get("results", [])
    col1, col2, col3, col4 = st.columns(4)
    n_results = len(results)
    if n_results > 0:
        dets = [r.get("determination", "") for r in results]
        counts = Counter(dets)
        top_det = counts.most_common(1)[0][0] if counts else "N/A"
        avg_conf = sum(float(r.get("confidence", 0)) for r in results) / n_results
    else:
        top_det = "N/A"
        avg_conf = 0.0

    with col1:
        st.metric("Total Results", n_results)
    with col2:
        st.metric("Top Determination", top_det)
    with col3:
        st.metric("Avg Confidence", f"{avg_conf:.2f}")
    with col4:
        st.metric("Mode", st.session_state.get("mode", "auto"))

    st.markdown("---")

    # ── Data Source Section ────────────
    with st.expander("\U0001f4c1 Data Source", expanded=True):
        tab_text, tab_file = st.tabs(["Single Text", "File Upload"])

        with tab_text:
            text_input = st.text_area(
                "Enter text to analyze",
                height=150,
                placeholder="Paste text here for analysis...",
            )
            analyze_text_btn = st.button(
                "\u25b6 Analyze Text", type="primary", key="analyze_text"
            )

        with tab_file:
            uploaded = st.file_uploader(
                "Upload CSV or XLSX", type=["csv", "xlsx", "xlsm"]
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                prompt_col = st.text_input("Prompt column", value="prompt")
            with c2:
                sheet_name = st.text_input("Sheet name (XLSX)", value="")
            with c3:
                attempter_filter = st.text_input("Attempter filter", value="")
            analyze_file_btn = st.button(
                "\U0001f4c1 Analyze File", type="primary", key="analyze_file"
            )

    # ── Detection Settings ────────────
    with st.expander("\u2699\ufe0f Detection Settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            mode = st.selectbox("Mode", ["auto", "task_prompt", "generic_aigt"])
            st.session_state["mode"] = mode
        with c2:
            ppl_model = st.selectbox(
                "PPL Model",
                [
                    "Qwen/Qwen2.5-0.5B",
                    "HuggingFaceTB/SmolLM2-360M",
                    "HuggingFaceTB/SmolLM2-135M",
                    "distilgpt2",
                    "gpt2",
                ],
            )
        with c3:
            workers = st.number_input("Workers", min_value=1, max_value=16, value=4)

        c1, c2, c3 = st.columns(3)
        with c1:
            show_details = st.checkbox("Show details", value=True)
        with c2:
            verbose = st.checkbox("Verbose", value=False)
        with c3:
            no_layer3 = st.checkbox("Skip Layer 3", value=False)

        st.markdown("**Channel Ablation**")
        abl_cols = st.columns(len(_CHANNELS))
        disabled_channels = []
        for i, ch in enumerate(_CHANNELS):
            with abl_cols[i]:
                if st.checkbox(f"Disable {ch}", key=f"abl_{ch}"):
                    disabled_channels.append(ch)

    # ── Execute analysis ────────────
    def _build_kwargs():
        api_key = st.session_state.get("api_key", "").strip() or None
        return {
            "run_l3": not no_layer3,
            "api_key": api_key,
            "dna_provider": st.session_state.get("dna_provider", "anthropic"),
            "dna_model": st.session_state.get("dna_model", "").strip() or None,
            "dna_samples": st.session_state.get("dna_samples", 3),
            "mode": mode,
            "disabled_channels": disabled_channels or None,
            "cal_table": st.session_state.get("cal_table"),
            "memory_store": st.session_state.get("memory_store"),
            "ppl_model": ppl_model or None,
        }

    if analyze_text_btn and text_input.strip():
        with st.spinner("Analyzing text..."):
            kwargs = _build_kwargs()
            result = analyze_prompt(text_input, **kwargs)

            mem = st.session_state.get("memory_store")
            if mem:
                dis = mem.check_shadow_disagreement(result)
                result["shadow_disagreement"] = dis
                result["shadow_ai_prob"] = (dis or {}).get("shadow_ai_prob")

            st.session_state["results"] = [result]
            st.session_state["text_map"] = {"_single": text_input}
            st.session_state["run_count"] += 1
        st.rerun()

    if analyze_file_btn and uploaded is not None:
        with st.spinner("Analyzing file..."):
            # Save upload to temp file
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            try:
                if suffix in (".xlsx", ".xlsm"):
                    tasks = load_xlsx(
                        tmp_path,
                        sheet=sheet_name.strip() or None,
                        prompt_col=prompt_col.strip() or "prompt",
                    )
                else:
                    tasks = load_csv(
                        tmp_path, prompt_col=prompt_col.strip() or "prompt"
                    )
            finally:
                os.unlink(tmp_path)

            if attempter_filter.strip():
                needle = attempter_filter.strip().lower()
                tasks = [
                    t
                    for t in tasks
                    if needle in t.get("attempter", "").lower()
                ]

            if not tasks:
                st.warning("No qualifying prompts found.")
            else:
                kwargs = _build_kwargs()
                all_results = []
                text_map = {}
                progress = st.progress(0, text="Processing...")

                for i, task in enumerate(tasks):
                    tid = task.get("task_id", f"_row{i+1}")
                    text_map[tid] = task["prompt"]
                    r = analyze_prompt(
                        task["prompt"],
                        task_id=task.get("task_id", ""),
                        occupation=task.get("occupation", ""),
                        attempter=task.get("attempter", ""),
                        stage=task.get("stage", ""),
                        **kwargs,
                    )
                    all_results.append(r)
                    progress.progress(
                        (i + 1) / len(tasks),
                        text=f"Processing {i+1}/{len(tasks)}...",
                    )

                st.session_state["results"] = all_results
                st.session_state["text_map"] = text_map
                st.session_state["run_count"] += 1
                progress.empty()
                st.rerun()

    # ── Display Results ────────────
    if results:
        st.markdown("---")
        st.markdown("### \U0001f4cb Results")

        # Summary bar
        counts = Counter(r["determination"] for r in results)
        summary_cols = st.columns(len(_DET_COLORS))
        for i, det in enumerate(["RED", "AMBER", "MIXED", "YELLOW", "REVIEW", "GREEN"]):
            ct = counts.get(det, 0)
            with summary_cols[i]:
                color = _DET_COLORS[det]
                emoji = _DET_EMOJI[det]
                st.markdown(
                    f"<div style='text-align:center; padding:8px; "
                    f"background:{color}15; border-left:3px solid {color}; "
                    f"border-radius:4px'>"
                    f"<span style='font-size:0.8rem; color:{color}'>"
                    f"{emoji} {det}</span><br>"
                    f"<strong style='font-size:1.2rem'>{ct}</strong></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("")

        # Results table
        for idx, r in enumerate(results):
            det = r.get("determination", "GREEN")
            conf = r.get("confidence", 0)
            task_id = r.get("task_id", f"#{idx+1}")
            reason = r.get("reason", "")
            color = _DET_COLORS.get(det, "#6b7280")
            emoji = _DET_EMOJI.get(det, "")

            with st.expander(
                f"{emoji} **{det}** — {task_id}  |  conf={conf:.2f}",
                expanded=(len(results) == 1),
            ):
                # Top-line info
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("Determination", det)
                with mc2:
                    st.metric("Confidence", f"{conf:.2f}")
                with mc3:
                    st.metric("Word Count", r.get("word_count", 0))

                st.markdown(f"**Reason:** {reason}")

                # Calibrated confidence
                cal_conf = r.get("calibrated_confidence")
                if cal_conf is not None:
                    st.info(
                        f"Calibrated: {cal_conf:.2f}  "
                        f"({r.get('calibration_stratum', '')} / "
                        f"{r.get('conformity_level', '')})"
                    )

                # Channel details
                if show_details:
                    cd = r.get("channel_details", {})
                    channels = cd.get("channels", {})
                    if channels:
                        st.markdown("**Channel Details:**")
                        ch_data = []
                        for ch_name, info in channels.items():
                            ch_data.append({
                                "Channel": ch_name,
                                "Severity": info.get("severity", "GREEN"),
                                "Score": f"{info.get('score', 0):.2f}",
                                "Data Sufficient": (
                                    "\u2705" if info.get("data_sufficient", True)
                                    else "\u274c"
                                ),
                                "Status": (
                                    "disabled" if info.get("disabled") else
                                    "ineligible" if not info.get("mode_eligible", True)
                                    else "active"
                                ),
                            })
                        st.dataframe(
                            pd.DataFrame(ch_data),
                            use_container_width=True,
                            hide_index=True,
                        )

                # Verbose details
                if verbose:
                    with st.expander("Verbose Details"):
                        vcols = {
                            "Preamble": f"{r.get('preamble_score', 0):.2f} ({r.get('preamble_severity', '-')})",
                            "Fingerprints": f"{r.get('fingerprint_score', 0):.2f} ({r.get('fingerprint_hits', 0)} hits)",
                            "PromptSig": (
                                f"composite={r.get('prompt_signature_composite', 0):.2f} "
                                f"CFD={r.get('prompt_signature_cfd', 0):.3f}"
                            ),
                            "IDI": f"{r.get('instruction_density_idi', 0):.1f}",
                            "VSD": f"{r.get('voice_dissonance_vsd', 0):.1f}",
                        }
                        for k, v in vcols.items():
                            st.text(f"{k}: {v}")

                        bscore = r.get("continuation_bscore", 0)
                        if bscore > 0:
                            st.text(
                                f"DNA-GPT: BScore={bscore:.4f} "
                                f"(det={r.get('continuation_determination', 'n/a')})"
                            )

                # Detection spans
                spans = r.get("detection_spans", [])
                if spans:
                    span_sources = Counter(
                        s.get("source", "?") for s in spans
                    )
                    st.caption(
                        f"Detection spans: {len(spans)} "
                        f"({', '.join(f'{src}={ct}' for src, ct in span_sources.items())})"
                    )

                # Attack types
                attacks = r.get("norm_attack_types", [])
                if attacks:
                    st.warning(
                        f"Attacks neutralized: {', '.join(attacks)}"
                    )

                # Shadow model
                shadow = r.get("shadow_disagreement")
                if shadow:
                    st.error(
                        f"Shadow: {shadow.get('interpretation', 'disagrees')} — "
                        f"Rule={shadow.get('rule_determination', '?')}, "
                        f"Model={shadow.get('shadow_ai_prob', 0):.1%} AI"
                    )

        # Export actions
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            # CSV export
            flat = []
            for r in results:
                row = {
                    k: v for k, v in r.items() if k != "preamble_details"
                }
                row["preamble_details"] = str(
                    r.get("preamble_details", [])
                )
                flat.append(row)
            csv_data = pd.DataFrame(flat).to_csv(index=False)
            st.download_button(
                "\U0001f4be Download CSV",
                csv_data,
                "results.csv",
                "text/csv",
            )

        with c2:
            # HTML report
            flagged = [
                r
                for r in results
                if r["determination"] in ("RED", "AMBER", "MIXED")
            ]
            if flagged:
                try:
                    from llm_detector.html_report import (
                        generate_batch_html_report,
                    )
                    html = generate_batch_html_report(
                        flagged, st.session_state.get("text_map", {})
                    )
                    st.download_button(
                        "\U0001f4c4 Download HTML Report",
                        html,
                        "report.html",
                        "text/html",
                    )
                except Exception:
                    st.caption("HTML report generation unavailable")
            else:
                st.caption("No flagged results for HTML report")


# ── Page: Configuration ──────────────────────────────────────────────────────

def _page_configuration():
    st.markdown("### \u2699\ufe0f Configuration")
    st.caption("API keys, similarity settings, and output options")

    # Continuation Analysis
    with st.expander("\U0001f9ec Continuation Analysis (DNA-GPT)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            provider = st.selectbox(
                "Provider",
                ["anthropic", "openai"],
                key="dna_provider",
            )
        with c2:
            api_key = st.text_input(
                "API Key", type="password", key="api_key"
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            dna_model = st.text_input(
                "Model (optional)", key="dna_model"
            )
        with c2:
            dna_samples = st.number_input(
                "Samples",
                min_value=1,
                max_value=10,
                value=3,
                key="dna_samples",
            )
        with c3:
            batch_api = st.checkbox(
                "Batch API (50% cheaper)", key="batch_api"
            )

    # Similarity
    with st.expander("\U0001f50d Similarity Analysis", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            no_similarity = st.checkbox(
                "Disable similarity", key="no_similarity"
            )
        with c2:
            sim_threshold = st.number_input(
                "Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.40,
                step=0.05,
                key="sim_threshold",
            )

        sim_store = st.text_input(
            "Sim store (JSONL path)", key="sim_store"
        )
        instructions = st.text_input(
            "Instructions file path", key="instructions_file"
        )

    # Output Options
    with st.expander("\U0001f4e4 Output Options", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            cost = st.number_input(
                "Cost per prompt ($)",
                min_value=0.0,
                value=400.0,
                step=50.0,
                key="cost_per_prompt",
            )
        with c2:
            collect_path = st.text_input(
                "Collect baselines to JSONL", key="collect_path"
            )

    st.success("Configuration is saved in the session automatically.")


# ── Page: Memory & Learning ──────────────────────────────────────────────────

def _page_memory():
    st.markdown("### \U0001f9e0 Memory & Learning")
    st.caption("BEET memory store, ground truth, and learning tools")

    # Memory Store
    with st.expander("\U0001f4be BEET Memory Store", expanded=True):
        mem_dir = st.text_input(
            "Store directory",
            placeholder="/path/to/.beet",
            key="memory_dir",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Load Memory Store"):
                if mem_dir.strip():
                    try:
                        from llm_detector.memory import MemoryStore

                        st.session_state["memory_store"] = MemoryStore(
                            mem_dir.strip()
                        )
                        st.success(f"Loaded: {mem_dir}")
                    except Exception as e:
                        st.error(str(e))
                else:
                    st.warning("Set a directory first.")

        with c2:
            if st.button("Print Summary"):
                mem = st.session_state.get("memory_store")
                if mem:
                    buf = io.StringIO()
                    old = sys.stdout
                    sys.stdout = buf
                    try:
                        mem.print_summary()
                    finally:
                        sys.stdout = old
                    st.code(buf.getvalue())
                else:
                    st.warning("Load a memory store first.")

    # Ground Truth Confirmation
    with st.expander("\u2705 Record Ground Truth Confirmation", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            confirm_task = st.text_input("Task ID", key="confirm_task")
        with c2:
            confirm_label = st.selectbox(
                "Label", ["ai", "human"], key="confirm_label"
            )
        with c3:
            confirm_reviewer = st.text_input(
                "Reviewer", key="confirm_reviewer"
            )

        if st.button("Confirm"):
            mem = st.session_state.get("memory_store")
            if not mem:
                st.warning("Load a memory store first.")
            elif not confirm_task or not confirm_reviewer:
                st.warning("Task ID and Reviewer are required.")
            else:
                mem.record_confirmation(
                    confirm_task, confirm_label, verified_by=confirm_reviewer
                )
                st.success(
                    f"Confirmed: {confirm_task} = {confirm_label} "
                    f"by {confirm_reviewer}"
                )

    # Attempter History
    with st.expander("\U0001f464 Attempter History", expanded=False):
        attempter_name = st.text_input("Attempter name", key="hist_attempter")
        if st.button("Show History"):
            mem = st.session_state.get("memory_store")
            if not mem:
                st.warning("Load a memory store first.")
            elif attempter_name.strip():
                buf = io.StringIO()
                old = sys.stdout
                sys.stdout = buf
                try:
                    mem.print_attempter_history(attempter_name.strip())
                finally:
                    sys.stdout = old
                st.code(buf.getvalue())

    # Learning Tools
    with st.expander("\U0001f4da Learning Tools", expanded=False):
        corpus_path = st.text_input(
            "Labeled corpus (JSONL)",
            placeholder="/path/to/corpus.jsonl",
            key="corpus_path",
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            rebuild_shadow_btn = st.button("Rebuild Shadow")
        with c2:
            rebuild_centroids_btn = st.button("Rebuild Centroids")
        with c3:
            discover_lexicon_btn = st.button("Discover Lexicon")
        with c4:
            rebuild_all_btn = st.button("Rebuild All")

        mem = st.session_state.get("memory_store")
        if rebuild_shadow_btn:
            if not mem:
                st.warning("Load a memory store first.")
            else:
                with st.spinner("Rebuilding shadow model..."):
                    pkg = mem.rebuild_shadow_model()
                if pkg:
                    st.success(f"Shadow model rebuilt: AUC={pkg['cv_auc']:.3f}")
                else:
                    st.error("Insufficient labeled data")

        if rebuild_centroids_btn:
            if not mem:
                st.warning("Load a memory store first.")
            elif not corpus_path.strip():
                st.warning("Set a labeled corpus path.")
            else:
                with st.spinner("Rebuilding centroids..."):
                    result = mem.rebuild_semantic_centroids(corpus_path.strip())
                if result:
                    st.success(
                        f"Centroids rebuilt: separation={result['separation']:.4f}"
                    )
                else:
                    st.error("Insufficient labeled text")

        if discover_lexicon_btn:
            if not mem:
                st.warning("Load a memory store first.")
            elif not corpus_path.strip():
                st.warning("Set a labeled corpus path.")
            else:
                with st.spinner("Discovering lexicon..."):
                    candidates = mem.discover_lexicon_candidates(
                        corpus_path.strip()
                    )
                n_new = sum(
                    1
                    for c in candidates
                    if not c.get("already_in_fingerprints")
                    and not c.get("already_in_packs")
                )
                st.success(
                    f"Lexicon discovery: {len(candidates)} candidates "
                    f"({n_new} new)"
                )

        if rebuild_all_btn:
            if not mem:
                st.warning("Load a memory store first.")
            else:
                with st.spinner("Rebuilding all artifacts..."):
                    msgs = []
                    cal = mem.rebuild_calibration()
                    if cal:
                        st.session_state["cal_table"] = cal
                        msgs.append(
                            f"Calibration: {cal['n_calibration']} samples"
                        )
                    else:
                        msgs.append("Calibration: insufficient data")

                    shadow = mem.rebuild_shadow_model()
                    if shadow:
                        msgs.append(
                            f"Shadow model: AUC={shadow['cv_auc']:.3f}"
                        )
                    else:
                        msgs.append("Shadow model: insufficient data")

                    if corpus_path.strip():
                        centroids = mem.rebuild_semantic_centroids(
                            corpus_path.strip()
                        )
                        if centroids:
                            msgs.append(
                                f"Centroids: separation="
                                f"{centroids['separation']:.4f}"
                            )
                        cands = mem.discover_lexicon_candidates(
                            corpus_path.strip()
                        )
                        n_new = sum(
                            1
                            for c in cands
                            if not c.get("already_in_fingerprints")
                            and not c.get("already_in_packs")
                        )
                        msgs.append(
                            f"Lexicon: {len(cands)} candidates ({n_new} new)"
                        )
                for m in msgs:
                    st.info(m)


# ── Page: Calibration ────────────────────────────────────────────────────────

def _page_calibration():
    st.markdown("### \u2696\ufe0f Calibration & Baselines")
    st.caption("Conformal calibration and baseline analysis")

    # Calibration
    with st.expander("\U0001f4cf Conformal Calibration", expanded=True):
        cal_path = st.text_input(
            "Calibration table (JSON)",
            placeholder="/path/to/calibration.json",
            key="cal_table_path",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Load Calibration"):
                if cal_path.strip() and os.path.exists(cal_path.strip()):
                    from llm_detector.calibration import load_calibration

                    cal = load_calibration(cal_path.strip())
                    st.session_state["cal_table"] = cal
                    st.success(
                        f"Loaded: {cal['n_calibration']} records, "
                        f"{len(cal.get('strata', {}))} strata"
                    )
                else:
                    st.warning("Select a valid JSON file.")

        with c2:
            if st.button("Rebuild from Memory"):
                mem = st.session_state.get("memory_store")
                if not mem:
                    st.warning("Load a memory store first.")
                else:
                    cal = mem.rebuild_calibration()
                    if cal:
                        st.session_state["cal_table"] = cal
                        st.success(
                            f"Rebuilt: {cal['n_calibration']} samples"
                        )
                    else:
                        st.error("Insufficient data")

        build_from = st.text_input(
            "Build from JSONL",
            placeholder="/path/to/baselines.jsonl",
            key="cal_build_jsonl",
        )
        if st.button("Build & Save Calibration"):
            if not build_from.strip() or not os.path.exists(
                build_from.strip()
            ):
                st.warning("Select a valid baselines JSONL file.")
            else:
                from llm_detector.calibration import (
                    calibrate_from_baselines,
                    save_calibration,
                )

                cal = calibrate_from_baselines(build_from.strip())
                if cal is None:
                    st.error(
                        "Need >= 20 labeled human samples for calibration"
                    )
                else:
                    out_path = build_from.strip().replace(
                        ".jsonl", "_calibration.json"
                    )
                    save_calibration(cal, out_path)
                    st.session_state["cal_table"] = cal
                    st.success(
                        f"Built: {cal['n_calibration']} records, "
                        f"{len(cal.get('strata', {}))} strata "
                        f"→ {out_path}"
                    )

    # Baseline Analysis
    with st.expander("\U0001f4ca Baseline Analysis", expanded=True):
        bl_jsonl = st.text_input(
            "Baselines JSONL",
            placeholder="/path/to/baselines.jsonl",
            key="bl_jsonl_path",
        )
        bl_csv = st.text_input(
            "Output CSV (optional)",
            key="bl_csv_path",
        )
        if st.button("Analyze Baselines"):
            if not bl_jsonl.strip() or not os.path.exists(bl_jsonl.strip()):
                st.warning("Select a valid baselines JSONL file.")
            else:
                buf = io.StringIO()
                old = sys.stdout
                sys.stdout = buf
                try:
                    from llm_detector.baselines import analyze_baselines

                    analyze_baselines(
                        bl_jsonl.strip(),
                        output_csv=bl_csv.strip() or None,
                    )
                finally:
                    sys.stdout = old
                st.code(buf.getvalue())


# ── Page: Reports ────────────────────────────────────────────────────────────

def _page_reports():
    st.markdown("### \U0001f4ca Reports")
    st.caption("Batch summaries, attempter profiles, and financial impact")

    results = st.session_state.get("results", [])
    if not results:
        st.info(
            "No results available. Run a batch analysis on the "
            "**Analysis** page first."
        )
        return

    # Determination Distribution
    with st.expander("\U0001f4ca Determination Distribution", expanded=True):
        counts = Counter(r["determination"] for r in results)
        det_order = ["RED", "AMBER", "MIXED", "YELLOW", "REVIEW", "GREEN"]
        chart_data = pd.DataFrame(
            {
                "Determination": det_order,
                "Count": [counts.get(d, 0) for d in det_order],
            }
        )
        chart_data = chart_data[chart_data["Count"] > 0]
        st.bar_chart(chart_data.set_index("Determination"))

        # Percentage breakdown
        total = len(results)
        for det in det_order:
            ct = counts.get(det, 0)
            if ct > 0:
                pct = ct / total * 100
                emoji = _DET_EMOJI.get(det, "")
                st.markdown(
                    f"{emoji} **{det}**: {ct} ({pct:.1f}%)"
                )

    # Attempter Profiles
    if len(results) >= 5:
        with st.expander("\U0001f464 Attempter Risk Profiles", expanded=True):
            try:
                from llm_detector.reporting import profile_attempters

                profiles = profile_attempters(results)
                if profiles:
                    df = pd.DataFrame(profiles[:20])
                    display_cols = [
                        c
                        for c in [
                            "attempter",
                            "n_submissions",
                            "flag_rate",
                            "mean_confidence",
                        ]
                        if c in df.columns
                    ]
                    st.dataframe(
                        df[display_cols],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No attempter profiles available.")
            except Exception:
                st.info("Attempter profiling unavailable.")

    # Channel Patterns
    flagged = [
        r
        for r in results
        if r["determination"] in ("RED", "AMBER", "MIXED")
    ]
    if flagged:
        with st.expander(
            "\U0001f50d Channel Patterns (flagged)", expanded=True
        ):
            channel_counts = Counter()
            for r in flagged:
                cd = r.get("channel_details", {}).get("channels", {})
                for ch_name, info in cd.items():
                    if info.get("severity") not in ("GREEN", None):
                        channel_counts[ch_name] += 1
            if channel_counts:
                df = pd.DataFrame(
                    channel_counts.most_common(),
                    columns=["Channel", "Flags"],
                )
                st.bar_chart(df.set_index("Channel"))
            else:
                st.info("No channel patterns detected.")

    # Financial Impact
    if len(results) >= 10:
        with st.expander(
            "\U0001f4b0 Financial Impact Estimate", expanded=True
        ):
            try:
                from llm_detector.reporting import financial_impact

                cost = st.session_state.get("cost_per_prompt", 400.0)
                impact = financial_impact(results, cost_per_prompt=cost)

                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric(
                        "Total Submissions",
                        impact["total_submissions"],
                    )
                with mc2:
                    st.metric(
                        "Flag Rate", f"{impact['flag_rate']:.1%}"
                    )
                with mc3:
                    st.metric(
                        "Waste Estimate",
                        f"${impact['waste_estimate']:,.0f}",
                    )
                with mc4:
                    st.metric(
                        "Projected Annual",
                        f"${impact.get('projected_annual_waste', 0):,.0f}",
                    )
            except Exception:
                st.info("Financial impact calculation unavailable.")

    # Export
    st.markdown("---")
    if st.button("Export Baselines"):
        try:
            from llm_detector.baselines import collect_baselines

            buf = io.BytesIO()
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".jsonl"
            ) as tmp:
                collect_baselines(results, tmp.name)
                with open(tmp.name, "r") as f:
                    data = f.read()
                os.unlink(tmp.name)
            st.download_button(
                "\U0001f4be Download Baselines JSONL",
                data,
                "baselines.jsonl",
                "application/jsonl",
            )
        except Exception as e:
            st.error(str(e))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Streamlit dashboard entry point."""
    _configure_page()
    _init_state()
    page = _render_sidebar()

    if page.endswith("Analysis"):
        _page_analysis()
    elif page.endswith("Configuration"):
        _page_configuration()
    elif page.endswith("Memory & Learning"):
        _page_memory()
    elif page.endswith("Calibration"):
        _page_calibration()
    elif page.endswith("Reports"):
        _page_reports()


if __name__ == "__main__":
    main()
