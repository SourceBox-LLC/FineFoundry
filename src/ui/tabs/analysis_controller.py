"""Dataset Analysis tab controller for FineFoundry.

This module builds the Dataset Analysis tab controls and wires up all
analysis handlers, keeping `src/main.py` slimmer. Layout composition
still lives in `tab_analysis.py` and its section builders.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import asyncio
import os
import random
import re
from collections import Counter

import flet as ft

from helpers.common import safe_update
from helpers.datasets import guess_input_output_columns
from helpers.logging_config import get_logger
from helpers.theme import ACCENT_COLOR, BORDER_BASE, ICONS
from helpers.ui import WITH_OPACITY

# Optional datasets dependency (for HF datasets backend)
try:  # pragma: no cover - optional
    from datasets import load_dataset, Dataset, get_dataset_config_names
except Exception:  # pragma: no cover - fallback
    load_dataset = None  # type: ignore
    Dataset = None  # type: ignore
    get_dataset_config_names = None  # type: ignore

# save_dataset utilities (local, with PYTHONPATH pointing to project src)
try:  # pragma: no cover - normal path
    pass
except Exception:  # pragma: no cover - fallback for alternate runtimes
    import sys as _sys

    _sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ui.tabs.tab_analysis import build_analysis_tab


logger = get_logger(__name__)


def build_analysis_tab_with_logic(
    page: ft.Page,
    *,
    section_title,
    _mk_help_handler,
) -> ft.Control:
    """Build the Dataset Analysis tab UI and attach all related handlers.

    This mirrors the previous inline Analysis tab setup from ``main.py``, but
    keeps the behavior localized to this module.
    """

    # ---- Dataset Analysis tab: UI controls for builder ----
    def kpi_tile(title: str, value, subtitle: str = "", icon=None):
        # Accept either a string or a Flet control for value, so we can update it dynamically later.
        val_ctrl = value if isinstance(value, ft.Control) else ft.Text(str(value), size=18, weight=ft.FontWeight.W_600)
        return ft.Container(
            content=ft.Row(
                [
                    ft.Icon(
                        icon or getattr(ICONS, "INSIGHTS", ICONS.SEARCH),
                        size=20,
                        color=ACCENT_COLOR,
                    ),
                    ft.Column(
                        [
                            ft.Text(
                                title,
                                size=12,
                                color=WITH_OPACITY(0.7, BORDER_BASE),
                            ),
                            val_ctrl,
                            ft.Text(
                                subtitle,
                                size=11,
                                color=WITH_OPACITY(0.6, BORDER_BASE),
                            )
                            if subtitle
                            else ft.Container(),
                        ],
                        spacing=2,
                    ),
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            width=230,
            padding=12,
            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
            border_radius=8,
        )

    analysis_overview_note = ft.Text(
        "Click Analyze to compute dataset insights: totals, lengths, duplicates, sentiment, class balance, and samples.",
        size=12,
        color=WITH_OPACITY(0.7, BORDER_BASE),
    )

    # Sentiment controls (dynamic)
    sent_pos_label = ft.Text("Positive", width=90)
    sent_pos_bar = ft.ProgressBar(value=0.0, width=240)
    sent_pos_pct = ft.Text("0%", width=50, text_align=ft.TextAlign.END)
    sent_neu_label = ft.Text("Neutral", width=90)
    sent_neu_bar = ft.ProgressBar(value=0.0, width=240)
    sent_neu_pct = ft.Text("0%", width=50, text_align=ft.TextAlign.END)
    sent_neg_label = ft.Text("Negative", width=90)
    sent_neg_bar = ft.ProgressBar(value=0.0, width=240)
    sent_neg_pct = ft.Text("0%", width=50, text_align=ft.TextAlign.END)
    sentiment_row = ft.Column(
        [
            ft.Row(
                [sent_pos_label, sent_pos_bar, sent_pos_pct],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            ft.Row(
                [sent_neu_label, sent_neu_bar, sent_neu_pct],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            ft.Row(
                [sent_neg_label, sent_neg_bar, sent_neg_pct],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        ],
        spacing=6,
    )

    # Class balance proxy (dynamic) — we use input length buckets: Short/Medium/Long
    class_a_label = ft.Text("Short", width=90)
    class_a_bar = ft.ProgressBar(value=0.0, width=240)
    class_a_pct = ft.Text("0%", width=50, text_align=ft.TextAlign.END)
    class_b_label = ft.Text("Medium", width=90)
    class_b_bar = ft.ProgressBar(value=0.0, width=240)
    class_b_pct = ft.Text("0%", width=50, text_align=ft.TextAlign.END)
    class_c_label = ft.Text("Long", width=90)
    class_c_bar = ft.ProgressBar(value=0.0, width=240)
    class_c_pct = ft.Text("0%", width=50, text_align=ft.TextAlign.END)
    class_balance_row = ft.Column(
        [
            ft.Row([class_a_label, class_a_bar, class_a_pct]),
            ft.Row([class_b_label, class_b_bar, class_b_pct]),
            ft.Row([class_c_label, class_c_bar, class_c_pct]),
        ],
        spacing=6,
    )

    # Wrap Sentiment and Class Balance into sections to toggle visibility later
    sentiment_section = ft.Container(
        sentiment_row,
        padding=8,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        visible=False,
    )
    class_balance_section = ft.Container(
        class_balance_row,
        padding=8,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        visible=False,
    )

    # Grid table view for detailed samples (dynamic)
    SAMPLE_INPUT_W = 420
    SAMPLE_OUTPUT_W = 420
    SAMPLE_LEN_W = 70
    samples_grid = ft.DataTable(
        column_spacing=12,
        data_row_min_height=40,
        heading_row_height=40,
        columns=[
            ft.DataColumn(ft.Container(width=SAMPLE_INPUT_W, content=ft.Text("Input"))),
            ft.DataColumn(ft.Container(width=SAMPLE_OUTPUT_W, content=ft.Text("Output"))),
            ft.DataColumn(
                ft.Container(
                    width=SAMPLE_LEN_W,
                    content=ft.Text("In len", text_align=ft.TextAlign.END),
                )
            ),
            ft.DataColumn(
                ft.Container(
                    width=SAMPLE_LEN_W,
                    content=ft.Text("Out len", text_align=ft.TextAlign.END),
                )
            ),
        ],
        rows=[],
    )

    # Extra metrics table (for optional modules)
    extra_metrics_table = ft.DataTable(
        column_spacing=12,
        data_row_min_height=32,
        heading_row_height=36,
        columns=[
            ft.DataColumn(
                ft.Container(width=220, content=ft.Text("Metric")),
            ),
            ft.DataColumn(
                ft.Container(width=560, content=ft.Text("Value")),
            ),
        ],
        rows=[],
    )
    extra_metrics_section = ft.Container(
        extra_metrics_table,
        padding=8,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        visible=False,
    )

    # Samples section wrapper (hidden until results are available)
    samples_section = ft.Container(
        samples_grid,
        padding=8,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        visible=False,
    )

    # Dataset selector controls for Analysis (Database or HF)
    analysis_source_dd = ft.Dropdown(
        label="Dataset source",
        options=[
            ft.dropdown.Option("Database"),
            ft.dropdown.Option("Hugging Face"),
        ],
        value="Database",
        width=180,
    )
    # Database session selector
    analysis_db_session_dd = ft.Dropdown(
        label="Scrape session",
        options=[],
        width=360,
        visible=True,
        tooltip="Select a scrape session from the database",
    )
    analysis_db_refresh_btn = ft.IconButton(
        icon=ft.Icons.REFRESH,
        tooltip="Refresh sessions",
        visible=True,
    )
    analysis_hf_repo = ft.TextField(
        label="Dataset repo (e.g., username/dataset)",
        width=360,
        visible=False,
    )
    analysis_hf_split = ft.TextField(
        label="Split",
        value="train",
        width=120,
        visible=False,
    )
    analysis_hf_config = ft.TextField(
        label="Config (optional)",
        width=180,
        visible=False,
    )
    # JSON path kept for internal use only
    analysis_json_path = ft.TextField(
        label="JSON path",
        width=360,
        visible=False,
    )

    analysis_dataset_hint = ft.Text(
        "Select a dataset to analyze.",
        size=12,
        color=WITH_OPACITY(0.7, BORDER_BASE),
    )
    # Analysis runtime settings (UI only)
    analysis_backend_dd = ft.Dropdown(
        label="Backend",
        options=[
            ft.dropdown.Option("HF Inference API"),
            ft.dropdown.Option("Local (Transformers)"),
        ],
        value="HF Inference API",
        width=220,
    )
    analysis_hf_token_tf = ft.TextField(
        label="HF token (optional)",
        width=360,
        password=True,
        can_reveal_password=True,
        visible=True,
    )
    analysis_sample_size_tf = ft.TextField(
        label="Sample size",
        value="5000",
        width=140,
    )

    # Analysis module toggles
    cb_basic_stats = ft.Checkbox(
        label="Basic Stats",
        value=True,
        tooltip=("Record count and average input/output lengths."),
    )
    cb_duplicates = ft.Checkbox(
        label="Duplicates & Similarity",
        tooltip=("Approximate duplicate/similarity detection via hashing heuristics."),
    )
    cb_sentiment = ft.Checkbox(
        label="Sentiment",
        value=True,
        tooltip="Heuristic sentiment distribution over sampled records.",
    )
    cb_class_balance = ft.Checkbox(
        label="Class balance",
        value=True,
        tooltip="Distribution of labels/classes if present.",
    )
    cb_coverage_overlap = ft.Checkbox(
        label="Coverage Overlap",
        tooltip=("Overlap of input and output tokens (higher may indicate copying)."),
    )
    cb_data_leakage = ft.Checkbox(
        label="Data Leakage Check",
        tooltip="Flags potential target text appearing in inputs.",
    )
    cb_conversation_depth = ft.Checkbox(
        label="Conversation Depth",
        tooltip="Estimated turns/exchanges in dialogue-like data.",
    )
    cb_speaker_balance = ft.Checkbox(
        label="Speaker Balance",
        tooltip=("Balance of speakers/roles when such tags exist."),
    )
    cb_question_statement = ft.Checkbox(
        label="Question vs Statement",
        tooltip="Ratio of questions to statements in inputs.",
    )
    cb_readability = ft.Checkbox(
        label="Readability",
        tooltip="Simple readability proxy (length, punctuation).",
    )
    cb_ner = ft.Checkbox(
        label="NER",
        tooltip=("Counts of proper nouns/capitalized tokens as NER proxy."),
    )
    cb_toxicity = ft.Checkbox(
        label="Toxicity / Safety",
        tooltip="Flags profanity or unsafe terms (heuristic).",
    )
    cb_politeness = ft.Checkbox(
        label="Politeness / Formality",
        tooltip="Presence of polite markers (please, thanks, etc.).",
    )
    cb_dialogue_acts = ft.Checkbox(
        label="Dialogue Acts",
        tooltip="Heuristic dialogue acts (question/command/statement).",
    )
    cb_topics = ft.Checkbox(
        label="Topics / Clustering",
        tooltip="Top keywords proxy for topics.",
    )
    cb_alignment = ft.Checkbox(
        label="Alignment (Similarity/NLI)",
        tooltip="Rough input/output semantic alignment proxy.",
    )
    # Select-all toggle for analysis modules
    select_all_modules_cb = ft.Checkbox(label="Select all", value=False)

    # Analyze button; enabled only when dataset is selected.
    # Stub delegates to implementation defined later.
    async def on_analyze(_=None):
        await _on_analyze_impl(_)

    analyze_btn = ft.ElevatedButton(
        "Analyze dataset",
        icon=getattr(
            ICONS,
            "INSIGHTS",
            getattr(ICONS, "ANALYTICS", ICONS.SEARCH),
        ),
        disabled=True,
        on_click=lambda e: page.run_task(on_analyze),
    )
    # Ensure there's always a snackbar to open (handle older Flet without attribute)
    if not getattr(page, "snack_bar", None):
        page.snack_bar = ft.SnackBar(ft.Text("Analysis ready."))

    def _validate_analysis_dataset(_=None):
        try:
            src = analysis_source_dd.value or "Database"
        except Exception:
            src = "Database"
        repo = (analysis_hf_repo.value or "").strip()
        db_session_id = (analysis_db_session_dd.value or "").strip()
        if src == "Database":
            valid = bool(db_session_id)
            desc = f"Selected: DB Session {db_session_id}" if db_session_id else "Select a scrape session"
        elif src == "Hugging Face":
            valid = bool(repo)
            desc = f"Selected: HF {repo} [{(analysis_hf_split.value or 'train').strip()}]"
        else:
            valid = False
            desc = "Select a dataset to analyze."
        analyze_btn.disabled = not valid
        analysis_dataset_hint.value = desc if valid else "Select a dataset to analyze."
        try:
            page.update()
        except Exception:
            pass

    def _refresh_analysis_db_sessions(_=None):
        """Refresh the database session dropdown."""
        try:
            from db.scraped_data import list_scrape_sessions

            sessions = list_scrape_sessions(limit=50)
            options = []
            for s in sessions:
                label = f"{s['source']} - {s['pair_count']} pairs ({s['created_at'][:10]})"
                if s.get("source_details"):
                    label = f"{s['source']}: {s['source_details'][:30]} - {s['pair_count']} pairs"
                options.append(ft.dropdown.Option(key=str(s["id"]), text=label))
            analysis_db_session_dd.options = options
            if options and not analysis_db_session_dd.value:
                analysis_db_session_dd.value = options[0].key
        except Exception as e:
            analysis_db_session_dd.options = [ft.dropdown.Option(key="", text=f"Error: {e}")]
        _validate_analysis_dataset()
        try:
            page.update()
        except Exception:
            pass

    analysis_db_refresh_btn.on_click = _refresh_analysis_db_sessions
    analysis_db_session_dd.on_change = _validate_analysis_dataset

    def _update_analysis_source(_=None):
        src = getattr(analysis_source_dd, "value", "Database") or "Database"
        is_db = src == "Database"
        is_hf = src == "Hugging Face"
        analysis_db_session_dd.visible = is_db
        analysis_db_refresh_btn.visible = is_db
        analysis_hf_repo.visible = is_hf
        analysis_hf_split.visible = is_hf
        analysis_hf_config.visible = is_hf
        analysis_json_path.visible = False  # Always hidden
        if is_db:
            _refresh_analysis_db_sessions()
        _validate_analysis_dataset()

    def _update_analysis_backend(_=None):
        use_api = (
            getattr(analysis_backend_dd, "value", "HF Inference API") or "HF Inference API"
        ) == "HF Inference API"
        analysis_hf_token_tf.visible = use_api
        try:
            page.update()
        except Exception:
            pass

    # Wire up events
    analysis_source_dd.on_change = _update_analysis_source
    analysis_hf_repo.on_change = _validate_analysis_dataset
    analysis_hf_split.on_change = _validate_analysis_dataset
    analysis_backend_dd.on_change = _update_analysis_backend

    # Initialize database sessions on load
    try:
        _refresh_analysis_db_sessions()
    except Exception:
        pass

    # Helpers for analysis modules selection
    def _all_analysis_modules() -> List[ft.Checkbox]:
        return [
            cb_basic_stats,
            cb_duplicates,
            cb_sentiment,
            cb_class_balance,
            cb_coverage_overlap,
            cb_data_leakage,
            cb_conversation_depth,
            cb_speaker_balance,
            cb_question_statement,
            cb_readability,
            cb_ner,
            cb_toxicity,
            cb_politeness,
            cb_dialogue_acts,
            cb_topics,
            cb_alignment,
        ]

    def _sync_select_all_modules():
        try:
            select_all_modules_cb.value = all(bool(getattr(m, "value", False)) for m in _all_analysis_modules())
            page.update()
        except Exception:
            pass

    def _on_select_all_modules_change(_):
        try:
            val = bool(getattr(select_all_modules_cb, "value", False))
            for m in _all_analysis_modules():
                m.value = val
            page.update()
        except Exception:
            pass

    def _on_module_cb_change(_):
        _sync_select_all_modules()

    # Attach module checkbox events
    try:
        select_all_modules_cb.on_change = _on_select_all_modules_change
        for _m in _all_analysis_modules():
            _m.on_change = _on_module_cb_change
    except Exception:
        pass

    # --- Analysis backend state & handler ---
    analysis_state: Dict[str, Any] = {"running": False}
    analysis_busy_ring = ft.ProgressRing(
        value=None,
        visible=False,
        width=18,
        height=18,
    )

    # KPI dynamic value controls
    kpi_total_value = ft.Text("—", size=18, weight=ft.FontWeight.W_600)
    kpi_avg_in_value = ft.Text("—", size=18, weight=ft.FontWeight.W_600)
    kpi_avg_out_value = ft.Text("—", size=18, weight=ft.FontWeight.W_600)
    kpi_dupe_value = ft.Text("—", size=18, weight=ft.FontWeight.W_600)

    async def _on_analyze_impl(_=None) -> None:
        if analysis_state.get("running"):
            return
        analysis_state["running"] = True
        try:
            analyze_btn.disabled = True
            analysis_busy_ring.visible = True
            # Hide results while computing a fresh run
            overview_block.visible = False
            sentiment_block.visible = False
            class_balance_block.visible = False
            extra_metrics_block.visible = False
            samples_block.visible = False
            samples_section.visible = False
            div_overview.visible = False
            div_sentiment.visible = False
            div_class.visible = False
            div_extra.visible = False
            div_samples.visible = False
            await safe_update(page)

            src = analysis_source_dd.value or "Database"
            repo = (analysis_hf_repo.value or "").strip()
            split = (analysis_hf_split.value or "train").strip()
            cfg = (analysis_hf_config.value or "").strip() or None
            db_session_id = (analysis_db_session_dd.value or "").strip()
            try:
                sample_size = int(float((analysis_sample_size_tf.value or "5000").strip()))
                sample_size = max(1, min(250000, sample_size))
            except Exception:
                sample_size = 5000

            # Load examples as list[{input, output}]
            examples: List[Dict[str, Any]] = []
            total_records = 0

            if src == "Hugging Face":
                if load_dataset is None:
                    raise RuntimeError("datasets library not available — cannot load from Hub")

                async def _load_hf(repo_id: str, sp: str, name: Optional[str]) -> Any:
                    def do_load():
                        return load_dataset(repo_id, split=sp, name=name)

                    try:
                        return await asyncio.to_thread(do_load)
                    except Exception as e:  # pragma: no cover - runtime path
                        msg = str(e).lower()
                        auto_loaded = False
                        if get_dataset_config_names is not None and (
                            "config name is missing" in msg or "config name is required" in msg
                        ):
                            try:
                                cfgs = await asyncio.to_thread(lambda: get_dataset_config_names(repo_id))
                            except Exception:
                                cfgs = []
                            pick = None
                            for pref in ("main", "default", "socratic"):
                                if pref in cfgs:
                                    pick = pref
                                    break
                            if not pick and cfgs:
                                pick = cfgs[0]
                            if pick:

                                def do_load_cfg():
                                    return load_dataset(repo_id, split=sp, name=pick)

                                obj = await asyncio.to_thread(do_load_cfg)
                                auto_loaded = True
                                return obj
                        if not auto_loaded:
                            raise

                ds = await _load_hf(repo, split, cfg)
                try:
                    names = list(getattr(ds, "column_names", []) or [])
                except Exception:
                    names = []
                inn, outn = guess_input_output_columns(names)
                if not inn or not outn:
                    # If already in expected schema, allow it
                    if "input" in names and "output" in names:
                        inn, outn = "input", "output"
                    else:
                        raise RuntimeError(
                            f"Could not resolve input/output columns for {repo} (have: {', '.join(names)})"
                        )

                # Prepare two-column view
                def mapper(batch):
                    srcs = batch.get(inn, [])
                    tgts = batch.get(outn, [])
                    return {
                        "input": ["" if v is None else str(v).strip() for v in srcs],
                        "output": ["" if v is None else str(v).strip() for v in tgts],
                    }

                try:
                    mapped = await asyncio.to_thread(
                        lambda: ds.map(
                            mapper,
                            batched=True,
                            remove_columns=list(getattr(ds, "column_names", []) or []),
                        )
                    )
                except Exception:
                    # Fallback: iterate to python list
                    tmp: List[Dict[str, Any]] = []
                    for r in ds:
                        tmp.append(
                            {
                                "input": "" if r.get(inn) is None else str(r.get(inn)).strip(),
                                "output": "" if r.get(outn) is None else str(r.get(outn)).strip(),
                            }
                        )
                    from_list = await asyncio.to_thread(lambda: Dataset.from_list(tmp) if Dataset is not None else None)
                    mapped = from_list if from_list is not None else tmp

                # Select sample
                try:
                    total_records = len(mapped)
                except Exception:
                    total_records = 0
                if hasattr(mapped, "select"):
                    k = min(sample_size, total_records)
                    idxs = list(range(total_records)) if k >= total_records else random.sample(range(total_records), k)
                    batch = await asyncio.to_thread(lambda: mapped.select(idxs))
                    examples = [
                        {
                            "input": (r.get("input", "") or ""),
                            "output": (r.get("output", "") or ""),
                        }
                        for r in batch
                    ]
                else:
                    # mapped is already a python list
                    total_records = len(mapped)
                    if total_records > sample_size:
                        idxs = random.sample(range(total_records), sample_size)
                        examples = [mapped[i] for i in idxs]
                    else:
                        examples = list(mapped)

            elif src == "Database":
                # Database session
                if not db_session_id:
                    raise RuntimeError("Select a database session")
                from db.scraped_data import get_pairs_for_session

                records = await asyncio.to_thread(lambda: get_pairs_for_session(int(db_session_id)))
                if not records:
                    raise RuntimeError(f"No pairs found in session {db_session_id}")
                ex0 = [{"input": r["input"], "output": r["output"]} for r in records]
                total_records = len(ex0)
                if total_records > sample_size:
                    idxs = random.sample(range(total_records), sample_size)
                    examples = [ex0[i] for i in idxs]
                else:
                    examples = ex0
            else:
                raise RuntimeError("Invalid dataset source")

            used_n = len(examples)
            if used_n == 0:
                raise RuntimeError("No examples found to analyze")

            # Compute metrics (gated by module toggles where applicable)
            do_basic = bool(getattr(cb_basic_stats, "value", True))
            do_dupe = bool(getattr(cb_duplicates, "value", False))
            do_sent = bool(getattr(cb_sentiment, "value", True))
            do_cls = bool(getattr(cb_class_balance, "value", True))
            do_cov = bool(getattr(cb_coverage_overlap, "value", False))
            do_leak = bool(getattr(cb_data_leakage, "value", False))
            do_depth = bool(getattr(cb_conversation_depth, "value", False))
            do_speaker = bool(getattr(cb_speaker_balance, "value", False))
            do_qstmt = bool(getattr(cb_question_statement, "value", False))
            do_read = bool(getattr(cb_readability, "value", False))
            do_ner = bool(getattr(cb_ner, "value", False))
            do_toxic = bool(getattr(cb_toxicity, "value", False))
            do_polite = bool(getattr(cb_politeness, "value", False))
            do_dacts = bool(getattr(cb_dialogue_acts, "value", False))
            do_topics = bool(getattr(cb_topics, "value", False))
            do_align = bool(getattr(cb_alignment, "value", False))

            in_lens = [len(str(x.get("input", ""))) for x in examples]
            out_lens = [len(str(x.get("output", ""))) for x in examples]

            avg_in = avg_out = 0.0
            if do_basic:
                avg_in = sum(in_lens) / max(1, used_n)
                avg_out = sum(out_lens) / max(1, used_n)

            dup_pct: Optional[float] = None
            if do_dupe:
                unique_pairs = len({(str(x.get("input", "")), str(x.get("output", ""))) for x in examples})
                dup_pct = 100.0 * (1.0 - (unique_pairs / max(1, used_n)))

            # Sentiment proxy via tiny lexicon (gated)
            POS = {
                "good",
                "great",
                "love",
                "awesome",
                "nice",
                "excellent",
                "happy",
                "lol",
                "thanks",
                "cool",
            }
            NEG = {
                "bad",
                "hate",
                "terrible",
                "awful",
                "angry",
                "sad",
                "stupid",
                "dumb",
                "wtf",
                "idiot",
                "trash",
            }
            pos = neu = neg = 0
            if do_sent:
                for ex in examples:
                    txt = f"{ex.get('input', '')} {ex.get('output', '')}".lower()
                    score = sum(1 for w in POS if w in txt) - sum(1 for w in NEG if w in txt)
                    if score > 0:
                        pos += 1
                    elif score < 0:
                        neg += 1
                    else:
                        neu += 1
                pos_p = pos / used_n
                neu_p = neu / used_n
                neg_p = neg / used_n
            else:
                pos_p = neu_p = neg_p = 0.0

            # Length buckets (Short/Medium/Long) for input (gated)
            if do_cls:
                short = sum(1 for L in in_lens if L <= 128)
                medium = sum(1 for L in in_lens if 129 <= L <= 512)
                long = used_n - short - medium
                a_p = short / used_n
                b_p = medium / used_n
                c_p = long / used_n
            else:
                a_p = b_p = c_p = 0.0

            # Update UI controls
            kpi_total_value.value = f"{used_n:,}" if do_basic else "—"
            kpi_avg_in_value.value = f"{avg_in:.0f} chars" if do_basic else "—"
            kpi_avg_out_value.value = f"{avg_out:.0f} chars" if do_basic else "—"
            kpi_dupe_value.value = f"{dup_pct:.1f}%" if (do_dupe and dup_pct is not None) else "—"

            # Sentiment section
            sentiment_section.visible = do_sent
            sent_pos_bar.value = pos_p
            sent_pos_pct.value = f"{int(pos_p * 100)}%"
            sent_neu_bar.value = neu_p
            sent_neu_pct.value = f"{int(neu_p * 100)}%"
            sent_neg_bar.value = neg_p
            sent_neg_pct.value = f"{int(neg_p * 100)}%"

            # Class balance section
            class_balance_section.visible = do_cls
            class_a_label.value = "Short"
            class_a_bar.value = a_p
            class_a_pct.value = f"{int(a_p * 100)}%"
            class_b_label.value = "Medium"
            class_b_bar.value = b_p
            class_b_pct.value = f"{int(b_p * 100)}%"
            class_c_label.value = "Long"
            class_c_bar.value = c_p
            class_c_pct.value = f"{int(c_p * 100)}%"

            # Compute Extra metrics based on selected modules
            extra_rows: List[ft.DataRow] = []

            def _tokens(s: str) -> List[str]:
                return re.findall(r"[A-Za-z0-9']+", s.lower())

            def _token_set(s: str) -> set[str]:
                return set(_tokens(s))

            def _jaccard(a: set[str], b: set[str]) -> float:
                if not a and not b:
                    return 1.0
                inter = len(a & b)
                union = len(a | b)
                return inter / union if union else 0.0

            if any(
                [
                    do_cov,
                    do_leak,
                    do_depth,
                    do_speaker,
                    do_qstmt,
                    do_read,
                    do_ner,
                    do_toxic,
                    do_polite,
                    do_dacts,
                    do_topics,
                    do_align,
                ]
            ):
                # Precompute tokens
                in_tokens = [_token_set(str(ex.get("input", ""))) for ex in examples]
                out_tokens = [_token_set(str(ex.get("output", ""))) for ex in examples]

                if do_cov:
                    cover_vals: List[float] = []
                    for ti, to in zip(in_tokens, out_tokens):
                        cover = (len(ti & to) / max(1, len(to))) if to else 0.0
                        cover_vals.append(cover)
                    cover_avg = sum(cover_vals) / len(cover_vals)
                    extra_rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text("Coverage overlap")),
                                ft.DataCell(ft.Text(f"{cover_avg * 100:.1f}%")),
                            ]
                        )
                    )

                if do_align:
                    jac_vals: List[float] = []
                    for ti, to in zip(in_tokens, out_tokens):
                        jac_vals.append(_jaccard(ti, to))
                    jac_avg = sum(jac_vals) / len(jac_vals)
                    extra_rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text("Alignment (Jaccard)")),
                                ft.DataCell(ft.Text(f"{jac_avg * 100:.1f}%")),
                            ]
                        )
                    )

                if do_leak:
                    leak = 0
                    for ex in examples:
                        a = str(ex.get("input", "")).lower()
                        b = str(ex.get("output", "")).lower()
                        if (a and b) and (a in b or b in a):
                            leak += 1
                    leak_p = leak / used_n
                    extra_rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text("Data leakage risk")),
                                ft.DataCell(ft.Text(f"{leak_p * 100:.1f}%")),
                            ]
                        )
                    )

                if do_depth:

                    def _turns(text: str) -> int:
                        tl = text.lower()
                        m = len(re.findall(r"\b(user|assistant|system)\s*:", tl))
                        if m:
                            return m
                        lines = [ln for ln in text.splitlines() if ln.strip()]
                        return max(1, len(lines))

                    turns = [max(_turns(str(ex.get("input", ""))), 1) for ex in examples]
                    avg_turns = sum(turns) / len(turns)
                    extra_rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text("Avg turns (approx)")),
                                ft.DataCell(ft.Text(f"{avg_turns:.1f}")),
                            ]
                        )
                    )

                if do_speaker:
                    shares: List[float] = []
                    for ex in examples:
                        a = str(ex.get("input", ""))
                        b = str(ex.get("output", ""))
                        tot = len(a) + len(b)
                        shares.append((len(a) / tot) if tot else 0.0)
                    share_avg = sum(shares) / len(shares)
                    extra_rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text("Speaker balance (input share)")),
                                ft.DataCell(ft.Text(f"{share_avg * 100:.1f}%")),
                            ]
                        )
                    )

                if do_qstmt:
                    q = 0
                    for ex in examples:
                        a = str(ex.get("input", ""))
                        if a.strip().endswith("?"):
                            q += 1
                    q_p = q / used_n
                    extra_rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text("Questions (inputs)")),
                                ft.DataCell(ft.Text(f"{q_p * 100:.1f}%")),
                            ]
                        )
                    )

                if do_read:

                    def _syllables(word: str) -> int:
                        w = word.lower()
                        groups = re.findall(r"[aeiouy]+", w)
                        return max(1, len(groups))

                    def _readability(text: str) -> float:
                        toks = _tokens(text)
                        words = max(1, len(toks))
                        sentences = max(
                            1,
                            len(re.findall(r"[.!?]", text)),
                        )
                        syll = sum(_syllables(t) for t in toks)
                        # Flesch Reading Ease (approx)
                        return 206.835 - 1.015 * (words / sentences) - 84.6 * (syll / words)

                    scores = [_readability(str(ex.get("input", ""))) for ex in examples]
                    score_avg = sum(scores) / len(scores)
                    extra_rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(
                                    ft.Text("Readability (Flesch approx)"),
                                ),
                                ft.DataCell(ft.Text(f"{score_avg:.1f}")),
                            ]
                        )
                    )

                if do_ner:

                    def _capwords(text: str) -> int:
                        # Count capitalized words not at sentence start (rough proxy)
                        toks = re.findall(r"\b[A-Z][a-z]+\b", text)
                        return len(toks)

                    ents = [_capwords(str(ex.get("input", ""))) for ex in examples]
                    ents_avg = sum(ents) / len(ents)
                    extra_rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text("NER (capwords avg)")),
                                ft.DataCell(ft.Text(f"{ents_avg:.2f}")),
                            ]
                        )
                    )

                if do_toxic:
                    tox = 0
                    for ex in examples:
                        txt = f"{ex.get('input', '')} {ex.get('output', '')}".lower()
                        if any(w in txt for w in NEG):
                            tox += 1
                    tox_p = tox / used_n
                    extra_rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text("Toxicity flagged")),
                                ft.DataCell(ft.Text(f"{tox_p * 100:.1f}%")),
                            ]
                        )
                    )

                if do_polite:
                    POLITE = {
                        "please",
                        "thank",
                        "thanks",
                        "kindly",
                        "sir",
                        "madam",
                        "regards",
                    }
                    pol = 0
                    for ex in examples:
                        txt = f"{ex.get('input', '')} {ex.get('output', '')}".lower()
                        if any(w in txt for w in POLITE):
                            pol += 1
                    pol_p = pol / used_n
                    extra_rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text("Politeness flagged")),
                                ft.DataCell(ft.Text(f"{pol_p * 100:.1f}%")),
                            ]
                        )
                    )

                if do_dacts:
                    q = c = s = 0
                    for ex in examples:
                        a = str(ex.get("input", "")).strip()
                        al = a.lower()
                        if a.endswith("?"):
                            q += 1
                        elif al.startswith(
                            (
                                "please ",
                                "do ",
                                "go ",
                                "make ",
                                "provide ",
                                "give ",
                                "show ",
                            )
                        ):
                            c += 1
                        else:
                            s += 1
                    q_p = q / used_n
                    c_p = c / used_n
                    s_p = s / used_n
                    extra_rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text("Dialogue acts (Q/C/S)")),
                                ft.DataCell(ft.Text(f"{int(q_p * 100)}/{int(c_p * 100)}/{int(s_p * 100)}%")),
                            ]
                        )
                    )

                if do_topics:
                    STOP = {
                        "the",
                        "a",
                        "an",
                        "and",
                        "or",
                        "to",
                        "is",
                        "are",
                        "was",
                        "were",
                        "of",
                        "for",
                        "in",
                        "on",
                        "at",
                        "it",
                        "this",
                        "that",
                        "i",
                        "you",
                        "he",
                        "she",
                        "they",
                        "we",
                        "with",
                    }
                    freq: Counter[str] = Counter()
                    for ex in examples:
                        freq.update([t for t in _tokens(str(ex.get("input", ""))) if t not in STOP and len(t) > 2])
                    top = ", ".join([w for w, _ in freq.most_common(5)]) or "(none)"
                    extra_rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text("Top keywords")),
                                ft.DataCell(ft.Text(top)),
                            ]
                        )
                    )

            extra_metrics_table.rows = extra_rows
            extra_metrics_section.visible = len(extra_rows) > 0

            # Reveal result blocks now that real values are computed
            kpi_total_tile.visible = do_basic
            kpi_avg_in_tile.visible = do_basic
            kpi_avg_out_tile.visible = do_basic
            kpi_dupe_tile.visible = do_dupe
            overview_block.visible = do_basic or do_dupe
            sentiment_block.visible = do_sent
            class_balance_block.visible = do_cls
            extra_metrics_block.visible = len(extra_rows) > 0
            samples_section.visible = True
            samples_block.visible = True
            # Toggle dividers to match block visibility
            div_overview.visible = overview_block.visible
            div_sentiment.visible = sentiment_block.visible
            div_class.visible = class_balance_block.visible
            div_extra.visible = extra_metrics_block.visible
            div_samples.visible = samples_block.visible

            # Samples grid (up to 10)
            try:
                show_n = min(10, used_n)
                rows: List[ft.DataRow] = []
                for i in range(show_n):
                    ex = examples[i]
                    a = str(ex.get("input", ""))
                    b = str(ex.get("output", ""))
                    # Scrollable text cells with fixed width for neat column layout
                    a_cell = ft.Container(
                        width=SAMPLE_INPUT_W,
                        content=ft.Row(
                            [
                                ft.Text(
                                    a,
                                    no_wrap=True,
                                    selectable=True,
                                )
                            ],
                            scroll=ft.ScrollMode.AUTO,
                        ),
                    )
                    b_cell = ft.Container(
                        width=SAMPLE_OUTPUT_W,
                        content=ft.Row(
                            [
                                ft.Text(
                                    b,
                                    no_wrap=True,
                                    selectable=True,
                                )
                            ],
                            scroll=ft.ScrollMode.AUTO,
                        ),
                    )
                    inlen_cell = ft.Container(
                        width=SAMPLE_LEN_W,
                        content=ft.Text(
                            str(len(a)),
                            text_align=ft.TextAlign.END,
                        ),
                    )
                    outlen_cell = ft.Container(
                        width=SAMPLE_LEN_W,
                        content=ft.Text(
                            str(len(b)),
                            text_align=ft.TextAlign.END,
                        ),
                    )
                    rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(a_cell),
                                ft.DataCell(b_cell),
                                ft.DataCell(inlen_cell),
                                ft.DataCell(outlen_cell),
                            ]
                        )
                    )
                samples_grid.rows = rows
            except Exception:
                pass

            try:
                modules_used: List[str] = []
                if do_basic:
                    modules_used.append("Basic Stats")
                if do_dupe:
                    modules_used.append("Duplicates")
                if do_sent:
                    modules_used.append("Sentiment")
                if do_cls:
                    modules_used.append("Class balance")
                if do_cov:
                    modules_used.append("Coverage Overlap")
                if do_leak:
                    modules_used.append("Data Leakage Check")
                if do_depth:
                    modules_used.append("Conversation Depth")
                if do_speaker:
                    modules_used.append("Speaker Balance")
                if do_qstmt:
                    modules_used.append("Question vs Statement")
                if do_read:
                    modules_used.append("Readability")
                if do_ner:
                    modules_used.append("NER")
                if do_toxic:
                    modules_used.append("Toxicity / Safety")
                if do_polite:
                    modules_used.append("Politeness / Formality")
                if do_dacts:
                    modules_used.append("Dialogue Acts")
                if do_topics:
                    modules_used.append("Topics / Clustering")
                if do_align:
                    modules_used.append("Alignment (Similarity/NLI)")
                mod_txt = " | Modules: " + ", ".join(modules_used) if modules_used else ""
                analysis_overview_note.value = (
                    f"Analyzed {used_n:,} records"
                    + (f" (sampled from {total_records:,})" if total_records > used_n else "")
                    + mod_txt
                )
            except Exception:
                pass

            await safe_update(page)
        except Exception as e:  # pragma: no cover - runtime error path
            page.snack_bar = ft.SnackBar(ft.Text(f"Analysis failed: {e}"))
            page.snack_bar.open = True
            await safe_update(page)
        finally:
            analysis_busy_ring.visible = False
            analyze_btn.disabled = False
            analysis_state["running"] = False
            await safe_update(page)

    # Helper: build a table layout for module checkboxes (3 columns)
    def _build_modules_table() -> ft.DataTable:
        mods = _all_analysis_modules()
        columns = [
            ft.DataColumn(ft.Text("")),
            ft.DataColumn(ft.Text("")),
            ft.DataColumn(ft.Text("")),
        ]
        rows: List[ft.DataRow] = []

        def _cell_with_help(ctrl: ft.Control) -> ft.Control:
            try:
                tip = getattr(ctrl, "tooltip", None)
            except Exception:
                tip = None
            # Try to add a small clickable info icon next to control
            try:
                _info_icon_name = getattr(
                    ICONS,
                    "INFO_OUTLINE",
                    getattr(
                        ICONS,
                        "INFO",
                        getattr(
                            ICONS,
                            "HELP_OUTLINE",
                            getattr(ICONS, "HELP", None),
                        ),
                    ),
                )

                def _on_help_click(e, text=tip):  # noqa: ARG001
                    try:
                        dlg = ft.AlertDialog(
                            title=ft.Text("About module"),
                            content=ft.Text(text or ""),
                        )
                        page.dialog = dlg
                        dlg.open = True
                        page.update()
                    except Exception:
                        try:
                            page.snack_bar = ft.SnackBar(ft.Text(text or ""))
                            page.snack_bar.open = True
                            page.update()
                        except Exception:
                            pass

                help_btn = None
                try:
                    help_btn = ft.IconButton(
                        icon=_info_icon_name,
                        icon_color=WITH_OPACITY(0.6, BORDER_BASE),
                        tooltip=tip or "Module help",
                        on_click=_on_help_click,
                    )
                except Exception:
                    try:
                        help_btn = ft.Icon(_info_icon_name, size=16, color=WITH_OPACITY(0.6, BORDER_BASE))
                        help_btn = ft.Tooltip(
                            message=tip or "Module help",
                            content=help_btn,
                        )
                    except Exception:
                        help_btn = None
                if help_btn is not None:
                    return ft.Row(
                        [ctrl, help_btn],
                        spacing=4,
                        alignment=ft.MainAxisAlignment.START,
                    )
            except Exception:
                pass
            # Fallback: return control as-is
            return ctrl

        for i in range(0, len(mods), 3):
            c1 = ft.DataCell(_cell_with_help(mods[i]))
            c2 = ft.DataCell(_cell_with_help(mods[i + 1])) if i + 1 < len(mods) else ft.DataCell(ft.Container())
            c3 = ft.DataCell(_cell_with_help(mods[i + 2])) if i + 2 < len(mods) else ft.DataCell(ft.Container())
            rows.append(ft.DataRow(cells=[c1, c2, c3]))
        return ft.DataTable(columns=columns, rows=rows)

    # Blocks for results sections: hidden until real results are computed
    kpi_total_tile = kpi_tile(
        "Total records",
        kpi_total_value,
        icon=getattr(
            ICONS,
            "TABLE_VIEW",
            getattr(ICONS, "LIST", ICONS.SEARCH),
        ),
    )
    kpi_avg_in_tile = kpi_tile(
        "Avg input length",
        kpi_avg_in_value,
        icon=getattr(
            ICONS,
            "TEXT_FIELDS",
            getattr(ICONS, "TEXT_FIELDS_OUTLINED", ICONS.SEARCH),
        ),
    )
    kpi_avg_out_tile = kpi_tile(
        "Avg output length",
        kpi_avg_out_value,
        icon=getattr(
            ICONS,
            "TEXT_FIELDS",
            getattr(ICONS, "TEXT_FIELDS_OUTLINED", ICONS.SEARCH),
        ),
    )
    kpi_dupe_tile = kpi_tile(
        "Duplicates",
        kpi_dupe_value,
        icon=getattr(
            ICONS,
            "CONTENT_COPY",
            getattr(ICONS, "COPY_ALL", ICONS.SEARCH),
        ),
    )
    overview_row = ft.Row(
        [
            kpi_total_tile,
            kpi_avg_in_tile,
            kpi_avg_out_tile,
            kpi_dupe_tile,
        ],
        wrap=True,
        spacing=12,
    )
    overview_block = ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Overview",
                    getattr(
                        ICONS,
                        "DASHBOARD",
                        getattr(ICONS, "INSIGHTS", ICONS.SEARCH),
                    ),
                    "Key KPIs such as total records, average input/output lengths, and duplicate rate (if enabled).",
                    on_help_click=_mk_help_handler(
                        "Key KPIs such as total records, average input/output lengths, and duplicate rate (if enabled).",
                    ),
                ),
                overview_row,
            ],
            spacing=6,
        ),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
        border_radius=8,
        padding=10,
        visible=False,
    )

    sentiment_block = ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Sentiment",
                    getattr(
                        ICONS,
                        "EMOJI_EMOTIONS",
                        getattr(ICONS, "INSERT_EMOTICON", ICONS.SEARCH),
                    ),
                    "Heuristic sentiment distribution computed over sampled records.",
                    on_help_click=_mk_help_handler(
                        "Heuristic sentiment distribution computed over sampled records.",
                    ),
                ),
                sentiment_section,
            ],
            spacing=6,
        ),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
        border_radius=8,
        padding=10,
        visible=False,
    )

    class_balance_block = ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Class balance",
                    getattr(
                        ICONS,
                        "DONUT_SMALL",
                        getattr(ICONS, "PIE_CHART", ICONS.SEARCH),
                    ),
                    "Distribution of labels/classes if present in your dataset.",
                    on_help_click=_mk_help_handler(
                        "Distribution of labels/classes if present in your dataset.",
                    ),
                ),
                class_balance_section,
            ],
            spacing=6,
        ),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
        border_radius=8,
        padding=10,
        visible=False,
    )

    extra_metrics_block = ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Extra metrics",
                    getattr(
                        ICONS,
                        "INSIGHTS",
                        getattr(ICONS, "ANALYTICS", ICONS.SEARCH),
                    ),
                    "Lightweight proxies: coverage overlap, leakage check, depth, speaker balance, Q vs statement, readability, NER proxy, toxicity, politeness, dialogue acts, topics, alignment.",
                    on_help_click=_mk_help_handler(
                        "Lightweight proxies: coverage overlap, leakage check, depth, speaker balance, Q vs statement, readability, NER proxy, toxicity, politeness, dialogue acts, topics, alignment.",
                    ),
                ),
                extra_metrics_section,
            ],
            spacing=6,
        ),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
        border_radius=8,
        padding=10,
        visible=False,
    )

    samples_block = ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Samples",
                    getattr(
                        ICONS,
                        "LIST",
                        getattr(ICONS, "LIST_ALT", ICONS.SEARCH),
                    ),
                    "Random sample rows for quick spot checks (input/output and lengths).",
                    on_help_click=_mk_help_handler(
                        "Random sample rows for quick spot checks (input/output and lengths).",
                    ),
                ),
                samples_section,
            ],
            spacing=6,
        ),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
        border_radius=8,
        padding=10,
        visible=False,
    )

    # Named dividers for each results block (hidden until analysis produces output)
    div_overview = ft.Divider(visible=False)
    div_sentiment = ft.Divider(visible=False)
    div_class = ft.Divider(visible=False)
    div_extra = ft.Divider(visible=False)
    div_samples = ft.Divider(visible=False)

    analysis_tab = build_analysis_tab(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        analyze_btn=analyze_btn,
        analysis_busy_ring=analysis_busy_ring,
        analysis_source_dd=analysis_source_dd,
        analysis_db_session_dd=analysis_db_session_dd,
        analysis_db_refresh_btn=analysis_db_refresh_btn,
        analysis_hf_repo=analysis_hf_repo,
        analysis_hf_split=analysis_hf_split,
        analysis_hf_config=analysis_hf_config,
        analysis_dataset_hint=analysis_dataset_hint,
        select_all_modules_cb=select_all_modules_cb,
        _build_modules_table=_build_modules_table,
        analysis_backend_dd=analysis_backend_dd,
        analysis_hf_token_tf=analysis_hf_token_tf,
        analysis_sample_size_tf=analysis_sample_size_tf,
        analysis_overview_note=analysis_overview_note,
        div_overview=div_overview,
        overview_block=overview_block,
        div_sentiment=div_sentiment,
        sentiment_block=sentiment_block,
        div_class=div_class,
        class_balance_block=class_balance_block,
        div_extra=div_extra,
        extra_metrics_block=extra_metrics_block,
        div_samples=div_samples,
        samples_block=samples_block,
    )

    # Initialize analysis-specific visibility and selection state
    try:
        _update_analysis_source()
    except Exception:
        pass
    try:
        _update_analysis_backend()
    except Exception:
        pass
    try:
        _sync_select_all_modules()
    except Exception:
        pass

    return analysis_tab
