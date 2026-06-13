# src/main.py
import os
import sys
import time
import argparse
import yaml
import traceback
from typing import Callable, Dict, List, Tuple

# -----------------------------
# Config loader
# -----------------------------
def load_config() -> Tuple[str, dict]:
    cfg_path = os.getenv("CONFIG_PATH", "config/pipeline_config.yaml")
    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[ERROR] CONFIG_PATH points to missing file: {cfg_path}", file=sys.stderr)
        sys.exit(2)
    os.environ["CONFIG_PATH"] = cfg_path
    return cfg_path, cfg

# -----------------------------
# Missing-step helper
# -----------------------------
def _missing(module_file: str, func_sig: str, err_msg: str):
    msg = (
        f"[ERROR] Missing or invalid step: need {module_file} with a public function {func_sig}\n"
        f"        Import error: {err_msg}"
    )
    print(msg, file=sys.stderr)
    sys.exit(3)

# -----------------------------
# Step registry (imports are lazy & error-wrapped)
# -----------------------------
def _load_step_funcs() -> Dict[str, Callable[[], None]]:
    steps: Dict[str, Callable[[], None]] = {}

    # data_fetch
    try:
        from .data_source import data_fetch as _data_fetch
        steps["data_fetch"] = _data_fetch
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        steps["data_fetch"] = (lambda err_msg=err_msg: _missing("data_source.py", "data_fetch()", err_msg))

    # data_extract
    try:
        from .extractor import data_extract as _data_extract
        steps["data_extract"] = _data_extract
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        steps["data_extract"] = (lambda err_msg=err_msg: _missing("extractor.py", "data_extract()", err_msg))

    # data_process (processor.py)
    try:
        from .processor import data_process as _process
        steps["data_process"] = _process
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        steps["data_process"] = (lambda err_msg=err_msg: _missing("processor.py", "data_process()", err_msg))

    # data_model (data_model.py)
    try:
        from .data_model import build_data_model as _build_data_model
        steps["data_model"] = _build_data_model
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        steps["data_model"] = (lambda err_msg=err_msg: _missing("data_model.py", "build_data_model()", err_msg))

    # data_query (uses env: PIPELINE_SQL or SQL, optional PIPELINE_OUT)
    try:
        from .data_model import run_duckdb_query as _run_duckdb_query
        def _dq():
            sql = os.getenv("PIPELINE_SQL") or os.getenv("SQL")
            out = os.getenv("PIPELINE_OUT")
            if not sql:
                raise ValueError("data_query requires SQL via env var PIPELINE_SQL (or SQL).")
            _run_duckdb_query(sql, out)
        steps["data_query"] = _dq
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        steps["data_query"] = (lambda err_msg=err_msg: _missing("data_model.py", "run_duckdb_query(sql, out=None)", err_msg))

    # graph_construct
    try:
        from .graph_constructor import graph_construct as _graph_construct
        steps["graph_construct"] = _graph_construct
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        steps["graph_construct"] = (lambda err_msg=err_msg: _missing("graph_constructor.py", "graph_construct()", err_msg))

    # graph_ai_model
    try:
        from .graph_ai import graph_ai_model as _graph_ai_model
        steps["graph_ai_model"] = _graph_ai_model
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        steps["graph_ai_model"] = (lambda err_msg=err_msg: _missing("graph_ai.py", "graph_ai_model()", err_msg))

    # post_analysis
    try:
        from .post_analysis import run_post_analysis as _post
        steps["post_analysis"] = _post
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        steps["post_analysis"] = (lambda err_msg=err_msg: _missing("post_analysis.py", "run_post_analysis()", err_msg))

    return steps

DEFAULT_ORDER = [
    "data_fetch",
    "data_extract",
    "data_process",
    "data_model",
    "graph_construct",
    "graph_ai_model",
    "post_analysis",
]

# -----------------------------
# CLI & plan resolution
# -----------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GAIPO pipeline driver")
    p.add_argument("--call", help="Single step or comma-list, e.g. data_fetch,data_extract")
    p.add_argument("--all", action="store_true", help=f"Run: {', '.join(DEFAULT_ORDER)}")
    p.add_argument("--until", help=f"Run default sequence up to this step")
    p.add_argument("--continue-on-error", action="store_true", help="Do not stop on first failing step")
    p.add_argument("--debug", action="store_true", help="Show full Python tracebacks on errors")
    return p.parse_args()

def _resolve_plan(args: argparse.Namespace) -> List[str]:
    if args.all:
        return DEFAULT_ORDER.copy()
    if args.until:
        if args.until not in DEFAULT_ORDER:
            print(f"[ERROR] --until must be one of: {', '.join(DEFAULT_ORDER)}", file=sys.stderr)
            sys.exit(2)
        return DEFAULT_ORDER[: DEFAULT_ORDER.index(args.until) + 1]
    if args.call:
        return [s.strip() for s in args.call.split(",") if s.strip()]
    print("[ERROR] Specify one of: --call, --all, or --until", file=sys.stderr)
    sys.exit(2)

# -----------------------------
# Step runner (with optional tracebacks)
# -----------------------------
def _run_step(name: str, fn: Callable[[], None], *, debug: bool) -> Tuple[bool, str]:
    print(f"\n===== RUN {name} =====")
    t0 = time.time()
    status = "OK"
    try:
        fn()
    except SystemExit as se:
        status = f"FAIL (SystemExit {se.code})"
        if debug:
            print(traceback.format_exc(), file=sys.stderr)
        else:
            print(f"[ERROR] {name} raised SystemExit({se.code})", file=sys.stderr)
    except Exception as e:
        status = f"FAIL ({type(e).__name__}: {e})"
        if debug:
            print(traceback.format_exc(), file=sys.stderr)
        else:
            print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
            print("  (run with --debug to see full traceback)", file=sys.stderr)
    dt = time.time() - t0
    print(f"===== DONE {name} [{status}] in {dt:.1f}s =====")
    return (status == "OK"), status

# -----------------------------
# Entry point
# -----------------------------
def main():
    cfg_path, _cfg = load_config()
    args = _parse_args()
    step_funcs = _load_step_funcs()
    plan = _resolve_plan(args)

    unknown = [s for s in plan if s not in step_funcs]
    if unknown:
        print(f"[ERROR] Unknown step(s): {', '.join(unknown)}", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Using config: {cfg_path}")
    print(f"[INFO] Plan: {' -> '.join(plan)}")

    start_all = time.time()
    any_fail = False

    for step in plan:
        fn = step_funcs[step]
        ok, _ = _run_step(step, fn, debug=args.debug)
        if not ok:
            any_fail = True
            if not args.continue_on_error:
                print(f"\nTotal: {time.time()-start_all:.1f}s")
                sys.exit(1)

    print(f"\nTotal: {time.time()-start_all:.1f}s")
    sys.exit(0 if not any_fail else 1)

if __name__ == "__main__":
    main()