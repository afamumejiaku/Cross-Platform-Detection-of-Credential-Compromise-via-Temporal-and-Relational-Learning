import os
import json
import traceback
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


# =============================================================================
# USER CONFIG (SET THIS)
# =============================================================================
OUT_DIR = "Detection_Output"
MODELS = ["fta_lstm", "fta_gru", "lstm", "gru", "gnn", "ifcnn_tpp", "cep3", "gcn", "gat"]
SETTINGS = ["per_platform", "cross_platform"]

STAGE3_DIR = os.path.join(OUT_DIR, "stage3")
DELAY_DIR = os.path.join(STAGE3_DIR, "delay")
BENEFIT_DIR = os.path.join(STAGE3_DIR, "benefit")
REPORT_PATH = os.path.join(STAGE3_DIR, "summary_report.json")
SUMMARY_TABLE_CSV = os.path.join(STAGE3_DIR, "summary_table.csv")

os.makedirs(DELAY_DIR, exist_ok=True)
os.makedirs(BENEFIT_DIR, exist_ok=True)


# =============================================================================
# SAFE HELPERS
# =============================================================================
def _exists_file(path: str) -> bool:
    if not path or not os.path.exists(path):
        print(f"[WARNING] Missing file: {path}")
        return False
    return True

def _exists_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        print(f"[WARNING] Missing directory: {path}")
        return False
    return True

def _safe_json_load(path: str) -> Optional[Dict[str, Any]]:
    if not _exists_file(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to read JSON {path}: {e}")
        return None

def _safe_json_dump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def _safe_run(step_name: str, fn, **kwargs) -> Tuple[bool, Optional[Any]]:
    print(f"\n--- {step_name} ---")
    try:
        out = fn(**kwargs)
        print(f"[OK] {step_name}")
        return True, out
    except Exception as e:
        print(f"[ERROR] {step_name}: {e}")
        traceback.print_exc()
        return False, None

def _fmt(x, nd=4):
    if x is None:
        return "N/A"
    try:
        if isinstance(x, (int, float, np.floating)) and np.isnan(x):
            return "N/A"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def _run_dir(setting: str, model: str) -> str:
    return os.path.join(OUT_DIR, f"{setting}__{model}")

def _results_json_path(setting: str, model: str) -> str:
    return os.path.join(_run_dir(setting, model), f"results_{setting}_{model}.json")

def _predictions_csv_path(setting: str, model: str) -> str:
    return os.path.join(_run_dir(setting, model), "predictions_test.csv")

def load_stage2_run_index() -> List[Dict[str, Any]]:
    master = os.path.join(OUT_DIR, "all_runs_summary.json")
    data = _safe_json_load(master)
    if isinstance(data, list) and data:
        ok = [r for r in data if r.get("status") == "ok" and r.get("out_dir")]
        print(f"[INFO] Loaded {len(ok)} successful runs from {master}")
        return ok

    runs = []
    if _exists_dir(OUT_DIR):
        for name in os.listdir(OUT_DIR):
            if "__" in name:
                full = os.path.join(OUT_DIR, name)
                if os.path.isdir(full):
                    s, m = name.split("__", 1)
                    runs.append({"setting": s, "model": m, "out_dir": full, "status": "unknown"})
    print(f"[INFO] Scanned {len(runs)} run dirs from OUT_DIR")
    return runs

def get_predictions_jobs(setting: str, model: str) -> List[Dict[str, Any]]:
    res = _safe_json_load(_results_json_path(setting, model))
    jobs = []

    if not res:
        pred = _predictions_csv_path(setting, model)
        if _exists_file(pred):
            jobs.append({
                "tag": f"{setting}__{model}",
                "predictions_csv": pred,
                "threshold": 0.5,
                "output_json": os.path.join(DELAY_DIR, f"{setting}__{model}.json"),
            })
        return jobs

    if setting == "per_platform":
        pred_map = res.get("per_platform_predictions", {}) or {}
        for plat, info in pred_map.items():
            jobs.append({
                "tag": f"{setting}__{model}__{plat}",
                "predictions_csv": info.get("predictions_csv"),
                "threshold": float(info.get("threshold", 0.5)),
                "output_json": os.path.join(DELAY_DIR, f"{setting}__{model}__{plat}.json"),
            })
        return jobs

    # cross_platform
    jobs.append({
        "tag": f"{setting}__{model}",
        "predictions_csv": res.get("predictions_csv") or _predictions_csv_path(setting, model),
        "threshold": float(res.get("threshold", 0.5)),
        "output_json": os.path.join(DELAY_DIR, f"{setting}__{model}.json"),
    })
    return jobs


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None

def compute_detection_delay_from_predictions(predictions_path: str, threshold: float) -> Dict[str, Any]:
   df = pd.read_csv(predictions_path)
    y_col = _pick_col(df, ["y_true", "label", "target", "y", "gt", "ground_truth"])
    s_col = _pick_col(df, ["y_score", "score", "prob", "proba", "y_pred_proba", "pred_prob", "prediction", "pred"])
    t_col = _pick_col(df, ["AttackTime", "attack_time", "time", "t", "Timestamp", "timestamp"])
    e_col = _pick_col(df, ["Email", "email", "user", "userid", "uid"])

    if y_col is None or s_col is None:
        raise ValueError(
            f"Predictions CSV missing required columns. Found columns: {list(df.columns)}. "
            f"Need y_col in {['y_true','label','target','y']} and s_col in {['y_score','prob','pred_prob','prediction']}."
        )

    y = df[y_col].astype(int).values
    s = df[s_col].astype(float).values
    pred = (s >= float(threshold)).astype(int)

    # time axis
    if t_col is None:
        t = np.arange(len(df), dtype=float)
        time_type = "index"
    else:
        series = df[t_col]
        try:
            tt = pd.to_datetime(series, errors="raise")
            t = tt.view("int64") / 1e9  # seconds
            time_type = "datetime"
        except Exception:
            try:
                t = pd.to_numeric(series, errors="raise").astype(float).values
                time_type = "numeric"
            except Exception:
                t = np.arange(len(df), dtype=float)
                time_type = "index"

    def delay_for_slice(idx: np.ndarray) -> Optional[float]:
        yy = y[idx]
        pp = pred[idx]
        tt = np.asarray(t)[idx]

        pos_idx = np.where(yy == 1)[0]
        if len(pos_idx) == 0:
            return None
        t_first_pos = tt[pos_idx].min()

        tp_idx = np.where((yy == 1) & (pp == 1))[0]
        if len(tp_idx) == 0:
            return None
        t_first_detect = tt[tp_idx].min()

        return float(t_first_detect - t_first_pos)

    delays = []
    detected_count = 0
    total_with_positive = 0

    if e_col is not None:
        for _, g in df.groupby(e_col, sort=False):
            idx = g.index.values
            if (y[idx] == 1).any():
                total_with_positive += 1
                d = delay_for_slice(idx)
                if d is not None:
                    detected_count += 1
                    delays.append(d)
    else:
        if (y == 1).any():
            total_with_positive = 1
            d = delay_for_slice(np.arange(len(df)))
            if d is not None:
                detected_count = 1
                delays.append(d)

    out = {
        "predictions_path": predictions_path,
        "threshold": float(threshold),
        "time_axis": time_type,
        "n_rows": int(len(df)),
        "n_positive_rows": int((y == 1).sum()),
        "n_predicted_positive_rows": int((pred == 1).sum()),
        "n_entities_with_positive": int(total_with_positive),
        "n_entities_detected": int(detected_count),
        "detection_rate_entities": float(detected_count / total_with_positive) if total_with_positive else None,
        "delay_mean": float(np.mean(delays)) if delays else None,
        "delay_median": float(np.median(delays)) if delays else None,
        "delay_p90": float(np.quantile(delays, 0.90)) if delays else None,
        "delay_min": float(np.min(delays)) if delays else None,
        "delay_max": float(np.max(delays)) if delays else None,
    }
    return out

def extract_primary_metrics(results_json: Dict[str, Any], setting: str) -> Dict[str, Any]:
    """
    Supports:
      A) Your updated stage2:
         - cross_platform: results_json["metrics"] = {accuracy,f1,auroc,auprc,threshold,...}
         - per_platform: results_json["metrics"] contains AVERAGES + best_* + n_platforms
      B) Older temp.py-like:
         - metrics_test: {f1, roc_auc, ...}, threshold at top-level
    """
    out: Dict[str, Any] = {"setting": setting}

    # New schema
    if isinstance(results_json.get("metrics"), dict):
        m = results_json["metrics"]
        out.update({
            "accuracy": m.get("accuracy"),
            "f1": m.get("f1"),
            "auroc": m.get("auroc"),
            "auprc": m.get("auprc"),
            "threshold": m.get("threshold", results_json.get("threshold")),
        })
        # per-platform extras (if present)
        out.update({
            "n_platforms": m.get("n_platforms"),
            "best_accuracy": m.get("best_accuracy"),
            "best_accuracy_platform": m.get("best_accuracy_platform"),
            "best_f1": m.get("best_f1"),
            "best_f1_platform": m.get("best_f1_platform"),
            "best_auroc": m.get("best_auroc"),
            "best_auroc_platform": m.get("best_auroc_platform"),
            "best_auprc": m.get("best_auprc"),
            "best_auprc_platform": m.get("best_auprc_platform"),
        })
        return {k: v for k, v in out.items() if v is not None}

    # Old schema
    mtest = results_json.get("metrics_test") or results_json.get("metrics") or {}
    out.update({
        "threshold": results_json.get("threshold"),
        "f1": mtest.get("f1"),
        "auroc": mtest.get("roc_auc") or mtest.get("auroc"),
        "auprc": mtest.get("pr_auc") or mtest.get("auprc"),
        "accuracy": mtest.get("accuracy"),
        "precision": mtest.get("precision"),
        "recall": mtest.get("recall"),
    })
    return {k: v for k, v in out.items() if v is not None}


def compute_cross_platform_benefit(model: str) -> Dict[str, Any]:
    """
    Compare cross_platform vs per_platform for a given model using results JSONs.
    - If per_platform contains averaged metrics, deltas will be computed against those averages.
    """
    per_path = _results_json_path("per_platform", model)
    cross_path = _results_json_path("cross_platform", model)

    per = _safe_json_load(per_path)
    cross = _safe_json_load(cross_path)

    if not per or not cross:
        return {
            "model": model,
            "status": "missing_inputs",
            "per_platform_results_json": per_path,
            "cross_platform_results_json": cross_path,
        }

    per_m = extract_primary_metrics(per, setting="per_platform")
    cross_m = extract_primary_metrics(cross, setting="cross_platform")

    def delta(k):
        if k in per_m and k in cross_m:
            try:
                return float(cross_m[k]) - float(per_m[k])
            except Exception:
                return None
        return None

    benefit = {
        "model": model,
        "status": "ok",
        "per_platform": per_m,
        "cross_platform": cross_m,
        "delta": {
            "accuracy": delta("accuracy"),
            "f1": delta("f1"),
            "auroc": delta("auroc"),
            "auprc": delta("auprc"),
        }
    }
    return benefit

def run_all_detection_delay():
    runs = load_stage2_run_index()
    seen = set()

    for r in runs:
        setting = r.get("setting")
        model = r.get("model")
        if setting not in SETTINGS or model not in MODELS:
            continue
        key = (setting, model)
        if key in seen:
            continue
        seen.add(key)

        jobs = get_predictions_jobs(setting, model)
        if not jobs:
            print(f"[SKIP] No prediction jobs found for {setting}__{model}")
            continue

        for j in jobs:
            pred_csv = j["predictions_csv"]
            if not _exists_file(pred_csv):
                print(f"[SKIP] Missing predictions for {j['tag']}")
                continue

            ok, metrics = _safe_run(
                f"Detection Delay: {j['tag']}",
                compute_detection_delay_from_predictions,
                predictions_path=pred_csv,
                threshold=float(j["threshold"]),
            )
            if ok and metrics is not None:
                _safe_json_dump(metrics, j["output_json"])


def run_all_benefits():
    for m in MODELS:
        out_json = os.path.join(BENEFIT_DIR, f"benefit__{m}.json")
        ok, benefit = _safe_run(f"Cross-Platform Benefit: {m}", compute_cross_platform_benefit, model=m)
        if ok and benefit is not None:
            _safe_json_dump(benefit, out_json)


def generate_summary_report() -> Dict[str, Any]:
    """
    Aggregates:
      - benefits per model
      - delay stats per run (from stage3/delay/*.json)
      - stage2 extracted metrics for table printing
    """
    report: Dict[str, Any] = {"out_dir": OUT_DIR, "models": MODELS, "benefits": {}, "delays": {}, "stage2_metrics": {}}

    # benefits
    for m in MODELS:
        p = os.path.join(BENEFIT_DIR, f"benefit__{m}.json")
        report["benefits"][m] = _safe_json_load(p) or {"status": "missing", "path": p}

    # delays
    if _exists_dir(DELAY_DIR):
        for fn in sorted(os.listdir(DELAY_DIR)):
            if fn.endswith(".json"):
                p = os.path.join(DELAY_DIR, fn)
                report["delays"][fn.replace(".json", "")] = _safe_json_load(p) or {"status": "missing", "path": p}

    # stage2 metrics
    for setting in SETTINGS:
        for m in MODELS:
            p = _results_json_path(setting, m)
            res = _safe_json_load(p)
            key = f"{setting}__{m}"
            if res:
                report["stage2_metrics"][key] = extract_primary_metrics(res, setting=setting)
            else:
                report["stage2_metrics"][key] = {"status": "missing", "path": p}

    # leaderboard by delta f1 if available
    leaderboard = []
    for m, b in report["benefits"].items():
        d = (b.get("delta") or {}).get("f1") if isinstance(b, dict) else None
        if d is not None:
            leaderboard.append((m, d))
    leaderboard.sort(key=lambda x: x[1], reverse=True)
    report["leaderboard_by_delta_f1"] = [{"model": m, "delta_f1": d} for m, d in leaderboard]

    return report


# =============================================================================
# SUMMARY TABLE PRINTING (requested)
# =============================================================================
def _read_delay_for_tag(tag: str, report_delays: Dict[str, Any]) -> Dict[str, Any]:
    d = report_delays.get(tag)
    if not isinstance(d, dict):
        return {}
    return {
        "det_rate_entities": d.get("detection_rate_entities"),
        "delay_mean": d.get("delay_mean"),
        "delay_median": d.get("delay_median"),
        "delay_p90": d.get("delay_p90"),
    }

def _aggregate_per_platform_delay(report_delays: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Aggregate per-platform delay JSONs:
      keys look like: per_platform__{model}__{platform}
    Return mean of means for delays, and mean det_rate_entities across platforms.
    """
    prefix = f"per_platform__{model}__"
    rows = []
    for k, v in report_delays.items():
        if k.startswith(prefix) and isinstance(v, dict) and v.get("predictions_path"):
            rows.append(v)

    if not rows:
        return {}

    det_rates = [r.get("detection_rate_entities") for r in rows if r.get("detection_rate_entities") is not None]
    mean_delays = [r.get("delay_mean") for r in rows if r.get("delay_mean") is not None]
    med_delays = [r.get("delay_median") for r in rows if r.get("delay_median") is not None]
    p90_delays = [r.get("delay_p90") for r in rows if r.get("delay_p90") is not None]

    return {
        "det_rate_entities": float(np.mean(det_rates)) if det_rates else None,
        "delay_mean": float(np.mean(mean_delays)) if mean_delays else None,
        "delay_median": float(np.mean(med_delays)) if med_delays else None,
        "delay_p90": float(np.mean(p90_delays)) if p90_delays else None,
        "n_delay_platforms": int(len(rows)),
    }

def print_summary_table(report: Dict[str, Any]) -> pd.DataFrame:
    """
    Prints a clean summary table across:
      - per_platform (averages + best scores if present)
      - cross_platform
    Adds delay metrics:
      - cross_platform uses its delay json directly
      - per_platform uses mean across per-platform delay jsons (if available)
    """
    stage2 = report.get("stage2_metrics", {}) or {}
    delays = report.get("delays", {}) or {}
    benefits = report.get("benefits", {}) or {}

    rows = []
    for setting in SETTINGS:
        for m in MODELS:
            key = f"{setting}__{m}"
            m2 = stage2.get(key, {})
            if not isinstance(m2, dict):
                m2 = {}

            row = {
                "setting": setting,
                "model": m,
                "accuracy": m2.get("accuracy"),
                "f1": m2.get("f1"),
                "auroc": m2.get("auroc"),
                "auprc": m2.get("auprc"),
                "threshold": m2.get("threshold"),
            }

            # per-platform extra columns (from stage2 aggregates, if available)
            row.update({
                "n_platforms_trained": m2.get("n_platforms"),
                "best_accuracy": m2.get("best_accuracy"),
                "best_accuracy_platform": m2.get("best_accuracy_platform"),
                "best_f1": m2.get("best_f1"),
                "best_f1_platform": m2.get("best_f1_platform"),
            })

            # delays
            if setting == "cross_platform":
                row.update(_read_delay_for_tag(key, delays))
            else:
                row.update(_aggregate_per_platform_delay(delays, model=m))

            # benefit deltas (for convenience on each row)
            b = benefits.get(m, {})
            if isinstance(b, dict) and b.get("status") == "ok":
                d = b.get("delta") or {}
                row.update({
                    "delta_accuracy": d.get("accuracy"),
                    "delta_f1": d.get("f1"),
                    "delta_auroc": d.get("auroc"),
                    "delta_auprc": d.get("auprc"),
                })
            else:
                row.update({
                    "delta_accuracy": None,
                    "delta_f1": None,
                    "delta_auroc": None,
                    "delta_auprc": None,
                })

            rows.append(row)

    df = pd.DataFrame(rows)

    # Order rows: cross_platform first within each model, then per_platform
    setting_order = {"cross_platform": 0, "per_platform": 1}
    df["setting_rank"] = df["setting"].map(setting_order).fillna(99).astype(int)
    df = df.sort_values(["model", "setting_rank"], ascending=[True, True]).drop(columns=["setting_rank"])

    # Print
    print("\n" + "=" * 140)
    print("SUMMARY TABLE (Stage2 + Stage3)")
    print(" - cross_platform: metrics from the global model; delays from cross_platform__model delay JSON")
    print(" - per_platform:   metrics are (if available) averaged in Stage2 results; delays are averaged across platforms")
    print("=" * 140)MODELS = ["fta_lstm", "fta_gru", "lstm", "gru", "gnn", "gcn", "gat"] 

    cols = [
        "model", "setting",
        "accuracy", "f1", "auroc", "auprc", "threshold",
        "n_platforms_trained", "best_accuracy", "best_accuracy_platform", "best_f1", "best_f1_platform",
        "det_rate_entities", "delay_mean", "delay_median", "delay_p90",
        "delta_accuracy", "delta_f1", "delta_auroc", "delta_auprc",
    ]

    # ensure columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = None

    # pretty-print with formatting (avoid scientific notation)
    df_print = df[cols].copy()
    for c in ["accuracy","f1","auroc","auprc","threshold","det_rate_entities","delay_mean","delay_median","delay_p90",
              "delta_accuracy","delta_f1","delta_auroc","delta_auprc","best_accuracy","best_f1"]:
        if c in df_print.columns:
            df_print[c] = df_print[c].apply(lambda x: _fmt(x, nd=4))

    print(df_print.to_string(index=False))
    print("\n" + "-" * 140)
    print(f"Saved CSV: {SUMMARY_TABLE_CSV}")
    df.to_csv(SUMMARY_TABLE_CSV, index=False)

    # Quick “best” views
    def _best(df0: pd.DataFrame, setting: str, col: str) -> Optional[pd.Series]:
        sub = df0[df0["setting"] == setting].copy()
        sub = sub[pd.to_numeric(sub[col], errors="coerce").notna()]
        if sub.empty:
            return None
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        return sub.sort_values(col, ascending=False).iloc[0]

    best_cross_acc = _best(df, "cross_platform", "accuracy")
    best_per_avg_acc = _best(df, "per_platform", "accuracy")

    print("\nBEST RUNS (by Accuracy):")
    if best_cross_acc is not None:
        print(f"  cross_platform best: model={best_cross_acc['model']} ACC={_fmt(best_cross_acc['accuracy'])} F1={_fmt(best_cross_acc['f1'])}")
    else:
        print("  cross_platform best: N/A")

    if best_per_avg_acc is not None:
        print(f"  per_platform best (avg): model={best_per_avg_acc['model']} ACC={_fmt(best_per_avg_acc['accuracy'])} F1={_fmt(best_per_avg_acc['f1'])}")
        if pd.notna(best_per_avg_acc.get("best_accuracy_platform")):
            print(f"    best platform for that model: {best_per_avg_acc.get('best_accuracy_platform')} "
                  f"(best_acc={_fmt(best_per_avg_acc.get('best_accuracy'))})")
    else:
        print("  per_platform best (avg): N/A")

    return df


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("STAGE 3 (SELF-CONTAINED) — updated for models:")
    print("  " + ", ".join(MODELS))
    print(f"Stage2 OUT_DIR: {OUT_DIR}")
    print(f"Stage3 OUT:     {STAGE3_DIR}")
    print("=" * 80)

    run_all_detection_delay()
    run_all_benefits()

    ok, rep = _safe_run("Summary Report", generate_summary_report)
    if ok and rep is not None:
        _safe_json_dump(rep, REPORT_PATH)
        print(f"[OK] Saved report: {REPORT_PATH}")

        # print the requested summary table
        _ = print_summary_table(rep)


if __name__ == "__main__":
    main()
