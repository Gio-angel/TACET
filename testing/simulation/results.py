import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "testing" / "simulation" / "metrics.csv"


def _float_or_none(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean(values):
    return statistics.mean(values) if values else None


def _done_offset(row):
    for key in ("gt_user_done_offset", "user_done_offset", "actual_done_offset"):
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _decision_offset(row):
    value = _float_or_none(row.get("decision_offset"))
    if value is not None:
        return value

    decision_time = _float_or_none(row.get("decision_time"))
    turn_start_time = _float_or_none(row.get("turn_start_time"))
    if decision_time is not None and turn_start_time is not None:
        return decision_time - turn_start_time
    return None


def load_rows(path=DEFAULT_CSV):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"metrics CSV not found at: {path}")

    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def calculate_metrics(rows):
    logged_turns = len(rows)
    auto_rows = [row for row in rows if row.get("decision") == "auto_response"]

    bert_calls = [
        int(float(row["bert_calls"]))
        for row in rows
        if _float_or_none(row.get("bert_calls")) is not None
    ]

    annotated_auto_rows = []
    wait_after_done = []
    false_interruptions = 0

    for row in auto_rows:
        done_offset = _done_offset(row)
        decision_offset = _decision_offset(row)
        if done_offset is None or decision_offset is None:
            continue

        annotated_auto_rows.append(row)
        wait = decision_offset - done_offset
        wait_after_done.append(wait)
        if wait < 0:
            false_interruptions += 1

    annotated_count = len(annotated_auto_rows)
    false_interruption_rate = None
    if annotated_count:
        false_interruption_rate = false_interruptions / annotated_count

    return {
        "logged_turns": logged_turns,
        "auto_turns": len(auto_rows),
        "annotated_auto_turns": annotated_count,
        "false_interruptions": false_interruptions,
        "false_interruption_rate": false_interruption_rate,
        "average_wait_after_done": _mean(wait_after_done),
        "bert_calls_per_turn": _mean(bert_calls),
    }


def print_metrics(metrics):
    print("=" * 60)
    print("TACET TURN-TAKING PERFORMANCE")
    print("=" * 60)
    print(f"Logged turns:          {metrics['logged_turns']}")
    print(f"Auto-response turns:   {metrics['auto_turns']}")
    print(f"BERT calls per turn:   {_format_number(metrics['bert_calls_per_turn'])}")

    if metrics["annotated_auto_turns"] == 0:
        print("False interruption:    unavailable (add gt_user_done_offset)")
        print("Avg wait after done:   unavailable (add gt_user_done_offset)")
        return

    rate = metrics["false_interruption_rate"] * 100
    print(
        "False interruption:    "
        f"{rate:.2f}% ({metrics['false_interruptions']}/{metrics['annotated_auto_turns']})"
    )
    print(f"Avg wait after done:   {_format_number(metrics['average_wait_after_done'])} sec")


def _format_number(value):
    if value is None:
        return "unavailable"
    return f"{value:.3f}"


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    rows = load_rows(path)
    metrics = calculate_metrics(rows)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
