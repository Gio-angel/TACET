import csv
import sys
import time
from pathlib import Path

# Setup root path matching your original project structure
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # Allows imports of tacet modules relative to execution path

import config as C
from tacet.encoder import BertEncoder
from tacet.infer import TacetEndpointer
from tacet.gate import should_respond

ROOT_DIR = HERE.parent  

BERT_DIR = ROOT_DIR / "models" / "bert"
DEFAULT_CSV = ROOT_DIR / "artifacts" / "simulation.csv"

def fetch_csv(file=DEFAULT_CSV):
    """Reads evaluation rows from a targeted CSV spreadsheet path

    and yields sanitized text data paired with an expected classification label.
    """
    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation file artifact at: {path}")

    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        # DictReader maps row indices directly to 'text' and 'label' string headers
        reader = csv.DictReader(f)

        # Confirm the column schema is present
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise KeyError("Your CSV must contain 'text' and 'label' headers.")

        for row in reader:
            dataset.append(row)

    return dataset


def print_status(msg):
    """Clean, synchronous status printer."""
    print(f"[*] {msg}")


def run_batch_evaluation(dataset, encoder, endpointer):
    """Iterates through the text lines, feeds them into the BERT-TACET
    pipeline, and calculates explicit structural breakdown performance.
    """
    print(f"\n[*] Instantiating execution loop across {len(dataset)} items...")
    print("=" * 65)

    correct_predictions = 0
    total_valid_samples = 0
    
    # Initialize the specific metric tracking states
    correct_finished = 0      # Model=1, Truth=1
    correct_unfinished = 0    # Model=0, Truth=0
    incorrect_finished = 0    # Model=1, Truth=0
    incorrect_unfinished = 0  # Model=0, Truth=1

    # Dictionary to store accuracy statistics per word count:
    # Structure: { word_count: { "complete_total": 0, "complete_correct": 0, "incomplete_total": 0, "incomplete_correct": 0 } }
    len_stats = {}

    for idx, row in enumerate(dataset, 1):
        text_line = row["text"].strip()
        raw_label = row["label"].strip()

        if not text_line or not raw_label:
            print(f"[!] Skipped Row #{idx}: Text or Label field is empty.")
            continue

        try:
            ground_truth = int(raw_label)
        except ValueError:
            print(f"[!] Skipped Row #{idx}: Label must be 0 or 1 (got '{raw_label}')")
            continue

        total_valid_samples += 1
        
        # Calculate word count for this sentence
        word_len = len(text_line.split())
        if word_len not in len_stats:
            len_stats[word_len] = {
                "complete_total": 0, "complete_correct": 0,
                "incomplete_total": 0, "incomplete_correct": 0
            }

        start_time = time.monotonic()

        # Step A: Transform raw text into standard 768-d [CLS] space
        embedding = encoder.encode(text_line)

        # Step B: Score via local TACET classification head
        probability = endpointer.probability(embedding)

        # Step C: Threshold check (TAU)
        decision = should_respond(probability)
        model_decision = 1 if decision else 0

        latency = (time.monotonic() - start_time) * 1000

        # Evaluate Correctness metrics
        is_correct = model_decision == ground_truth
        if is_correct:
            correct_predictions += 1

        # Populate performance tracking metrics grouped by ground truth class
        if ground_truth == 1:
            len_stats[word_len]["complete_total"] += 1
            if is_correct:
                correct_finished += 1
                len_stats[word_len]["complete_correct"] += 1
            else:
                incorrect_unfinished += 1 # Predicted unfinished (0), actually finished (1)
        else:
            len_stats[word_len]["incomplete_total"] += 1
            if is_correct:
                correct_unfinished += 1
                len_stats[word_len]["incomplete_correct"] += 1
            else:
                incorrect_finished += 1   # Predicted finished (1), actually unfinished (0)

        match_marker = "✅ MATCH" if is_correct else "❌ MISMATCH"

        # Log details per sentence pass
        print(f"Sample #{idx} (Words: {word_len})")
        print(f" ├── Input Text   : '{text_line}'")
        print(f" ├── P(Complete)  : {probability:.4f} (Threshold: {C.TAU})")
        print(f" ├── Output/Truth : Model={model_decision} | Baseline={ground_truth}")
        print(f" └── Status       : {match_marker} ({latency:.1f} ms)")
        print("-" * 50)

    # Print Final Summary Metrics
    if total_valid_samples > 0:
        accuracy = (correct_predictions / total_valid_samples) * 100
        print("\n" + "=" * 65)
        print("📊 TACET PERFORMANCE METRICS SUMMARY")
        print(f" └── Processed Dataset Size  : {total_valid_samples} rows")
        print(f" └── Total Correct Decisions : {correct_predictions} rows")
        
        print(f" ├── Total Correctly Classified   : Finished (1) = {correct_finished} | Unfinished (0) = {correct_unfinished}")
        print(f" ├── Total Incorrectly Classified : Finished (1) = {incorrect_finished} | Unfinished (0) = {incorrect_unfinished}")
        print(f" └── Calculated Overall Accuracy : {accuracy:.2f}%")
        print("=" * 65)
        
        # Print the detailed word-length breakdown table
        print("\n📊 WORD ACCURACY BREAKDOWN TABLE")
        print("-" * 52)
        print(f"{'Word Length':^13} | {'Complete Accuracy':^17} | {'Incomplete Accuracy':^17}")
        print("-" * 52)
        
        # Sort keys to present word counts sequentially
        for length in sorted(len_stats.keys()):
            stats = len_stats[length]
            
            # Compute complete percentages safely
            if stats["complete_total"] > 0:
                comp_pct = (stats["complete_correct"] / stats["complete_total"]) * 100
                comp_str = f"{comp_pct:>6.1f}% ({stats['complete_correct']}/{stats['complete_total']})"
            else:
                comp_str = "  N/A  "

            # Compute incomplete percentages safely    
            if stats["incomplete_total"] > 0:
                incomp_pct = (stats["incomplete_correct"] / stats["incomplete_total"]) * 100
                incomp_str = f"{incomp_pct:>6.1f}% ({stats['incomplete_correct']}/{stats['incomplete_total']})"
            else:
                incomp_str = "  N/A  "
                
            print(f"{length:^13} | {comp_str:^17} | {incomp_str:^17}")
        print("-" * 52)
        
    else:
        print("[!] No valid records were successfully evaluated.")


def main():
    print("--- TACET Terminal Batch Processing Pipeline ---")

    try:
        # 1. Fetch data from artifacts subfolder
        print_status("Checking for validation dataset file...")
        dataset = fetch_csv()
        print_status(f"Imported {len(dataset)} records from CSV storage.")

        # 2. Initialize and eagerly load the models locally
        print_status("Initializing BertEncoder...")
        encoder = BertEncoder(model_dir=BERT_DIR)
        encoder.ensure_loaded(status=print_status)

        print_status("Initializing TacetEndpointer...")
        endpointer = TacetEndpointer()
        print_status("Local pipeline modules fully online.\n" + "=" * 40)

        # 3. Process the validation dataset batch
        run_batch_evaluation(dataset, encoder, endpointer)

    except FileNotFoundError as fnf:
        print(f"\n[ERROR] File missing error: {fnf}")
    except Exception as e:
        print(f"\n[FATAL] Pipeline cracked unexpectedly: {e}")


if __name__ == "__main__":
    main()