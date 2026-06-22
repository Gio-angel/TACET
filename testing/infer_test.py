import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C
from tacet.infer import run_model_inference, TacetEndpointer   # probability
from tacet.gate import should_respond                          #  decision


def test_pipeline():
    print("=" * 60)
    print("TACET INFERENCE PIPELINE VERIFICATION")
    print("=" * 60)
    print(f"Target Checkpoint:  {C.HEAD_PATH}")
    print(f"Expected Size Dim:  {C.EMB_DIM}")
    print(f"Default Threshold:  {C.TAU}\n")

    if not C.HEAD_PATH.exists():
        print(f"[CRITICAL ERROR] weights file not found at: {C.HEAD_PATH}")
        return

    print("Generating standard mock embedding vector...")
    mock_bert_vector = np.random.uniform(-1.0, 1.0, C.EMB_DIM).astype(np.float32)

    print("-" * 60)
    print("Stage 4 (inference) -> probability, then gate -> decision")
    print("-" * 60)
    try:
        prob = run_model_inference(mock_bert_vector)     # probability only
        decision = should_respond(prob)                  # gate applied separately

        print("\nExecution Finished!")
        print(f"\nProbability Score:        {prob:.3f}")
        print(f" -> Turn-Taking Decision: {decision}")
        print("-" * 60)
    except Exception as exc:
        print(f"\n[CRITICAL FAILURE] crashed during execution: {exc}")


if __name__ == "__main__":
    test_pipeline()
