import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C
# Import your actual inference loop function and endpointer wrapper
from tacet.infer import run_model_inference, TacetEndpointer

def test_pipeline():
    print("=" * 60)
    print("TACET INFERENCE PIPELINE VERIFICATION")
    print("=" * 60)
    
    print(f"Target Checkpoint:  {C.HEAD_PATH}")
    print(f"Expected Size Dim:  {C.EMB_DIM}")
    print(f"Default Threshold:  {C.TAU}\n")

    if not C.HEAD_PATH.exists():
        print(f"[CRITICAL ERROR] Cannot run test. Your trained weights file does not exist at: {C.HEAD_PATH}")
        return

    print("Generating standard mock embedding vector...")
    mock_bert_vector = np.random.uniform(-1.0, 1.0, C.EMB_DIM).astype(np.float32)

    print("-" * 60)
    print("Executing forward pass through 'run_model_inference'...")
    print("-" * 60)

    try:
        decision = run_model_inference(mock_bert_vector)
        
        endpointer = TacetEndpointer()
        prob = endpointer.probability(mock_bert_vector)
        
        print("\nExecution Finished!")
        print(f"\nPropability Score: {prob:.3f}")
        print(f" -> Turn-Taking Decision: {decision}")
        print("-" * 60)

    except Exception as exc:
        print(f"\n[CRITICAL FAILURE] System crashed during calculation execution thread: {exc}")
        print("This usually means there is a matrix multiplication shape issue inside your model layer.")

if __name__ == "__main__":
    test_pipeline()