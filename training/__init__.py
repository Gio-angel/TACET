import sys
from pathlib import Path
import config as C
from training.train import train_model, train_model_finetune
from training.visualize import visualize_results

def run() -> bool:
    args = sys.argv[1:]
    
    if args and args[0] == "finetune":
        # Execute your 20k custom sample trainer
        train_result = train_model_finetune()
        # Set explicitly requested folder destination for fine-tuning graphs
        target_plots_dir = C.ARTIFACTS / "plots_finetune"
    else:
        # Standard baseline training execution route
        train_result = train_model()
        # Default destination path
        target_plots_dir = C.ARTIFACTS / "plots"

    if train_result is None:
        print("\n[FAILED] Pipeline run failed — skipping visualization steps.")
        return False

    # Pass the matching targeted paths directly to the visualization engine
    ok = visualize_results(train_result, target_plots_dir=target_plots_dir)
    if not ok:
        print("\n[FAILED] Plots generation engine failed.")
        return False

    print(f"\n[SUCCESS] Pipeline completed successfully.")
    return True

if __name__ == "__main__":
    success = run()
    raise SystemExit(0 if success else 1)