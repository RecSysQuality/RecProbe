import argparse
import pandas as pd
from io.loader import load_dataset
from io.saver import save_dataset
from noise.orchestrator import NoiseOrchestrator
from config import load_config

def main():
    parser = argparse.ArgumentParser(description="Noise Injector for Reviews")
    parser.add_argument("--input", required=True, help="Path to input CSV/JSONL")
    parser.add_argument("--output", required=True, help="Path to save noisy dataset")
    parser.add_argument("--config", required=False, help="Optional YAML config file")
    parser.add_argument("--level", choices=["easy","medium","complex"], default="medium")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Load dataset
    df = load_dataset(args.input)

    # 2. Load config or default profile
    config = load_config(args.config) if args.config else {
        "level": args.level,
        "metadata": {"enabled": True, "intensity": 0.1},
        "aggregation": {"enabled": True, "intensity": 0.05},
        "instrumentation": {"enabled": True, "intensity": 0.0},
        "seed": args.seed
    }

    # 3. Create orchestrator e applica rumore
    orchestrator = NoiseOrchestrator(config)
    df_noisy = orchestrator.apply(df)

    # 4. Salva output
    save_dataset(df_noisy, args.output)
    print(f"Noisy dataset saved to {args.output}")

if __name__ == "__main__":
    main()
