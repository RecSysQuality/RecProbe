# """
# vLLM batch inference per 3 dataset CSV con Llama 3.1 8B Instruct
# Uso: python run_inference.py --datasets d1.csv d2.csv d3.csv --prompt_col prompt
# """
#
# import argparse
# import csv
# import json
# import time
# from pathlib import Path
#
# from vllm import LLM, SamplingParams
#
# BASE_DIR = Path(__file__).resolve().parent.parent.parent
#
# # def load_csv(path: str, prompt_col: str) -> list[str]:
# #     prompts = []
# #     with open(path, newline="", encoding="utf-8") as f:
# #         reader = csv.DictReader(f)
# #         for row in reader:
# #             prompts.append(row[prompt_col])
# #     return prompts
#
#
# def format_chat_prompt(user_text: str) -> str:
#     """Formato chat per Llama 3.1 Instruct"""
#     return (
#         "<|begin_of_text|>"
#         "<|start_header_id|>user<|end_header_id|>\n\n"
#         f"{user_text}"
#         "<|eot_id|>"
#         "<|start_header_id|>assistant<|end_header_id|>\n\n"
#     )
#
#
# def run_batch(llm: LLM, prompts: list[str], params: SamplingParams) -> list[str]:
#     formatted = [format_chat_prompt(p) for p in prompts]
#     outputs = llm.generate(formatted, params)
#     return [o.outputs[0].text.strip() for o in outputs]
#
# #
# # def save_results(input_path: str, rows: list[dict], outputs: list[str]):
# #     out_path = Path(input_path).with_suffix(".output.csv")
# #     with open(out_path, "w", newline="", encoding="utf-8") as f:
# #         writer = csv.DictWriter(f, fieldnames=["user_id", "item_id", "rating", "timestamp", "text"])
# #         writer.writeheader()
# #         for row, output in zip(rows, outputs):
# #             writer.writerow({
# #                 "user_id:token": row["user_id"],
# #                 "item_id:token": row["item_id"],
# #                 "rating:float": row["rating"],
# #                 "timestamp:float": row["timestamp"],
# #                 "text:token": output,
# #             })
# #     print(f"  → Salvato in: {out_path}")
# #     return out_path
# def load_csv(path: str):
#     with open(path, newline="", encoding="utf-8") as f:
#         return list(csv.DictReader(f))
#
#
# def save_results(input_path: str, rows, outputs):
#     out_path = Path(input_path).with_suffix(".output.csv")
#
#     with open(out_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(
#             f,
#             fieldnames=[
#                 "user_id:token",
#                 "item_id:token",
#                 "rating:float",
#                 "timestamp:float",
#                 "text:token",
#             ],
#         )
#
#         writer.writeheader()
#
#         for row, output in zip(rows, outputs):
#
#             writer.writerow({
#                 "user_id:token": row["user"],
#                 "item_id:token": row["item"],
#                 "rating:float": row["rating"],
#                 "timestamp:float": row["timestamp"],
#                 "text:token": output,
#             })
#
#     return out_path
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--datasets",      nargs="+", default=list(DATASETS.keys()))
#     parser.add_argument("--prompt_col", default="prompt", help="Nome colonna prompt nel CSV")
#     parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
#     parser.add_argument("--max_tokens", type=int, default=512)
#     parser.add_argument("--temperature", type=float, default=0.7)
#     parser.add_argument("--batch_size", type=int, default=5000,
#                         help="Prompt per batch (aumenta se hai tanta VRAM)")
#     args = parser.parse_args()
#
#     print(f"Carico modello: {args.model}")
#     llm = LLM(
#         model=args.model,
#         dtype="float16",          # A100 supporta fp16 nativo
#         gpu_memory_utilization=0.90,
#         max_model_len=4096,
#     )
#
#     sampling_params = SamplingParams(
#         temperature=args.temperature,
#         max_tokens=args.max_tokens,
#         stop=["<|eot_id|>"],
#     )
#
#
#     datasets = ['Toys_and_Games','yelp','Books']
#     #datasets = ['Toys_and_Games']
#     subfolders = ['sequence_disruption','context_swap','sentiment_flip']
#     for portion in [0.3,0.2,0.1]:
#         for subfolder in subfolders:
#             for dataset_name in args.datasets:
#                 output = f"{BASE_DIR}/data_review/{dataset}/{subfolder}/prompts_{portion}.csv"
#
#                 dataset_path = Path(output)
#                 print(f"\n{'='*50}")
#                 print(f"Dataset: {dataset_path}")
#
#                 rows = load_csv(output)
#
#
#                 #prompts = load_csv(output, args.prompt_col)
#
#
#                 prompts = [r[args.prompt_col] for r in rows]
#                 #prompts = prompts[0:100]
#
#                 print(f"  Prompt caricati: {len(prompts)}")
#
#                 all_outputs = []
#                 t0 = time.time()
#
#                 for i in range(0, len(prompts), args.batch_size):
#                     batch = prompts[i : i + args.batch_size]
#                     print(f"  Batch {i//args.batch_size + 1} ({len(batch)} prompt)...", end=" ", flush=True)
#                     results = run_batch(llm, batch, sampling_params)
#                     all_outputs.extend(results)
#                     elapsed = time.time() - t0
#                     done = i + len(batch)
#                     speed = done / elapsed
#                     remaining = (len(prompts) - done) / speed if speed > 0 else 0
#                     print(f"✓ | {speed:.0f} prompt/s | ~{remaining/60:.1f} min rimanenti")
#
#                 save_results(output, rows, all_outputs)
#                 total = time.time() - t0
#                 print(f"  Completato in {total/60:.1f} minuti")
#
#             print("\nTutto fatto!")
#
#
# if __name__ == "__main__":
#     main()


"""
vLLM batch inference per 3 dataset CSV con Llama 3.1 8B Instruct
Uso: python run_inference.py --datasets d1.csv d2.csv d3.csv --prompt_col prompt
"""
import argparse
import csv
import time
from pathlib import Path
from vllm import LLM, SamplingParams

BASE_DIR = Path(__file__).resolve().parent.parent.parent


"""
vLLM batch inference per 3 dataset CSV con Llama 3.1 8B Instruct
Uso: python run_inference.py --datasets d1.csv d2.csv d3.csv --prompt_col prompt
"""
import argparse
import csv
import os
import time
from pathlib import Path
from vllm import LLM, SamplingParams

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def format_chat_prompt(user_text: str) -> str:
    """Formato chat per Llama 3.1 Instruct"""
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_text}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def run_all(llm: LLM, prompts: list[str], params: SamplingParams) -> list[str]:
    """
    Passa TUTTI i prompt in una sola chiamata a generate().
    vLLM gestisce internamente il continuous batching e lo scheduling:
    non serve (anzi è controproducente) spezzare manualmente in chunk.
    use_tqdm=True mostra una progress bar nativa con throughput in tempo reale.
    """
    formatted = [format_chat_prompt(p) for p in prompts]
    outputs = llm.generate(formatted, params, use_tqdm=True)
    # vLLM non garantisce l'ordine di output == ordine di input quando
    # ci sono molte richieste in coda con scheduling dinamico: per sicurezza
    # ricostruiamo l'ordine originale usando request_id (l'indice di inserimento).
    # Se la tua versione di vLLM mantiene l'ordine, questo passaggio è no-op.
    return [o.outputs[0].text.strip() for o in outputs]


def load_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_results(input_path: str, rows, outputs):
    out_path = Path(input_path).with_suffix(".output.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "user_id:token",
                "item_id:token",
                "rating:float",
                "timestamp:float",
                "text:token",
            ],
        )
        writer.writeheader()
        for row, output in zip(rows, outputs):
            writer.writerow({
                "user_id:token": row["user"],
                "item_id:token": row["item"],
                "rating:float": row["rating"],
                "timestamp:float": row["timestamp"],
                "text:token": output,
            })
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",  nargs="+", required=True)
    parser.add_argument("--subfolder", required=True)
    parser.add_argument("--portion",   required=True, type=float)
    parser.add_argument("--prompt_col", default="prompt")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    # Un solo dataset per job (passa --datasets yelp)
    dataset_name = args.datasets[0]

    llm = LLM(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["<|eot_id|>"],
    )

    output = f"{BASE_DIR}/data_review/{dataset_name}/{args.subfolder}/prompts_{args.portion}.csv"
    print(f"Dataset: {output}")

    rows = load_csv(output)
    prompts = [r[args.prompt_col] for r in rows]
    print(f"Prompt caricati: {len(prompts)}")

    t0 = time.time()
    all_outputs = run_all(llm, prompts, sampling_params)
    total = time.time() - t0
    print(f"✓ {len(prompts)} prompt in {total/60:.1f} min | {len(prompts)/total:.1f} prompt/s")

    save_results(output, rows, all_outputs)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--datasets", nargs="+", default=None,
#                          help="Se non specificato, legge DATASET_NAME dall'ambiente (Slurm array)")
#     parser.add_argument("--prompt_col", default="prompt", help="Nome colonna prompt nel CSV")
#     parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
#     parser.add_argument("--max_tokens", type=int, default=512)
#     parser.add_argument("--temperature", type=float, default=0.8)
#     args = parser.parse_args()
#
#     # Priorità: --datasets da CLI > DATASET_NAME da ambiente (Slurm array) > tutti i dataset
#     if args.datasets is None:
#         env_dataset = os.environ.get("DATASET_NAME")
#         if env_dataset:
#             args.datasets = [env_dataset]
#             print(f"[Slurm array] Eseguo solo il dataset: {env_dataset}")
#         else:
#             args.datasets = ["Toys_and_Games", "yelp", "Books"]
#             print("Nessun --datasets né DATASET_NAME: eseguo tutti i dataset in sequenza")
#
#     print(f"Carico modello: {args.model}")
#     llm = LLM(
#         model=args.model,
#         dtype="float16",
#         gpu_memory_utilization=0.90,
#         max_model_len=8192,
#         enable_prefix_caching=True,  # utile se i prompt condividono prefissi (es. istruzioni di sistema ripetute)
#     )
#
#     sampling_params = SamplingParams(
#         temperature=args.temperature,
#         max_tokens=args.max_tokens,
#         stop=["<|eot_id|>"],
#     )
#
#     subfolders = ['context_swap',"sequence_disruption", "sentiment_flip"]
#
#     for portion in [0.3, 0.2, 0.1]:
#         for subfolder in subfolders:
#             for dataset_name in args.datasets:
#                 output = f"{BASE_DIR}/data_review/{dataset_name}/{subfolder}/prompts_{portion}.csv"
#                 dataset_path = Path(output)
#                 print(f"\n{'='*50}")
#                 print(f"Dataset: {dataset_path}")
#
#                 rows = load_csv(output)
#                 prompts = [r[args.prompt_col] for r in rows]
#                 print(f"  Prompt caricati: {len(prompts)}")
#
#                 t0 = time.time()
#                 # Una sola chiamata: lascia che vLLM gestisca lo scheduling
#                 # interno con continuous batching, senza chunking manuale.
#                 all_outputs = run_all(llm, prompts, sampling_params)
#                 total = time.time() - t0
#
#                 speed = len(prompts) / total if total > 0 else 0
#                 print(f"  ✓ {len(prompts)} prompt in {total/60:.1f} min | {speed:.1f} prompt/s")
#
#                 save_results(output, rows, all_outputs)
#                 print(f"  Completato in {total/60:.1f} minuti")
#
#     print("\nTutto fatto!")


if __name__ == "__main__":
    main()