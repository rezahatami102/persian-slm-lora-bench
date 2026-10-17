import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Fewer examples -> more epochs so each run gets a comparable number of optimizer steps
EPOCHS = {0.1: 12, 0.25: 6, 0.5: 4}

# Seeds are the outer loop: after the first pass (seed 42) you already have a
# usable low-data curve; later seeds only add statistical precision.
JOBS = [
    ["--model_id", model, "--lr", "2e-4", "--train_frac", str(frac),
     "--epochs", str(EPOCHS[frac]), "--batch_size", "8", "--grad_accum", "4",
     "--seed", str(seed)]
    for seed in (42, 43, 44)
    for model in ("Qwen/Qwen2.5-0.5B", "google/gemma-3-1b-pt")
    for frac in (0.1, 0.25, 0.5)
]

STOP_FILE = Path("STOP")  # create this file to stop cleanly after the current job
RESULTS = Path("results/results.jsonl")


def job_key(argv):
    """Identify a job by the settings that define a run (defaults mirror train.py)."""
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=1.0)
    a, _ = p.parse_known_args(argv)
    return (a.model_id, a.lora_r, a.lr, a.epochs, a.seed, a.train_frac)


def finished_keys():
    """Keys of all runs that already have a record in results.jsonl."""
    keys = set()
    if RESULTS.exists():
        with open(RESULTS, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                keys.add((r["model_id"], r["lora_r"], r["lr"], r["epochs"],
                          r["seed"], r.get("train_frac", 1.0)))
    return keys


Path("logs").mkdir(exist_ok=True)
done = finished_keys()

for i, job in enumerate(JOBS, start=1):
    if STOP_FILE.exists():
        print("STOP file found: stopping before the next job.", flush=True)
        break
    if job_key(job) in done:
        print(f"[{i}/{len(JOBS)}] already finished, skipping", flush=True)
        continue
    log_path = Path("logs") / f"job{i}_{int(time.time())}.log"
    print(f"[{i}/{len(JOBS)}] starting: {' '.join(job)}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        code = subprocess.call(
            [sys.executable, "scripts/train.py", *job],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    print(f"[{i}/{len(JOBS)}] finished with exit code {code} (log: {log_path})", flush=True)
    # Continue with the next job even if this one failed (e.g. OOM)