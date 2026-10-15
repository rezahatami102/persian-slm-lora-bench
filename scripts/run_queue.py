import subprocess
import sys
import time
from pathlib import Path

# Each entry is a list of CLI arguments for scripts/train.py
JOBS = [
    ["--model_id", "Qwen/Qwen2.5-1.5B", "--lr", "2e-4",
     "--batch_size", "8", "--grad_accum", "4", "--seed", "44"],
]

Path("logs").mkdir(exist_ok=True)

for i, job in enumerate(JOBS, start=1):
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