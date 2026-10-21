"""Run five independent training seeds and evaluate each on all profiles."""
import pathlib
import shutil
import subprocess
import sys
import time

SEEDS = [1, 2, 3, 4, 5]
PROFILES = ["train", "congestion", "unseen", "outage"]
EPISODES = 500
EVAL_EPISODES = 100
EVAL_SEED = 100007

artifacts = pathlib.Path("artifacts")
artifacts.mkdir(exist_ok=True)


def run(arguments):
    print(">>", " ".join(arguments), flush=True)
    subprocess.run([sys.executable, "experiment.py", *arguments], check=True)


for seed in SEEDS:
    checkpoint = artifacts / f"dqn_seed{seed}.pt"
    if not checkpoint.exists():
        run(
            [
                "train",
                "--episodes",
                str(EPISODES),
                "--seed",
                str(seed),
                "--checkpoint",
                str(checkpoint),
            ]
        )
    for profile in PROFILES:
        started = time.time()
        run(
            [
                "evaluate",
                "--checkpoint",
                str(checkpoint),
                "--episodes",
                str(EVAL_EPISODES),
                "--seed",
                str(EVAL_SEED),
                "--profile",
                profile,
            ]
        )
        output = pathlib.Path("runs") / f"seed{seed}" / profile
        output.mkdir(parents=True, exist_ok=True)
        for path in artifacts.rglob("*"):
            if (
                path.is_file()
                and path.suffix in (".json", ".csv")
                and path.stat().st_mtime >= started - 1
            ):
                shutil.copy2(path, output / path.name)

print("done: now run python aggregate_results.py")
