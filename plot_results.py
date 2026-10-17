
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Folder containing evaluation JSON files
ARTIFACTS = Path("artifacts")

# Evaluation conditions
PROFILES = ["train", "congestion", "unseen"]

# Policies being compared
POLICIES = ["dqn", "local_only", "random", "latency_aware"]

# Metrics to plot
METRICS = [
    (
        "average_latency_ms",
        "Average latency (ms)",
        "latency_comparison.png",
    ),
    (
        "average_estimated_energy_mj",
        "Estimated energy (mJ)",
        "energy_comparison.png",
    ),
    (
        "deadline_miss_rate",
        "Deadline miss rate",
        "deadline_miss_comparison.png",
    ),
]


def load_results():
    """Load evaluation results from JSON files."""
    results = {}

    for profile in PROFILES:
        path = ARTIFACTS / f"evaluation_{profile}.json"

        if not path.exists():
            print(f"WARNING: Missing file: {path}")
            continue

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        # Evaluation metrics are nested under the "results" key.
        profile_results = data.get("results")

        if not isinstance(profile_results, dict):
            print(f"WARNING: Unexpected JSON structure in {path}")
            continue

        missing = [
            policy
            for policy in POLICIES
            if policy not in profile_results
        ]

        if missing:
            print(
                f"WARNING: Missing policies in {path}: "
                f"{', '.join(missing)}"
            )
            continue

        results[profile] = profile_results
        print(f"Loaded: {path}")

    if not results:
        raise SystemExit(
            "No valid evaluation results found. "
            "Check that the JSON files exist in artifacts/."
        )

    return results


def create_graph(results, metric, ylabel, output_filename):
    """Create and save a comparison chart."""
    profiles = list(results.keys())
    x = np.arange(len(profiles))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, policy in enumerate(POLICIES):
        values = [
            results[profile][policy][metric]
            for profile in profiles
        ]

        positions = x + (i - 1.5) * width

        bars = ax.bar(
            positions,
            values,
            width,
            label=policy.replace("_", " ").title(),
        )

        # Add values above bars for readability.
        if metric == "deadline_miss_rate":
            labels = [f"{value:.1%}" for value in values]
        else:
            labels = [f"{value:.1f}" for value in values]

        ax.bar_label(
            bars,
            labels=labels,
            padding=3,
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [profile.title() for profile in profiles]
    )
    ax.set_ylabel(ylabel)
    ax.set_title(f"Policy Comparison: {ylabel}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()

    output_path = ARTIFACTS / output_filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved graph: {output_path}")


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    results = load_results()

    for metric, ylabel, filename in METRICS:
        create_graph(
            results=results,
            metric=metric,
            ylabel=ylabel,
            output_filename=filename,
        )

    print("\nAll graphs generated successfully!")
    print("Note: latency and energy values are simulated MEC estimates.")
    print(f"Open the graphs folder with: explorer {ARTIFACTS}")


if __name__ == "__main__":
    main()
