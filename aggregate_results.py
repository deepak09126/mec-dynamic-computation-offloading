"""Aggregate seeded experiment JSON files into LaTeX tables and figures."""
import glob
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9})

PROFILES = ["train", "congestion", "unseen", "outage"]
PNAME = {
    "train": "Train",
    "congestion": "Congestion",
    "unseen": "Unseen",
    "outage": "Outage",
}
POLS = ["DQN", "Local-only", "Random", "Latency-aware"]
COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
SEEDS = [1, 2, 3, 4, 5]
H = 30
PH = "--placeholder" in sys.argv


def canon(name):
    name = str(name).lower()
    if "dqn" in name:
        return "DQN"
    if "local" in name:
        return "Local-only"
    if "random" in name:
        return "Random"
    if "latency" in name or "heuristic" in name:
        return "Latency-aware"
    return None


def metric_key(key):
    key = key.lower()
    if any(
        ignored in key
        for ignored in ("std", "sd", "ci", "min", "max", "median", "count", "n_")
    ):
        return None
    if "miss" in key:
        return "miss"
    if "fail" in key:
        return None
    if "latency" in key or "delay" in key:
        return "latency"
    if "energy" in key:
        return "energy"
    if "reward" in key:
        return "reward"
    return None


def metrics(data):
    found = {}
    for key, value in data.items():
        metric = metric_key(key)
        if not metric:
            continue
        if isinstance(value, dict):
            value = value.get("mean")
        if isinstance(value, (int, float)):
            found.setdefault(metric, float(value))
    required = {"latency", "energy", "miss", "reward"}
    return found if required <= set(found) else None


def extract(obj):
    found = {}

    def visit(value, name=None):
        if isinstance(value, dict):
            result = metrics(value)
            policy = (
                canon(value["policy"])
                if isinstance(value.get("policy"), str)
                else (canon(name) if name else None)
            )
            if result and policy:
                found[policy] = result
                return
            for key, child in value.items():
                visit(child, key)
        elif isinstance(value, list):
            for child in value:
                visit(child, name)

    visit(obj)
    return found


def load():
    data = {}
    for profile in PROFILES:
        data[profile] = {
            policy: {metric: [] for metric in ("latency", "energy", "miss", "reward")}
            for policy in POLS
        }
        for seed in SEEDS:
            found = {}
            for path in sorted(glob.glob(f"runs/seed{seed}/{profile}/*.json")):
                try:
                    with open(path, encoding="utf-8") as file:
                        found = extract(json.load(file))
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
                    print(f"Could not read {path}: {error}", file=sys.stderr)
                    continue
                if len(found) == 4:
                    print(
                        f"seed{seed}/{profile}: {os.path.basename(path)} -> "
                        f"{sorted(found)}"
                    )
                    break
            if len(found) != 4:
                raise SystemExit(
                    f"Could not read all 4 policies for seed {seed}, profile "
                    f"{profile}. Check runs/seed{seed}/{profile}/ and edit extract()."
                )
            for policy in POLS:
                for metric in data[profile][policy]:
                    data[profile][policy][metric].append(found[policy][metric])

    all_misses = [
        value
        for profile in data.values()
        for policy in profile.values()
        for value in policy["miss"]
    ]
    if max(all_misses) <= 1.0:
        print("miss rate looks like a fraction -> converting to %")
        for profile in data.values():
            for policy in profile.values():
                policy["miss"] = [100 * value for value in policy["miss"]]
    return {
        profile: {
            policy: {metric: np.array(values) for metric, values in metrics.items()}
            for policy, metrics in profile_data.items()
        }
        for profile, profile_data in data.items()
    }


def mean_sd(values):
    return float(np.mean(values)), float(np.std(values, ddof=1))


def format_value(values, digits=1):
    if PH:
        return "TBD"
    mean, deviation = mean_sd(values)
    return f"${mean:.{digits}f}\\pm{deviation:.{digits}f}$"


def build(data):
    table = [
        r"""\begin{table}[t]
\centering\small
\caption{Evaluation results: mean $\pm$ sample standard deviation over five training seeds (1--5); each run is the mean of 100 evaluation episodes. Latency and energy are simulated estimates; the task-failure rate is 0 for every entry. Local-only, random and latency-aware do not depend on the training seed and are evaluated on the same scenarios in every run, so their deviation reflects only the evaluation seeding.}\label{tab:results}
\begin{tabular}{llrrrr}
\toprule
Profile & Policy & Latency (ms) & Energy (mJ) & Miss rate (\%) & Cum.\ reward\\
\midrule""",
    ]
    for profile_index, profile in enumerate(PROFILES):
        for policy_index, policy in enumerate(POLS):
            lead = (
                f"\\multirow{{4}}{{*}}{{{PNAME[profile]}}}"
                if policy_index == 0
                else ""
            )
            values = data[profile][policy]
            table.append(
                f"{lead} & {policy} & {format_value(values['latency'])} & "
                f"{format_value(values['energy'])} & {format_value(values['miss'])} & "
                f"{format_value(values['reward'], 2)}\\\\"
            )
        table.append("\\midrule" if profile_index < 3 else "\\bottomrule")
    table.append("\\end{tabular}\n\\end{table}\n")

    table.extend(
        [
            r"""\begin{table}[t]
\centering\small
\caption{DQN relative to the latency-aware heuristic (LA) and to local-only (Loc), computed per seed and then summarised as mean $\pm$ SD over the five seeds. Latency and energy as percentage change (negative = DQN lower); miss rate as percentage-point difference.}\label{tab:rel2}
\begin{tabular}{lrrrrrr}
\toprule
 & \multicolumn{3}{c}{vs.\ LA} & \multicolumn{3}{c}{vs.\ Loc}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Profile & Latency (\%) & Energy (\%) & Miss (pp) & Latency (\%) & Energy (\%) & Miss (pp)\\
\midrule""",
        ]
    )
    for profile in PROFILES:
        comparisons = []
        for baseline in ("Latency-aware", "Local-only"):
            result = data[profile]["DQN"]
            base = data[profile][baseline]
            comparisons.extend(
                [
                    format_value(
                        100 * (result["latency"] - base["latency"]) / base["latency"]
                    ),
                    format_value(
                        100 * (result["energy"] - base["energy"]) / base["energy"]
                    ),
                    format_value(result["miss"] - base["miss"], 2),
                ]
            )
        table.append(f"{PNAME[profile]} & " + " & ".join(comparisons) + "\\\\")
    table.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    table.extend(
        [
            r"""\begin{table}[t]
\centering\small
\caption{Cost gap (DQN minus latency-aware; positive = DQN worse) split by reward term, $H=30$; mean $\pm$ SD over five seeds. The total is the negative difference of the cumulative rewards.}\label{tab:gap}
\begin{tabular}{lrrrr}
\toprule
Profile & Latency term & Energy term & Deadline term & Total\\
\midrule""",
        ]
    )
    gaps = {}
    for profile in PROFILES:
        result = data[profile]["DQN"]
        baseline = data[profile]["Latency-aware"]
        latency = H * (result["latency"] - baseline["latency"]) / 150.0
        energy = H * 0.35 * (result["energy"] - baseline["energy"]) / 100.0
        deadline = H * 2.0 * (result["miss"] - baseline["miss"]) / 100.0
        total = -(result["reward"] - baseline["reward"])
        gaps[profile] = (latency, energy, deadline)
        table.append(
            f"{PNAME[profile]} & {format_value(latency, 2)} & "
            f"{format_value(energy, 2)} & {format_value(deadline, 2)} & "
            f"{format_value(total, 2)}\\\\"
        )
    table.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    with open("results_tables.tex", "w", encoding="utf-8") as file:
        file.write("\n".join(table))
    return gaps


def figures(data, gaps):
    for key, ylabel, filename in [
        ("latency", "Mean latency (ms)", "fig_latency.png"),
        ("energy", "Mean energy (mJ)", "fig_energy.png"),
        ("miss", "Deadline-miss rate (%)", "fig_miss.png"),
    ]:
        figure, axis = plt.subplots(figsize=(6.4, 3.1))
        width = 0.2
        for index, policy in enumerate(POLS):
            positions = [x + index * width for x in range(4)]
            if PH:
                axis.bar(positions, [0] * 4, width, label=policy, color=COL[index])
                continue
            values = [mean_sd(data[profile][policy][key]) for profile in PROFILES]
            axis.bar(
                positions,
                [mean for mean, _ in values],
                width,
                yerr=[deviation for _, deviation in values],
                capsize=2,
                label=policy,
                color=COL[index],
                error_kw={"lw": 0.8},
            )
        axis.set_xticks([x + 1.5 * width for x in range(4)])
        axis.set_xticklabels([PNAME[profile] for profile in PROFILES])
        axis.set_ylabel(ylabel)
        if PH:
            axis.text(
                0.5,
                0.5,
                "PLACEHOLDER - run aggregate_results.py",
                transform=axis.transAxes,
                ha="center",
                color="red",
            )
        axis.legend(fontsize=7, ncol=2, frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        figure.savefig(filename, dpi=220)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.8, 2.9))
    width = 0.25
    for index, (name, color) in enumerate(
        [("Latency", "#4c72b0"), ("Energy", "#dd8452"), ("Deadline", "#55a868")]
    ):
        positions = [x + index * width for x in range(4)]
        if PH:
            axis.bar(positions, [0] * 4, width, label=name, color=color)
            continue
        values = [mean_sd(gaps[profile][index]) for profile in PROFILES]
        axis.bar(
            positions,
            [mean for mean, _ in values],
            width,
            yerr=[deviation for _, deviation in values],
            capsize=2,
            label=name,
            color=color,
            error_kw={"lw": 0.8},
        )
    axis.axhline(0, color="k", lw=0.6)
    axis.set_xticks([x + width for x in range(4)])
    axis.set_xticklabels([PNAME[profile] for profile in PROFILES])
    axis.set_ylabel("Cost gap, DQN $-$ LA")
    if PH:
        axis.text(
            0.5,
            0.5,
            "PLACEHOLDER - run aggregate_results.py",
            transform=axis.transAxes,
            ha="center",
            color="red",
        )
    axis.legend(fontsize=7, frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig("fig_gap.png", dpi=220)
    plt.close(figure)


if PH:
    DATA = {
        profile: {
            policy: {metric: np.zeros(5) for metric in ("latency", "energy", "miss", "reward")}
            for policy in POLS
        }
        for profile in PROFILES
    }
else:
    DATA = load()

GAPS = build(DATA)
figures(DATA, GAPS)
print("wrote results_tables.tex, fig_latency/energy/miss/gap.png")
