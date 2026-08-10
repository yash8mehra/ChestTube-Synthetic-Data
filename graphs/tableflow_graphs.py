from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tableFlow import (
    add_time_since_surgery,
    build_flow_table,
    build_hourly_feature_summary,
    generate_synthetic_data,
    predict_removals_hourly,
)


# Create a multi-panel chart showing patient removal trends, probability distribution, and outcome summaries.
def plot_removal_metrics(removal_df, hourly_df, output_file="removal_analysis.png"):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    if not hourly_df.empty:
        hourly_decisions = hourly_df.groupby("Hour")["RemovalDecision"].sum()
        axes[0, 0].plot(hourly_decisions.index, hourly_decisions.values, marker="o", linewidth=2, color="steelblue")
        axes[0, 0].axvline(27, color="green", linestyle="--", alpha=0.7, label="Hour 27")
        axes[0, 0].axvline(30, color="red", linestyle="--", alpha=0.7, label="Hour 30")
        axes[0, 0].set_xlabel("Hours After Surgery")
        axes[0, 0].set_ylabel("Number of Removals")
        axes[0, 0].set_title("Chest Tube Removals per Hour")
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].legend()

    axes[0, 1].hist(removal_df["RemovalProbability"], bins=20, color="coral", alpha=0.7, edgecolor="black")
    axes[0, 1].set_xlabel("Removal Probability")
    axes[0, 1].set_ylabel("Number of Patients")
    axes[0, 1].set_title("Distribution of Removal Probabilities")
    axes[0, 1].grid(axis="y", alpha=0.3)

    removal_counts = removal_df["Removed"].value_counts()
    axes[0, 2].pie(
        removal_counts.values,
        labels=["Not Removed", "Removed"],
        autopct="%1.1f%%",
        colors=["#ff9999", "#90ee90"],
        startangle=90,
    )
    axes[0, 2].set_title("Overall Removal Rate")

    if not hourly_df.empty:
        hourly_prob = hourly_df.groupby("Hour")["RemovalProbability"].mean()
        axes[1, 0].plot(hourly_prob.index, hourly_prob.values, marker="o", linewidth=2, color="purple", markersize=6)
        axes[1, 0].axvline(27, color="green", linestyle="--", alpha=0.7, label="Hour 27")
        axes[1, 0].axvline(30, color="red", linestyle="--", alpha=0.7, label="Hour 30")
        axes[1, 0].set_xlabel("Hours After Surgery")
        axes[1, 0].set_ylabel("Average Removal Probability")
        axes[1, 0].set_title("Removal Probability Trend")
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].legend()

    if "Profile" in removal_df.columns:
        profile_removal = removal_df.groupby("Profile")["Removed"].agg(["sum", "count"])
        profile_removal["rate"] = profile_removal["sum"] / profile_removal["count"]
        axes[1, 1].bar(profile_removal.index, profile_removal["rate"], color=["#ff6b6b", "#ffd93d", "#6bcf7f"], alpha=0.7)
        axes[1, 1].set_ylabel("Removal Rate")
        axes[1, 1].set_title("Removal Rate by Patient Profile")
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(axis="y", alpha=0.3)
        for i, v in enumerate(profile_removal["rate"]):
            axes[1, 1].text(i, v + 0.02, f"{v:.2%}", ha="center")

    if "ForceNoRemoval" in removal_df.columns:
        long_stay_counts = removal_df["ForceNoRemoval"].value_counts()
        axes[1, 2].bar(["<120h", "≥120h"], [long_stay_counts.get(False, 0), long_stay_counts.get(True, 0)], color=["steelblue", "orange"], alpha=0.7)
        axes[1, 2].set_ylabel("Number of Patients")
        axes[1, 2].set_title("Patient Stay Duration")
        axes[1, 2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


# Plot hourly air-leak summary values for removed vs. non-removed patients.
def plot_summary_stats(summary_stats, output_file="summary_feature_analysis.png"):
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    if not summary_stats.empty:
        air_leak = summary_stats[summary_stats["Feature"] == "AirLeakFlow"]
        for group_name, group in air_leak.groupby("Group"):
            axes.plot(group["Hour"], group["Mean"], marker="o", linewidth=2, label=f"{group_name} (mean)")
            axes.fill_between(
                group["Hour"],
                group["Mean"] - group["Std"],
                group["Mean"] + group["Std"],
                alpha=0.15,
            )
        axes.set_xlabel("Hours After Surgery")
        axes.set_ylabel("AirLeakFlow")
        axes.set_title("AirLeakFlow: Removed vs Not Removed")
        axes.grid(alpha=0.3)
        axes.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


# Generate the synthetic data, build the relevant tables, and save all graph outputs in a target directory.
def generate_graph_outputs(num_patients=None, output_dir=None):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_data, cxr_data = generate_synthetic_data(num_patients)
    flow = build_flow_table(manifest_data)
    removal_predictions, hourly_removals = predict_removals_hourly(flow, cxr_data, manifest_data)

    plot_removal_metrics(removal_predictions, hourly_removals, output_dir / "removal_analysis.png")

    flow_with_time = add_time_since_surgery(flow, manifest_data, "Timestamp")
    summary_stats = build_hourly_feature_summary(flow_with_time, removal_predictions)
    plot_summary_stats(summary_stats, output_dir / "summary_feature_analysis.png")

    return {
        "manifest": manifest_data,
        "cxr": cxr_data,
        "flow": flow,
        "removal_predictions": removal_predictions,
        "hourly_removals": hourly_removals,
        "summary_stats": summary_stats,
    }


# Run the graph-generation workflow from the command line.
def main(num_patients=None):
    generate_graph_outputs(num_patients=num_patients)


if __name__ == "__main__":
    main()
