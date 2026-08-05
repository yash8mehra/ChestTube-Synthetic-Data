import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

GRADES = ["Z", "O", "T", "Th"]


def make_manifest(num_patients=None):
    if num_patients is None:
        num_patients = 100

    study_ids = [f"{i:03d}" for i in range(1, num_patients + 1)]
    all_possible_starts = pd.date_range("2026-01-01", "2026-05-28", freq="h")
    surgery_starts = np.random.choice(all_possible_starts, size=num_patients)
    surgery_starts = pd.Series(surgery_starts).reset_index(drop=True)
    surgery_type = np.random.choice(["VATS", "Open"], size=num_patients, p=[0.7, 0.3])

    duration_hours = np.random.randint(24, 120, size=num_patients)
    long_stay_count = min(max(1, round(num_patients * 0.1)), num_patients)
    long_stay_idx = np.random.choice(num_patients, size=long_stay_count, replace=False)
    duration_hours[long_stay_idx] = np.random.randint(120, 125, size=long_stay_count)

    return pd.DataFrame({
        "StudyID": study_ids,
        "SurgeryStart": surgery_starts,
        "DurationHours": duration_hours.astype(int),
        "SurgeryType": surgery_type,
    })


def generate_cxr(manifest_data):
    all_cxr = []

    for _, row in manifest_data.iterrows():
        studyid = row["StudyID"]
        start = row["SurgeryStart"]
        duration_hrs = row["DurationHours"]

        current = start + pd.Timedelta(hours=4)
        end = start + pd.Timedelta(hours=duration_hrs)
        times = []

        while current < end:
            times.append(current.round("10s"))
            gap_hours = np.random.uniform(12, 168)
            current = current + pd.Timedelta(hours=gap_hours)

        rows = []
        for t in times:
            rows.append({
                "StudyID": studyid,
                "EventDate": t,
                "Effusion": np.random.choice(GRADES),
                "PneumothoraxSize": np.random.choice(GRADES),
                "SubcuEmphysema": np.random.choice(GRADES),
            })

        all_cxr.append(pd.DataFrame(rows))

    cxr = pd.concat(all_cxr, ignore_index=True)
    cxr = cxr.sort_values(by=["StudyID", "EventDate"])
    cxr["EventDate"] = cxr["EventDate"].dt.strftime("%Y-%m-%d %H:%M")
    return cxr


def generate_synthetic_data(num_patients=None):
    manifest_data = make_manifest(num_patients)
    cxr_data = generate_cxr(manifest_data)
    return manifest_data, cxr_data


def make_timestamps(start, duration_hours):
    end = start + pd.Timedelta(hours=duration_hours)
    return pd.date_range(start=start, end=end, freq="10min")


def sample_segment_rates(n):
    air_base = np.zeros(n)
    fluid_base = np.zeros(n)
    readings_per_hour = 6
    i = 0

    while i < n:
        seg_end = min(i + np.random.randint(readings_per_hour, 2 * readings_per_hour + 1), n)
        progress = i / n

        if progress < 0.083:
            air_base[i:seg_end] = np.random.uniform(2.0, 5.0)
            fluid_base[i:seg_end] = np.random.uniform(5, 20)
        elif progress < 0.333:
            air_base[i:seg_end] = np.random.uniform(1.0, 2.0)
            fluid_base[i:seg_end] = np.random.uniform(1, 5)
        else:
            air_base[i:seg_end] = np.random.uniform(0.5, 1.0)
            fluid_base[i:seg_end] = np.random.uniform(0, 1)

        i = seg_end

    return air_base, fluid_base


def make_air_leak(air_base):
    n = len(air_base)
    air_leak = air_base * np.abs(np.random.normal(1.0, 0.4, size=n))
    spike_mask = np.random.random(n) < 0.03
    air_leak[spike_mask] += np.random.exponential(scale=150, size=spike_mask.sum())
    return np.clip(np.round(air_leak, 2), 0, 1000)


def make_fluid_output(fluid_base):
    noise = np.random.normal(0, fluid_base * 0.25 + 0.5, size=fluid_base.shape)
    fluid_output = np.round(fluid_base + noise).astype(int)
    return np.clip(fluid_output, 0, 500)


def make_pressure(n):
    target = np.random.uniform(-1.5, 0.5)
    drift = np.cumsum(np.random.normal(0, 0.01, size=n))
    noise = np.random.normal(0, 0.08, size=n)
    pressure = target + drift + noise
    return np.clip(np.round(pressure, 2), -3.0, 1.5)


def generate_flow(row):
    studyid = row["StudyID"]
    start = row["SurgeryStart"]
    duration = row["DurationHours"]

    timestamps = make_timestamps(start, duration)
    n = len(timestamps)

    air_base, fluid_base = sample_segment_rates(n)
    air_leak = make_air_leak(air_base)
    fluid_output = make_fluid_output(fluid_base)
    pleural_pressure = make_pressure(n)

    return pd.DataFrame({
        "StudyID": studyid,
        "Timestamp": timestamps,
        "MeasuredPleuralPressure": pleural_pressure,
        "AirLeakFlow": air_leak,
        "LOWESSFluidOutput": fluid_output,
    })


def classify_patient_profile(patient_flow, start_time):
    window = patient_flow[
        (patient_flow["Timestamp"] >= start_time) &
        (patient_flow["Timestamp"] < start_time + pd.Timedelta(hours=24))
    ]

    if window.empty:
        return "UNKNOWN"

    avg_air = window["AirLeakFlow"].mean()
    avg_fluid_rate = window["LOWESSFluidOutput"].mean() / 10.0

    if avg_air > 10.0 and avg_fluid_rate > 3.0:
        return "SEVERE"
    if avg_air > 10.0 or avg_fluid_rate > 3.0:
        return "MODERATE"
    return "MILD"


def validate_cxr_for_removal(patient_cxr, removal_time):
    if patient_cxr.empty:
        return True

    patient_cxr = patient_cxr.copy()
    patient_cxr["EventDate"] = pd.to_datetime(patient_cxr["EventDate"])
    before = patient_cxr[patient_cxr["EventDate"] <= removal_time]

    if before.empty:
        return True

    grade_map = {"Z": 0, "O": 1, "T": 2, "Th": 3}
    latest = before.iloc[-1]
    return (
        grade_map.get(latest["Effusion"], 2) <= 1 and
        grade_map.get(latest["PneumothoraxSize"], 2) <= 1
    )


def normalized_flow_limits(window_data):
    air_max = window_data["AirLeakFlow"].max()
    fluid_rate = window_data["LOWESSFluidOutput"].max() / 10.0
    pressure_mean = window_data["MeasuredPleuralPressure"].mean()
    pressure_min = window_data["MeasuredPleuralPressure"].min()
    pressure_ok = pressure_mean < 0.0 and pressure_min < 0.0
    return air_max, fluid_rate, pressure_mean, pressure_ok, air_max <= 10.0 and fluid_rate <= 3.0 and pressure_ok


def hourly_removal_probability(hour):
    if hour < 12:
        return 0.0
    if hour < 120:
        return 0.05
    return 0.0


def predict_removals_hourly(flow, cxr_data, manifest_data):
    records = []
    hourly = []
    flow = flow.copy()
    flow["Timestamp"] = pd.to_datetime(flow["Timestamp"])

    for _, patient in manifest_data.iterrows():
        studyid = patient["StudyID"]
        start_time = patient["SurgeryStart"]
        duration_hours = patient["DurationHours"]

        patient_flow = flow[flow["StudyID"] == studyid]
        patient_cxr = cxr_data[cxr_data["StudyID"] == studyid]
        profile = classify_patient_profile(patient_flow, start_time)
        force_no_removal = duration_hours >= 120

        removal_time = None
        removal_hour = None
        removal_probability_final = 0.0

        for hour in range(1, int(duration_hours) + 1):
            hour_end = start_time + pd.Timedelta(hours=hour)
            window_data = patient_flow[
                (patient_flow["Timestamp"] >= hour_end - pd.Timedelta(hours=8)) &
                (patient_flow["Timestamp"] <= hour_end)
            ]

            if window_data.empty:
                continue

            air_max, fluid_rate, pressure_mean, pressure_ok, meets_criteria = normalized_flow_limits(window_data)
            prob = 0.0 if force_no_removal else hourly_removal_probability(hour)

            cxr_valid = True
            if meets_criteria and hour <= 72:
                cxr_valid = validate_cxr_for_removal(patient_cxr, hour_end)

            removal_decision = bool(
                meets_criteria and
                cxr_valid and
                not removal_time and
                np.random.random() < prob
            )

            hourly.append({
                "StudyID": studyid,
                "Hour": hour,
                "Timestamp": hour_end,
                "AirLeakFlow_Max_8h": air_max,
                "FluidOutput_Max_PerMin_8h": fluid_rate,
                "PressureMean_8h": pressure_mean,
                "PressureOK_8h": pressure_ok,
                "MeetsFlowCriteria": meets_criteria,
                "CXRValid": cxr_valid,
                "RemovalProbability": prob,
                "RemovalDecision": removal_decision,
                "Profile": profile,
            })

            if removal_decision:
                removal_time = hour_end
                removal_hour = hour
                removal_probability_final = prob

        records.append({
            "StudyID": studyid,
            "SurgeryStart": start_time,
            "DurationHours": duration_hours,
            "Profile": profile,
            "RemovalTime": removal_time,
            "RemovalHour": removal_hour,
            "Removed": removal_time is not None,
            "RemovalProbability": removal_probability_final,
            "HoursUntilRemoval": removal_hour if removal_hour else None,
            "ForceNoRemoval": force_no_removal,
        })

    return pd.DataFrame(records), pd.DataFrame(hourly)


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


def build_flow_table(manifest_data):
    return pd.concat([generate_flow(row) for _, row in manifest_data.iterrows()], ignore_index=True).sort_values(
        by=["StudyID", "Timestamp"]
    )


def add_time_since_surgery(dataframe, manifest_data, timestamp_column):
    result = dataframe.copy()
    result = result.merge(manifest_data[["StudyID", "SurgeryStart"]], on="StudyID", how="left")
    result[timestamp_column] = pd.to_datetime(result[timestamp_column])
    result["SurgeryStart"] = pd.to_datetime(result["SurgeryStart"])
    result["HoursSinceSurgery"] = (result[timestamp_column] - result["SurgeryStart"]).dt.total_seconds() / 3600.0
    return result.drop(columns=["SurgeryStart"])


def build_hourly_feature_summary(flow_with_time, removal_predictions):
    feature_cols = ["AirLeakFlow", "LOWESSFluidOutput", "MeasuredPleuralPressure"]
    summary_input = flow_with_time.merge(
        removal_predictions[["StudyID", "Removed"]], on="StudyID", how="left"
    )
    summary_input = summary_input.copy()
    summary_input["Hour"] = np.floor(summary_input["HoursSinceSurgery"]).astype(int)
    summary_input["Group"] = np.where(summary_input["Removed"], "Removed", "NotRemoved")
    long_summary = summary_input.melt(
        id_vars=["Hour", "Group"],
        value_vars=feature_cols,
        var_name="Feature",
        value_name="Value",
    )
    summary_stats = (
        long_summary.groupby(["Hour", "Feature", "Group"], as_index=False)["Value"]
        .agg(Mean="mean", Std="std")
    )
    return summary_stats


def build_auc_over_time(hourly_df, removal_predictions, next_n_hours=24):
    feature_cols = ["AirLeakFlow_Max_8h", "FluidOutput_Max_PerMin_8h"]
    auc_rows = []
    feature_input = hourly_df.merge(
        removal_predictions[["StudyID", "RemovalHour"]], on="StudyID", how="left"
    )

    for hour in sorted(feature_input["Hour"].dropna().unique()):
        hour_frame = feature_input[feature_input["Hour"] == hour].copy()
        if hour_frame.empty:
            continue
        hour_frame["RemovedSoon"] = (
            hour_frame["RemovalHour"].notna()
            & hour_frame["RemovalHour"].between(hour, hour + next_n_hours)
        )
        for feature in feature_cols:
            values = hour_frame[feature].astype(float)
            target = hour_frame["RemovedSoon"].astype(int)
            if values.notna().sum() < 2 or target.sum() < 2 or target.sum() == len(target):
                continue
            try:
                auc = roc_auc_score(target, values)
            except ValueError:
                auc = np.nan
            auc_rows.append({"Hour": hour, "Feature": feature, "AUC": auc})

    return pd.DataFrame(auc_rows)


def plot_summary_and_auc(summary_stats, auc_over_time, output_file="summary_auc_analysis.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if not auc_over_time.empty:
        for feature, group in auc_over_time.groupby("Feature"):
            axes[0].plot(group["Hour"], group["AUC"], marker="o", linewidth=2, label=feature)
        axes[0].axvline(27, color="green", linestyle="--", alpha=0.7, label="Hour 27")
        axes[0].axvline(30, color="red", linestyle="--", alpha=0.7, label="Hour 30")
        axes[0].axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        axes[0].set_xlabel("Hours After Surgery")
        axes[0].set_ylabel("AUC")
        axes[0].set_title("Separation (AUC) Over Time")
        axes[0].set_ylim([0, 1])
        axes[0].grid(alpha=0.3)
        axes[0].legend()

    if not summary_stats.empty:
        air_leak = summary_stats[summary_stats["Feature"] == "AirLeakFlow"]
        for group_name, group in air_leak.groupby("Group"):
            axes[1].plot(group["Hour"], group["Mean"], marker="o", linewidth=2, label=f"{group_name} (mean)")
            axes[1].fill_between(
                group["Hour"],
                group["Mean"] - group["Std"],
                group["Mean"] + group["Std"],
                alpha=0.15,
            )
        axes[1].set_xlabel("Hours After Surgery")
        axes[1].set_ylabel("AirLeakFlow")
        axes[1].set_title("AirLeakFlow: Removed vs Not Removed")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def save_flow_table(flow, filename="flow_output.tab"):
    flow.to_csv(filename, sep="\t", index=False)


def main(num_patients=None):
    manifest_data, cxr_data = generate_synthetic_data(num_patients)

    flow = build_flow_table(manifest_data)
    removal_predictions, hourly_removals = predict_removals_hourly(flow, cxr_data, manifest_data)

    plot_removal_metrics(removal_predictions, hourly_removals)

    flow_with_time = add_time_since_surgery(flow, manifest_data, "Timestamp")
    cxr_with_time = add_time_since_surgery(cxr_data, manifest_data, "EventDate")

    ct = removal_predictions[["StudyID", "HoursUntilRemoval"]].rename(columns={"HoursUntilRemoval": "TOptimal"})
    ct.to_csv("ct.csv", index=False)

    summary_stats = build_hourly_feature_summary(flow_with_time, removal_predictions)
    summary_stats.to_csv("hourly_feature_summary.csv", index=False)

    auc_over_time = build_auc_over_time(hourly_removals, removal_predictions)
    auc_over_time.to_csv("auc_over_time.csv", index=False)

    plot_summary_and_auc(summary_stats, auc_over_time)

    force_no_removal_count = int(removal_predictions["ForceNoRemoval"].sum())
    pd.DataFrame({"ForceNoRemovalCount": [force_no_removal_count]}).to_csv("force_no_removal_report.csv", index=False)

    flow_with_time["Timestamp"] = flow_with_time["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    cxr_with_time["EventDate"] = cxr_with_time["EventDate"].dt.strftime("%Y-%m-%d %H:%M")

    save_flow_table(flow_with_time, "flow_output.tab")
    removal_predictions.to_csv("removal_predictions.csv", index=False)
    hourly_removals.to_csv("hourly_removal_decisions.csv", index=False)
    cxr_with_time.to_csv("cxr_output.csv", index=False)

print("done")
if __name__ == "__main__":
    main()