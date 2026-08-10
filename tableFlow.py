import numpy as np
import pandas as pd
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


def save_flow_table(flow, filename="flow_output.tab"):
    flow.to_csv(filename, sep="\t", index=False)


def export_csv_outputs(num_patients=None, output_dir="."):
    manifest_data, cxr_data = generate_synthetic_data(num_patients)

    flow = build_flow_table(manifest_data)
    removal_predictions, hourly_removals = predict_removals_hourly(flow, cxr_data, manifest_data)

    flow_with_time = add_time_since_surgery(flow, manifest_data, "Timestamp")
    cxr_with_time = add_time_since_surgery(cxr_data, manifest_data, "EventDate")

    ct = removal_predictions[["StudyID", "HoursUntilRemoval"]].rename(columns={"HoursUntilRemoval": "TOptimal"})
    ct.to_csv(f"{output_dir}/ct.csv", index=False)

    summary_stats = build_hourly_feature_summary(flow_with_time, removal_predictions)
    summary_stats.to_csv(f"{output_dir}/hourly_feature_summary.csv", index=False)

    auc_over_time = build_auc_over_time(hourly_removals, removal_predictions)
    auc_over_time.to_csv(f"{output_dir}/auc_over_time.csv", index=False)

    force_no_removal_count = int(removal_predictions["ForceNoRemoval"].sum())
    pd.DataFrame({"ForceNoRemovalCount": [force_no_removal_count]}).to_csv(f"{output_dir}/force_no_removal_report.csv", index=False)

    flow_with_time["Timestamp"] = flow_with_time["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    cxr_with_time["EventDate"] = cxr_with_time["EventDate"].dt.strftime("%Y-%m-%d %H:%M")

    save_flow_table(flow_with_time, f"{output_dir}/flow_output.tab")
    removal_predictions.to_csv(f"{output_dir}/removal_predictions.csv", index=False)
    hourly_removals.to_csv(f"{output_dir}/hourly_removal_decisions.csv", index=False)
    cxr_with_time.to_csv(f"{output_dir}/cxr_output.csv", index=False)

    return {
        "manifest": manifest_data,
        "cxr": cxr_data,
        "flow": flow,
        "removal_predictions": removal_predictions,
        "hourly_removals": hourly_removals,
        "flow_with_time": flow_with_time,
        "cxr_with_time": cxr_with_time,
        "summary_stats": summary_stats,
        "auc_over_time": auc_over_time,
    }


def main(num_patients=None):
    export_csv_outputs(num_patients=num_patients)
    print("done")


if __name__ == "__main__":
    main()
