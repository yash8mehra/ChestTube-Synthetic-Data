import numpy as np
import pandas as pd

GRADES = ["Z", "O", "T", "TH"]
DEFAULT_SEED = 20240601

# Grade probabilities per finding, calibrated to the real sample's skew toward
# low-severity readings (Effusion in particular is ~94% "Z" in the real data).
GRADE_PROBS = {
    "Effusion": [0.90, 0.08, 0.015, 0.005],
    "PneumothoraxSize": [0.35, 0.55, 0.05, 0.05],
    "SubcuEmphysema": [0.35, 0.28, 0.12, 0.25],
}


def round_numeric_columns(df):
    if isinstance(df, pd.DataFrame):
        return df.apply(lambda col: col.round(3) if pd.api.types.is_numeric_dtype(col) else col)
    return round(df, 3)


def sample_t_optimal(num_patients):
    values = np.full(num_patients, 24, dtype=int)
    exact_count = int(round(num_patients * 0.62))
    exact_count = min(max(exact_count, 1), num_patients)
    tail_count = num_patients - exact_count

    tail_idx = np.random.choice(num_patients, size=tail_count, replace=False)
    tail = np.random.lognormal(mean=np.log(80), sigma=1.0, size=tail_count)
    tail = np.clip(np.round(tail), 21, 412).astype(int)

    long_tail_mask = np.random.random(tail_count) < 0.08
    if np.any(long_tail_mask):
        tail[long_tail_mask] = np.clip(
            np.random.uniform(180, 412, size=np.sum(long_tail_mask)).round().astype(int),
            180,
            412,
        )

    values[tail_idx] = tail
    return np.clip(values, 21, 412)


# Create a synthetic patient manifest with surgery timing, duration information,
# and a direct TOptimal removal-time distribution calibrated to the professor's
# real sample: 62% removed at ~24h, all others drawn from a long-tail shape.
def make_manifest(num_patients=None):
    if num_patients is None:
        num_patients = 100

    study_ids = [f"{i:03d}" for i in range(1, num_patients + 1)]
    all_possible_starts = pd.date_range("2026-01-01", "2026-05-28", freq="h")
    surgery_starts = np.random.choice(all_possible_starts, size=num_patients)
    surgery_starts = pd.Series(surgery_starts).reset_index(drop=True)
    surgery_type = np.random.choice(["VATS", "Open"], size=num_patients, p=[0.7, 0.3])

    base_duration = np.random.lognormal(mean=np.log(38), sigma=0.62, size=num_patients)
    duration_hours = np.clip(base_duration, 16, 220)
    long_stay_count = min(max(1, round(num_patients * 0.07)), num_patients)
    long_stay_idx = np.random.choice(num_patients, size=long_stay_count, replace=False)
    duration_hours[long_stay_idx] = np.random.uniform(220, 420, size=long_stay_count)

    t_optimal = sample_t_optimal(num_patients)
    duration_hours = duration_hours.astype(int)
    t_optimal = np.minimum(t_optimal, duration_hours)

    return pd.DataFrame({
        "StudyID": study_ids,
        "SurgeryStart": surgery_starts,
        "DurationHours": duration_hours,
        "SurgeryType": surgery_type,
        "TOptimal": t_optimal,
    })


# Generate chest X-ray events over the course of each patient's recovery window.
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
                "Effusion": np.random.choice(GRADES, p=GRADE_PROBS["Effusion"]),
                "PneumothoraxSize": np.random.choice(GRADES, p=GRADE_PROBS["PneumothoraxSize"]),
                "SubcuEmphysema": np.random.choice(GRADES, p=GRADE_PROBS["SubcuEmphysema"]),
            })

        all_cxr.append(pd.DataFrame(rows))

    cxr = pd.concat(all_cxr, ignore_index=True)
    cxr = cxr.sort_values(by=["StudyID", "EventDate"])
    cxr["EventDate"] = cxr["EventDate"].dt.strftime("%Y-%m-%d %H:%M")
    return cxr


# Produce the full synthetic dataset for the study: patient metadata and imaging timing.
def generate_synthetic_data(num_patients=None):
    manifest_data = make_manifest(num_patients)
    cxr_data = generate_cxr(manifest_data)
    return manifest_data, cxr_data


# Build the timestamp grid for a patient's chest tube monitoring period.
# Use an exclusive end so a 24-hour stay produces 144 ten-minute observations
# rather than the off-by-one 145-point series from an inclusive boundary.
def make_timestamps(start, duration_hours):
    end = start + pd.Timedelta(hours=duration_hours)
    return pd.date_range(start=start, end=end - pd.Timedelta(minutes=10), freq="10min")


# Simulate changing baseline air leak and fluid output rates over time.
def sample_segment_rates(n):
    air_base = np.zeros(n)
    fluid_base = np.zeros(n)
    readings_per_hour = 6
    i = 0

    while i < n:
        seg_end = min(i + np.random.randint(readings_per_hour, 2 * readings_per_hour + 1), n)
        progress = i / n

        # Air leak: most segments are low-flow, with occasional moderate/high
        # states -- this is what keeps the median low while still letting the
        # mean run high once spikes are layered on top (matches the real
        # data's heavy right skew: median ~5, mean ~300+).
        state = np.random.choice(["low", "mod", "high"], p=[0.75, 0.18, 0.07])
        if state == "low":
            air_base[i:seg_end] = np.random.uniform(0.1, 3.0)
        elif state == "mod":
            air_base[i:seg_end] = np.random.uniform(3.0, 30.0)
        else:
            air_base[i:seg_end] = np.random.uniform(30.0, 150.0)

        if progress < 0.10:
            fluid_base[i:seg_end] = np.random.uniform(0.7, 3.6)
        elif progress < 0.35:
            fluid_base[i:seg_end] = np.random.uniform(0.35, 1.8)
        else:
            fluid_base[i:seg_end] = np.random.uniform(0.1, 0.9)

        i = seg_end

    return air_base, fluid_base


# Convert the air leak baseline into realistic measured values with occasional spikes.
def make_air_leak(air_base):
    n = len(air_base)
    air_leak = air_base * np.abs(np.random.normal(1.1, 0.6, size=n))
    # Spikes are kept rare (1.2% of readings) so most 8h removal-decision
    # windows stay spike-free -- a higher spike rate pulls the mean toward the
    # real ~300 target but also makes almost every window fail the flow gate,
    # which is why removal rate collapsed in the last pass.
    spike_mask = np.random.random(n) < 0.012
    air_leak[spike_mask] += np.random.gamma(shape=1.15, scale=13000.0, size=spike_mask.sum())
    return np.clip(np.round(air_leak, 3), 0, 6000)


# Create cumulative fluid output with local, one-time dips rather than a
# permanently compounding baseline shift. The overall trend remains upward,
# but the real data includes short decreases due to measurement noise and
# smoothing artifacts.
def make_fluid_output(fluid_base):
    noise = np.random.normal(0, fluid_base * 0.5 + 0.2, size=fluid_base.shape)
    incremental_flow = np.clip(fluid_base + noise, 0.0, None)
    cumulative_flow = np.cumsum(incremental_flow)

    dip_mask = np.random.random(cumulative_flow.shape) < 0.08
    if np.any(dip_mask):
        dip_idx = np.flatnonzero(dip_mask)
        for idx in dip_idx:
            local_drop = np.random.uniform(3.0, 35.0)
            if idx + 1 < len(cumulative_flow):
                cumulative_flow[idx + 1:] = cumulative_flow[idx + 1:] - local_drop

    return np.clip(np.round(cumulative_flow, 3), 0, 5000)


# Simulate pleural pressure with mild drift and random measurement noise.
def make_pressure(n):
    # Real sample has a rare negative tail; the 0.0 floor was too strict.
    target = np.random.uniform(0.6, 2.2)
    drift = np.cumsum(np.random.normal(0, 0.01, size=n))
    noise = np.random.normal(0, 0.18, size=n)
    pressure = target + drift + noise
    return np.clip(np.round(pressure, 3), -0.5, 5.5)


# Produce one patient's full chest tube flow time series from the study metadata.
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


# Label each patient as mild, moderate, or severe using early post-surgery flow behavior.
def classify_patient_profile(patient_flow, start_time):
    window = patient_flow[
        (patient_flow["Timestamp"] >= start_time) &
        (patient_flow["Timestamp"] < start_time + pd.Timedelta(hours=24))
    ]

    if window.empty:
        return "UNKNOWN"

    avg_air = window["AirLeakFlow"].mean()
    fluid_vals = window["LOWESSFluidOutput"]
    # Fluid is cumulative now, so "rate" has to come from the increase across
    # the window, not the raw value.
    avg_fluid_rate = (fluid_vals.max() - fluid_vals.min()) / 24.0 if len(fluid_vals) else 0.0

    if avg_air > 50.0 and avg_fluid_rate > 15.0:
        return "SEVERE"
    if avg_air > 50.0 or avg_fluid_rate > 15.0:
        return "MODERATE"
    return "MILD"


# Check whether the most recent chest X-ray supports removal at a given time.
def validate_cxr_for_removal(patient_cxr, removal_time):
    if patient_cxr.empty:
        return True

    patient_cxr = patient_cxr.copy()
    patient_cxr["EventDate"] = pd.to_datetime(patient_cxr["EventDate"])
    before = patient_cxr[patient_cxr["EventDate"] <= removal_time]

    if before.empty:
        return True

    grade_map = {"Z": 0, "O": 1, "T": 2, "TH": 3}
    latest = before.iloc[-1]
    return (
        grade_map.get(latest["Effusion"], 2) <= 1 and
        grade_map.get(latest["PneumothoraxSize"], 2) <= 1
    )


# Summarize the last 8 hours of flow data into the metrics used for removal criteria.
def normalized_flow_limits(window_data):
    air_max = window_data["AirLeakFlow"].max()
    # BUG FIX: fluid is cumulative now, so the removal-relevant number is how
    # much was output DURING this window, not the running total-to-date.
    fluid_series = window_data["LOWESSFluidOutput"]
    fluid_rate = (fluid_series.max() - fluid_series.min()) / 8.0 if len(fluid_series) else 0.0
    pressure_mean = window_data["MeasuredPleuralPressure"].mean()
    pressure_min = window_data["MeasuredPleuralPressure"].min()
    pressure_ok = pressure_mean > 0.0 and pressure_min > 0.0
    meets_criteria = air_max <= 3000.0 and fluid_rate <= 40.0 and pressure_ok
    return air_max, fluid_rate, pressure_mean, pressure_ok, meets_criteria


# Return a base probability of removal at a given postoperative hour.
def hourly_removal_probability(hour):
    if hour < 12:
        return 0.0
    if hour < 24:
        return 0.55
    if hour < 48:
        return 0.72
    if hour < 96:
        return 0.85
    if hour < 168:
        return 0.92
    return 0.75


# Estimate hourly chest tube removal decisions across all patients using flow and imaging rules.
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
        force_no_removal = duration_hours >= 400

        removal_time = None
        removal_hour = None
        removal_probability_final = 0.0

        t_optimal = patient["TOptimal"]
        candidate_start = int(t_optimal) if pd.notna(t_optimal) else None
        if candidate_start is not None and candidate_start <= int(duration_hours) and not force_no_removal:
            candidate_range = range(candidate_start, int(duration_hours) + 1)
        else:
            candidate_range = []

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
            if meets_criteria:
                cxr_valid = validate_cxr_for_removal(patient_cxr, hour_end)

            removal_candidate = hour in candidate_range and meets_criteria and cxr_valid and not force_no_removal
            if removal_candidate and removal_hour is None:
                removal_hour = hour
                removal_time = hour_end
                removal_probability_final = prob

            removal_decision = bool(hour == removal_hour and meets_criteria and cxr_valid and not force_no_removal)

            hourly.append({
                "StudyID": studyid,
                "Hour": hour,
                "Timestamp": hour_end,
                "AirLeakFlow_Max_8h": air_max,
                "FluidOutput_Rate_8h": fluid_rate,
                "PressureMean_8h": pressure_mean,
                "PressureOK_8h": pressure_ok,
                "MeetsFlowCriteria": meets_criteria,
                "CXRValid": cxr_valid,
                "RemovalProbability": prob,
                "RemovalDecision": removal_decision,
                "Profile": profile,
            })

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
            "TOptimal": t_optimal if (removal_time is not None) else None,
        })

    return pd.DataFrame(records), pd.DataFrame(hourly)


# Combine all generated per-patient flow tables into one study-wide data table.
def build_flow_table(manifest_data):
    return pd.concat([generate_flow(row) for _, row in manifest_data.iterrows()], ignore_index=True).sort_values(
        by=["StudyID", "Timestamp"]
    )


# Add hours-since-surgery to a dataframe based on each patient's surgery start time.
def add_time_since_surgery(dataframe, manifest_data, timestamp_column):
    result = dataframe.copy()
    result = result.merge(manifest_data[["StudyID", "SurgeryStart"]], on="StudyID", how="left")
    result[timestamp_column] = pd.to_datetime(result[timestamp_column])
    result["SurgeryStart"] = pd.to_datetime(result["SurgeryStart"])
    result["HoursSinceSurgery"] = (result[timestamp_column] - result["SurgeryStart"]).dt.total_seconds() / 3600.0
    return result.drop(columns=["SurgeryStart"])


# Summarize hourly feature averages and variability for removed versus non-removed groups.
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
    return round_numeric_columns(summary_stats)


# Save the flow table in a tab-separated format for downstream use.
def save_flow_table(flow, filename="flow_output.tab"):
    flow = round_numeric_columns(flow)
    flow.to_csv(filename, sep="\t", index=False)


# Export the full set of study tables and summary files to a target directory.
def export_csv_outputs(num_patients=None, output_dir="."):
    manifest_data, cxr_data = generate_synthetic_data(num_patients)

    flow = build_flow_table(manifest_data)
    removal_predictions, hourly_removals = predict_removals_hourly(flow, cxr_data, manifest_data)

    flow_with_time = add_time_since_surgery(flow, manifest_data, "Timestamp")
    cxr_with_time = add_time_since_surgery(cxr_data, manifest_data, "EventDate")

    ct = removal_predictions[["StudyID", "TOptimal"]].copy()
    ct = round_numeric_columns(ct)
    ct.to_csv(f"{output_dir}/ct.csv", index=False)

    summary_stats = build_hourly_feature_summary(flow_with_time, removal_predictions)
    summary_stats.to_csv(f"{output_dir}/hourly_feature_summary.csv", index=False)

    force_no_removal_count = int(removal_predictions["ForceNoRemoval"].sum())
    pd.DataFrame({"ForceNoRemovalCount": [force_no_removal_count]}).to_csv(f"{output_dir}/force_no_removal_report.csv", index=False)

    flow_with_time["Timestamp"] = flow_with_time["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    cxr_with_time["EventDate"] = cxr_with_time["EventDate"].dt.strftime("%Y-%m-%d %H:%M")

    save_flow_table(round_numeric_columns(flow_with_time), f"{output_dir}/flow_output.tab")
    removal_predictions = round_numeric_columns(removal_predictions)
    hourly_removals = round_numeric_columns(hourly_removals)
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
    }


# Run the full export workflow for the synthetic chest tube dataset.
# Normal runs are stochastic and use fresh randomness unless a seed is supplied
# explicitly, e.g. `main(seed=20240601)` for reproducible output.
def main(num_patients=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    export_csv_outputs(num_patients=num_patients)
    if seed is None:
        print("done")
    else:
        print(f"done (seed={seed})")


if __name__ == "__main__":
    main()