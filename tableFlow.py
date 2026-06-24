
import numpy as np
import pandas as pd
from tableCXR import manifest
def generate_flow(row):
    """
    Generate a FLOW table for one patient.

    Clinical reasoning
    ------------------
    After thoracic surgery the chest drain is connected immediately (time 0).
    Two physiological processes produce measurable output:

    1. AirLeakFlow (mL/min or arbitrary pump units):
       The visceral pleural fissure leaks air as the lung re-expands.
       Leakage is not constant — it varies with breathing depth, patient
       movement, and healing state.  We model it as a piecewise-stable
       base rate that changes every 1–2 hours (the "segment" approach),
       with small per-reading noise added on top.  The base rate decays
       over the full hospital stay to simulate gradual sealing of the leak.
       Occasional spikes (coughing, deep breaths) are injected randomly.
       Real data shows ~50 % of readings below 5 units, median ~1–2,
       with sporadic large spikes up to several hundred.

    2. LOWESSFluidOutput (integer mL per 10-minute interval):
       Surgical trauma causes inflammation; lymphatic fluid and blood serum
       accumulate in the pleural space and are drained continuously.
       Output is highest immediately post-op and declines as healing
       progresses.  The LOWESS label implies the values represent a
       smoothed (locally-weighted) estimate of instantaneous drainage —
       so each row is that interval's output, not a running total.
       Real-world drainage ranges from ~0–10 mL per 10 min during the
       later post-op period, but can be 10–30 mL/10 min early on.

    3. MeasuredPleuralPressure (cmH2O, negative pressure):
       The drain maintains sub-atmospheric pressure in the pleural space
       to re-expand the lung.  Typical targets are −0.75 to −5 cmH2O.
       Small breath-by-breath fluctuations occur around the set point.
       We represent the magnitude (stored as a positive float per the
       real dataset convention) with a slowly-drifting mean and small
       Gaussian noise.
    """

    studyid  = row["StudyID"]
    start    = row["SurgeryStart"]
    duration = row["DurationHours"]

    # -------------------------------------------------------------------------
    # Build the timestamp spine — every 10 minutes from surgery start
    # -------------------------------------------------------------------------
    end = start + pd.Timedelta(hours=duration)

    # pd.date_range guarantees exact 10-minute alignment
    timestamps = pd.date_range(start=start, end=end, freq="10min")

    n = len(timestamps)   # total number of readings for this patient

    # -------------------------------------------------------------------------
    # SEGMENT STRUCTURE: leakage rate changes every 1–2 hours
    # Each segment has its own base AirLeakFlow rate and fluid drainage rate.
    # This simulates clinical reality: the patient coughs, changes position,
    # or the wound shifts, causing a step-change in drain output.
    # -------------------------------------------------------------------------
    readings_per_hour = 6               # 6 readings × 10 min = 60 min
    min_seg = 1 * readings_per_hour     # shortest segment: 1 hour
    max_seg = 2 * readings_per_hour     # longest segment:  2 hours

    # Pre-allocate arrays for efficiency (matches the existing CXR style of
    # building a list of rows, but we compute arrays first then zip them)
    air_leak_base   = np.zeros(n)
    fluid_base      = np.zeros(n)

    # Fill segment by segment
    i = 0
    while i < n:
        seg_len = np.random.randint(min_seg, max_seg + 1)
        seg_end = min(i + seg_len, n)

        # --- AirLeakFlow base for this segment ---
        # The base rate decays over time: early segments can be higher,
        # later segments tend toward zero as the leak seals.
        # progress ∈ [0, 1] tracks how far through the stay we are.
        progress = i / n

        # Exponential decay envelope: starts up to ~40 units, falls toward 0.
        # The random multiplier gives each patient a unique severity profile.
        decay_envelope = np.random.uniform(2, 40) * np.exp(-2 * progress)

        # Base rate is drawn from a half-normal (most values near 0,
        # occasional moderate values) scaled by the decay envelope.
        seg_air = abs(np.random.normal(0, decay_envelope * 0.3))
        seg_air = np.clip(seg_air, 0, 1000)

        # --- LOWESSFluidOutput base for this segment (mL per 10 min) ---
        # Fluid drainage is heaviest early post-op and declines.
        # Typical post-thoracotomy drainage: 5–20 mL/10 min early,
        # dropping to 1–5 mL/10 min by day 2–3.
        fluid_envelope = np.random.uniform(5, 25) * np.exp(-3 * progress)
        seg_fluid = max(0, np.random.normal(fluid_envelope, fluid_envelope * 0.3))

        air_leak_base[i:seg_end]  = seg_air
        fluid_base[i:seg_end]     = seg_fluid

        i = seg_end

    # -------------------------------------------------------------------------
    # PER-READING NOISE on top of the segment base
    # -------------------------------------------------------------------------

    # AirLeakFlow: multiplicative noise so quiet periods stay quiet and
    # active periods have proportionally larger swings.
    air_noise = np.abs(np.random.normal(1.0, 0.4, size=n))
    air_leak  = air_leak_base * air_noise

    # Random spikes: ~3 % of readings get a large cough/movement spike.
    # Spike magnitude is drawn from a heavy-tailed distribution to match
    # the real data's occasional values in the 100–1000 range.
    spike_mask = np.random.random(n) < 0.03
    spike_vals = np.random.exponential(scale=150, size=n)
    air_leak   = np.where(spike_mask, air_leak + spike_vals, air_leak)
    air_leak   = np.clip(np.round(air_leak, 2), 0, 1000)

    # LOWESSFluidOutput: additive Gaussian noise; clipped to non-negative
    # integer mL (matches the integer dtype in the real dataset).
    fluid_noise  = np.random.normal(0, fluid_base * 0.25 + 0.5, size=n)
    fluid_output = np.round(fluid_base + fluid_noise).astype(int)
    fluid_output = np.clip(fluid_output, 0, 500)   # physiological ceiling

    # -------------------------------------------------------------------------
    # MeasuredPleuralPressure
    # Pleural pressure is set by the drain unit, typically −1 to −2 cmH2O,
    # with breath-to-breath fluctuations of ±0.1–0.3 cmH2O.
    # We store the magnitude as a positive float (matching real dataset).
    # A slowly-drifting mean (random walk) keeps it from being constant.
    # -------------------------------------------------------------------------
    # Each patient has their own target pressure set by the clinician
    target_pressure = np.random.uniform(1.0, 2.5)

    # Small random walk for drift (cumulative sum of tiny steps)
    drift_steps = np.random.normal(0, 0.01, size=n)
    drift       = np.cumsum(drift_steps)

    # Breath-by-breath noise
    breath_noise = np.random.normal(0, 0.08, size=n)

    pleural_pressure = target_pressure + drift + breath_noise
    # Clip to realistic physiological range (0.75–5.0 cmH2O magnitude)
    pleural_pressure = np.clip(np.round(pleural_pressure, 2), 0.75, 5.0)

    rows = []

    for j in range(n):

        row_data = {
            "StudyID":                studyid,
            "Timestamp":              timestamps[j],
            "MeasuredPleuralPressure": pleural_pressure[j],
            "AirLeakFlow":             air_leak[j],
            "LOWESSFluidOutput":       fluid_output[j]
        }

        rows.append(row_data)

    return pd.DataFrame(rows)


# =============================================================================
# Loop through the manifest — mirrors the CXR loop exactly
# =============================================================================

all_flow = []

for _, row in manifest.iterrows():

    patient_flow = generate_flow(row)

    all_flow.append(patient_flow)

flow = pd.concat(all_flow, ignore_index=True)

flow = flow.sort_values(
    by=["StudyID", "Timestamp"]
)

# Format timestamp to match CXR EventDate style: YYYY-MM-DD HH:MM
flow["Timestamp"] = flow["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")

print("")
output_choice = input("Output FLOW table to a file? (yes/no): ")

if output_choice == "yes":
    filename = input("Enter filename (e.g. flow_output.tab): ").strip()
    if not filename:
        filename = "flow_output.tab"
    with open(filename, "w") as f:
        f.write("\t".join(flow.columns) + "\n")
        for _, r in flow.iterrows():
            line = f"{r['StudyID']}\t{r['Timestamp']}\t{r['MeasuredPleuralPressure']}\t\t\t{r['AirLeakFlow']}\t\t\t{r['LOWESSFluidOutput']}\n"
            f.write(line)
    print(f"FLOW table saved to {filename}")
else:
    print(flow.to_string(index=False))