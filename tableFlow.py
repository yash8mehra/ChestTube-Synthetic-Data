
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        # Fixed clinical rates based on post-op healing trajectory.
        # progress ∈ [0, 1] tracks how far through the stay we are.
        progress = i / n

        # Fixed rates based on clinical data:
        # Early (0-24h): 2.0-5.0 mL/min
        # Mid (24-72h): 1.0-2.0 mL/min
        # Late (72h+): 0.5-1.0 mL/min (approaching removal threshold)
        if progress < 0.083:  # First 24 hours (0.083 ≈ 1/12)
            seg_air = np.random.uniform(2.0, 5.0)
        elif progress < 0.333:  # 24-72 hours
            seg_air = np.random.uniform(1.0, 2.0)
        else:  # 72+ hours
            seg_air = np.random.uniform(0.5, 1.0)

        # --- LOWESSFluidOutput base for this segment (mL per 10 min) ---
        # Fixed clinical rates based on post-thoracotomy drainage patterns.
        # Early post-op: 5–20 mL/10 min
        # Mid post-op (day 2-3): 1–5 mL/10 min
        # Late post-op (day 4+): <1 mL/10 min
        if progress < 0.083:  # First 24 hours
            seg_fluid = np.random.uniform(5, 20)
        elif progress < 0.333:  # 24-72 hours
            seg_fluid = np.random.uniform(1, 5)
        else:  # 72+ hours
            seg_fluid = np.random.uniform(0, 1)

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
# REMOVAL READINESS LOGIC
# =============================================================================


# =============================================================================
# PATIENT CLUSTERING & PROFILING
# =============================================================================

def classify_patient_profile(patient_flow, start_time):
    """
    Classify patient into clinical profile based on early (first 24h) flow characteristics.
    
    Profiles:
    - SEVERE: High air leak + high fluid output
    - MODERATE: Mixed air leak and fluid output
    - MILD: Low air leak + low fluid output
    """
    early_window = patient_flow[
        (patient_flow['Timestamp'] >= start_time) &
        (patient_flow['Timestamp'] < start_time + pd.Timedelta(hours=24))
    ]
    
    if len(early_window) == 0:
        return 'UNKNOWN'
    
    avg_air = early_window['AirLeakFlow'].mean()
    avg_fluid = early_window['LOWESSFluidOutput'].mean()
    
    # Thresholds
    air_threshold = 3.0
    fluid_threshold = 10.0
    
    if avg_air > air_threshold and avg_fluid > fluid_threshold:
        return 'SEVERE'
    elif avg_air > air_threshold or avg_fluid > fluid_threshold:
        return 'MODERATE'
    else:
        return 'MILD'


# =============================================================================
# CXR-BASED REMOVAL VALIDATION
# =============================================================================

def validate_cxr_for_removal(patient_cxr, removal_time, early_only=True):
    """
    Validate CXR findings for removal readiness.
    
    CXR values (0 or 1 indicate readiness):
    - Effusion: 0 = resolved, 1 = minimal
    - PneumothoraxSize: 0 = resolved, 1 = minimal
    
    Early post-op only: Check CXR only if within first 72 hours
    """
    if len(patient_cxr) == 0:
        return True  # No CXR data, assume valid
    
    patient_cxr = patient_cxr.copy()
    patient_cxr['EventDate'] = pd.to_datetime(patient_cxr['EventDate'])
    
    # Get closest CXR before removal time
    cxr_before_removal = patient_cxr[patient_cxr['EventDate'] <= removal_time]
    
    if len(cxr_before_removal) == 0:
        return True
    
    latest_cxr = cxr_before_removal.iloc[-1]
    hours_since_surgery = (removal_time - patient_cxr['EventDate'].min()).total_seconds() / 3600
    
    # Only validate CXR in early post-op (first 72 hours)
    if early_only and hours_since_surgery > 72:
        return True
    
    # Map grades to numeric (Z=0, O=1, T=2, Th=3)
    grade_map = {'Z': 0, 'O': 1, 'T': 2, 'Th': 3}
    effusion_score = grade_map.get(latest_cxr['Effusion'], 2)
    pneumo_score = grade_map.get(latest_cxr['PneumothoraxSize'], 2)
    
    # Valid for removal if both are 0 or 1 (Z or O)
    is_valid = effusion_score <= 1 and pneumo_score <= 1
    
    return is_valid


# =============================================================================
# PER-HOUR REMOVAL DECISIONS WITH AUC TRACKING
# =============================================================================

def predict_removals_hourly(flow, cxr, manifest_data):
    """
    Predict chest tube removal with HOURLY granularity.
    
    Returns: 
    - removal_predictions: DataFrame with per-patient removal summary
    - hourly_removals: DataFrame with per-hour removal decisions
    """
    removal_predictions = []
    hourly_removals = []
    
    for _, patient_row in manifest_data.iterrows():
        studyid = patient_row['StudyID']
        start_time = patient_row['SurgeryStart']
        duration_hours = patient_row['DurationHours']
        
        patient_flow = flow[flow['StudyID'] == studyid].copy()
        patient_cxr = cxr[cxr['StudyID'] == studyid].copy()
        patient_flow['Timestamp'] = pd.to_datetime(patient_flow['Timestamp'])
        
        # Classify patient profile early
        profile = classify_patient_profile(patient_flow, start_time)
        
        # Force non-removal for 120+ hour stays
        force_no_removal = duration_hours >= 120
        
        removal_time = None
        removal_hour = None
        removal_probability_final = 0.0
        removal_hour_decision = []
        
        # Check EVERY HOUR (not 12-hour segments)
        for hour in range(1, int(duration_hours) + 1):
            hour_start = start_time + pd.Timedelta(hours=hour - 1)
            hour_end = start_time + pd.Timedelta(hours=hour)
            eight_hours_ago = hour_end - pd.Timedelta(hours=8)
            
            # Get data from this hour and 8-hour lookback
            hour_data = patient_flow[
                (patient_flow['Timestamp'] >= hour_start) &
                (patient_flow['Timestamp'] < hour_end)
            ]
            
            window_data = patient_flow[
                (patient_flow['Timestamp'] >= eight_hours_ago) &
                (patient_flow['Timestamp'] <= hour_end)
            ]
            
            if len(window_data) == 0:
                continue
            
            # Check 3ml/min threshold over 8-hour window
            air_leak_max = window_data['AirLeakFlow'].max()
            fluid_max_per_min = window_data['LOWESSFluidOutput'].max() / 10.0
            meets_flow_criteria = (air_leak_max <= 3.0) and (fluid_max_per_min <= 3.0)
            
            # Calculate removal probability for this hour
            if hour < 12:
                prob = 0.0  # First 12 hours: no removal
            elif hour < 24:
                prob = 0.04  # Hours 12-24: 4% per hour
            elif hour < 120:
                # Hours 24-120: increasing rate (5% per hour average)
                prob = 0.04 + (0.01 * (hour - 24) / 96)
            else:
                prob = 0.05  # After 120h: 5% per hour (if not forced no-removal)
            
            # If 120+ hours, force no removal
            if force_no_removal:
                prob = 0.0
            
            # Check CXR validation (only if meets flow criteria and early post-op)
            cxr_valid = True
            if meets_flow_criteria and hour <= 72:
                cxr_valid = validate_cxr_for_removal(patient_cxr, hour_end, early_only=True)
            
            # Generate removal decision: random 0-1
            removal_decision = np.random.random() < prob if (meets_flow_criteria and cxr_valid and not removal_time) else False
            
            hourly_removals.append({
                'StudyID': studyid,
                'Hour': hour,
                'Timestamp': hour_end,
                'AirLeakFlow_Max_8h': air_leak_max,
                'FluidOutput_Max_PerMin_8h': fluid_max_per_min,
                'MeetsFlowCriteria': meets_flow_criteria,
                'CXRValid': cxr_valid,
                'RemovalProbability': prob,
                'RemovalDecision': removal_decision,
                'Profile': profile
            })
            
            if removal_decision and not removal_time:
                removal_time = hour_end
                removal_hour = hour
                removal_probability_final = prob
        
        # Determine if patient gets removed
        removed = removal_time is not None
        
        removal_predictions.append({
            'StudyID': studyid,
            'SurgeryStart': start_time,
            'DurationHours': duration_hours,
            'Profile': profile,
            'RemovalTime': removal_time,
            'RemovalHour': removal_hour,
            'Removed': removed,
            'RemovalProbability': removal_probability_final,
            'HoursUntilRemoval': removal_hour if removal_hour else None,
            'ForceNoRemoval': force_no_removal
        })
    
    removal_pred_df = pd.DataFrame(removal_predictions)
    hourly_df = pd.DataFrame(hourly_removals)
    
    return removal_pred_df, hourly_df


def predict_removals(flow, cxr, manifest_data):
    """
    Predict chest tube removal times for all patients.
    
    Returns: DataFrame with removal predictions and metrics
    """
    removal_results = []
    
    for _, patient_row in manifest_data.iterrows():
        studyid = patient_row['StudyID']
        start_time = patient_row['SurgeryStart']
        duration_hours = patient_row['DurationHours']
        
        patient_flow = flow[flow['StudyID'] == studyid].copy()
        patient_cxr = cxr[cxr['StudyID'] == studyid].copy()
        patient_flow['Timestamp'] = pd.to_datetime(patient_flow['Timestamp'])
        
        removal_time = None
        removal_probability = 0.0
        
        # Check each 12-hour segment
        for hour_segment in range(12, duration_hours + 12, 12):
            segment_start = start_time + pd.Timedelta(hours=hour_segment - 12)
            segment_end = start_time + pd.Timedelta(hours=hour_segment)
            
            # Get readings in this segment
            segment_data = patient_flow[
                (patient_flow['Timestamp'] >= segment_start) &
                (patient_flow['Timestamp'] <= segment_end)
            ]
            
            if len(segment_data) == 0:
                continue
            
            # Calculate removal probability for this segment
            prob = calculate_removal_probability(segment_data, hour_segment, base_rate=0.05)
            
            # Generate random decision: remove or not?
            if np.random.random() < prob:
                # Find first timestamp in segment that meets 8-hour criterion
                for idx, row in segment_data.iterrows():
                    removal_time = row['Timestamp']
                    removal_probability = prob
                    break
                
                if removal_time:
                    break
        
        # Determine if patient gets removed
        removed = removal_time is not None
        
        removal_results.append({
            'StudyID': studyid,
            'SurgeryStart': start_time,
            'DurationHours': duration_hours,
            'RemovalTime': removal_time,
            'Removed': removed,
            'RemovalProbability': removal_probability,
            'HoursUntilRemoval': (removal_time - start_time).total_seconds() / 3600 if removal_time else None
        })
    
    return pd.DataFrame(removal_results)


def plot_removal_metrics(removal_df, hourly_df, output_file='removal_analysis.png'):
    """
    Plot removal rate metrics and per-hour analysis.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Removal rate per hour
    if len(hourly_df) > 0:
        hourly_decisions = hourly_df.groupby('Hour')['RemovalDecision'].sum()
        axes[0, 0].plot(hourly_decisions.index, hourly_decisions.values, marker='o', linewidth=2, color='steelblue')
        axes[0, 0].axvline(27, color='green', linestyle='--', alpha=0.7, label='Hour 27 (Total Separation)')
        axes[0, 0].axvline(30, color='red', linestyle='--', alpha=0.7, label='Hour 30 (Breakdown)')
        axes[0, 0].set_xlabel('Hours After Surgery')
        axes[0, 0].set_ylabel('Number of Removals')
        axes[0, 0].set_title('Chest Tube Removals per Hour')
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].legend()
    
    # 2. Removal probability distribution
    axes[0, 1].hist(removal_df['RemovalProbability'], bins=20, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Removal Probability')
    axes[0, 1].set_ylabel('Number of Patients')
    axes[0, 1].set_title('Distribution of Removal Probabilities')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Removal vs Non-Removal counts
    removal_counts = removal_df['Removed'].value_counts()
    colors = ['#ff9999', '#90ee90']
    axes[0, 2].pie(removal_counts.values, labels=['Not Removed', 'Removed'], autopct='%1.1f%%',
                   colors=colors, startangle=90)
    axes[0, 2].set_title('Overall Removal Rate')
    
    # 4. Removal probability over time
    if len(hourly_df) > 0:
        hourly_prob = hourly_df.groupby('Hour')['RemovalProbability'].mean()
        axes[1, 0].plot(hourly_prob.index, hourly_prob.values, marker='o', linewidth=2, color='purple', markersize=6)
        axes[1, 0].axvline(27, color='green', linestyle='--', alpha=0.7, label='Hour 27')
        axes[1, 0].axvline(30, color='red', linestyle='--', alpha=0.7, label='Hour 30')
        axes[1, 0].set_xlabel('Hours After Surgery')
        axes[1, 0].set_ylabel('Average Removal Probability')
        axes[1, 0].set_title('Removal Probability Trend')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].legend()
    
    # 5. Removals by patient profile
    if 'Profile' in removal_df.columns:
        profile_removal = removal_df.groupby('Profile')['Removed'].agg(['sum', 'count'])
        profile_removal['rate'] = profile_removal['sum'] / profile_removal['count']
        axes[1, 1].bar(profile_removal.index, profile_removal['rate'], color=['#ff6b6b', '#ffd93d', '#6bcf7f'], alpha=0.7)
        axes[1, 1].set_ylabel('Removal Rate')
        axes[1, 1].set_title('Removal Rate by Patient Profile')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(profile_removal['rate']):
            axes[1, 1].text(i, v + 0.02, f'{v:.2%}', ha='center')
    
    # 6. Force no-removal (120+ hours)
    if 'ForceNoRemoval' in removal_df.columns:
        long_stay_counts = removal_df['ForceNoRemoval'].value_counts()
        axes[1, 2].bar(['<120h', '≥120h'], 
                       [long_stay_counts.get(False, 0), long_stay_counts.get(True, 0)],
                       color=['steelblue', 'orange'], alpha=0.7)
        axes[1, 2].set_ylabel('Number of Patients')
        axes[1, 2].set_title('Patient Stay Duration')
        axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Removal analysis plot saved to {output_file}")
    plt.close()


# =============================================================================
# REMOVAL ANALYSIS & AUC CALCULATION
# =============================================================================




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

# Load CXR data for removal analysis
from tableCXR import cxr

# =============================================================================
# GENERATE REMOVAL PREDICTIONS (HOURLY WITH FULL ANALYSIS)
# =============================================================================

print("\n" + "="*80)
print("REMOVAL READINESS ANALYSIS (HOURLY)")
print("="*80)

removal_predictions, hourly_removals = predict_removals_hourly(flow, cxr, manifest)

print(f"\nTotal patients: {len(removal_predictions)}")
print(f"Removed: {removal_predictions['Removed'].sum()}")
print(f"Not removed: {(~removal_predictions['Removed']).sum()}")
print(f"Overall removal rate: {removal_predictions['Removed'].mean():.2%}")

# Removal rate by profile
print("\nRemoval rates by patient profile:")
profile_stats = removal_predictions.groupby('Profile').agg({
    'Removed': ['sum', 'count', 'mean']
}).round(3)
print(profile_stats)

# Find non-removable patients (120+ hours)
long_stay = removal_predictions[removal_predictions['DurationHours'] >= 120]
not_removed_long_stay = long_stay[~long_stay['Removed']]
print(f"\nPatients with 120+ hour stays: {len(long_stay)}")
print(f"Not removed (120+ hours): {len(not_removed_long_stay)}")

if len(not_removed_long_stay) > 0:
    print("\nNon-removable patients (120+ hours):")
    print(not_removed_long_stay[['StudyID', 'DurationHours', 'Profile', 'RemovalProbability']].head(10))

# Plot analysis
plot_removal_metrics(removal_predictions, hourly_removals, output_file='removal_analysis.png')

print("\n" + "="*80)
print("HOURLY REMOVAL DECISIONS")
print("="*80)
print(f"Total hourly decisions: {len(hourly_removals)}")
print(f"Removals by hour (first 50 hours):")
hourly_summary = hourly_removals[hourly_removals['Hour'] <= 50].groupby('Hour').agg({
    'RemovalDecision': 'sum',
    'MeetsFlowCriteria': 'sum',
    'CXRValid': 'sum',
    'RemovalProbability': 'mean'
}).round(3)
print(hourly_summary)

# =============================================================================
# OUTPUT OPTIONS
# =============================================================================

flow["Timestamp"] = flow["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")

print("\n" + "="*80)
print("FLOW DATA OUTPUT")
print("="*80 + "\n")

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

# Output removal predictions
output_removals = input("\nSave removal predictions to CSV? (yes/no): ")
if output_removals == "yes":
    removal_predictions.to_csv('removal_predictions.csv', index=False)
    print("Removal predictions saved to removal_predictions.csv")
    
    # Also save hourly removals for detailed analysis
    hourly_removals.to_csv('hourly_removal_decisions.csv', index=False)
    print("Hourly removal decisions saved to hourly_removal_decisions.csv")