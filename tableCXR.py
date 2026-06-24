import pandas as pd
import numpy as np

GRADES = ["Z", "O", "T", "Th"]

num_patients = int(input("Enter number of patients to simulate: "))

#atleast 100
if num_patients < 100:
    num_patients = 100

study_ids = []
for i in range(1, num_patients + 1):
    study_ids.append(f"{i:03d}") #padding leading zeros to 3 digits

all_possible_starts = pd.date_range("2026-01-01", "2026-05-28", freq="h")

surgery_starts = np.random.choice(
    all_possible_starts,
    size=num_patients,
    #replace=False
)

surgery_starts = pd.Series(surgery_starts).reset_index(drop=True)

duration_hours = np.random.randint(24, 125, size=num_patients)

manifest = pd.DataFrame({
    "StudyID": study_ids,
    "SurgeryStart": surgery_starts,
    "DurationHours": duration_hours
})

#CXR GENERATION

def generate_cxr(row):

    studyid = row["StudyID"]
    start = row["SurgeryStart"]
    duration_hrs = row["DurationHours"]

    #first xray is 4 hours after surgery
    current = start + pd.Timedelta(hours=4)

    end = start + pd.Timedelta(hours=duration_hrs)

    #list of xray timestamps
    times = []

    while current < end:

        times.append(current.round('10s'))

        gap_hours = np.random.uniform(24, 168)

        current = current + pd.Timedelta(hours=gap_hours)

    # build one row per xray
    rows = []

    for t in times:

        row_data = {
            "StudyID": studyid,
            "EventDate": t,
            "Effusion": np.random.choice(GRADES),
            "PneumothoraxSize": np.random.choice(GRADES),
            "SubcuEmphysema": np.random.choice(GRADES)
        }

        rows.append(row_data)

    return pd.DataFrame(rows)

all_cxr = []

for _, row in manifest.iterrows():

    patient_cxr = generate_cxr(row)

    all_cxr.append(patient_cxr)

cxr = pd.concat(all_cxr, ignore_index=True)

cxr = cxr.sort_values(
    by=["StudyID", "EventDate"]
)

cxr["EventDate"] = cxr["EventDate"].dt.strftime("%Y-%m-%d %H:%M")

print(cxr.to_string(index=False))
