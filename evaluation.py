import numpy as np
import rtamt
import os
from glob import glob
from triangulation import compute_areas
from tqdm import tqdm

# Folder with patient files
dataset_folder = 'datasets/WT/'
patient_files = glob(os.path.join(dataset_folder, '*.nii'))

# Counters
total_patients = len(patient_files)
formula1_satisfied = 0
formula2_satisfied = 0
formula3_satisfied = 0
formula4_satisfied = 0
formula5_satisfied = 0
formula6_satisfied = 0

for patient_path in tqdm(patient_files):
    patient_id = os.path.basename(patient_path).replace('.nii', '')

    # Compute tumor area per slice
    areas_1, _, _, _ = compute_areas(patient_path)
    trace = np.array(areas_1)

    # Skip empty masks (patients with no tumor)
    if np.sum(trace) == 0:
        continue

    # Prepare time/area signals
    time_signal = [float(i) for i in range(len(trace))]
    area_signal = [float(val) for val in trace]
    dataset = {'time': time_signal, 'area': area_signal}

    # --- Formula 1: Tumor always under control ---
    spec1 = rtamt.STLSpecification()
    spec1.declare_var('time', 'float')
    spec1.declare_var('area', 'float')
    spec1.spec = 'G[0,155](area < 5000)'
    spec1.parse()
    result1 = spec1.evaluate(dataset)
    satisfied1 = all(r[1] > 0 for r in result1)
    if satisfied1:
        formula1_satisfied += 1

    # --- Formula 2: Tumor eventually shrinks below 200 ---
    spec2 = rtamt.STLSpecification()
    spec2.declare_var('time', 'float')
    spec2.declare_var('area', 'float')
    spec2.spec = 'F[0,155](area < 200)'
    spec2.parse()
    result2 = spec2.evaluate(dataset)
    satisfied2 = any(r[1] > 0 for r in result2)
    if satisfied2:
        formula2_satisfied += 1

    # --- Formula 3: No sudden unrealistic shrinkage ---
    area_diff = np.diff(area_signal, prepend=area_signal[0])
    dataset_diff = {'time': time_signal, 'area_diff': area_diff}

    spec3 = rtamt.STLSpecification()
    spec3.declare_var('time', 'float')
    spec3.declare_var('area_diff', 'float')
    spec3.spec = 'G[0,155](area_diff > -300)'
    spec3.parse()
    result3 = spec3.evaluate(dataset_diff)
    satisfied3 = all(r[1] > 0 for r in result3)
    if satisfied3:
        formula3_satisfied += 1

    # --- Formula 4: Rapid growth detection ---
    # Robust check if initial area is zero
    initial_area = next((a for a in area_signal if a > 0), 1)  # avoid multiplying by zero
    threshold_growth = 1.5 * initial_area

    spec4 = rtamt.STLSpecification()
    spec4.declare_var('time', 'float')
    spec4.declare_var('area', 'float')
    spec4.spec = f'F[0,50](area > {threshold_growth})'
    spec4.parse()
    result4 = spec4.evaluate(dataset)
    satisfied4 = any(r[1] > 0 for r in result4)
    if satisfied4:
        formula4_satisfied += 1

    # --- Formula 5: The tumor area decreases by at least 50% compared to the previous measurement--
    pr_events = sum([area_signal[i] < area_signal[i - 1] * 0.5 for i in range(1, len(area_signal))])
    if pr_events >= 2:
        formula5_satisfied += 1

    # --- Formula 6: The tumor area increases by at least 25% compared to the previous measurement--
    pd_events = sum([area_signal[i] > area_signal[i - 1] * 1.25 for i in range(1, len(area_signal))])
    if pd_events >= 10:
        formula6_satisfied += 1

# --- Print Global Results ---
print("\n--- Temporal Logic Evaluation Results ---")
print(f"Total patients processed: {total_patients}")
print(f"Formula 1 (Always under control): {formula1_satisfied}/{total_patients}")
print(f"Formula 2 (Shrinkage detected): {formula2_satisfied}/{total_patients}")
print(f"Formula 3 (No sudden drop): {formula3_satisfied}/{total_patients}")
print(f"Formula 4 (Rapid growth detected): {formula4_satisfied}/{total_patients}")
print(f"Formula 5 (Partial Response - PR, ≥50% reduction): {formula5_satisfied}/{total_patients}")
print(f"Formula 6 (Progressive Disease - PD, ≥25% increase): {formula6_satisfied}/{total_patients}")
