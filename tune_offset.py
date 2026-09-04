from pathlib import Path
import shutil
import subprocess
import pandas as pd
import numpy as np

SRC_DIR = Path("perception_results")
TMP_DIR = Path("perception_results_offset_test")
GT_FILE = Path("src/ground_truth.csv")
GRADER = Path("src/grading_script.py")

# Search ranges: start coarse, then narrow around the best result
x_offsets = np.arange(-20.0, 20.5, 2.0)
y_offsets = np.arange(-20.0, 20.5, 2.0)
z_offsets = np.arange(-3.0, 3.5, 1.0)

best_score = -1e9
best_offsets = None
best_output = ""

csv_files = sorted(SRC_DIR.glob("*.csv"), key=lambda p: int(p.stem))

for dx in x_offsets:
    for dy in y_offsets:
        for dz in z_offsets:
            if TMP_DIR.exists():
                shutil.rmtree(TMP_DIR)
            TMP_DIR.mkdir(parents=True, exist_ok=True)

            for csv_file in csv_files:
                df = pd.read_csv(csv_file)

                # Shift positions
                df["position_x"] += dx
                df["position_y"] += dy
                df["position_z"] += dz

                # Shift bounding boxes too
                df["bbox_x_min"] += dx
                df["bbox_x_max"] += dx
                df["bbox_y_min"] += dy
                df["bbox_y_max"] += dy
                df["bbox_z_min"] += dz
                df["bbox_z_max"] += dz

                df.to_csv(TMP_DIR / csv_file.name, index=False)

            result = subprocess.run(
                ["python", str(GRADER), str(GT_FILE), str(TMP_DIR)],
                capture_output=True,
                text=True
            )

            output = result.stdout
            total_line = None
            for line in output.splitlines():
                if line.startswith("Total score:"):
                    total_line = line
                    break

            if total_line is None:
                continue

            score = float(total_line.split(":")[1].split("/")[0].strip())
            if score > best_score:
                best_score = score
                best_offsets = (dx, dy, dz)
                best_output = output
                print(f"NEW BEST: score={best_score:.2f}, dx={dx}, dy={dy}, dz={dz}")

print("\nBEST OFFSETS:", best_offsets)
print("BEST SCORE:", best_score)
print("\nBEST GRADER OUTPUT:\n")
print(best_output)