"""
Reads a manifest.csv (from make_grid_manifest.py) and submits one individual SLURM job
per row via `sbatch --export=...` — no job array, exactly the pattern already used by
hand for the 6-point boundary-weight sweep this session. Writes the resulting job IDs
back into a copy of the manifest so each point can be tracked.

Usage:
python submit_grid.py --manifest manifests/a1.csv
python submit_grid.py --manifest manifests/a1.csv --dry-run   # print, don't submit
"""

import argparse
import csv
import subprocess
import sys

ENV_KEYS = [
    "run_id", "kind", "run_dir", "base_run", "images_dir", "labels_dir",
    "shells_dir", "splits_json", "p1_checkpoint", "band_mm",
    "epochs_p1", "lr_p1", "loss_p1", "bw_floor_p1", "bw_thr_p1", "bw_coef_p1", "gd_coef_p1",
    "epochs_p2", "lr_p2", "loss_p2", "bw_floor_p2", "bw_coef_p2",
    "seed",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--job-script", default="run_grid_task.sh")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"{len(rows)} rows in {args.manifest}")

    submitted = []
    for row in rows:
        export_pairs = [f"{k.upper()}={row[k]}" for k in ENV_KEYS if row.get(k, "") != ""]
        export_str = "ALL," + ",".join(export_pairs)

        cmd = ["sbatch", f"--export={export_str}", args.job_script]

        if args.dry_run:
            print(" ".join(cmd))
            job_id = "DRYRUN"
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FAILED to submit {row['run_id']}: {result.stderr.strip()}", file=sys.stderr)
                job_id = "FAILED"
            else:
                # sbatch prints "Submitted batch job 12345"
                job_id = result.stdout.strip().split()[-1]
                print(f"{row['run_id']}: job {job_id}")

        submitted.append({**row, "job_id": job_id})

    out_path = args.manifest.replace(".csv", "_submitted.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["job_id"])
        writer.writeheader()
        writer.writerows(submitted)
    print(f"\nJob IDs recorded in {out_path}")


if __name__ == "__main__":
    main()
