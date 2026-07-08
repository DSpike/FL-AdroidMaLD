"""
Re-run the 4 GWO seeds (42, 123, 456, 789) that were missing from sota_resume.log
due to completing before the resume point (42/123/456) or being incomplete (789).

Captures stdout to gwo_missing_seeds.log to extract selection events.
Seeds 42/123/456 CSVs already exist and are complete (50 rows) -- the experiment
script will overwrite them, which is fine since they are deterministic given the seed.
"""

import subprocess, sys, os

BASE         = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT       = os.path.join(BASE, 'run_experiment_20c_dir01.py')
RESULTS_BASE = os.path.join(BASE, 'results', 'sota_comparison')
LOG_OUT      = os.path.join(BASE, 'gwo_missing_seeds.log')
MISSING_SEEDS = [42, 123, 456, 789]

print(f"Re-running GWO for seeds: {MISSING_SEEDS}")
print(f"Selection events will be captured to: {LOG_OUT}")

with open(LOG_OUT, 'w') as logf:
    for seed in MISSING_SEEDS:
        rdir = os.path.join(RESULTS_BASE, 'gwo8', f'seed{seed}')
        csv_path = os.path.join(rdir, f'gwo8_seed{seed}_rounds.csv')

        # Remove existing CSV so it gets re-run and we get fresh stdout
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"Removed existing CSV for seed {seed} (will re-run)")

        print(f"\n{'='*60}")
        print(f"  Running GWO seed={seed}")
        print(f"{'='*60}")
        logf.write(f"\n{'='*60}\n")
        logf.write(f" GWO seed={seed}\n")
        logf.write(f"{'='*60}\n")
        logf.flush()

        result = subprocess.run(
            [sys.executable, SCRIPT,
             '--method',      'gwo8',
             '--seed',        str(seed),
             '--k_select',    '8',
             '--lam',         '0.2',
             '--delta',       '0.3',
             '--results_dir', rdir],
            cwd=BASE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        logf.write(result.stdout)
        logf.flush()
        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        print(f"  Seed {seed}: {status}")

print(f"\nDone. Selection events in: {LOG_OUT}")
