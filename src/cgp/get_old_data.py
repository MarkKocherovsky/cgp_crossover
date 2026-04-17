import pickle
import pandas as pd
from pathlib import Path
import re

def extract_and_save_p_fits_autodiscover():
    input_dir = Path("/mnt/gs21/scratch/kocherov/Documents/cgp/output/intermediate_results")
    output_dir = Path("../output/")
    output_dir.mkdir(exist_ok=True)

    # Define known methods and problems
    methods = {'cgp_base', 'cgp_1x', 'cgp_uniform', 'cgp_dnc', 'cgp_dnc_1x', 'cgp_sgx', 'cgp_unx'}
    problems = {'Koza 1', 'Koza 2', 'Koza 3', 'Nguyen 4', 'Nguyen 5', 'Nguyen 6', 'Nguyen 7',
                'Ackley_1D', 'Rastrigin_1D', 'Levy_1D', 'Griewank_1D'}

    for path in input_dir.glob("*.pkl"):
        match = re.match(rf"({'|'.join(methods)})_({'|'.join(re.escape(p) for p in problems)})_results\.pkl", path.name)
        if not match:
            print(f"⚠️ Skipping unrecognized file: {path.name}")
            continue

        method, problem = match.groups()

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"❌ Failed to load {path.name}: {e}")
            continue

        if 'p_size' not in data:
            print(f"⚠️ 'nodes' missing in {path.name}")
            continue

        out_path = output_dir / f"intermediate_results_{problem}_{method}_size_old.csv"
        try:
            pd.DataFrame(data['p_size'][:, 3001]).to_csv(out_path, index=False)
            print(f"✅ Saved: {out_path}")
        except Exception as e:
            print(f"❌ Failed to save {out_path}: {e}")

extract_and_save_p_fits_autodiscover()

