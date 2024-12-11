import os
import subprocess
import time
from test_problems import Collection

# SLURM settings
MAX_JOBS = 1000  # Maximum jobs allowed in queue/running

# Problem configuration
functions = Collection()
function_list = functions.function_list.keys()

xovers = ['n_point', 'uniform', 'subgraph', 'semantic n_point', 'semantic uniform']
mutation = 'point'
selection = 'elite tournament'

output_dir = "../output/"
os.makedirs(output_dir, exist_ok=True)

# parameters
max_g = 100
max_p = 12
max_c = 12
max_n = 24
x_rate = 0.5
m_rate = 0.025
n_points = 1
n_elites = 1
t_size = 4
p_dim = 1
step_size = 10

job_count = 0


def count_user_jobs():
    """Counts the number of jobs in the SLURM queue for the current user."""
    user = os.getenv("USER")
    result = subprocess.run(
        ["squeue", "-u", user],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    # Exclude header and count jobs
    return len(result.stdout.strip().split("\n")) - 1


for function in function_list:
    for xover in xovers:
        for i in range(50):  # Create 50 jobs per function/xover combination
            # Wait until jobs in queue are below MAX_JOBS
            while count_user_jobs() >= MAX_JOBS:
                print("Max job limit reached. Waiting...")
                time.sleep(60)  # Check every 60 seconds

            job_name = f"{function}_{xover}_{i}"
            slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}{job_name}.out
#SBATCH --error={output_dir}{job_name}.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G

module purge
module load Conda/3
source ~/.bashrc
conda activate cgp

# Verify activation
if [[ "$(which python3)" != "/mnt/ufs18/home-220/kocherov/miniforge3/envs/cgp/bin/python3" ]]; then
    echo "Error: Conda environment 'cgp' not properly activated!"
    exit 1
fi

# Run the script
cd /mnt/home/kocherov/Documents/cgp/src/

# Run the script using the absolute path to Python in the activated environment
/mnt/ufs18/home-220/kocherov/miniforge3/envs/cgp/bin/python3 -u run.py {i} {max_g} {max_n} {max_p} {max_c} {xover} {x_rate} {mutation} {m_rate} {selection} {function} --n_points {n_points} --n_elites {n_elites} --problem_dimensions {p_dim} --step_size {step_size} --tournament_size {t_size}
conda deactivate
"""
            # Write the SLURM script
            script_path = os.path.join(output_dir, f"{job_name}.slurm")
            with open(script_path, "w") as f:
                f.write(slurm_script)

            # Submit the job
            os.system(f"sbatch {script_path}")
            print(f"Submitted job {job_name}")
            job_count += 1
