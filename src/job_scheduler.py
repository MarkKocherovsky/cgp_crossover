import os
import subprocess
import time
import json
from datetime import timedelta

from test_problems import Collection
from pathlib import Path


def get_job_duration(scheduler_job_name):
    """Fetch the elapsed time of a job using sacct and return it as a timedelta."""
    result = subprocess.run(
        ["sacct", "-j", str(scheduler_job_name), "--format=Elapsed", "--noheader"],
        stdout=subprocess.PIPE,
        text=True
    )
    elapsed = result.stdout.strip()  # Remove leading/trailing whitespace and newlines

    if not elapsed:
        raise ValueError("No elapsed time found. The job might not have started or is not tracked.")

    elapsed_times = elapsed.split()

    # Use the first valid elapsed time (you can modify this if you need to handle multiple entries)
    elapsed = elapsed_times[0]

    # Clean any extra whitespace/newlines
    elapsed = " ".join(elapsed.split())  # Remove excess whitespace and newlines

    # Split the elapsed time into parts
    time_parts = elapsed.split(":")

    try:
        if len(time_parts) == 3:  # Format: HH:MM:SS
            hours, minutes, seconds = map(int, time_parts)
        elif len(time_parts) == 2:  # Format: MM:SS
            hours = 0
            minutes, seconds = map(int, time_parts)
        elif len(time_parts) == 1:  # Format: SS
            hours = 0
            minutes = 0
            seconds = int(time_parts[0])
        else:
            raise ValueError(f"Unexpected elapsed time format: {elapsed}")
    except ValueError as e:
        raise ValueError(f"Error parsing elapsed time '{elapsed}': {e}")

    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def get_job_id():
    """Fetch the SLURM job ID from the environment."""
    job_id = os.getenv("SLURM_JOB_ID")
    if job_id is None:
        raise EnvironmentError("SLURM_JOB_ID not found. Are you running inside a SLURM job?")
    return job_id


import json

import json


def get_parameters(cfg: str = 'canonical'):
    """Load parameters from a JSON config file with default values."""
    defaults = {
        "mutation": "point",
        "selection": "elite",
        "max_g": 6000,
        "max_p": 1,
        "max_c": 4,
        "max_n": 64,
        "x_rate": 0.0,
        "m_rate": 1.0,
        "n_points": 1,
        "n_elites": 1,
        "t_size": 4,
        "p_dim": 1,
        "step_size": 100,
        "asexual_reproduction": False,
        "one_d": False,
    }

    cfg_src = f'configs/{cfg}.json'
    try:
        with open(cfg_src, 'r') as f:
            user_cfg = json.load(f)
            print('Config Loaded')
    except FileNotFoundError:
        print(f'{cfg_src} not found')
        exit()
        user_cfg = {}

    # Merge defaults with user config, giving priority to user values
    params = {**defaults, **user_cfg}

    return params


job_id = get_job_id()

# SLURM settings
MAX_JOBS = 990  # Maximum jobs allowed in queue/running
CHECKPOINT_FILE = "checkpoint.json"

# Problem configuration
functions = Collection()
# function_list = ['Koza1', 'Koza2', 'Koza3', 'Nguyen4', 'Nguyen5', 'Nguyen6', 'Nguyen7', 'Levy', 'Rastrigin', 'Ackley']
# function_list = ['Diabetes']
# function_list = ['California', 'Abalone', 'Airfoil']
# function_list = ['Rastrigin', 'Levy', 'Ackley', 'Nguyen7']
# function_list = ['Koza1', 'Koza2']
# function_list = ['Koza3', 'Nguyen4']
# function_list = ['Koza2', 'Nguyen6']
# function_list = ['Nguyen5', 'Nguyen4']
# function_list = ['Nguyen5', 'Nguyen6']
# function_list = ['Nguyen7', 'Ackley']
# function_list = ['Rastrigin', 'Levy']
# function_list = ['Nguyen6, Nguyen7']
# , 'Koza3', 'Nguyen4', 'Nguyen5', 'Nguyen6', 'Nguyen7']
# function_list = ['Griewank']
# function_list = ['Koza1']
# function_list = ['Levy', 'Ackley']
# function_list = ['Rastrigin']
function_list = ['Ackley', 'Levy', 'Rastrigin']
# xovers = ['n_point', 'uniform', 'subgraph', 'semantic_n_point', 'semantic_uniform', 'homologous_semantic_n_point',
#          'homologous_semantic_uniform']
# xovers = ['dnc_n_point', 'dnc_uniform', 'dnc_semantic_uniform', 'dnc_semantic_n_point']
# xovers = ['aligned_homologous_semantic_n_point']
# xovers = ['semantic_uniform', 'homologous_semantic_uniform', 'semantic_n_point', 'homologous_semantic_n_point']
# xovers = ['homologous_semantic_n_point', 'homologous_semantic_uniform']
# xovers = ['subgraph']
# xovers = ['homologous_semantic_n_point'] #, 'semantic_uniform']
# xovers = ['None']
# xovers = ['n_point']
# xovers = ['n_point', 'uniform', 'subgraph']
#          'semantic_n_point', 'aligned_homologous_semantic_n_point', 'aligned_semantic_uniform',
#          'aligned_homologous_semantic_uniform', 'homologous_semantic_uniform']
xovers = ['n_point', 'uniform']
# function_list = ['Koza1', 'Koza2']

# xovers = ['homologous_semantic_uniform', 'homologous_semantic_n_point']
# mutation = 'full'
# selection = 'elite_tournament'
# selection = 'competent_tournament'
output_dir = "/mnt/gs21/scratch/kocherov/Documents/cgp/output/logs/"
error_dir = "/mnt/gs21/scratch/kocherov/Documents/cgp/output/err/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(error_dir, exist_ok=True)

# params = get_parameters('one_d')
# params = get_parameters('canonical')
# params = get_parameters('crossover')
# params = get_parameters('crossover_full_mutation')
params = get_parameters('one_d_full_mutation')
# params = get_parameters('canonical_full_mutation')

# Parameters
mutation = params['mutation']
selection = params['selection']
max_g = params['max_g']  # max generations
max_p = params['max_p']  # max number of parents
max_c = params['max_c']  # max number of children
max_n = params['max_n']  # max number of function nodes
x_rate = params['x_rate']  # crossover rate
m_rate = params['m_rate']  # mutation rate
n_points = params['n_points']  # n-point crossover
n_elites = params['n_elites']  # number of elites for elitism
t_size = params['t_size']  # tournament size
p_dim = params['p_dim']  # problem dimension
step_size = params['step_size']  # report step size
asexual_reproduction = params['asexual_reproduction']  # whether asexual reproduction is permitted
one_d = params['one_d']  # whether crossover preserves nodes (False) or not (True)

print(params)
job_count = 0


def load_checkpoint():
    """Load progress from checkpoint, handling interrupted jobs separately."""
    if not os.path.exists(CHECKPOINT_FILE) or os.stat(CHECKPOINT_FILE).st_size == 0:
        return {"completed_jobs": [], "checkpointed_jobs": []}

    try:
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"completed_jobs": [], "checkpointed_jobs": []}


def save_checkpoint(state):
    """Safely save progress to the checkpoint file."""
    temp_file = CHECKPOINT_FILE + ".tmp"
    try:
        with open(temp_file, "w") as f:
            json.dump(state, f)
        os.replace(temp_file, CHECKPOINT_FILE)  # Atomic replace to prevent corruption
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


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


def resubmit():
    """Resubmit job_scheduler if SLURM time limit is nearly reached or job failed."""
    duration = get_job_duration(job_id)

    if duration >= timedelta(hours=3, minutes=50):
        print('Time limit nearly reached, checking logs before resubmitting...')

        # Check job log file for SLURM errors
        log_file = f"{output_dir}{job_id}.out"
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                log_contents = f.read()
                if "slurmstepd: error:" in log_contents:
                    print("Detected SLURM time limit error. Resubmitting...")
                    os.system('sbatch schedule_jobs.sb')
                    exit(0)
        else:
            print("Log file not found, assuming normal execution.")


# Load checkpoint state


state = load_checkpoint()
completed_jobs = set(state["completed_jobs"])

for function in function_list:
    f_no_space = function.replace(' ', '')
    for xover in xovers:
        Path(f'../output/{f_no_space}/{xover}/').mkdir(parents=True, exist_ok=True)
        for i in range(50):  # Create 50 jobs per function/xover combination
            job_name = f"kocherov_{f_no_space}_{xover}_{i}"

            # Skip already completed jobs
            # Before submitting a job, check if it was previously checkpointed
            if job_name in state["completed_jobs"]:
                continue  # Skip completed jobs

            # Wait until jobs in queue are below MAX_JOBS
            resubmit()
            while count_user_jobs() >= MAX_JOBS:
                print("Max job limit reached. Waiting...")
                time.sleep(60)  # Check every 60 seconds
                resubmit()
            # print(one_d)
            slurm_script = f"""#!/bin/bash --login
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}{job_name}.out
#SBATCH --error={error_dir}{job_name}.err
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --qos=scavenger  # Uncomment if using preemptible jobs

export MAMBA_EXE="/mnt/ffs24/home/kocherov/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="/mnt/ffs24/home/kocherov/micromamba"
micromamba activate cgp

# Move to working directory
cd /mnt/home/kocherov/Documents/cgp/src/

python3 -u run.py {i} {max_g} {max_n} {max_p} {max_c} {xover} {x_rate} {mutation} {m_rate} {selection} {function} --n_points {n_points} --n_elites {n_elites} --problem_dimensions {p_dim} --step_size {step_size} --tournament_size {t_size} --asexual_reproduction {asexual_reproduction} --one_dimensional_xover {one_d}

# Capture exit code
ret=$?

# Resubmit on timeout
if [[ -f "$SLURM_JOB_ID.out" ]]; then
    if grep -q "slurmstepd: error:" "$SLURM_JOB_ID.out"; then
        echo "Job exceeded time limit. Resubmitting..."
        sbatch $0
    fi
fi

conda deactivate
exit $ret
"""

            print(f"Preparing job {job_name}")

            # Write the SLURM script
            script_path = os.path.join('../output/slurm_files/', f"{job_name}.slurm")
            print(script_path)
            with open(script_path, "w") as f:
                f.write(slurm_script)

            # ✅ Ensure the file is flushed and closed before using it
            time.sleep(0.1)  # Optional: ensure the filesystem syncs

            # ✅ Check file exists and is non-empty
            if os.path.exists(script_path) and os.path.getsize(script_path) > 0:
                if job_name in state["checkpointed_jobs"]:
                    print(f"Resuming checkpointed job {job_name}...")
                    state["checkpointed_jobs"].remove(job_name)
                    save_checkpoint(state)

                # Submit the job
                os.system(f"sbatch {script_path}")

                # Mark job as checkpointed
                state["checkpointed_jobs"].append(job_name)
                save_checkpoint(state)

                print(f"Submitted job {job_name}")
                job_count += 1
            else:
                print(f"ERROR: SLURM script {script_path} is empty or missing. Skipping submission.")

                # Mark job as checkpointed (so we track incomplete jobs)
                state["checkpointed_jobs"].append(job_name)
                save_checkpoint(state)

                print(f"Submitted job {job_name}")
                job_count += 1
print('All Jobs Complete')
