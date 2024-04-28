# multi-view-foul-recognition


## Generate pdf from markdown
```bash
pandoc koncepcja.md -s -o koncepcja.pdf -V colorlinks=true
```

## Running experiments on Athena

### Activate Python virtual environment
To ensure that all dependencies are correctly managed and isolated, activate the Python virtual environment using the following commands:
```bash
python3.9 -m venv /net/tscratch/people/plgmiloszl/.venv
source /net/tscratch/people/plgmiloszl/.venv/bin/activate
```

### Installing Requirements
Before running experiments, it's essential to install all required Python packages. This includes general dependencies listed in a `requirements.txt` file, if available, and specific GPU-accelerated libraries for PyTorch.
#### General Dependencies
To install the general dependencies, ensure you are in the activated virtual environment and run:
```bash
pip install -r requirements.txt
```

#### GPU-Accelerated Libraries
For experiments requiring GPU acceleration, such as those involving deep learning frameworks like PyTorch, use the provided script which installs the GPU-specific versions of PyTorch, torchvision, and torchaudio:
```bash
source install_gpu_cuda.sh
```

This script executes the following command to install the CUDA-enabled versions of the libraries, ensuring compatibility with NVIDIA GPUs:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Ensure that the CUDA version (`cu118` in this case) matches the CUDA version installed on your system or the one available in the cluster environment.

### Deactivating environments
```bash
deactivate  # For Python venv
```

### Check available storage
Since the home directory is limited to 10GB, it's important to use the temporary storage directory for larger datasets or outputs. Check the contents and available space using:
```bash
ls /net/tscratch/people/plgmiloszl/
```

### Submitting jobs to Slurm
To run experiments using the Slurm job scheduler, you can submit jobs as follows. Adjust the script and resource specifications according to your needs:
```bash
sbatch -A your_allocation -o slurm_output.log -p partition_name -t time_limit --array job_array -c num_cpus --gres gpu:num_gpus --mem memory_size --nodes num_nodes run_train_vars.sh
```

Replace placeholders with actual values:
- `your_allocation`: Your project's allocation ID.
- `partition_name`: The Slurm partition to use.
- `time_limit`: Maximum time per job (e.g., `360` for 360 minutes).
- `job_array`: Job array settings (e.g., `0-19` for 20 jobs).
- `num_cpus`: Number of CPUs per job.
- `num_gpus`: Number of GPUs per job.
- `memory_size`: Memory per job (e.g., `40G` for 40 GB).
- `num_nodes`: Number of nodes per job.

#### Example
Here is an example of a job submission command that was used based on the bash history:
```bash
sbatch -A plgzzsn2024-gpu-a100 -o slurm_%a.log -p plgrid-gpu-a100 -t 360 --array 0-19 -c 4 --gres gpu:1 --mem 40G --nodes 1 run_train_vars.sh
```

This command submits a job to the Slurm scheduler with the following specifications:
- Allocation ID: `plgzzsn2024-gpu-a100`
- Output log file: `slurm_%a.log` (where `%a` is replaced by the array job ID)
- Partition: `plgrid-gpu-a100`
- Time limit: `360` minutes
- Job array: `0-19` (20 jobs in total)
- CPUs per job: `4`
- GPUs per job: `1`
- Memory per job: `40G`
- Number of nodes per job: `1`

### Managing Slurm Jobs

#### Viewing Jobs
To view all your submitted jobs, use:
```bash
squeue -u your_username
```

Example:
```bash
squeue -u plgmiloszl
```

#### Cancelling Jobs
To cancel a specific job or all your jobs:
```bash
scancel job_id  # Replace 'job_id' with your specific job ID
scancel -u your_username  # Cancels all your jobs
```

#### Modifying Jobs
To modify the attributes of a job:
```bash
scontrol update JobId=job_id AttributeName=new_value  # Replace 'AttributeName' and 'new_value' appropriately
```
