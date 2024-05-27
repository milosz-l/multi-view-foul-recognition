# multi-view-foul-recognition

## Generate pdf from markdown
```bash
pandoc koncepcja.md -s -o koncepcja.pdf -V colorlinks=true
```

## Submitting predictions to Soccernet evalai challange - example
```bash
source /net/tscratch/people/$USER/.venv/bin/activate
evalai challenge 2201 phase 4380 submit --file "/net/tscratch/people/plgmiloszl/outputs/predicitions_test_2024-05-19 18:51:29.542772.json"  --large --public
```

## Running experiments on Athena (first log in)

### Step 1: Choosing right folder
In home directory for each user there is only 10GB of space. It is advised to run experiments from scratch directory (there is risk of data loss, but the risk is low and the experiments are stored in wandb anyway).

So after logging to plgrid, go to scratch directory and clone the project from git.
```bash
cd /net/tscratch/people/$USER/
git clone git@github.com:milosz-l/multi-view-foul-recognition.git
cd multi-view-foul-recognition
```


### Step 2: Activate Python virtual environment
To ensure that all dependencies are correctly managed and isolated, activate the Python virtual environment using the following commands:
```bash
python3.9 -m venv /net/tscratch/people/$USER/.venv
source /net/tscratch/people/$USER/.venv/bin/activate
```

### Step 3: Installing Requirements
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

### (Optional) Deactivating environments
```bash
deactivate  # For Python venv
```

### (Optional) Check available storage
Since the home directory is limited to 10GB, it's important to use the temporary storage directory for larger datasets or outputs. Check the contents and available space using:
```bash
ls /net/tscratch/people/$USER/
```

### Step 4: Download data
Make sure that you created the .env file with the SoccerNet password given in the following format:
```
SNDL_PASSWORD="123"
```

Run `download_data.sh` to download the data.
```bash
source download_data.sh
```

### Step 5: Set the wandb cache folder path to scratch instead of home
We do this only for wandb, because its cache takes up much more space compared to other packages.
```bash
export WANDB_CACHE_DIR=/net/tscratch/people/$USER/.cache/wandb
```

### Step 6: Submitting jobs to Slurm
To run experiments using the Slurm job scheduler, you can submit jobs as follows. Adjust the script and resource specifications according to your needs:
```bash
sbatch -A plgzzsn2024-gpu-a100 -o slurm_%a.log -p plgrid-gpu-a100 -t 360 -c 4 --gres gpu:1 --mem 40G --nodes 1 run_train_vars.sh
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
source sbatch.sh
```
<!-- ```bash
sbatch -A plgzzsn2024-gpu-a100 -o slurm_%a.log -p plgrid-gpu-a100 -t 360 --array 0-1 -c 4 --gres gpu:1 --mem 40G --nodes 1 run_train_vars.sh
``` -->

### Managing Slurm Jobs

#### Viewing Jobs
To view all your submitted jobs, use:
```bash
squeue -u $USER
```

#### Cancelling Jobs
To cancel a specific job or all your jobs:
```bash
scancel job_id  # Replace 'job_id' with your specific job ID
scancel -u $USER  # Cancels all your jobs
```

#### Modifying Jobs
To modify the attributes of a job:
```bash
scontrol update JobId=job_id AttributeName=new_value  # Replace 'AttributeName' and 'new_value' appropriately
```

### Additional Commands

- **hpc-fs**: Manages and inspects the HPC file system. Usage: `hpc-fs`.
- **hpc-jobs**: Manages and views the status of HPC jobs. Usage: `hpc-jobs`.
- **du**: Estimates file space usage. Usage: `du -h --max-depth=1 $HOME | sort -rh`.

## Running experiments on Athena (after the first log in)
You only need these to run experiments:
```bash
cd /net/tscratch/people/$USER/multi-view-foul-recognition
export WANDB_CACHE_DIR=/net/tscratch/people/$USER/.cache/wandb
source sbatch.sh
```


# Additional: Running vanilla VARS
Example for running tests in `VARS_test` directory:
```bash
cd /net/tscratch/people/$USER/VARS_test
python3.9 -m venv vars
source /net/tscratch/people/$USER/VARS_test/vars/bin/activate
source /net/tscratch/people/$USER/multi-view-foul-recognition/install_gpu_cuda.sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install SoccerNet
```

Create a file with [these](https://github.com/SoccerNet/sn-mvfoul/blob/main/VARS%20model/requirements.txt) requirements and install them:
```
pip install -r requirements.txt
pip install pyav
```

## Baseline hyperaparameters
if you use the code from the GitHub repository and run:

```
python main.py --LR 5e-5 --step_size 3 --gamma 0.3 --pooling_type "attention" --start_frame 63 --end_frame 87 --fps 17 --path "path/to/dataset" --pre_model "mvit_v2_s"
```

you should get similar results.