module load cuda
conda activate ~/.conda_envs/dynpartition

export PYTHONPATH="${HOME}/DynPartition"
python3 "${HOME}/DynPartition/dynpartition/partitioner/run_single_device.py"
python3 "${HOME}/DynPartition/dynpartition/partitioner/run_single_device.py" --with-cpu
python3 "${HOME}/DynPartition/dynpartition/x/test_data_trasfer_speeds.py" --with-cpu
python3 "${HOME}/DynPartition/dynpartition/partitioner/run_random_distribution.py"
python3 "${HOME}/DynPartition/dynpartition/partitioner/run_random_distribution.py" --with-cpu
