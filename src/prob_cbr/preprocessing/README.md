## Preprocessing
There are several preprocessing steps which has to be done once before running the model. They include
1. Collecting subgraphs around each entity
2. Cluster entities
3. Compute ent-ent similarity
4. Compute the prior maps
5. Compute the precision maps

### 1. Collecting subgraphs around each entity
We collect a subgraph around each entity in the KG. In practice, we gather a set of paths around each entity. This needs to be done once offline. If your KG 
is relatively small, you can simply run
```
python src/prob_cbr/preprocessing/preprocessing.py --get_paths_parallel --add_inv_edges --current_job=0 --total_jobs=1 --dataset_name=obl2021 --num_paths_to_collect=10000 --data_dir=/home/rajarshi/Dropbox/research/Open-BIo-Link/
```
However, for large KGs, it would save a lot of time to run multiple processes in parallel.
Use the `job_id` and `total_jobs` arguments to run parallel process.
```
python src/prob_cbr/preprocessing/preprocessing.py --get_paths_parallel --add_inv_edges --current_job=0 --total_jobs=100 --dataset_name=obl2021 --num_paths_to_collect=10000 --data_dir=/home/rajarshi/Dropbox/research/Open-BIo-Link/ 
``` 
For our setup we use wandb and slurm to parallelize. If you have a similar setup refer to `src/prob_cbr/preprocessing/{processing_sweep_config.yaml, sbatch_run.sh}`.

### 2. Cluster entities
TBD (righnow linkage is set to 0, i.e. 1 cluster for all entities)
```
srun --mem=10G --partition=longq python src/prob_cbr/preprocessing/preprocessing.py --dataset_name obl2021 --data_dir ./   --linkage 0 --do_clustering  --use_wandb 1
```

### 3. Compute ent-ent similarity
```
srun --mem=10G --partition=longq python src/prob_cbr/preprocessing/preprocessing.py --dataset_name obl2021 --data_dir ./   --linkage 0 --calculate_ent_similarity  --use_wandb 1 --sim_batch_size 1024
```

### 4. Compute the prior maps
```
python src/prob_cbr/preprocessing/preprocessing.py --calculate_prior_map_parallel --add_inv_edges --current_job=0 --total_jobs=1 --dataset_name=obl2021 --num_paths_to_collect=10000 --data_dir=/home/rajarshi/Dropbox/research/Open-BIo-Link/ 
``` 
You can again, parallelize this by running multiple processes and setting ``total_jobs`` equal to the number of processes and setting ``current_job`` respectively
To combine the prior maps, run:
```
python src/prob_cbr/preprocessing/preprocessing.py --combine_prior_map --dataset_name=obl2021 --num_paths_to_collect=10000 --data_dir=/home/rajarshi/Dropbox/research/Open-BIo-Link/ 
```
### 5. Compute the precision maps
```
``` 
You can again, parallelize this by running multiple processes and setting ``total_jobs`` equal to the number of processes and setting ``current_job`` respectively
To combine the precision maps, run:
```
python src/prob_cbr/preprocessing/preprocessing.py --combine_precision_map --dataset_name=obl2021 --num_paths_to_collect=10000 --data_dir=/home/rajarshi/Dropbox/research/Open-BIo-Link/ 

