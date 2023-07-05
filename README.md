### Note:

In this repo, we present the application of CASE-BASED REASONING FOR DRUG REPURPOSING

******************

Code for the AKBC'20 paper -- [A Simple Approach to Case-Based Reasoning in Knowledge Bases](https://openreview.net/forum?id=AEY9tRqlU7)

![CBR](image/teaser.png)

## Setup
### Download the experiment data from [here](http://iesl.cs.umass.edu/downloads/cbr-akbc-data), untar it in the root directory of the project.
The compressed size is ~401M and uncompressed size is ~716M. 
```bash
wget  http://iesl.cs.umass.edu/downloads/cbr-akbc-data/cbr-akbc-data.tar.gz && tar xvf cbr-akbc-data.tar.gz
```
### Set the pythonpath in root directory of the project
```
export PYTHONPATH=`pwd`
```

### To reproduce results from the paper, run the following:

```
WN18RR
python code/cbr.py --dataset_name WN18RR --test --k_adj 5 --max_num_programs 25

Sample output

[2020-06-30 01:48:46     Avg number of programs 22.41]
[2020-06-30 01:48:46     Avg number of answers after executing the programs: 88.83012354651163]
[2020-06-30 01:48:46     Accuracy (Loose): 0.6043395022335674]
[2020-06-30 01:48:46     Hits@1 0.3940650925335035]
[2020-06-30 01:48:46     Hits@3 0.4610721123165284]
[2020-06-30 01:48:46     Hits@5 0.4834077855775367]
[2020-06-30 01:48:46     Hits@10 0.5113273771537971]
[2020-06-30 01:48:46     MRR 0.4353462886214267]
[2020-06-30 01:48:46     Avg number of nn, that do not have the query relation: 0.0]
[2020-06-30 01:48:46     Avg num of returned nearest neighbors: 4.8146]
[2020-06-30 01:48:46     Avg number of programs that do not execute per query: 9.7958]
``` 


### To run for a new dataset, first a random subgraph around each entity needs to be collected and written to disk.
### In our experiments, we just collect 1000 random paths aorund each entity by doing DFS.
```
python code/data/get_paths.py --dataset_name <insert_dataset_name> --num_paths_to_collect 1000 --data_dir cbr_akbc_data
```

### Citation
````
@inproceedings{cbr_akbc,
  title = {A Simple Approach to Case-Based Reasoning in Knowledge Bases},
  author = {Das, Rajarshi and Godbole, Ameya and Dhuliawala, Shehzaad and Zaheer, Manzil and McCallum, Andrew},
  booktitle = {AKBC},
  year = 2020
}
````
 
# CBRonMRN
