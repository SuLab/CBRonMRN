import os
import sys
import pandas as pd

data_graph = pd.read_csv("/home/msinha/ProbCBRforMRN/prob-cbr-data/data/MRN_ind_with_CtD_len4/graph.txt", sep="\t", names=['e1','r','e2'])
data_test = pd.read_csv("/home/msinha/ProbCBRforMRN/prob-cbr-data/data/MRN_ind_with_CtD_len4/test.txt", sep="\t", names=['e1','r','e2'])

data = data_graph.append(data_test)

e1 = sys.argv[1]
rel = sys.argv[2]

#e1 = "CHEBI:6640"
#rel = "indication"

e2 =data[(data.e1==e1) & (data["r"]==rel)].e2.head(1)
e2=e2.to_string(index=False)

if e2 == 'Series([], )':
    print("not found")
else:
    with open('/home/msinha/ProbCBRforMRN/prob-cbr-data/data/MRN_ind_with_CtD_len4/test5.txt', 'w') as f:
        f.write(e1 + "\t" + rel + "\t" + e2)
    os.system("python /home/msinha/ProbCBRforMRN/src/prob_cbr/pr_cbr10.py --dataset MRN_ind_with_CtD_len4 --data_dir prob-cbr-data/ --expt_dir ../prob-cbr-expts/ --test --test_file_name test5.txt   --max_num_programs 80  --k_adj 10 --linkage 0 --max_num_programs 5000 --max_path_len 4 --prevent_loops 1")
    
    