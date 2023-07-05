import argparse
import numpy as np
from scipy.special import logsumexp
import os
from tqdm import tqdm
import scipy.sparse
from collections import defaultdict
import pickle
import torch
import uuid
from typing import *
import logging
import json
import pandas as pd
import sys
import wandb
from src.prob_cbr.preprocessing.preprocessing import combine_path_splits
from src.prob_cbr.utils import get_programs, create_sparse_adj_mats, execute_one_program
from src.prob_cbr.data.data_utils import create_vocab, load_vocab, load_data, get_unique_entities, \
    read_graph, get_entities_group_by_relation, get_inv_relation, load_data_all_triples, create_adj_list

logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

MRN_nodes = pd.read_csv('/home/msinha/CBR-AKBC/cbr-akbc-data/data/MRN/node_biolink.csv', dtype=str)
class ProbCBR(object):
    def __init__(self, args, train_map, eval_map, entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, eval_vocab,
                 eval_rev_vocab, all_paths, rel_ent_map, per_relation_config: Union[None, dict]):
        self.args = args
        self.eval_map = eval_map
        self.train_map = train_map
        self.all_zero_ctr = []
        self.all_num_ret_nn = []
        self.entity_vocab, self.rev_entity_vocab, self.rel_vocab, self.rev_rel_vocab = entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab
        self.eval_vocab, self.eval_rev_vocab = eval_vocab, eval_rev_vocab
        self.all_paths = all_paths
        self.rel_ent_map = rel_ent_map
        self.per_relation_config = per_relation_config
        self.num_non_executable_programs = []
        self.nearest_neighbor_1_hop = None
        #logger.info("Building sparse adjacency matrices")
        self.sparse_adj_mats = create_sparse_adj_mats(self.train_map, self.entity_vocab, self.rel_vocab)
        self.top_query_preds = {}

    def set_nearest_neighbor_1_hop(self, nearest_neighbor_1_hop):
        self.nearest_neighbor_1_hop = nearest_neighbor_1_hop

    def get_nearest_neighbor_inner_product(self, e1: str, r: str, k: Optional[int] = 5) -> Union[List[str], None]:
        try:
            nearest_entities = [self.rev_entity_vocab[e] for e in
                                self.nearest_neighbor_1_hop[self.eval_vocab[e1]].tolist()]
            # remove e1 from the set of k-nearest neighbors if it is there.
            nearest_entities = [nn for nn in nearest_entities if nn != e1]
            # making sure, that the similar entities also have the query relation
            ctr = 0
            temp = []
            for nn in nearest_entities:
                if ctr == k:
                    break
                if len(self.train_map[nn, r]) > 0:
                    temp.append(nn)
                    ctr += 1
            nearest_entities = temp
        except KeyError:
            return None
        return nearest_entities

    def get_programs_from_nearest_neighbors(self, e1: str, r: str, nn_func: Callable, num_nn: Optional[int] = 5):
        all_programs = []
        nearest_entities = nn_func(e1, r, k=num_nn)
        use_cheat_neighbors_for_r = self.args.cheat_neighbors if self.per_relation_config is None else \
            self.per_relation_config[r]["cheat_neighbors"]
        if (nearest_entities is None or len(nearest_entities) == 0) and use_cheat_neighbors_for_r:
            num_ent_with_r = len(self.rel_ent_map[r])
            if num_ent_with_r > 0:
                if num_ent_with_r < num_nn:
                    nearest_entities = self.rel_ent_map[r]
                else:
                    random_idx = np.random.choice(num_ent_with_r, num_nn, replace=False)
                    nearest_entities = [self.rel_ent_map[r][r_idx] for r_idx in random_idx]
        if nearest_entities is None or len(nearest_entities) == 0:
            self.all_num_ret_nn.append(0)
            return []
        self.all_num_ret_nn.append(len(nearest_entities))
        zero_ctr = 0
        for e in nearest_entities:
            if len(self.train_map[(e, r)]) > 0:
                paths_e = self.all_paths[e]  # get the collected 3 hop paths around e
                nn_answers = self.train_map[(e, r)]
                for nn_ans in nn_answers:
                    all_programs += get_programs(e, nn_ans, paths_e)
            elif len(self.train_map[(e, r)]) == 0:
                zero_ctr += 1
        self.all_zero_ctr.append(zero_ctr)
        return all_programs

    def rank_programs(self, list_programs: List[List[str]], r: str) -> List[List[str]]:
        """
        Rank programs.
        """
        # sort it by the path score
        unique_programs = set()
        for p in list_programs:
            unique_programs.add(tuple(p))
        # now get the score of each path
        path_and_scores = []
        use_only_precision_scores_for_r = self.args.use_only_precision_scores if self.per_relation_config is None \
            else self.per_relation_config[r]["use_only_precision_scores"]
        for p in unique_programs:
            try:
                if use_only_precision_scores_for_r:
                    path_and_scores.append((p, self.args.precision_map[self.c][r][p]))
                else:
                    path_and_scores.append((p, self.args.path_prior_map_per_relation[self.c][r][p] *
                                            self.args.precision_map[self.c][r][p]))
            except KeyError:
                # TODO: Fix key error
                if len(p) == 1 and p[0] == r:
                    continue  # ignore query relation
                else:
                    # use the fall back score
                    try:
                        c = 0
                        if use_only_precision_scores_for_r:
                            score = self.args.precision_map_fallback[c][r][p]
                        else:
                            score = self.args.path_prior_map_per_relation_fallback[c][r][p] * \
                                    self.args.precision_map_fallback[c][r][p]
                        path_and_scores.append((p, score))
                    except KeyError:
                        # still a path or rel is missing.
                        path_and_scores.append((p, 0))

        # sort wrt counts
        sorted_programs = [k for k, v in sorted(path_and_scores, key=lambda item: -item[1])]

        return sorted_programs

    def execute_programs(self, e: str, r: str, path_list: List[List[str]], max_branch: Optional[int] = 1000) \
            -> Tuple[List[Tuple[np.ndarray, float, List[str]]], List[List[str]]]:

        def _fall_back(r, p):
            """
            When a cluster does not have a query relation (because it was not seen during counting)
            or if a path is not found, then fall back to no cluster statistics
            :param r:
            :param p:
            :return:
            """
            c = 0  # one cluster for all entity
            try:
                score = self.args.path_prior_map_per_relation_fallback[c][r][p] * \
                        self.args.precision_map_fallback[c][r][p]
            except KeyError:
                # either the path or relation is missing from the fall back map as well
                score = 0
            return score

        all_answers = []
        not_executed_paths = []
        execution_fail_counter = 0
        executed_path_counter = 0
        max_num_programs_for_r = self.args.max_num_programs if self.per_relation_config is None else \
            self.per_relation_config[r]["max_num_programs"]
        for path in path_list:
            if executed_path_counter == max_num_programs_for_r:
                break
            ans = execute_one_program(self.sparse_adj_mats, self.entity_vocab, e, path)
            if self.args.use_path_counts:
                try:
                    if path in self.args.path_prior_map_per_relation[self.c][r] and path in \
                            self.args.precision_map[self.c][r]:
                        path_score = self.args.path_prior_map_per_relation[self.c][r][path] * \
                                     self.args.precision_map[self.c][r][path]
                    else:
                        # logger.info("This path was not there in the cluster for the relation.")
                        path_score = _fall_back(r, path)
                except KeyError:
                    # logger.info("Looks like the relation was not found in the cluster, have to fall back")
                    # fallback to the global scores
                    path_score = _fall_back(r, path)
            else:
                path_score = 1
            path = tuple(path)
            if len(np.nonzero(ans)[0]) == 0:
                not_executed_paths.append(path)
                execution_fail_counter += 1
            else:
                executed_path_counter += 1
            all_answers += [(ans, path_score, path)]
        #np.savetxt("all_answers.csv", all_answers, delimiter=",", fmt='%s')
        self.num_non_executable_programs.append(execution_fail_counter)
        return all_answers, not_executed_paths

    def rank_answers(self, list_answers: List[Tuple[np.ndarray, float, List[str]]], aggr_type1="none",
                     aggr_type2="sum") -> List[
        str]:
        """
        Different ways to re-rank answers
        """

        def rank_entities_by_max_score(score_map):
            """
            sorts wrt top value. If there are same values, then it sorts wrt the second value
            :param score_map:
            :return:
            """
            # sort wrt the max value
            if len(score_map) == 0:
                return []
            sorted_score_map = sorted(score_map.items(), key=lambda kv: -kv[1][0])
            sorted_score_map_second_round = []
            temp = []
            curr_val = sorted_score_map[0][1][0]  # value of the first
            for (k, v) in sorted_score_map:
                if v[0] == curr_val:
                    temp.append((k, v))
                else:
                    sorted_temp = sorted(temp, key=lambda kv: -kv[1][1] if len(
                        kv[1]) > 1 else 1)  # sort wrt second highest score
                    sorted_score_map_second_round += sorted_temp
                    temp = [(k, v)]  # clear temp and add new val
                    curr_val = v[0]  # calculate new curr_val
            # do the same for remaining elements in temp
            if len(temp) > 0:
                sorted_temp = sorted(temp,
                                     key=lambda kv: -kv[1][1] if len(kv[1]) > 1 else 1)  # sort wrt second highest score
                sorted_score_map_second_round += sorted_temp
            return sorted_score_map_second_round

        count_map = {}
        uniq_entities = set()
        for e_vec, e_score, path in list_answers:
            path_answers = [(self.rev_entity_vocab[d_e], e_vec[d_e]) for d_e in np.nonzero(e_vec)[0]]
            for e, e_c in path_answers:
                if e not in count_map:
                    count_map[e] = {}
                if aggr_type1 == "none":
                    count_map[e][path] = e_score  # just count once for a path type.
                elif aggr_type1 == "sum":
                    count_map[e][path] = e_score * e_c  # aggregate for each path
                else:
                    raise NotImplementedError("{} aggr_type1 is invalid".format(aggr_type1))
                uniq_entities.add(e)
        score_map = defaultdict(int)
        for e, path_scores_map in count_map.items():
            p_scores = [v for k, v in path_scores_map.items()]
            if aggr_type2 == "sum":
                score_map[e] = np.sum(p_scores)
            elif aggr_type2 == "max":
                score_map[e] = sorted(p_scores, reverse=True)
            elif aggr_type2 == "noisy_or":
                score_map[e] = 1 - np.prod(1 - np.asarray(p_scores))
            elif aggr_type2 == "logsumexp":
                score_map[e] = logsumexp(p_scores)
            else:
                raise NotImplementedError("{} aggr_type2 is invalid".format(aggr_type2))
        if aggr_type2 == "max":
            sorted_entities_by_val = rank_entities_by_max_score(score_map)
        else:
            sorted_entities_by_val = sorted(score_map.items(), key=lambda kv: -kv[1])
        return sorted_entities_by_val

    @staticmethod
    def get_rank_in_list(e, predicted_answers):
        for i, e_to_check in enumerate(predicted_answers):
            if e == e_to_check:
                return i + 1
        return -1



    def do_symbolic_case_based_reasoning(self):
        num_programs = []
        num_answers = []
        all_acc = []
        non_zero_ctr = 0
        hits_10, hits_5, hits_3, hits_1, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        per_relation_scores = {}  # map of performance per relation
        per_relation_query_count = {}
        total_examples = 0
        learnt_programs = defaultdict(lambda: defaultdict(int))  # for each query relation, a map of programs to count
        all_data =[]
        for ex_ctr, ((e1, r), e2_list) in enumerate(tqdm(self.eval_map.items())):
            #logger.info("Executing query {}".format(ex_ctr))
            orig_train_e2_list = self.train_map[(e1, r)]
            temp_train_e2_list = []
            for e2 in orig_train_e2_list:
                if e2 in e2_list:
                    continue
                temp_train_e2_list.append(e2)
            self.train_map[(e1, r)] = temp_train_e2_list
             #also remove (e2, r^-1, e1)
            r_inv = get_inv_relation(r, args.dataset_name)
            temp_map = {}  # map from (e2, r_inv) -> outgoing nodes
            for e2 in e2_list:
                temp_map[(e2, r_inv)] = self.train_map[e2, r_inv]
                temp_list = []
                for e1_dash in self.train_map[e2, r_inv]:
                    if e1_dash == e1:
                        continue
                    else:
                        temp_list.append(e1_dash)
                self.train_map[e2, r_inv] = temp_list

            total_examples += len(e2_list)
            if e1 not in self.entity_vocab:
                all_acc += [0.0] * len(e2_list)
                 #put it back
                self.train_map[(e1, r)] = orig_train_e2_list
                for e2 in e2_list:
                    self.train_map[(e2, r_inv)] = temp_map[(e2, r_inv)]
                continue  # this entity was not seen during train; skip?
            self.c = self.args.cluster_assignments[self.entity_vocab[e1]]
            num_nn_for_r = self.args.k_adj if self.per_relation_config is None else self.per_relation_config[r]["k_adj"]
            all_programs = self.get_programs_from_nearest_neighbors(e1, r, self.get_nearest_neighbor_inner_product,
                                                                    num_nn=num_nn_for_r)
            for p in all_programs:
                if p[0] == r:
                    continue
                if r not in learnt_programs:
                    learnt_programs[r] = {}
                p = tuple(p)
                if p not in learnt_programs[r]:
                    learnt_programs[r][p] = 0
                learnt_programs[r][p] += 1
                
            temp = []
            for p in all_programs:
                if len(p) == 1 and p[0] == r:
                    continue
                temp.append(p)
            all_programs = temp



            all_uniq_programs = self.rank_programs(all_programs, r)
            

            
            num_programs.append(len(all_uniq_programs))
            # Now execute the program
            answers, not_executed_programs = self.execute_programs(e1, r, all_uniq_programs, max_branch=args.max_branch)
            aggr_type1_for_r = self.args.aggr_type1 if self.per_relation_config is None \
                else self.per_relation_config[r]["aggr_type1"]
            aggr_type2_for_r = self.args.aggr_type2 if self.per_relation_config is None \
                else self.per_relation_config[r]["aggr_type2"]
            answers = self.rank_answers(answers,
                                        aggr_type1_for_r,
                                        aggr_type2_for_r)
            
            predicted_answers = [(e, float(score)) for e, score in answers]
            predicted_answers_highscore = list()
            predicted_answers = dict(predicted_answers)
            print(predicted_answers)
            itemMaxValue = max(predicted_answers.items(), key=lambda x: x[1],default=0)
            for key, value in predicted_answers.items():
                if value == itemMaxValue[1]:
                     predicted_answers_highscore.append((key,value))
            predicted_answers_highscore = dict(predicted_answers_highscore)

            


def main(args):
    dataset_name = 'MRN_ind_with_CtD_len4'
    #logger.info("==========={}============".format(dataset_name))
    data_dir = 'prob-cbr-data/data/MRN_ind_with_CtD_len4'
    subgraph_dir = os.path.join(args.data_dir, "subgraphs", dataset_name,
                                "paths_{}".format(args.num_paths_around_entities))
    kg_file = 'prob-cbr-data/data/MRN_ind_with_CtD_len4/graph.txt'
    args.test_file = 'prob-cbr-data/data/MRN_ind_with_CtD_len4/test5.txt'
    args.dev_file = 'prob-cbr-data/data/MRN_ind_with_CtD_len4/dev.txt'
    #if args.small:
    #    if args.input_file_name is not None:
    #        if args.test:
    #            args.test_file = os.path.join(data_dir, "inputs", "test", args.input_file_name + ".small")
    #            args.dev_file = os.path.join(data_dir, "dev.txt.small")
    #        else:
    #            args.dev_file = os.path.join(data_dir, "inputs", "valid", args.input_file_name + ".small")
    #            args.test_file = os.path.join(data_dir, "test.txt")
    #    elif args.specific_rel is not None:
    #        args.dev_file = os.path.join(data_dir, f"dev.{args.specific_rel}.txt.small")
    #        args.test_file = os.path.join(data_dir, "test.txt")
    #    else:
     #       args.dev_file = os.path.join(data_dir, "dev.txt.small")
     #       args.test_file = os.path.join(data_dir, "test.txt")
    #else:
     #   if args.input_file_name is not None:
      #      if args.test:
       #         args.test_file = os.path.join(data_dir, "inputs", "test", args.input_file_name)
        #        args.dev_file = os.path.join(data_dir, "dev.txt")
         #   else:
          #      args.dev_file = os.path.join(data_dir, "inputs", "valid", args.input_file_name)
           #     args.test_file = os.path.join(data_dir, "test.txt")
        #elif args.specific_rel is not None:
        #    args.dev_file = os.path.join(data_dir, f"dev.{args.specific_rel}.txt")
        #    args.test_file = os.path.join(data_dir, "test.txt") if not args.test_file_name \
         #       else os.path.join(data_dir, args.test_file_name)
       # else:
         #   args.dev_file = os.path.join(data_dir, "dev.txt")
         #   args.test_file = os.path.join(data_dir, "test.txt") if not args.test_file_name \
          #      else os.path.join(data_dir, args.test_file_name)

    args.train_file = 'prob-cbr-data/data/MRN_ind_with_CtD_len4/train.txt'
    #logger.info("Loading train map")
    train_map = load_data(kg_file)
    #logger.info("Loading dev map")
    dev_map = load_data(args.dev_file, True if args.specific_rel is None else False)
    #logger.info("Loading test map")
    test_map = load_data(args.test_file)
    eval_map = dev_map
    eval_file = args.dev_file
    if args.test:
        eval_map = test_map
        eval_file = args.test_file
    rel_ent_map = get_entities_group_by_relation(args.train_file)

    #logger.info("=========Config:============")
    #logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    if args.per_relation_config_file is not None and os.path.exists(args.per_relation_config_file):
        per_relation_config = json.load(open(args.per_relation_config_file))
        #logger.info("=========Per Relation Config:============")
        #logger.info(json.dumps(per_relation_config, indent=1, sort_keys=True))
    else:
        per_relation_config = None
    #logger.info("Loading vocabs...")
    entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, eval_vocab, eval_rev_vocab = load_vocab(data_dir)
    # making these part of args for easier access #hack
    args.entity_vocab = entity_vocab
    args.rel_vocab = rel_vocab
    args.rev_entity_vocab = rev_entity_vocab
    args.rev_rel_vocab = rev_rel_vocab
    args.train_map = train_map
    args.dev_map = dev_map
    args.test_map = test_map

    #logger.info("Loading combined train/dev/test map for filtered eval")
    all_kg_map = load_data_all_triples(args.train_file, os.path.join(data_dir, 'dev.txt'),
                                       os.path.join(data_dir, 'test.txt'))
    args.all_kg_map = all_kg_map

    ########### Load all paths ###########
    file_prefix = "paths_{}_path_len_{}_".format(args.num_paths_around_entities, args.max_path_len)
    all_paths = combine_path_splits(subgraph_dir, file_prefix=file_prefix)

    prob_cbr_agent = ProbCBR(args, train_map, eval_map, entity_vocab, rev_entity_vocab, rel_vocab,
                             rev_rel_vocab, eval_vocab, eval_rev_vocab, all_paths, rel_ent_map, per_relation_config)
    ########### entity sim ###########
    if os.path.exists(os.path.join(args.data_dir, "data", args.dataset_name, "ent_sim.pkl")):
        with open(os.path.join(args.data_dir, "data", args.dataset_name, "ent_sim.pkl"), "rb") as fin:
            sim_and_ind = pickle.load(fin)
            sim = sim_and_ind["sim"]
            arg_sim = sim_and_ind["arg_sim"]
    else:
        logger.info(
            "Entity similarity matrix not found at {}. Please run the preprocessing script first to generate this matrix...".format(
                os.path.join(args.data_dir, "data", args.dataset_name, "ent_sim.pkl")))
        sys.exit(1)
    assert arg_sim is not None
    prob_cbr_agent.set_nearest_neighbor_1_hop(arg_sim)

    ########### cluster entities ###########
    dir_name = os.path.join(args.data_dir, "data", args.dataset_name, "linkage={}".format(args.linkage))
    cluster_file_name = os.path.join(dir_name, "cluster_assignments.pkl")
    if os.path.exists(cluster_file_name):
        with open(cluster_file_name, "rb") as fin:
            args.cluster_assignments = pickle.load(fin)
    else:
       # logger.info(
      #      "Clustering file not found at {}. Please run the preprocessing script first".format(cluster_file_name))
        sys.exit(1)

    ########### load prior maps ###########
    path_prior_map_filenm = os.path.join(data_dir, "linkage={}".format(args.linkage), "prior_maps",
                                         "path_{}".format(args.num_paths_around_entities), "path_prior_map1.pkl")
   # logger.info("Loading path prior weights")
    if os.path.exists(path_prior_map_filenm):
        with open(path_prior_map_filenm, "rb") as fin:
            args.path_prior_map_per_relation = pickle.load(fin)
    else:
        logger.info(
            "Path prior files not found at {}. Please run the preprocessing script".format(path_prior_map_filenm))

    ########### load prior maps (fall-back) ###########
    linkage_bck = args.linkage
    args.linkage = 0.0
    bck_dir_name = os.path.join(data_dir, "linkage={}".format(args.linkage), "prior_maps",
                                "path_{}".format(args.num_paths_around_entities))
    path_prior_map_filenm_fallback = os.path.join(bck_dir_name, "path_prior_map1.pkl")
    if os.path.exists(bck_dir_name):
     #   logger.info("Loading fall-back path prior weights")
        with open(path_prior_map_filenm_fallback, "rb") as fin:
            args.path_prior_map_per_relation_fallback = pickle.load(fin)
    else:
        logger.info("Fall-back path prior weights not found at {}. Please run the preprocessing script".format(
            path_prior_map_filenm_fallback))
    args.linkage = linkage_bck

    ########### load precision maps ###########
    precision_map_filenm = os.path.join(data_dir, "linkage={}".format(args.linkage), "precision_maps",
                                        "path_{}".format(args.num_paths_around_entities), "precision_map.pkl")
   # logger.info("Loading precision map")
    if os.path.exists(precision_map_filenm):
        with open(precision_map_filenm, "rb") as fin:
            args.precision_map = pickle.load(fin)
    else:
        logger.info(
            "Path precision files not found at {}. Please run the preprocessing script".format(precision_map_filenm))

    ########### load precision maps (fall-back) ###########
    linkage_bck = args.linkage
    args.linkage = 0.0
    precision_map_filenm_fallback = os.path.join(data_dir, "linkage={}".format(args.linkage), "precision_maps",
                                                 "path_{}".format(args.num_paths_around_entities), "precision_map.pkl")
   #logger.info("Loading fall-back precision map")
    if os.path.exists(precision_map_filenm_fallback):
        with open(precision_map_filenm_fallback, "rb") as fin:
            args.precision_map_fallback = pickle.load(fin)
    else:
        logger.info("Path precision fall-back files not found at {}. Please run the preprocessing script".format(
            precision_map_filenm_fallback))
    args.linkage = linkage_bck

    # Finally all files are loaded, do inference!
    prob_cbr_agent.do_symbolic_case_based_reasoning()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect subgraphs around entities")
    parser.add_argument("--dataset_name", type=str, default="nell")
    parser.add_argument("--data_dir", type=str, default="../prob_cbr_data/")
    parser.add_argument("--expt_dir", type=str,
                        default="./outputs/")
    parser.add_argument("--subgraph_file_name", type=str, default="")
    # Per relation config
    parser.add_argument("--per_relation_config_file", type=str, default=None)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_file_name", type=str, default='',
                        help="Useful to switch between test files for FB122")
    parser.add_argument("--input_file_name", type=str, default=None,
                        help="Input file name.")
    parser.add_argument("--use_path_counts", type=int, choices=[0, 1], default=1,
                        help="Set to 1 if want to weight paths during ranking")
    # Clustering args
    parser.add_argument("--linkage", type=float, default=0.8,
                        help="Clustering threshold")
    # CBR args
    parser.add_argument("--k_adj", type=int, default=5,
                        help="Number of nearest neighbors to consider based on adjacency matrix")
    parser.add_argument("--cheat_neighbors", type=int, default=0,
                        help="When adjacency fails to return neighbors, use any entities which have query relation")
    parser.add_argument("--max_num_programs", type=int, default=5000)
    # Output modifier args
    parser.add_argument("--name_of_run", type=str, default="unset")
    parser.add_argument("--output_per_relation_scores", action="store_true")
    parser.add_argument("--print_paths", action="store_true")
    parser.add_argument("--use_wandb", type=int, choices=[0, 1], default=0, help="Set to 1 if using W&B")
    # Path sampling args
    parser.add_argument("--num_paths_around_entities", type=int, default=1000)
    parser.add_argument("--max_path_len", type=int, default=3)
    parser.add_argument("--prevent_loops", type=int, choices=[0, 1], default=1)
    parser.add_argument("--max_branch", type=int, default=100)
    parser.add_argument("--aggr_type1", type=str, default="none", help="none/sum")
    parser.add_argument("--aggr_type2", type=str, default="sum", help="sum/max/noisy_or/logsumexp")
    parser.add_argument("--use_only_precision_scores", type=int, default=0)
    parser.add_argument("--specific_rel", type=int, default=None)
    parser.add_argument("--dump_paths", action="store_true", default='true')

    args = parser.parse_args()


    args.use_path_counts = (args.use_path_counts == 1)

    main(args)