import os
import sys
from struct import unpack
from typing import List, Set
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from load import read_fvecs
from db import DB


def read_ivecs(filename: str) -> List[Set[int]]:
    ground_truth_top_k = []
    with open(filename, 'rb') as file:
        while True:
            try:
                num_neighbors = np.fromfile(file, dtype=np.int32, count=1)[0]
                neighbors = np.fromfile(file, dtype=np.int32, count=num_neighbors)
                ground_truth_top_k.append(set(neighbors))
            except:
                break
    return ground_truth_top_k


def test_recall(db, query_vectors, ground_truth):
    top_k_found = 0
    top_k = 100
    num_query_vectors = len(query_vectors)

    def perform_query_and_count(query_vector, gt):
        result = db.query(query_vector.tolist(), top_k)
        n = sum(1 for pk in result if pk in gt)
        return n

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda x: perform_query_and_count(query_vectors[x], 
                                                                           ground_truth[x]), 
                                         range(num_query_vectors)), 
                            total=num_query_vectors))
        top_k_found = sum(results)

    return top_k_found / (num_query_vectors * top_k)

def test(sift_name):
    db = DB("demo", sift_name)

    query_vectors = read_fvecs(f"{sift_name}/{sift_name}_query.fvecs")
    ground_truth = read_ivecs(f"{sift_name}/{sift_name}_groundtruth.ivecs")
    assert len(query_vectors) == len(ground_truth), (len(query_vectors), len(ground_truth))

    memory_recall = test_recall(db, query_vectors, ground_truth)
    print(memory_recall)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load.py <sift_name>")
        exit(1)

    test(sys.argv[1])