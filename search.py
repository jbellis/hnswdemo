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


def test_recall(db, query_vectors):
    top_k = 10
    num_query_vectors = len(query_vectors)

    def perform_query_and_count(query_vector):
        db.query(query_vector.tolist(), top_k)

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda x: perform_query_and_count(query_vectors[x]), 
                                         range(num_query_vectors)), 
                            total=num_query_vectors))

def test(sift_name):
    db = DB("demo", sift_name)

    query_vectors = read_fvecs(f"{sift_name}/{sift_name}_query.fvecs")
    test_recall(db, query_vectors)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load.py <sift_name>")
        exit(1)

    test(sys.argv[1])