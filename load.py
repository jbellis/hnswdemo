import os
import sys
import cProfile, pstats

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from db import DB


def count_vectors(filepath):
    with open(filepath, 'rb') as file:
        dimension = np.fromfile(file, dtype=np.int32, count=1)[0]
        assert dimension > 0, dimension
        file_size = os.path.getsize(filepath)
        return file_size // (4 + dimension * 4)

def read_fvecs(filepath: str) -> np.ndarray:
    vectors = []
    n = count_vectors(filepath)
    with open(filepath, 'rb') as file:
        for i in tqdm(range(n), desc='Reading vectors'):
            dimension = np.fromfile(file, dtype=np.int32, count=1)[0]
            assert dimension > 0, dimension
            vector = np.fromfile(file, dtype=np.float32, count=dimension)
            vectors.append(vector)
    return vectors

def insert_vectors(db, base_vectors):
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda i, x: db.upsert_one(i, x.tolist()), 
                               range(len(base_vectors)), 
                               base_vectors)
        for _ in tqdm(results, total=len(base_vectors), desc='Inserting vectors'):
            pass

def load(sift_name):
    base_vectors = read_fvecs(f"{sift_name}/{sift_name}_base.fvecs")
    db = DB("demo", sift_name)
    insert_vectors(db, base_vectors)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load.py <sift_name>")
        exit(1)

    load(sys.argv[1])
