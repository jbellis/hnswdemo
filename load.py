import sys

import numpy as np
from struct import unpack
from typing import List, Set
import concurrent.futures


def read_fvecs(filepath: str) -> List[List[float]]:
    vectors = []
    with open(filepath, 'rb') as file:
        while True:
            try:
                dimension = unpack('i', file.read(4))[0]
                assert dimension > 0, dimension
                vector = unpack('f' * dimension, file.read(4 * dimension))
                vectors.append(list(vector))
            except:
                break
    return vectors

def insert_vectors(db, base_vectors):
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(db.upsert_one, range(len(base_vectors)), base_vectors), total=len(base_vectors)))

def load(sift_name):
    import db
    db = db.DB("demo", sift_name)
    base_vectors = read_fvecs(f"{sift_name}/{sift_name}_base.fvecs")
    insert_vectors(db, base_vectors)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load.py <sift_name>")
        exit(1)

    load(sys.argv[1])