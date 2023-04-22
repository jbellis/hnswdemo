# HNSW demos

This provides two classes illustrating the usage of Lucene's HNSW approximate nearest-neighbor (ANN) index.
See the [Lucene JIRA issue](https://issues.apache.org/jira/browse/LUCENE-9004) and [the Elastic blog post](https://www.elastic.co/blog/introducing-approximate-nearest-neighbor-search-in-elasticsearch-8-0) for more background, 
and [the original paper](https://arxiv.org/pdf/1603.09320.pdf) for how it works.

The classes provided are `SimpleExample`, which creates and searches a random graph, and `Texmex`, which tests the recall performance against a dataset with ground truth (i.e. known, exact nearest neighbors) precomputed for each query.

## Usage

```bash
$ ./gradlew runSimple
$ ./gradlew runTexmex -PsiftName=siftsmall
```

The Texmex datasets may be found [here](http://corpus-texmex.irisa.fr/). 
The Texmex class expects to find the data files in a subdirectory of the current working directory, as extracted from the dataset `tgz` archive (e.g. `siftsmall`, `sift`, etc.). The `siftsmall` dataset runs in about 2 seconds, as long as there are enough cores to give each of the 10 runs to a separate thread. The `sift` dataset runs in about 10.5 minutes.
