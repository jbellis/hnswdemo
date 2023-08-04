package org.example;

import io.jhdf.HdfFile;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.NamedThreadFactory;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.*;
import org.example.util.ListRandomAccessVectorValues;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 * Tests HNSW against vectors from the Texmex dataset
 */
public class Texmex {
    public static void testRecall(List<float[]> baseVectors, List<float[]> queryVectors, List<Set<Integer>> groundTruth) throws IOException, ExecutionException, InterruptedException {
        var ravv = new ListRandomAccessVectorValues(baseVectors, baseVectors.get(0).length);
        var topK = groundTruth.get(0).size();

        var start = System.nanoTime();
        var builder = ConcurrentHnswGraphBuilder.create(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.DOT_PRODUCT, 16, 100);
        int buildThreads = 24;
        var es = Executors.newFixedThreadPool(
                buildThreads, new NamedThreadFactory("Concurrent HNSW builder"));
        var hnsw = builder.buildAsync(ravv.copy(), es, buildThreads).get();
        var vBuilder = new VamanaGraphBuilder<>(hnsw, ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.DOT_PRODUCT, 100);
        long buildNanos = System.nanoTime() - start;

        int queryRuns = 10;
        start = System.nanoTime();
        var pqr = performQueries(queryVectors, groundTruth, ravv, hnsw::getView, topK, queryRuns);
        long queryNanos = System.nanoTime() - start;
        var recall = ((double) pqr.topKFound) / (queryRuns * queryVectors.size() * topK);
        System.out.format("HNSW: top %d recall %.4f, build %.2fs, query %.2fs. %s nodes visited%n",
                topK, recall, buildNanos / 1_000_000_000.0, queryNanos / 1_000_000_000.0, pqr.nodesVisited);

        IntStream.range(1, 5).forEach(i -> {
            float alpha = (float) Math.pow(2, i);
            var vStart = System.nanoTime();
            QueryResult vqr;
            long vBuildNanos;
            try {
                var vamona = es.submit(() -> vBuilder.buildVamana(alpha)).get();
                vBuildNanos = System.nanoTime() - vStart;
                vStart = System.nanoTime();
                vqr = performQueries(queryVectors, groundTruth, ravv, vamona::getView, topK, queryRuns);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            var vQueryNanos = System.nanoTime() - vStart;
            var vRecall = ((double) vqr.topKFound) / (queryRuns * queryVectors.size() * topK);
            System.out.format("With Vamona@%s: top %d recall %.4f, build %.2fs, query %.2fs. %s nodes visited%n",
                    alpha, topK, vRecall, vBuildNanos / 1_000_000_000.0, vQueryNanos / 1_000_000_000.0, vqr.nodesVisited);
        });

        es.shutdown();
    }

    private static float normOf(float[] baseVector) {
        float norm = 0;
        for (float v : baseVector) {
            norm += v * v;
        }
        return (float) Math.sqrt(norm);
    }

    private record QueryResult(int topKFound, int nodesVisited) { }

    private static QueryResult performQueries(List<float[]> queryVectors, List<Set<Integer>> groundTruth, ListRandomAccessVectorValues ravv, Supplier<HnswGraph> graphSupplier, int topK, int queryRuns) {
        LongAdder topKfound = new LongAdder();
        LongAdder nodesVisited = new LongAdder();
        for (int k = 0; k < queryRuns; k++) {
            IntStream.range(0, queryVectors.size()).parallel().forEach(i -> {
                var queryVector = queryVectors.get(i);
                NeighborQueue nn;
                try {
                    nn = HnswGraphSearcher.search(queryVector, topK, ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.DOT_PRODUCT, graphSupplier.get(), null, Integer.MAX_VALUE);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
                var gt = groundTruth.get(i);
                int[] resultNodes = nn.nodes();
                var n = IntStream.range(0, Math.min(nn.size(), topK)).filter(j -> gt.contains(resultNodes[j])).count();
                topKfound.add(n);
                nodesVisited.add(nn.visitedCount());
            });
        }
        return new QueryResult((int) topKfound.sum(), (int) nodesVisited.sum());
    }

    private static void computeRecallFor(String pathStr) throws IOException, ExecutionException, InterruptedException {
        float[][] baseVectors;
        float[][] queryVectors;
        int[][] groundTruth;
        try (HdfFile hdf = new HdfFile(Paths.get(pathStr))) {
            baseVectors = (float[][]) hdf.getDatasetByPath("train").getData();
            queryVectors = (float[][]) hdf.getDatasetByPath("test").getData();
            groundTruth = (int[][]) hdf.getDatasetByPath("neighbors").getData();
        }

        // verify that vectors are normalized and sane
        List<float[]> scrubbedBaseVectors = new ArrayList<>(baseVectors.length);
        List<float[]> scrubbedQueryVectors = new ArrayList<>(queryVectors.length);
        List<Set<Integer>> gtSet = new ArrayList<>(groundTruth.length);
        // remove zero vectors, noting that this will change the indexes of the ground truth answers
        Map<Integer, Integer> rawToScrubbed = new HashMap<>();
        {
            int j = 0;
            for (int i = 0; i < baseVectors.length; i++) {
                float[] v = baseVectors[i];
                if (Math.abs(normOf(v)) > 1e-5) {
                    scrubbedBaseVectors.add(v);
                    rawToScrubbed.put(i, j++);
                }
            }
        }
        for (int i = 0; i < queryVectors.length; i++) {
            float[] v = queryVectors[i];
            if (Math.abs(normOf(v)) > 1e-5) {
                scrubbedQueryVectors.add(v);
                var gt = new HashSet<Integer>();
                for (int j = 0; j < groundTruth[i].length; j++) {
                    gt.add(rawToScrubbed.get(groundTruth[i][j]));
                }
                gtSet.add(gt);
            }
        }
        // now that the zero vectors are removed, we can normalize
        if (Math.abs(normOf(baseVectors[0]) - 1.0) > 1e-5) {
            normalizeAll(scrubbedBaseVectors);
            normalizeAll(scrubbedQueryVectors);
        }
        assert scrubbedQueryVectors.size() == gtSet.size();
        baseVectors = null;
        queryVectors = null;
        groundTruth = null;

        System.out.format("%s: %d base and %d query vectors loaded, dimensions %d%n",
                pathStr, scrubbedBaseVectors.size(), scrubbedQueryVectors.size(), scrubbedBaseVectors.get(0).length);

        testRecall(scrubbedBaseVectors, scrubbedQueryVectors, gtSet);
    }

    private static void normalizeAll(Iterable<float[]> vectors) {
        for (float[] v : vectors) {
            VectorUtil.l2normalize(v);
        }
    }

    public static void main(String[] args) {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        new Thread(() -> {
            try {
                computeRecallFor("hdf5/nytimes-256-angular.hdf5");
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }).run();
//        new Thread(() -> {
//            try {
//                computeRecallFor("hdf5/glove-100-angular.hdf5");
//            } catch (Throwable e) {
//                throw new RuntimeException(e);
//            }
//        }).run();
//        new Thread(() -> {
//            try {
//                computeRecallFor("hdf5/glove-200-angular.hdf5");
//            } catch (Throwable e) {
//                throw new RuntimeException(e);
//            }
//        }).run();
//        new Thread(() -> {
//            try {
//                computeRecallFor("hdf5/deep-image-96-angular.hdf5");
//            } catch (Throwable e) {
//                throw new RuntimeException(e);
//            }
//        }).start();
    }
}
