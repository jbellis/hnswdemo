package org.example;

import io.jhdf.HdfFile;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.NamedThreadFactory;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.*;
import org.example.util.ListRandomAccessVectorValues;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.stream.IntStream;

/**
 * Tests HNSW against vectors from the Texmex dataset
 */
public class Texmex {
    public static void testRecall(List<float[]> baseVectors, List<float[]> queryVectors, List<Set<Integer>> groundTruth) throws IOException, ExecutionException, InterruptedException {
        var ravv = new ListRandomAccessVectorValues(baseVectors, baseVectors.get(0).length);
        var topK = 10; // groundTruth.get(0).size();

        var start = System.nanoTime();
        var builder = ConcurrentHnswGraphBuilder.create(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.DOT_PRODUCT, 128, 512);
        int buildThreads = 24;
        var es = Executors.newFixedThreadPool(
                buildThreads, new NamedThreadFactory("Concurrent HNSW builder"));
        var hnsw = builder.buildAsync(ravv.copy(), es, buildThreads).get().getView();
        es.shutdown();
        long buildNanos = System.nanoTime() - start;

        start = System.nanoTime();
        FingerMetadata<float[]> fm = new FingerMetadata<>(hnsw, ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.DOT_PRODUCT, 64);
        long fingerNanos = System.nanoTime() - start;

        int queryRuns = 10;
        start = System.nanoTime();
        var pqr = performQueries(queryVectors, groundTruth, ravv, hnsw, null, topK, queryRuns);
        long queryNanos = System.nanoTime() - start;
        var recall = ((double) pqr.topKFound) / (queryRuns * queryVectors.size() * topK);
        System.out.format("Without Finger: top %d recall %.4f, build %.2fs, finger %.2fs, query %.2fs. %s exact similarity evals and %s approx%n",
                topK, recall, buildNanos / 1_000_000_000.0, fingerNanos / 1_000_000_000.0, queryNanos / 1_000_000_000.0, pqr.exactSimilarities, pqr.approxSimilarities);

        start = System.nanoTime();
        pqr = performQueries(queryVectors, groundTruth, ravv, hnsw, fm, topK, queryRuns);
        queryNanos = System.nanoTime() - start;
        recall = ((double) pqr.topKFound) / (queryRuns * queryVectors.size() * topK);
        System.out.format("Witho Finger: top %d recall %.4f, build %.2fs, finger %.2fs, query %.2fs. %s exact similarity evals and %s approx%n",
                topK, recall, buildNanos / 1_000_000_000.0, fingerNanos / 1_000_000_000.0, queryNanos / 1_000_000_000.0, pqr.exactSimilarities, pqr.approxSimilarities);
    }

    private static float normOf(float[] baseVector) {
        float norm = 0;
        for (float v : baseVector) {
            norm += v * v;
        }
        return (float) Math.sqrt(norm);
    }

    private record QueryResult(int topKFound, int exactSimilarities, int approxSimilarities) { }

    private static QueryResult performQueries(List<float[]> queryVectors, List<Set<Integer>> groundTruth, ListRandomAccessVectorValues ravv, HnswGraph hnsw, FingerMetadata<float[]> fm, int topK, int queryRuns) throws IOException {
        int topKfound = 0;
        int exactSimilarities = 0;
        int approxSimilarities = 0;
        for (int k = 0; k < queryRuns; k++) {
            HnswSearcher<float[]> searcher = new HnswSearcher.Builder<>(hnsw, ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.DOT_PRODUCT)
                        .withFinger(fm)
                        .build();
            for (int i = 0; i < queryVectors.size(); i++) {
                var queryVector = queryVectors.get(i);
                NeighborQueue nn;
                nn = searcher.search(queryVector, topK, null, Integer.MAX_VALUE);

                var gt = groundTruth.get(i);
                int[] resultNodes = nn.nodes();
                var n = IntStream.range(0, Math.min(nn.size(), topK)).filter(j -> gt.contains(resultNodes[j])).count();
                topKfound += n;
            }
            exactSimilarities += searcher.exactSimilarityCalls.intValue();
            approxSimilarities += searcher.approxSimilarityCalls.intValue();
        }
        return new QueryResult(topKfound, exactSimilarities, approxSimilarities);
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
