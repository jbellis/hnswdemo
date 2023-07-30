package org.example;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.NamedThreadFactory;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.*;
import org.example.util.ListRandomAccessVectorValues;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.IntStream;

/**
 * Tests HNSW against vectors from the Texmex dataset
 */
public class Texmex {
    private static String siftName = "sift";

    public static ArrayList<float[]> readFvecs(String filePath) throws IOException {
        var vectors = new ArrayList<float[]>();
        try (var dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {
            while (dis.available() > 0) {
                var dimension = Integer.reverseBytes(dis.readInt());
                assert dimension > 0 : dimension;
                var buffer = new byte[dimension * Float.BYTES];
                dis.readFully(buffer);
                var byteBuffer = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);

                var vector = new float[dimension];
                for (var i = 0; i < dimension; i++) {
                    vector[i] = byteBuffer.getFloat();
                }
                VectorUtil.l2normalize(vector);
                vectors.add(vector);
            }
        }
        return vectors;
    }

    private static ArrayList<HashSet<Integer>> readIvecs(String filename) {
        var groundTruthTopK = new ArrayList<HashSet<Integer>>();

        try (var dis = new DataInputStream(new FileInputStream(filename))) {
            while (dis.available() > 0) {
                var numNeighbors = Integer.reverseBytes(dis.readInt());
                var neighbors = new HashSet<Integer>(numNeighbors);

                for (var i = 0; i < numNeighbors; i++) {
                    var neighbor = Integer.reverseBytes(dis.readInt());
                    neighbors.add(neighbor);
                }

                groundTruthTopK.add(neighbors);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return groundTruthTopK;
    }

    public static double testRecall(ArrayList<float[]> baseVectors, ArrayList<float[]> queryVectors, ArrayList<HashSet<Integer>> groundTruth) throws IOException, InterruptedException, ExecutionException {
        var ravv = new ListRandomAccessVectorValues(baseVectors, baseVectors.get(0).length);

        var start = System.nanoTime();
        var builder = ConcurrentHnswGraphBuilder.create(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.DOT_PRODUCT, 16, 100);
        int buildThreads = 24;
        var es = Executors.newFixedThreadPool(
                        buildThreads, new NamedThreadFactory("Concurrent HNSW builder"));
        var hnsw = builder.buildAsync(ravv.copy(), es, buildThreads).get();
        es.shutdown();
        System.out.printf("  Building index took %s seconds%n", (System.nanoTime() - start) / 1_000_000_000.0);

        start = System.nanoTime();
        FingerMetadata<float[]> fm = new FingerMetadata<>(hnsw.getView(), ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.DOT_PRODUCT, 64);
        System.out.printf("  Finger metadata created in %s seconds%n", (System.nanoTime() - start) / 1_000_000_000.0);

        var topKfound = new AtomicInteger(0);
        var topK = 100;
        start = System.nanoTime();

        int queryRuns = 100;
        performQueries(queryVectors, groundTruth, ravv, hnsw, fm, topKfound, topK, queryRuns);
        System.out.printf("  Querying %d vectors x10 in parallel took %s seconds%n", queryVectors.size(), (System.nanoTime() - start) / 1_000_000_000.0);
        return (double) topKfound.get() / (queryRuns * queryVectors.size() * topK);
    }

    private static void performQueries(ArrayList<float[]> queryVectors, ArrayList<HashSet<Integer>> groundTruth, ListRandomAccessVectorValues ravv, ConcurrentOnHeapHnswGraph hnsw, FingerMetadata<float[]> fm, AtomicInteger topKfound, int topK, int queryRuns) {
        for (int k = 0; k < queryRuns; k++) {
            ThreadLocal<HnswSearcher<float[]>> searchers = ThreadLocal.withInitial(() -> {
                return new HnswSearcher.Builder<>(hnsw.getView(), ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.DOT_PRODUCT)
                        .withFinger(fm)
                        .build();
            });
            IntStream.range(0, queryVectors.size()).parallel().forEach(i -> {
                var queryVector = queryVectors.get(i);
                NeighborQueue nn;
                try {
                    var searcher = searchers.get();
                    nn = searcher.search(queryVector, topK, null, Integer.MAX_VALUE);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

                var gt = groundTruth.get(i);
                int[] resultNodes = nn.nodes();
                var n = IntStream.range(0, Math.min(nn.size(), topK)).filter(j -> gt.contains(resultNodes[j])).count();
                topKfound.addAndGet((int) n);
            });
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length > 0) {
            siftName = args[0];
        }

        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());
        var baseVectors = readFvecs(String.format("%s/%s_base.fvecs", siftName, siftName));
        var queryVectors = readFvecs(String.format("%s/%s_query.fvecs", siftName, siftName));
        var groundTruth = readIvecs(String.format("%s/%s_groundtruth.ivecs", siftName, siftName));
        System.out.format("%d base and %d query vectors loaded, dimensions %d%n",
                baseVectors.size(), queryVectors.size(), baseVectors.get(0).length);

        // Average recall and standard deviation over multiple runs
        var numRuns = 1;

        var totalRecall = new DoubleAdder();
        var totalRecallSquared = new DoubleAdder();

        IntStream.range(0, numRuns)
                .mapToDouble(i -> {
                    System.out.printf("Run %d:%n", i);
                    try {
                        return testRecall(baseVectors, queryVectors, groundTruth);
                    } catch (Exception e) {
                        e.printStackTrace();
                        return 0;
                    }
                })
                .forEach(recall -> {
                    totalRecall.add(recall);
                    totalRecallSquared.add(recall * recall);
                });

        var averageRecall = totalRecall.doubleValue() / numRuns;
        var variance = (totalRecallSquared.doubleValue() / numRuns) - (averageRecall * averageRecall);
        var stdev = Math.sqrt(variance);

        System.out.println("Average Recall: " + averageRecall);
        System.out.println("Standard Deviation: " + stdev);
    }
}
