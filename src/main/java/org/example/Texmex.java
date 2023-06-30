package org.example;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.example.util.ListRandomAccessVectorValues;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
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

    public static double testRecall(ArrayList<float[]> baseVectors, ArrayList<float[]> queryVectors, ArrayList<HashSet<Integer>> groundTruth) throws IOException {
        var ravv = new ListRandomAccessVectorValues(baseVectors, baseVectors.get(0).length);

        var start = System.nanoTime();
        var builder = HnswGraphBuilder.create(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, 16, 100, ThreadLocalRandom.current().nextInt());
        var hnsw = builder.build(ravv.copy());
        System.out.printf("  Building index took %s seconds%n", (System.nanoTime() - start) / 1_000_000_000.0);

        var topKfound = new AtomicInteger(0);
        var topK = 100;
        start = System.nanoTime();
        for (int k = 0; k < 10; k++) {
            IntStream.range(0, queryVectors.size()).forEach(i -> {
                var queryVector = queryVectors.get(i);
                NeighborQueue nn;
                try {
                    nn = HnswGraphSearcher.search(queryVector, 100, ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, hnsw, null, Integer.MAX_VALUE);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

                var gt = groundTruth.get(i);
                var n = IntStream.range(0, topK).filter(j -> gt.contains(nn.nodes()[j])).count();
                topKfound.addAndGet((int) n);
            });
        }
        System.out.printf("  Querying %d vectors x10 took %s seconds%n", queryVectors.size(), (System.nanoTime() - start) / 1_000_000_000.0);

        return (double) topKfound.get() / (queryVectors.size() * topK);
    }

    public static void main(String[] args) throws IOException {
        if (args.length > 0) {
            siftName = args[0];
        }

        var baseVectors = readFvecs(String.format("%s/%s_base.fvecs", siftName, siftName));
        var queryVectors = readFvecs(String.format("%s/%s_query.fvecs", siftName, siftName));
        var groundTruth = readIvecs(String.format("%s/%s_groundtruth.ivecs", siftName, siftName));
        System.out.format("%d base and %d query vectors loaded, dimensions %d%n",
                baseVectors.size(), queryVectors.size(), baseVectors.get(0).length);

        // Average recall and standard deviation over multiple runs
        var numRuns = 5;

        var totalRecall = new DoubleAdder();
        var totalRecallSquared = new DoubleAdder();

        IntStream.range(0, numRuns).parallel()
                .mapToDouble(i -> {
                    System.out.printf("Run %d:%n", i);
                    try {
                        return testRecall(baseVectors, queryVectors, groundTruth);
                    } catch (IOException e) {
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
