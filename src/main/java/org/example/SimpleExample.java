package org.example;

import com.github.jelmerk.knn.DistanceFunctions;
import com.github.jelmerk.knn.Item;
import com.github.jelmerk.knn.hnsw.HnswIndex;
import org.apache.lucene.index.*;
import org.apache.lucene.util.hnsw.*;
import org.example.lucenecopy.MockVectorValues;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class SimpleExample {
    private static final VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.COSINE;
    private static final Random random = new Random();
    private static int universeSize = 50_000;
    private static int vectorDimensions = 1500;

    // search for the nearest neighbors of a random vector
    private static int queryIterations = 10_000;

    static class LongItem implements Item<Long, float[]> {
        private final float[] vector;
        private final Long id;

        public LongItem(Long id, float[] vector) {
            this.vector = vector;
            this.id = id;
        }

        @Override
        public Long id() {
            return id;
        }

        @Override
        public float[] vector() {
            return vector;
        }

        @Override
        public int dimensions() {
            return vector.length;
        }
    }

    public static void main(String[] args) throws IOException {
        testLucene();
        testHnswlib();
    }

    private static void testHnswlib() {
        System.out.println("Building hnswlib HNSW graph...");
        HnswIndex<Long, float[], LongItem, Float> hnsw = HnswIndex
                .newBuilder(vectorDimensions, DistanceFunctions.FLOAT_INNER_PRODUCT, universeSize)
                .withM(16)
                .withEf(100)
                .withEfConstruction(100)
                .build();
        for (int i = 0; i < universeSize; i++) {
            hnsw.add(randomItem());
        }

        System.out.println("Searching hnswlib graph...");
        var searchTime = timeLambda(() -> {
            for (int i = 0; i < queryIterations; i++) {
                var v = randomVector(vectorDimensions);
                hnsw.findNearest(v, 10);
            }
        });
        System.out.println("Execution time: " + searchTime + "ms");
    }

    static LongItem randomItem() {
        return new LongItem(random.nextLong(), randomVector(1500));
    }

    private static void testLucene() throws IOException {
        // Create a random vector universe
        var universe = new float[universeSize][];
        for (int i = 0; i < universeSize; i++) {
            universe[i] = randomVector(vectorDimensions);
        }

        System.out.println("Building Lucene HNSW graph...");
        var ravv = MockVectorValues.fromValues(universe);
        var builder = HnswGraphBuilder.create(ravv, VectorEncoding.FLOAT32, similarityFunction, 16, 100, random.nextInt());
        var hnsw = builder.build(ravv.copy());

        System.out.println("Searching Lucene graph...");
        var searchTime = timeLambda(() -> {
            for (int i = 0; i < queryIterations; i++) {
                var queryVector = randomVector(vectorDimensions);
                try {
                    luceneSearch(ravv, hnsw, queryVector, 10);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        });
        System.out.println("Execution time: " + searchTime + "ms");
    }

    public static int timeLambda(Runnable runnable) {
        long startTime = System.nanoTime();
        runnable.run();
        long endTime = System.nanoTime();
        return (int) ((endTime - startTime) / 1000_000);
    }

    private static void bruteSearch(float[][] universe, float[] queryVector, int topK) {
       var scored = Arrays.stream(universe).map(v -> new AbstractMap.SimpleEntry<>(v, similarityFunction.compare(v, queryVector)))
               .sorted(Comparator.comparingDouble(entry -> entry.getValue()))
               .limit(topK)
               .collect(Collectors.toList());
       System.out.println(scored.size() + " found");
    }

    private static void luceneSearch(MockVectorValues ravv, HnswGraph hnsw, float[] queryVector, int topK) throws IOException {
        NeighborQueue nn = HnswGraphSearcher.search(queryVector, topK, ravv, VectorEncoding.FLOAT32, similarityFunction, hnsw, null, Integer.MAX_VALUE);
//        System.out.println(nn.nodes().length + " found");
    }

     static float[] randomVector(int vectorDimension) {
        float[] queryVector = new float[vectorDimension];
        for (int i = 0; i < vectorDimension; i++) {
            queryVector[i] = random.nextFloat();
        }
        return queryVector;
    }
}
