package org.example;

import org.apache.lucene.index.*;
import org.apache.lucene.util.hnsw.*;
import org.example.lucenecopy.MockVectorValues;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class SimpleExample {
    private static final VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.COSINE;
    private static final Random random = new Random();

    public static void main(String[] args) throws IOException {
        // Create a random vector universe
        int vectorDimensions = 1500;
        int universeSize = 10_000;
        var universe = new float[universeSize][];
        for (int i = 0; i < universeSize; i++) {
            universe[i] = randomVector(vectorDimensions);
        }

        System.out.println("Building HNSW graph...");
        var ravv = MockVectorValues.fromValues(universe);
        var builder = HnswGraphBuilder.create(ravv, VectorEncoding.FLOAT32, similarityFunction, 16, 100, random.nextInt());
        var hnsw = builder.build(ravv.copy());

        // search for the nearest neighbors of a random vector
        for (int i = 0; i < 1000; i++) {
            var queryVector = randomVector(vectorDimensions);

//            bruteSearch(universe, queryVector, 10);
            hnswSearch(ravv, hnsw, queryVector, 10);
        }
    }

    private static void bruteSearch(float[][] universe, float[] queryVector, int topK) {
       var scored = Arrays.stream(universe).map(v -> new AbstractMap.SimpleEntry<>(v, similarityFunction.compare(v, queryVector)))
               .sorted(Comparator.comparingDouble(entry -> entry.getValue()))
               .limit(topK)
               .collect(Collectors.toList());
       System.out.println(scored.size() + " found");
    }

    private static void hnswSearch(MockVectorValues ravv, HnswGraph hnsw, float[] queryVector, int topK) throws IOException {
        NeighborQueue nn = HnswGraphSearcher.search(queryVector, topK, ravv, VectorEncoding.FLOAT32, similarityFunction, hnsw, null, Integer.MAX_VALUE);
        System.out.println(nn.nodes().length + " found");
    }

     static float[] randomVector(int vectorDimension) {
        float[] queryVector = new float[vectorDimension];
        for (int i = 0; i < vectorDimension; i++) {
            queryVector[i] = random.nextFloat();
        }
        return queryVector;
    }
}
