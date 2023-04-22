package org.example;

import org.apache.lucene.index.*;
import org.apache.lucene.util.hnsw.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class SimpleExample {
    private static final VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.COSINE;
    private static final Random random = new Random();

    public static void main(String[] args) throws IOException {
        // Create a random vector universe
        int vectorDimensions = 1500;
        int universeSize = 10_000;
        var universe = new ArrayList<float[]>(universeSize);
        for (int i = 0; i < universeSize; i++) {
            universe.add(randomVector(vectorDimensions));
        }

        // construct a HNSW graph of the universe
        var ravv = new ListRandomAccessVectorValues(universe, vectorDimensions);
        var builder = HnswGraphBuilder.create(ravv, VectorEncoding.FLOAT32, similarityFunction, 16, 100, random.nextInt());
        var hnsw = builder.build(ravv.copy());

        // search for the nearest neighbors of a random vector
        var queryVector = randomVector(vectorDimensions);
        NeighborQueue nn = HnswGraphSearcher.search(queryVector, 10, ravv.copy(), VectorEncoding.FLOAT32, similarityFunction, hnsw, null, Integer.MAX_VALUE);
        System.out.println("Nearest neighbors of " + Arrays.toString(queryVector) + ":");
        for (var i : nn.nodes()) {
            var neighbor = universe.get(i);
            var similarity = similarityFunction.compare(queryVector, neighbor);
            System.out.println("  " + Arrays.toString(neighbor) + " (similarity: " + similarity + ")");
        }
    }

    private static float[] randomVector(int vectorDimension) {
        float[] queryVector = new float[vectorDimension];
        for (int i = 0; i < vectorDimension; i++) {
            queryVector[i] = random.nextFloat();
        }
        return queryVector;
    }
}
