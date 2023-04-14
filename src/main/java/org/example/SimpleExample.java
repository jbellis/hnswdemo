package org.example;

import org.apache.lucene.index.*;
import org.apache.lucene.util.hnsw.*;
import org.example.lucenecopy.MockVectorValues;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class SimpleExample {
    private static final VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.COSINE;
    private static final Random random = new Random();

    public static void main(String[] args) throws IOException {
        // Create a random vector universe
        int vectorDimensions = 5;
        var universe = new float[1000][];
        for (int i = 0; i < 1000; i++) {
            universe[i] = randomVector(vectorDimensions);
        }

        // construct a HNSW graph of the universe
        var ravv = MockVectorValues.fromValues(universe);
        var builder = HnswGraphBuilder.create(ravv, VectorEncoding.FLOAT32, similarityFunction, 16, 100, random.nextInt());
        var hnsw = builder.build(ravv.copy());

        // search for the nearest neighbors of a random vector
        var queryVector = randomVector(vectorDimensions);
        NeighborQueue nn = HnswGraphSearcher.search(queryVector, 10, ravv.copy(), VectorEncoding.FLOAT32, similarityFunction, hnsw, null, Integer.MAX_VALUE);
        System.out.println("Nearest neighbors of " + Arrays.toString(queryVector) + ":");
        for (var i : nn.nodes()) {
            var neighbor = universe[i];
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
