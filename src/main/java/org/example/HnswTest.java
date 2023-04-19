package org.example;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.example.lucenecopy.MockVectorValues;

import java.io.IOException;
import java.util.AbstractMap;
import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.Collectors;

// tests behavior of inserting identical vectors
public class HnswTest {
    private static final VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.COSINE;

    public static void main(String[] args) throws IOException {
        int vectorDimensions = 1500;
        int universeSize = 10;
        var universe = new float[universeSize][];
        for (int i = 0; i < universeSize; i++) {
            universe[i] = new float[vectorDimensions];
        }

        System.out.println("Building HNSW graph...");
        var ravv = MockVectorValues.fromValues(universe);
        var builder = HnswGraphBuilder.create(ravv, VectorEncoding.FLOAT32, similarityFunction, 16, 100, 0);
        var hnsw = builder.build(ravv.copy());
        System.out.println("HNSW is " + hnsw.size()); // 10

        var queryVector = SimpleExample.randomVector(vectorDimensions);
        NeighborQueue nn = HnswGraphSearcher.search(queryVector, 5, ravv, VectorEncoding.FLOAT32, similarityFunction, hnsw, null, Integer.MAX_VALUE);
        System.out.println(nn.size());
    }
}
