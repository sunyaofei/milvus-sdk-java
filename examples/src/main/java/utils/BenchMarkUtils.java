package utils;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

/**
 * Description: BenchMarkUtils
 * Author: yaofei.sun
 * Create: yaofei.sun(2020-01-15 16:36)
 */
public class BenchMarkUtils {
    // Helper function that generates random vectors
    public static List<List<Float>> generateVectors(long vectorCount, long dimension) {
        SplittableRandom splittableRandom = new SplittableRandom();
        List<List<Float>> vectors = new ArrayList<>();
        for (int i = 0; i < vectorCount; ++i) {
            splittableRandom = splittableRandom.split();
            DoubleStream doubleStream = splittableRandom.doubles(dimension);
            List<Float> vector =
                    doubleStream.boxed().map(Double::floatValue).collect(Collectors.toList());
            vectors.add(vector);
        }
        return vectors;
    }

    // Helper function that normalizes a vector if you are using IP (Inner Product) as your metric
    // type
    public static List<Float> normalizeVector(List<Float> vector) {
        float squareSum = vector.stream().map(x -> x * x).reduce((float) 0, Float::sum);
        final float norm = (float) Math.sqrt(squareSum);
        vector = vector.stream().map(x -> x / norm).collect(Collectors.toList());
        return vector;
    }
}
