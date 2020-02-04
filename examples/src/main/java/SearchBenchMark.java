import com.alibaba.fastjson.JSON;
import com.google.common.base.Splitter;
import io.milvus.client.*;
import org.apache.commons.lang3.StringUtils;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

/**
 * Description: SearchBenchMark
 * Author: yaofei.sun
 * Create: yaofei.sun(2020-01-19 11:42)
 */
public class SearchBenchMark {
    // Helper function that generates random vectors
    static List<List<Float>> generateVectors(long vectorCount, long dimension) {
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
    static List<Float> normalizeVector(List<Float> vector) {
        float squareSum = vector.stream().map(x -> x * x).reduce((float) 0, Float::sum);
        final float norm = (float) Math.sqrt(squareSum);
        vector = vector.stream().map(x -> x / norm).collect(Collectors.toList());
        return vector;
    }

    public static void main(String[] args) throws ConnectFailedException, InterruptedException, IOException {
        final String host = args[0];
        final String port = args[1];
        final String table = args[2];
        final Long dimension = Long.valueOf(args[3]);
        final Long nprobe = Long.valueOf(args[4]);
        final Integer concurrency = Integer.valueOf(args[5]);

        // Create Milvus client
        MilvusClient client = new MilvusGrpcClient();

        // Connect to Milvus server
        ConnectParam connectParam = new ConnectParam.Builder().withHost(host).withPort(port).build();
        try {
            Response connectResponse = client.connect(connectParam);
        } catch (ConnectFailedException e) {
            System.out.println(e.toString());
            throw e;
        }

        // Check whether we are connected
        boolean connected = client.isConnected();
        System.out.println("Connected = " + connected);

        final int vectorCount = 10000;
        List<List<Float>> vectorsPools = generateVectors(vectorCount, dimension);
        vectorsPools = vectorsPools.stream().map(MilvusClientExample::normalizeVector).collect(Collectors.toList());


        final int searchBatchSize = 5;
        List<List<Float>> vectorsToSearch = vectorsPools.subList(0, searchBatchSize);
        final long topK = 10;

        Integer[] nq = new Integer[]{1, 10, 30, 50};
        Long[] topks = new Long[]{10L, 100L, 500L, 1024L};
        int querys = 10;
        String querysInEnv = System.getenv("QUERYS");
        if (StringUtils.isNumeric(querysInEnv)) {
            querys = Integer.parseInt(querysInEnv);
        }

        Properties conf = new Properties();
        File file = new File("conf.properties");
        if (file.exists()) {
            conf.load(new FileReader(file));
            if (conf.containsKey("querys")) {
                querys = Integer.parseInt(conf.getOrDefault("querys", querys).toString());
            }
            if (conf.containsKey("nq")) {
                nq = parseNqs(conf.get("nq").toString());
            }

            if (conf.containsKey("topks")) {
                topks = parseTopks(conf.get("topks").toString());
            }
        }

        int finalQuery = querys;
        System.out.println("nq=" + JSON.toJSONString(nq));
        System.out.println("topks=" + JSON.toJSONString(topks));
        System.out.println("Interator querys: " + querys);
        TimeUnit.SECONDS.sleep(5);
        FileWriter fileWriter = new FileWriter(String.format("%s_%s_%s_output.csv", table, nprobe, concurrency));
        for (int i = 0; i < nq.length; i++) {
            fileWriter.write("nq,topk,query_cost,avg_cost\n");
            int anq = nq[i];
            for (int j = 0; j < topks.length; j++) {
                long atopk = topks[j];
                CountDownLatch latch = new CountDownLatch(concurrency);

                List<List<Float>> queryVectors =vectorsPools.subList(0, nq[i]);
                long topk = topks[j];
                ExecutorService executorService = Executors.newFixedThreadPool(concurrency, new ThreadFactory() {
                    private AtomicInteger inc = new AtomicInteger();
                    @Override
                    public Thread newThread(Runnable r) {
                        Thread thread = new Thread(r, String.format("Worker-%s", inc.getAndIncrement()));
                        return thread;
                    }
                });
                long start = Calendar.getInstance().getTimeInMillis();
                for (int k = 0; k < concurrency; k++) {
                    executorService.submit(new Callable<String>() {
                        @Override
                        public String call() throws Exception {
                            for (int ii = 0; ii < finalQuery; ii++) {
                                doSearch(table, client, nprobe, queryVectors, topk);
                                System.out.println(String.format("worker=%s, nq=%s, topk=%s, counter=%s",
                                        Thread.currentThread().getName(), anq, atopk, ii));
                            }
                            latch.countDown();
                            return "ok";
                        }
                    });
                }

                latch.await();
                executorService.shutdownNow();
                long end = Calendar.getInstance().getTimeInMillis();
                double costSeconds = (double) (end - start) / 1000;
                long queryCnt = querys * concurrency;
                double costPerBatch = costSeconds / queryCnt;
                double costPerVector = costSeconds / (queryCnt * nq[i]);
                System.out.println(String.format("batch=%s, topk=%s, costPerBatch=%.4f, costPerVector=%.4f",
                        nq[i], topk, costPerBatch, costPerVector));
                fileWriter.write(String.format("%s,%s,%.4f,%.4f", nq[i], topks[j], costPerBatch, costPerVector));
                fileWriter.write("\n");
                fileWriter.flush();
            }
            fileWriter.write("\n");
        }
        fileWriter.close();
        client.disconnect();



    }

    private static Long[] parseTopks(String topks) {
        List<String> splits = Splitter.on(",").trimResults().omitEmptyStrings().splitToList(topks);
        return splits.stream().map(Long::parseLong).distinct().toArray(Long[]::new);
    }

    private static Integer[] parseNqs(String nq) {
        List<String> splits = Splitter.on(",").trimResults().omitEmptyStrings().splitToList(nq);
        return splits.stream().map(Integer::parseInt).distinct().toArray(Integer[]::new);
    }

    private static void doSearch(String table, MilvusClient client, Long nprobe, List<List<Float>> vectorsToSearch, long topK) {
        SearchParam searchParam =
                new SearchParam.Builder(table, vectorsToSearch).withTopK(topK).withNProbe(nprobe).build();
        SearchResponse searchResponse = client.search(searchParam);
        if (!searchResponse.ok()) {
            System.out.println(JSON.toJSONString(searchResponse.getResponse()));
        }
    }

}
