import io.milvus.client.*;
import utils.BenchMarkUtils;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Date;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * Description: BenchMark
 * Author: yaofei.sun
 * Create: yaofei.sun(2020-01-15 16:29)
 */
public class BenchMark {

    private static String tableName;
    private static long dimension;
    private static long indexFileSize;
    private static MetricType metricType;
    private static String sift1mPath;
    private static void init(MilvusClient client) {
        // Create a table with the following table schema
        tableName = "ann_1m_sq8"; // table name
        dimension = 128; // dimension of each vector
        indexFileSize = 1024; // maximum size (in MB) of each index file
        metricType = MetricType.IP; // we choose IP (Inner Product) as our metric type
        TableSchema tableSchema =
                new TableSchema.Builder(tableName, dimension)
                        .withIndexFileSize(indexFileSize)
                        .withMetricType(metricType)
                        .build();
        Response createTableResponse = client.createTable(tableSchema);
        System.out.println(createTableResponse);

        // Check whether the table exists
        HasTableResponse hasTableResponse = client.hasTable(tableName);
        System.out.println(hasTableResponse);

        // Describe the table
        DescribeTableResponse describeTableResponse = client.describeTable(tableName);
        System.out.println(describeTableResponse);

        // Create index for the table
        // We choose IVF_SQ8 as our index type here. Refer to IndexType javadoc for a
        // complete explanation of different index types
        final IndexType indexType = IndexType.IVF_SQ8;
        Index index = new Index.Builder().withIndexType(indexType).build();
        CreateIndexParam createIndexParam =
                new CreateIndexParam.Builder(tableName).withIndex(index).build();
        Response createIndexResponse = client.createIndex(createIndexParam);
        System.out.println(createIndexResponse);
    }

    /**
     * http://corpus-texmex.irisa.fr/
     * http://corpus-texmex.irisa.fr/fvecs_read.m
     *
     * @param client
     * @throws IOException
     */
    private static void loadData(MilvusClient client) throws IOException {
        sift1mPath = "/Users/sunyaofei/work/data/milvus/sift";
//

        File file = new File(sift1mPath + "/" + "sift_base.fvecs");
        long byteSize = file.length();
        System.out.println(String.format("length: %sB, %sM, %sG", byteSize, byteSize / (1024*1024), (float)byteSize / (1024*1024*1024)));
        FileInputStream fileInputStream = new FileInputStream(file);
        byte[] bytes = new byte[(int)byteSize];
        fileInputStream.read(bytes);

        // https://stackoverflow.com/questions/5616052/how-can-i-convert-a-4-byte-array-to-an-integer
        ByteBuffer byteBuffer = ByteBuffer.wrap(bytes, 0, 4).order(ByteOrder.LITTLE_ENDIAN);
        byteBuffer.asIntBuffer().get();




        // Insert randomly generated vectors to table
        //    final int vectorCount = 100000;
        final int vectorCount = 1000;
        List<List<Float>> vectors = BenchMarkUtils.generateVectors(vectorCount, dimension);
        vectors =
                vectors.stream().map(BenchMarkUtils::normalizeVector).collect(Collectors.toList());

        InsertParam insertParam = new InsertParam.Builder(tableName, vectors).build();
        InsertResponse insertResponse = client.insert(insertParam);
        System.out.println(insertResponse);
    }

    private static void clear() {

    }

    public static void main(String[] args) throws InterruptedException, ConnectFailedException, IOException {
//        10.26.9.4 31481 10.26.36.16 30059 10.26.21.51 19530
        final String host = "10.26.9.4 31481";
        final String port = "31481";

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

        init(client);
        loadData(client);

        // The data we just inserted won't be serialized and written to meta until the next second
        // wait 1 second here
        TimeUnit.SECONDS.sleep(1);


        // Get current row count of table
        GetTableRowCountResponse getTableRowCountResponse = client.getTableRowCount(tableName);
        System.out.println(getTableRowCountResponse);

        // Describe the index for your table
        DescribeIndexResponse describeIndexResponse = client.describeIndex(tableName);
        System.out.println(describeIndexResponse);

        System.out.println("ready to search " + new Date());

//        // Search vectors
//        // Searching the first 5 vectors of the vectors we just inserted
//        final int searchBatchSize = 5;
//        List<List<Float>> vectorsToSearch = vectors.subList(0, searchBatchSize);
//        final long topK = 10;
//        SearchParam searchParam =
//                new SearchParam.Builder(tableName, vectorsToSearch).withTopK(topK).build();
//        SearchResponse searchResponse = client.search(searchParam);
//        System.out.println(searchResponse);
//        if (searchResponse.ok()) {
//            List<List<SearchResponse.QueryResult>> queryResultsList =
//                    searchResponse.getQueryResultsList();
//            final double epsilon = 0.001;
//            for (int i = 0; i < searchBatchSize; i++) {
//                // Since we are searching for vector that is already present in the table,
//                // the first result vector should be itself and the distance (inner product) should be
//                // very close to 1 (some precision is lost during the process)
//                System.out.println(queryResultsList);
//                //        SearchResponse.QueryResult firstQueryResult = queryResultsList.get(i).get(0);
//                //        if (firstQueryResult.getVectorId() != vectorIds.get(i)
//                //            || Math.abs(1 - firstQueryResult.getDistance()) > epsilon) {
//                ////          throw new AssertionError("Wrong results!");
//                //        }
//            }
//        }

        System.out.println("search done " + new Date());

        //     Drop index for the table
        Response dropIndexResponse = client.dropIndex(tableName);
        System.out.println(dropIndexResponse);

        // Drop table
        Response dropTableResponse = client.dropTable(tableName);
        System.out.println(dropTableResponse);

        // Disconnect from Milvus server
        try {
            Response disconnectResponse = client.disconnect();
        } catch (InterruptedException e) {
            System.out.println("Failed to disconnect: " + e.toString());
            throw e;
        }
    }
}
