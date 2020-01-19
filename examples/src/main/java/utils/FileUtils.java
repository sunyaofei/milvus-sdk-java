package utils;

import com.google.common.collect.Lists;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

/**
 * Description: FileUtils
 * Author: yaofei.sun
 * Create: yaofei.sun(2020-01-15 17:50)
 */
public class FileUtils {

    public static class Data {
        private final List<Integer> ids;
        private final List<List<Float>> data;

        public Data(List<Integer> ids, List<List<Float>> data) {
            this.ids = ids;
            this.data = data;
        }
    }


    public static List<Data> fvecs() throws IOException {
        File file = new File("/Users/sunyaofei/work/data/milvus/sift/sift_base.fvecs");
        long byteSize = file.length();
        System.out.println(String.format("length: %sB, %sM, %sG", byteSize, byteSize / (1024*1024), (float)byteSize / (1024*1024*1024)));
        FileInputStream fileInputStream = new FileInputStream(file);
        byte[] bytes = new byte[(int)byteSize];
        fileInputStream.read(bytes);

        // https://stackoverflow.com/questions/5616052/how-can-i-convert-a-4-byte-array-to-an-integer
        ByteBuffer byteBuffer = ByteBuffer.wrap(bytes, 0, 4).order(ByteOrder.LITTLE_ENDIAN);
        int dimention = byteBuffer.asIntBuffer().get();
        int rowCount = ((int)byteSize - 4) / (dimention * 4 + 4);
        int left = ((int)byteSize - 4) % (dimention * 4 + 4);
        System.out.println(String.format("dim=%s, rows=%s, left=%s", dimention, rowCount, left));
//        length: 516000000B, 492M, 0.48056245G
//        dim=128, rows=999999, left=512

        int oneRowSize = dimention * 4 + 4;
        List<Data> result = Lists.newArrayList();
        for (int i = 0; i * 100000 + 4 < byteSize; i++) {
            int start = i * 100000 + 4;
            int end = (i + 1)  * 100000 + 4;
            end = Math.min(end, (int) byteSize);

            int row = (end -  start) / oneRowSize;
            int leftBytes = (end -  start) % oneRowSize;

            List<Integer> ids = Lists.newArrayList();
            List<List<Float>> oneData = Lists.newArrayList();

            for (int j = i * 100000 + 4; j < end; j = j + oneRowSize) {
                int oneId = ByteBuffer.wrap(bytes, j, 4).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get();
                float[] oneRow = ByteBuffer.wrap(bytes, j + 1, dimention).order(ByteOrder.LITTLE_ENDIAN)
                        .asFloatBuffer().array();
                ids.add(oneId);
                List<Float> oneRowData = Lists.newArrayList();
                for (int x = 0; x < oneRow.length; x++) {
                    oneRowData.add(oneRow[x]);
                }

                oneData.add(oneRowData);
            }

            result.add(new Data(ids, oneData));
        }




return null;

    }

    public static void main(String[] args) throws IOException {
        fvecs();
    }
}
