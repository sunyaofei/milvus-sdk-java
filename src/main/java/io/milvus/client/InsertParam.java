/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package io.milvus.client;

import com.google.protobuf.ByteString;
import io.milvus.client.exception.UnsupportedDataType;
import io.milvus.grpc.AttrRecord;
import io.milvus.grpc.FieldValue;
import io.milvus.grpc.VectorRecord;
import io.milvus.grpc.VectorRowRecord;

import java.nio.ByteBuffer;
import java.util.List;
import java.util.stream.Collectors;

/** Contains parameters for <code>insert</code> */
public class InsertParam {
  private io.milvus.grpc.InsertParam.Builder builder;

  public static InsertParam create(String collectionName) {
    return new InsertParam(collectionName);
  }

  private InsertParam(String collectionName) {
    this.builder = io.milvus.grpc.InsertParam.newBuilder();
    builder.setCollectionName(collectionName);
  }

  public InsertParam setEntityIds(List<Long> entityIds) {
    builder.addAllEntityIdArray(entityIds);
    return this;
  }

  public <T> InsertParam addField(String name, DataType type, List<T> values) {
    AttrRecord.Builder record = AttrRecord.newBuilder();
    switch (type) {
      case INT32:
        record.addAllInt32Value((List<Integer>) values);
        break;
      case INT64:
        record.addAllInt64Value((List<Long>) values);
        break;
      case FLOAT:
        record.addAllFloatValue((List<Float>) values);
        break;
      case DOUBLE:
        record.addAllDoubleValue((List<Double>) values);
        break;
      default:
        throw new UnsupportedDataType("Unsupported data type: " + type.name());
    }
    builder.addFields(FieldValue.newBuilder()
        .setFieldName(name)
        .setTypeValue(type.getVal())
        .setAttrRecord(record.build())
        .build());
    return this;
  }

  public <T> InsertParam addVectorField(String name, DataType type, List<T> values) {
    VectorRecord.Builder record = VectorRecord.newBuilder();
    switch (type) {
      case VECTOR_FLOAT:
        record.addAllRecords(
            ((List<List<Float>>) values).stream()
                .map(row -> VectorRowRecord.newBuilder().addAllFloatData(row).build())
                .collect(Collectors.toList()));
        break;
      case VECTOR_BINARY:
        record.addAllRecords(
            ((List<ByteBuffer>) values).stream()
                .map(row -> VectorRowRecord.newBuilder().setBinaryData(ByteString.copyFrom(row.slice())).build())
                .collect(Collectors.toList()));
        break;
      default:
        throw new UnsupportedDataType("Unsupported data type: " + type.name());
    }
    builder.addFields(FieldValue.newBuilder()
        .setFieldName(name)
        .setTypeValue(type.getVal())
        .setVectorRecord(record.build())
        .build());
    return this;
  }

  public InsertParam setPartitionTag(String partitionTag) {
    builder.setPartitionTag(partitionTag);
    return this;
  }

  io.milvus.grpc.InsertParam grpc() {
    return builder.build();
  }
}
