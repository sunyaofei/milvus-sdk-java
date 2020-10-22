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

import io.milvus.grpc.IndexParam;
import io.milvus.grpc.KeyValuePair;

import javax.annotation.Nonnull;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/** Represents an index containing <code>fieldName</code>, <code>indexName</code> and
 * <code>paramsInJson</code>, which contains index_type, params etc.
 */
public class Index {
  private final IndexParam.Builder builder;

  public static Index create(@Nonnull String collectionName, @Nonnull String fieldName) {
    return new Index(collectionName, fieldName);
  }

  private Index(String collectionName, String fieldName) {
    this.builder = IndexParam.newBuilder()
        .setCollectionName(collectionName)
        .setFieldName(fieldName);
  }

  public String getCollectionName() {
    return builder.getCollectionName();
  }

  public Index setCollectionName(@Nonnull String collectionName) {
    builder.setCollectionName(collectionName);
    return this;
  }

  public String getFieldName() {
    return builder.getFieldName();
  }

  public Index setFieldName(@Nonnull String collectionName) {
    builder.setFieldName(collectionName);
    return this;
  }

  public String getIndexName() {
    return builder.getIndexName();
  }

  public Map<String, String> getExtraParams() {
    return toMap(builder.getExtraParamsList());
  }

  public Index setIndexType(IndexType indexType) {
    return addParam("index_type", indexType.name());
  }

  public Index setMetricType(MetricType metricType) {
    return addParam("metric_type", metricType.name());
  }

  public Index setParamsInJson(String paramsInJson) {
    return addParam(MilvusClient.extraParamKey, paramsInJson);
  }

  private Index addParam(String key, Object value) {
    builder.addExtraParams(
        KeyValuePair.newBuilder()
            .setKey(key)
            .setValue(String.valueOf(value))
            .build());
    return this;
  }

  @Override
  public String toString() {
    return "Index {"
        + "collectionName="
        + getCollectionName()
        + ", fieldName="
        + getFieldName()
        + ", params="
        + getExtraParams()
        + '}';
  }

  IndexParam grpc() {
    return builder.build();
  }

  private Map<String, String> toMap(List<KeyValuePair> extraParams) {
    return extraParams.stream().collect(Collectors.toMap(KeyValuePair::getKey, KeyValuePair::getValue));
  }
}
