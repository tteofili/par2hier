/*
 * Copyright 2017 Tommaso Teofili
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
*/
package com.github.tteofili.hier2vec;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Utility class to implement hier2vec
 */
class Hier2VecUtils {

  enum Method {
    CLUSTER,
    SUM
  }

  /**
   * perform PCA using SVD
   *
   * @param weightTable the weights to scale
   * @param k the target vector dimensions
   * @return a weight table doc->reduced vector
   */
  static Map<String, INDArray> svdPCA(Map<String, INDArray> weightTable, int k) {
    Map<String, INDArray> reducedVectors = new TreeMap<>();

    INDArray matrix = null;
    int i = 0;
    for (Map.Entry<String, INDArray> entry : weightTable.entrySet()) {
      INDArray vector = entry.getValue();
      if (matrix == null) {
        matrix = Nd4j.zeros(weightTable.size(), vector.columns());
      }
      matrix.putRow(i, vector);
      i++;
    }

    int j = 0;
    INDArray reducedMatrix = Nd4j.create(getTruncatedUT(matrix, k));
    for (Map.Entry<String, INDArray> entry : weightTable.entrySet()) {
      reducedVectors.put(entry.getKey(), reducedMatrix.getRow(j));
      j++;
    }

    return reducedVectors;
  }

  /**
   * transforms paragraph vectors into hierarchical vectors
   * @param iterator iterator over docs
   * @param lookupTable the paragraph vector table
   * @param labels the labels
   * @param k the no. of centroids
   * @return a map doc->hierarchical vector
   */
  static Map<String, INDArray> getHier2Vec(LabelAwareIterator iterator,
                                           WeightLookupTable<VocabWord> lookupTable,
                                           List<String> labels, int k, Method method) {
    Collections.sort(labels);
    LabelsSource labelsSource = iterator.getLabelsSource();
    StringTrie trie = new StringTrie();
    trie.addAll(labels.toArray(new String[labels.size()]));

    Map<String, INDArray> hvs = new TreeMap<>();
    // for each doc
    for (String node : labelsSource.getLabels()) {
      Hier2VecUtils.getHv(lookupTable, trie, node, k, hvs, method);
    }
    return hvs;
  }

  /**
   * base case: on a leave hv = pv (+ k centroids of wv)
   * on a non-leave node with n childs: hv = pv + k centroids of the n hv
   */
  private static INDArray getHv(WeightLookupTable<VocabWord> lookupTable, StringTrie trie, String node,
                                int k, Map<String, INDArray> hvs, Method method) {
    if (hvs.containsKey(node)) {
      return hvs.get(node);
    }
    INDArray hv = lookupTable.vector(node);
    String[] split = node.split("\\.txt\\_(\\d\\.?)+");
    Collection<String> descendants;
    if (split.length == 2) {
      String prefix = node.substring(0, node.indexOf(split[1]));

      descendants = trie.search(prefix);
    } else {
      descendants = Collections.emptyList();
    }
    if (descendants.size() <= 1) {
      // just the pv
      hvs.put(node, hv);
      return hv;
    } else {
      INDArray chvs = Nd4j.zeros(descendants.size() - 1, hv.columns());
      int i = 0;
      for (String desc : descendants) {
        if (node.equals(desc)) {
          continue;
        }
        // child hierarchical vector
        INDArray chv = getHv(lookupTable, trie, desc, k, hvs, method);
        chvs.putRow(i, chv);
        i++;
      }

      double[][] centroids;
      if (chvs.rows() > k) {
        centroids = Hier2VecUtils.getTruncatedVT(chvs, k);
      } else if (chvs.rows() == 1) {
        centroids = Hier2VecUtils.getDoubles(chvs.getRow(0));
      } else {
        centroids = Hier2VecUtils.getTruncatedVT(chvs, 1);
      }
      switch (method) {

        case CLUSTER:
          INDArray matrix = Nd4j.zeros(centroids.length + 1, hv.columns());
          matrix.putRow(0, hv);
          for (int c = 0; c < centroids.length; c++) {
            matrix.putRow(c + 1, Nd4j.create(centroids[c]));
          }
          hv = Nd4j.create(Hier2VecUtils.getTruncatedVT(matrix, 1));
          break;
        case SUM:
          for (double[] centroid : centroids) {
            hv.addi(Nd4j.create(centroid));
          }
          break;
      }

      hvs.put(node, hv);
      return hv;
    }
  }

  /**
   * truncated SVD as taken from http://stackoverflow.com/questions/19957076/best-way-to-compute-a-truncated-singular-value-decomposition-in-java
   */
  static double[][] getTruncatedSVD(double[][] matrix, final int k) {
    SingularValueDecomposition svd = new SingularValueDecomposition(MatrixUtils.createRealMatrix(matrix));

    double[][] truncatedU = new double[svd.getU().getRowDimension()][k];
    svd.getU().copySubMatrix(0, truncatedU.length - 1, 0, k - 1, truncatedU);

    double[][] truncatedS = new double[k][k];
    svd.getS().copySubMatrix(0, k - 1, 0, k - 1, truncatedS);

    double[][] truncatedVT = new double[k][svd.getVT().getColumnDimension()];
    svd.getVT().copySubMatrix(0, k - 1, 0, truncatedVT[0].length - 1, truncatedVT);

    RealMatrix approximatedSvdMatrix = (MatrixUtils.createRealMatrix(truncatedU)).multiply(
        MatrixUtils.createRealMatrix(truncatedS)).multiply(MatrixUtils.createRealMatrix(truncatedVT));

    return approximatedSvdMatrix.getData();
  }

  public static double[][] getTruncatedSVD(INDArray matrix, final int k) {

    double[][] data = getDoubles(matrix);

    SingularValueDecomposition svd = new SingularValueDecomposition(MatrixUtils.createRealMatrix(data));

    double[][] truncatedU = new double[svd.getU().getRowDimension()][k];
    svd.getU().copySubMatrix(0, truncatedU.length - 1, 0, k - 1, truncatedU);

    double[][] truncatedS = new double[k][k];
    svd.getS().copySubMatrix(0, k - 1, 0, k - 1, truncatedS);

    double[][] truncatedVT = new double[k][svd.getVT().getColumnDimension()];
    svd.getVT().copySubMatrix(0, k - 1, 0, truncatedVT[0].length - 1, truncatedVT);

    RealMatrix approximatedSvdMatrix = (MatrixUtils.createRealMatrix(truncatedU)).multiply(
        MatrixUtils.createRealMatrix(truncatedS)).multiply(MatrixUtils.createRealMatrix(truncatedVT));

    return approximatedSvdMatrix.getData();
  }

  private static double[][] getTruncatedVT(INDArray matrix, int k) {
    double[][] data = getDoubles(matrix);

    SingularValueDecomposition svd = new SingularValueDecomposition(MatrixUtils.createRealMatrix(data));

    double[][] truncatedVT = new double[k][svd.getVT().getColumnDimension()];
    svd.getVT().copySubMatrix(0, k - 1, 0, truncatedVT[0].length - 1, truncatedVT);
    return truncatedVT;
  }

  private static double[][] getTruncatedUT(INDArray matrix, int k) {
    double[][] data = getDoubles(matrix);

    SingularValueDecomposition svd = new SingularValueDecomposition(MatrixUtils.createRealMatrix(data));

    double[][] truncatedU = new double[svd.getU().getRowDimension()][k];
    svd.getU().copySubMatrix(0, truncatedU.length - 1, 0, k - 1, truncatedU);
    return truncatedU;
  }

  private static double[][] getDoubles(INDArray matrix) {
    double[][] data = new double[matrix.rows()][matrix.columns()];
    for (int i = 0; i < data.length; i++) {
      for (int j = 0; j < data[0].length; j++) {
        data[i][j] = matrix.getDouble(i, j);
      }
    }
    return data;
  }
}
