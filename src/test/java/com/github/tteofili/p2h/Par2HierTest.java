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
package com.github.tteofili.p2h;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.io.IOUtils;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.FilenamesLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Tests for par2hier
 */
@RunWith(Parameterized.class)
public class Par2HierTest {

  private final Par2HierUtils.Method method;
  private int k;

  public Par2HierTest(Par2HierUtils.Method method, int k) {
    this.method = method;
    this.k = k;
  }

  @Parameterized.Parameters
  public static Collection<Object[]> data() {
    return Arrays.asList(new Object[][] {
        {Par2HierUtils.Method.CLUSTER, 3},
        {Par2HierUtils.Method.CLUSTER, 2},
        {Par2HierUtils.Method.SUM, 3},
        {Par2HierUtils.Method.SUM, 2},
    });
  }

  @Test
  public void testP2HOnMTPapers() throws Exception {
    ParagraphVectors paragraphVectors;
    LabelAwareIterator iterator;
    TokenizerFactory tokenizerFactory;
    ClassPathResource resource = new ClassPathResource("papers/sbc");

    // build a iterator for our MT papers dataset
    iterator = new FilenamesLabelAwareIterator.Builder()
        .addSourceFolder(resource.getFile())
        .build();

    tokenizerFactory = new DefaultTokenizerFactory();
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

    Map<String, INDArray> hvs = new TreeMap<>();
    Map<String, INDArray> pvs = new TreeMap<>();

    paragraphVectors = new ParagraphVectors.Builder()
        .iterate(iterator)
        .tokenizerFactory(tokenizerFactory)
        .build();

    // fit model
    paragraphVectors.fit();

    Par2Hier par2Hier = new Par2Hier(paragraphVectors, method, k);

    // fit model
    par2Hier.fit();

    Map<String, String[]> comparison = new TreeMap<>();

    // extract paragraph vectors similarities
    WeightLookupTable<VocabWord> lookupTable = paragraphVectors.getLookupTable();
    List<String> labels = paragraphVectors.getLabelsSource().getLabels();
    for (String label : labels) {
      INDArray vector = lookupTable.vector(label);
      pvs.put(label, vector);
      Collection<String> strings = paragraphVectors.nearestLabels(vector, 2);
      Collection<String> hstrings = par2Hier.nearestLabels(vector, 2);
      String[] stringsArray = new String[2];
      stringsArray[0] = new LinkedList<>(strings).get(1);
      stringsArray[1] = new LinkedList<>(hstrings).get(1);
      comparison.put(label, stringsArray);
      hvs.put(label, par2Hier.getLookupTable().vector(label));
    }

    System.out.println("--->func(args):pv,p2h");

    // measure similarity indexes
    double[] intraDocumentSimilarity = getIntraDocumentSimilarity(comparison);
    System.out.println("ids(" + k + "," + method + "):" + Arrays.toString(intraDocumentSimilarity));
    double[] depthSimilarity = getDepthSimilarity(comparison);
    System.out.println("ds(" + k + "," + method + "):" + Arrays.toString(depthSimilarity));

    // classification
    Map<Integer, Map<Integer, Long>> pvCounts = new HashMap<>();
    Map<Integer, Map<Integer, Long>> p2hCounts = new HashMap<>();
    for (String label : labels) {

      INDArray vector = lookupTable.vector(label);
      int topN = 1;
      Collection<String> strings = paragraphVectors.nearestLabels(vector, topN);
      Collection<String> hstrings = par2Hier.nearestLabels(vector, topN);
      int labelDepth = label.split("\\.").length - 1;

      int stringDepth = getClass(strings);
      int hstringDepth = getClass(hstrings);

      updateCM(pvCounts, labelDepth, stringDepth);
      updateCM(p2hCounts, labelDepth, hstringDepth);
    }

    ConfusionMatrix pvCM = new ConfusionMatrix(pvCounts);
    ConfusionMatrix p2hCM = new ConfusionMatrix(p2hCounts);

    System.out.println("mf1("+k+","+method+"):"+pvCM.getF1Measure()+","+p2hCM.getF1Measure());
    System.out.println("acc("+k+","+method+"):"+pvCM.getAccuracy()+","+p2hCM.getAccuracy());

    // create a CSV with a raw comparison
    File pvFile = Files.createFile(Paths.get("target/comparison-" + k + "-" + method + ".csv")).toFile();
    FileOutputStream pvOutputStream = new FileOutputStream(pvFile);

    try {
      Map<String, INDArray> pvs2 = Par2HierUtils.svdPCA(pvs, 2);
      Map<String, INDArray> hvs2 = Par2HierUtils.svdPCA(hvs, 2);
      String pvCSV = asStrings(pvs2, hvs2);
      IOUtils.write(pvCSV, pvOutputStream);
    } finally {
      pvOutputStream.flush();
      pvOutputStream.close();
    }
  }

  private void updateCM(Map<Integer, Map<Integer, Long>> pvCounts, int labelDepth, int stringDepth) {
    Map<Integer, Long> stringLongMap = pvCounts.get(labelDepth);
    if (stringLongMap != null) {
      Long aLong = stringLongMap.get(stringDepth);
      if (aLong != null) {
        stringLongMap.put(stringDepth, aLong + 1);
      } else {
        stringLongMap.put(stringDepth, 1L);
      }
    } else {
      stringLongMap = new HashMap<>();
      stringLongMap.put(stringDepth, 1L);
      pvCounts.put(labelDepth, stringLongMap);
    }
  }

  private int getClass(Collection<String> strings) {
    Map<Integer, Integer> m = new HashMap<>();
    for (String s : strings) {
      int depth = s.split("\\.").length - 1;
      m.put(depth, m.containsKey(depth) ? m.get(depth) + 1 : 1);
    }
    int max = 0;
    int md = 0;
    for (Map.Entry<Integer, Integer> e : m.entrySet()) {
      if (e.getValue() > max) {
        md = e.getKey();
        max = e.getValue();
      }
    }
    return md;
  }

  private String asStrings(Map<String, INDArray> pvs, Map<String, INDArray> hvs) {
    StringBuilder builder = new StringBuilder();
    builder.append("doc, depth, Paragraph, PV x, PV y, HV x, HV y\n");
    for (Map.Entry<String, INDArray> entry : pvs.entrySet()) {
      String key = entry.getKey();
      String depth = String.valueOf(key.split("\\.").length - 1);
      String c = String.valueOf(key.charAt(3));
      if (Character.isDigit(key.charAt(4))) {
        c += key.charAt(4);
      }
      String doc = String.valueOf(c);
      builder.append(doc).append(',').append(depth).append(", ").append(entry.toString().replace("=[", ",").replace("]", ","));
      String s = hvs.get(key).toString();
      builder.append(s.replace("[", "").replace("]", "")).append("\n");
    }
    return builder.toString();
  }

  private double[] getDepthSimilarity(Map<String, String[]> comparison) {
    double pvSimilarity = 0;
    double hvSimilarity = 0;
    for (Map.Entry<String, String[]> c : comparison.entrySet()) {
      String label = c.getKey();
      String nearestPV = c.getValue()[0];
      String nearestHV = c.getValue()[1];

      int labelDepth = label.split("\\.").length;
      int pvDepth = nearestPV.split("\\.").length;
      int hvDepth = nearestHV.split("\\.").length;
      if (labelDepth == hvDepth) {
        hvSimilarity++;
      }
      if (labelDepth == pvDepth) {
        pvSimilarity++;
      }
    }
    double size = comparison.keySet().size();
    pvSimilarity /= size;
    hvSimilarity /= size;
    return new double[] {pvSimilarity, hvSimilarity};
  }

  private double[] getIntraDocumentSimilarity(Map<String, String[]> comparison) {
    double pvSimilarity = 0;
    double hvSimilarity = 0;
    for (Map.Entry<String, String[]> c : comparison.entrySet()) {
      String label = c.getKey();
      String nearestPV = c.getValue()[0];
      String nearestHV = c.getValue()[1];
      if (label.charAt(3) == nearestHV.charAt(3) && label.charAt(4) == nearestHV.charAt(4)) {
        hvSimilarity++;
      }
      if (label.charAt(3) == nearestPV.charAt(3) && label.charAt(4) == nearestPV.charAt(4)) {
        pvSimilarity++;
      }
    }
    double size = comparison.keySet().size();
    pvSimilarity /= size;
    hvSimilarity /= size;
    return new double[] {pvSimilarity, hvSimilarity};
  }

  static class ConfusionMatrix {

    private final Map<Integer, Map<Integer, Long>> linearizedMatrix;
    private double accuracy = -1d;

    private ConfusionMatrix(Map<Integer, Map<Integer, Long>> linearizedMatrix) {
      this.linearizedMatrix = linearizedMatrix;
    }

    /**
     * get the linearized confusion matrix as a {@link Map}
     *
     * @return a {@link Map} whose keys are the correct classification answers and whose values are the actual answers'
     * counts
     */
    public Map<Integer, Map<Integer, Long>> getLinearizedMatrix() {
      return Collections.unmodifiableMap(linearizedMatrix);
    }

    /**
     * calculate precision on the given class
     *
     * @param klass the class to calculate the precision for
     * @return the precision for the given class
     */
    public double getPrecision(Integer klass) {
      Map<Integer, Long> classifications = linearizedMatrix.get(klass);
      double tp = 0;
      double den = 0; // tp + fp
      if (classifications != null) {
        for (Map.Entry<Integer, Long> entry : classifications.entrySet()) {
          if (klass.equals(entry.getKey())) {
            tp += entry.getValue();
          }
        }
        for (Map<Integer, Long> values : linearizedMatrix.values()) {
          if (values.containsKey(klass)) {
            den += values.get(klass);
          }
        }
      }
      return tp > 0 ? tp / den : 0;
    }

    /**
     * calculate recall on the given class
     *
     * @param klass the class to calculate the recall for
     * @return the recall for the given class
     */
    public double getRecall(Integer klass) {
      Map<Integer, Long> classifications = linearizedMatrix.get(klass);
      double tp = 0;
      double fn = 0;
      if (classifications != null) {
        for (Map.Entry<Integer, Long> entry : classifications.entrySet()) {
          if (klass.equals(entry.getKey())) {
            tp += entry.getValue();
          } else {
            fn += entry.getValue();
          }
        }
      }
      return tp + fn > 0 ? tp / (tp + fn) : 0;
    }

    /**
     * get the F-1 measure on this confusion matrix
     *
     * @return the F-1 measure
     */
    public double getF1Measure() {
      double recall = getRecall();
      double precision = getPrecision();
      return precision > 0 && recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
    }

    /**
     * Calculate accuracy on this confusion matrix using the formula:
     * {@literal accuracy = correctly-classified / (correctly-classified + wrongly-classified)}
     *
     * @return the accuracy
     */
    public double getAccuracy() {
      if (this.accuracy == -1) {
        double tp = 0d;
        double tn = 0d;
        double tfp = 0d; // tp + fp
        double fn = 0d;
        for (Map.Entry<Integer, Map<Integer, Long>> classification : linearizedMatrix.entrySet()) {
          Integer klass = classification.getKey();
          for (Map.Entry<Integer, Long> entry : classification.getValue().entrySet()) {
            if (klass.equals(entry.getKey())) {
              tp += entry.getValue();
            } else {
              fn += entry.getValue();
            }
          }
          for (Map<Integer, Long> values : linearizedMatrix.values()) {
            if (values.containsKey(klass)) {
              tfp += values.get(klass);
            } else {
              tn++;
            }
          }

        }
        this.accuracy = (tp + tn) / (tfp + fn + tn);
      }
      return this.accuracy;
    }

    /**
     * get the macro averaged precision (see {@link #getPrecision(Integer)}) over all the classes.
     *
     * @return the macro averaged precision as computed from the confusion matrix
     */
    public double getPrecision() {
      double p = 0;
      for (Map.Entry<Integer, Map<Integer, Long>> classification : linearizedMatrix.entrySet()) {
        Integer klass = classification.getKey();
        p += getPrecision(klass);
      }

      return p / linearizedMatrix.size();
    }

    /**
     * get the macro averaged recall (see {@link #getRecall(Integer)}) over all the classes
     *
     * @return the recall as computed from the confusion matrix
     */
    public double getRecall() {
      double r = 0;
      for (Map.Entry<Integer, Map<Integer, Long>> classification : linearizedMatrix.entrySet()) {
        Integer klass = classification.getKey();
        r += getRecall(klass);
      }

      return r / linearizedMatrix.size();
    }

    @Override
    public String toString() {
      return "ConfusionMatrix{" +
          "linearizedMatrix=" + linearizedMatrix +
          '}';
    }
  }
}
