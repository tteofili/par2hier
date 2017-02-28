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

import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

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

import static org.junit.Assert.assertEquals;

/**
 * Tests for par2hier
 */
@RunWith(Parameterized.class)
public class Par2HierTest {

  private final Hier2VecUtils.Method method;
  private int k;
  private int iterations;

  public Par2HierTest(Hier2VecUtils.Method method, int k, int iterations) {
    this.method = method;
    this.k = k;
    this.iterations = iterations;
  }

  @Parameterized.Parameters
  public static Collection<Object[]> data() {
    return Arrays.asList(new Object[][] {
        {Hier2VecUtils.Method.CLUSTER, 4, 5},
        {Hier2VecUtils.Method.CLUSTER, 3, 5},
        {Hier2VecUtils.Method.CLUSTER, 2, 5},
        {Hier2VecUtils.Method.CLUSTER, 1, 5},
        {Hier2VecUtils.Method.SUM, 4, 5},
        {Hier2VecUtils.Method.SUM, 3, 5},
        {Hier2VecUtils.Method.SUM, 2, 5},
        {Hier2VecUtils.Method.SUM, 1, 5},
    });
  }

  @Test
  public void testP2HOnMTPapers() throws Exception {
    for (int it = 0; it < iterations; it++) {
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

      // paragraph vectors training configuration
      double learningRate = 0.025;
      int iterations = 5;
      int windowSize = 5;
      int layerSize = 60;
      int numEpochs = 10;
      int minWordFrequency = 1;
      double minLearningRate = 0.001;
      int batchSize = 5;

      paragraphVectors = new ParagraphVectors.Builder()
          .minWordFrequency(minWordFrequency)
          .iterations(iterations)
          .epochs(numEpochs)
          .layerSize(layerSize)
          .minLearningRate(minLearningRate)
          .batchSize(batchSize)
          .learningRate(learningRate)
          .windowSize(windowSize)
          .iterate(iterator)
          .trainWordVectors(true)
          .tokenizerFactory(tokenizerFactory)
          .build();

      // fit model
      paragraphVectors.fit();

      Par2Hier par2Hier = new Par2Hier.Builder()
          .minWordFrequency(minWordFrequency)
          .iterations(iterations)
          .epochs(numEpochs)
          .layerSize(layerSize)
          .learningRate(learningRate)
          .minLearningRate(minLearningRate)
          .batchSize(batchSize)
          .windowSize(windowSize)
          .iterate(iterator)
          .trainWordVectors(true)
          .tokenizerFactory(tokenizerFactory)
          .centroids(k)
          .smoothing(method)
          .useExistingWordVectors(paragraphVectors) // enhance existing vectors rather than creating new ones, for more appropriate comparison
          .build();

      // fit model
      par2Hier.fit();

      Map<String, String[]> comparison = new TreeMap<>();

      // extract paragraph vectors similarities
      WeightLookupTable<VocabWord> lookupTable = paragraphVectors.getLookupTable();
      List<String> labels = paragraphVectors.getLabelsSource().getLabels();
      for (String label : labels) {
        INDArray vector = lookupTable.vector(label);
        Collection<String> strings = paragraphVectors.nearestLabels(vector, 2);
        Collection<String> hstrings = par2Hier.nearestLabels(vector, 2);
        String[] stringsArray = new String[2];
        stringsArray[0] = new LinkedList<>(strings).get(1);
        stringsArray[1] = new LinkedList<>(hstrings).get(1);
        comparison.put(label, stringsArray);
      }

      System.out.println("--->func(args):pv,p2h");
      // measure similarity indexes
      double[] intraDocumentSimilarity = getIntraDocumentSimilarity(comparison);
      System.out.println("ids(" + k + "," + method + "):" + Arrays.toString(intraDocumentSimilarity));
      double[] depthSimilarity = getDepthSimilarity(comparison);
      System.out.println("ds(" + k + "," + method + "):" + Arrays.toString(depthSimilarity));
      double[] accuracies = getDepthSimilarityAccuracy(comparison);
      System.out.println("acc(" + k + "," + method + "):" + Arrays.toString(accuracies));
    }
  }

  private double[] getDepthSimilarityAccuracy(Map<String, String[]> comparison) {
    double pvAcc = 0;
    double hvAcc = 0;
    for (Map.Entry<String, String[]> c : comparison.entrySet()) {
      String label = c.getKey();
      String nearestPV = c.getValue()[0];
      String nearestHV = c.getValue()[1];
      if (label.lastIndexOf('.') == nearestHV.lastIndexOf('.')) {
        hvAcc++;
      }
      if (label.lastIndexOf('.') == nearestPV.lastIndexOf('.')) {
        pvAcc++;
      }
    }
    pvAcc /= (comparison.keySet().size());
    hvAcc /= (comparison.keySet().size());
    return new double[] {pvAcc, hvAcc};
  }

  private double[] getDepthSimilarity(Map<String, String[]> comparison) {
    double pvSimilarity = 0;
    double hvSimilarity = 0;
    for (Map.Entry<String, String[]> c : comparison.entrySet()) {
      String label = c.getKey();
      String nearestPV = c.getValue()[0];
      String nearestHV = c.getValue()[1];
      if (label.indexOf(' ') == nearestHV.indexOf(' ')) {
        hvSimilarity++;
      }
      if (label.indexOf(' ') == nearestPV.indexOf(' ')) {
        pvSimilarity++;
      }
    }
    pvSimilarity /= (comparison.keySet().size() * 6);
    hvSimilarity /= (comparison.keySet().size() * 6);
    return new double[] {pvSimilarity, hvSimilarity};
  }

  private double[] getIntraDocumentSimilarity(Map<String, String[]> comparison) {
    double pvSimilarity = 0;
    double hvSimilarity = 0;
    for (Map.Entry<String, String[]> c : comparison.entrySet()) {
      String label = c.getKey();
      String nearestPV = c.getValue()[0];
      String nearestHV = c.getValue()[1];
      if (label.charAt(3) == nearestHV.charAt(3)) {
        hvSimilarity++;
      }
      if (label.charAt(3) == nearestPV.charAt(3)) {
        pvSimilarity++;
      }
    }
    pvSimilarity /= (comparison.keySet().size() * 6);
    hvSimilarity /= (comparison.keySet().size() * 6);
    return new double[] {pvSimilarity, hvSimilarity};
  }

}
