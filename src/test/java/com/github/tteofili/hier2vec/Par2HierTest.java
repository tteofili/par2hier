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

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import com.github.tteofili.hier2vec.utils.LabelSeeker;
import com.github.tteofili.hier2vec.utils.MeansBuilder;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.FilenamesLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.assertEquals;

/**
 * Tests for par2hier
 */
@RunWith(Parameterized.class)
public class Par2HierTest {

  private final Hier2VecUtils.Method method;
  private int k;

  public Par2HierTest(Hier2VecUtils.Method method, int k) {
    this.method = method;
    this.k = k;
  }

  @Parameterized.Parameters
  public static Collection<Object[]> data() {
    return Arrays.asList(new Object[][] {
        {Hier2VecUtils.Method.CLUSTER, 5},
        {Hier2VecUtils.Method.CLUSTER, 4},
        {Hier2VecUtils.Method.CLUSTER, 3},
        {Hier2VecUtils.Method.CLUSTER, 2},
        {Hier2VecUtils.Method.SUM, 5},
        {Hier2VecUtils.Method.SUM, 4},
        {Hier2VecUtils.Method.SUM, 3},
        {Hier2VecUtils.Method.SUM, 2},
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

    // paragraph vectors training configuration
    paragraphVectors = new ParagraphVectors.Builder()
        .minWordFrequency(1)
        .iterations(5)
        .epochs(1)
        .layerSize(100)
        .learningRate(0.025)
        .windowSize(5)
        .iterate(iterator)
        .trainWordVectors(true)
        .tokenizerFactory(tokenizerFactory)
        .sampling(0)
        .build();

    // fit model
    paragraphVectors.fit();

    Par2Hier par2Hier = new Par2Hier.Builder()
        .minWordFrequency(1)
        .iterations(5)
        .epochs(1)
        .layerSize(100)
        .learningRate(0.025)
        .windowSize(5)
        .iterate(iterator)
        .trainWordVectors(true)
        .tokenizerFactory(tokenizerFactory)
        .sampling(0)
        .centroids(k)
        .smoothing(method)
        .build();

    // fit model
    par2Hier.fit();

    Map<String, String[]> comparison = new TreeMap<>();

    // extract paragraph vectors similarities
    Map<String, INDArray> pvs = new TreeMap<>();
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
      pvs.put(label, vector);
    }

    System.out.println("comparison (" + k + "," + method + ")");
    // output comparison
    for (Map.Entry<String, String[]> c : comparison.entrySet()) {
      String[] values = c.getValue();
      if (!values[0].equals(values[1])) {
        System.out.println(c.getKey() + ": " + Arrays.toString(values));
      }
    }

    System.out.println("indexes (" + k + "," + method + ")");
    // measure similarity indexes
    double[] intraDocumentSimilarity = getIntraDocumentSimilarity(comparison);
    System.out.println("ids:" + Arrays.toString(intraDocumentSimilarity));
    double[] depthSimilarity = getDepthSimilarity(comparison);
    System.out.println("ds:" + Arrays.toString(depthSimilarity));

    // persist 2 dimensional vectors
//    String pvCSV = asStrings(Hier2VecUtils.svdPCA(pvs, 2));
//    File pvFile = Files.createFile(Paths.get("target/pvs" + k + "-" + method + ".csv")).toFile();
//    FileOutputStream pvOutputStream = new FileOutputStream(pvFile);
//    IOUtils.write(pvCSV, pvOutputStream);
//
//    String hvCSV = asStrings(Hier2VecUtils.svdPCA(hvs, 2));
//    File hvFile = Files.createFile(Paths.get("target/hvs.csv" + k + "-" + method + ")")).toFile();
//    FileOutputStream hvOutputStream = new FileOutputStream(hvFile);
//    IOUtils.write(hvCSV, hvOutputStream);

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

  private Collection<String> nearestVectors(Map<String, INDArray> hvs, INDArray vector) {
    List<BasicModelUtils.WordSimilarity> result = new ArrayList<>();

    for (Map.Entry<String, INDArray> current : hvs.entrySet()) {
      double sim = Transforms.cosineSim(vector, current.getValue());
      result.add(new BasicModelUtils.WordSimilarity(current.getKey(), sim));
    }
    Collections.sort(result, new BasicModelUtils.SimilarityComparator());
    return BasicModelUtils.getLabels(result, 2);
  }

  private String asStrings(Map<String, INDArray> vs) {
    StringBuilder builder = new StringBuilder();
    for (Map.Entry<String, INDArray> entry : vs.entrySet()) {
      builder.append(entry.getKey()).append(", ").append(Arrays.toString(entry.getValue().data().asDouble())).append("\n");
    }
    return builder.toString();
  }

  @Test
  public void testTruncatedSVD() throws Exception {
    Random random = new Random();
    // 10 word vectors with dimension 30
    double[][] data = new double[10][30];
    for (int i = 0; i < data.length; i++) {
      for (int j = 0; j < data[i].length; j++) {
        data[i][j] = random.nextDouble();
      }
    }

    double[][] truncatedSVD = Hier2VecUtils.getTruncatedSVD(data, 3);

    assertEquals(data.length, truncatedSVD.length);
    assertEquals(data[0].length, truncatedSVD[0].length);
  }

  @Test
  public void testClassification() throws Exception {
    ClassPathResource resource = new ClassPathResource("classification/labeled");
    LabelAwareIterator iterator;
    TokenizerFactory tokenizerFactory;

    // build a iterator for our dataset
    iterator = new FileLabelAwareIterator.Builder()
        .addSourceFolder(resource.getFile())
        .build();

    tokenizerFactory = new DefaultTokenizerFactory();
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

    // ParagraphVectors training configuration
    ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
        .learningRate(0.025)
        .minLearningRate(0.001)
        .batchSize(1000)
        .epochs(20)
        .iterate(iterator)
        .trainWordVectors(true)
        .tokenizerFactory(tokenizerFactory)
        .build();

    // Start model training
    paragraphVectors.fit();

    // par2hier training configuration
    Par2Hier par2Hier = new Par2Hier.Builder()
        .learningRate(0.025)
        .minLearningRate(0.001)
        .batchSize(1000)
        .epochs(20)
        .iterate(iterator)
        .trainWordVectors(true)
        .tokenizerFactory(tokenizerFactory)
        .smoothing(method)
        .centroids(k)
        .build();

    // Start model training
    par2Hier.fit();

    ClassPathResource unClassifiedResource = new ClassPathResource("classification/unlabeled");
    checkClassification(iterator, tokenizerFactory, paragraphVectors, unClassifiedResource);
    checkClassification(iterator, tokenizerFactory, par2Hier, unClassifiedResource);

  }

  private void checkClassification(LabelAwareIterator iterator, TokenizerFactory tokenizerFactory,
                                   Word2Vec vectors, ClassPathResource unClassifiedResource) throws FileNotFoundException {
    FileLabelAwareIterator unClassifiedIterator = new FileLabelAwareIterator.Builder()
        .addSourceFolder(unClassifiedResource.getFile())
        .build();

    MeansBuilder meansBuilder = new MeansBuilder(
        (InMemoryLookupTable<VocabWord>) vectors.getLookupTable(),
        tokenizerFactory);
    LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
        (InMemoryLookupTable<VocabWord>) vectors.getLookupTable());

    while (unClassifiedIterator.hasNextDocument()) {
      LabelledDocument document = unClassifiedIterator.nextDocument();
      INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
      List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

      System.out.println(vectors.getClass() + "-> Document '" + document.getLabel() + "' falls into the following categories: ");
      for (Pair<String, Double> score : scores) {
        System.out.println("        " + score.getFirst() + ": " + score.getSecond());
      }
    }
  }

}
