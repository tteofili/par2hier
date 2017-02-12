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

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.io.IOUtils;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.FilenamesLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.assertEquals;

/**
 * Tests for par2hier
 */
public class Par2HierTest {

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

    // ParagraphVectors training configuration
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

    Map<String, String[]> comparison = new TreeMap<>();

    // check similarities among paragraph vectors
    WeightLookupTable<VocabWord> lookupTable = paragraphVectors.getLookupTable();
    List<String> labels = paragraphVectors.getLabelsSource().getLabels();
    Map<String, INDArray> pvs = new TreeMap<>();
    for (String label : labels) {
      INDArray vector = lookupTable.vector(label);
      pvs.put(label, vector);
      Collection<String> strings = paragraphVectors.nearestLabels(vector, 2);
//      System.out.println(label + ": " + strings);
      String[] stringsArray = new String[2];
      stringsArray[0] = new LinkedList<>(strings).get(1);
      comparison.put(label, stringsArray);
    }


    System.out.println("****");
    System.out.println("****");
    System.out.println("****");
    System.out.println("****");
    System.out.println("****");
    System.out.println("****");


    // create hierarchical vectors
    Map<String, INDArray> hvs = Hier2VecUtils.getPar2Hier(iterator, lookupTable, labels, 3,
        Hier2VecUtils.Method.CLUSTER);

    // check similarity between hierarchical and paragraph vectors
//    for (Map.Entry<String, INDArray> entry : hvs.entrySet()) {
//      Collection<String> strings = paragraphVectors.nearestLabels(entry.getValue(), 2);
//      System.out.println(entry.getKey() + ": " + strings);
//    }

//    System.out.println("****");
//    System.out.println("****");
//    System.out.println("****");
//    System.out.println("****");
//    System.out.println("****");
//    System.out.println("****");


    // check similarity among hierarchical vectors
    for (Map.Entry<String, INDArray> entry : hvs.entrySet()) {

      INDArray vector = entry.getValue();
      Collection<String> strings = nearestVectors(hvs, vector);
      String label = entry.getKey();
//      System.out.println(label + ": " + strings);

      String[] stringsArray = comparison.get(label);
      stringsArray[1] = new LinkedList<>(strings).get(1);
      comparison.put(label, stringsArray);
    }

//    String pvCSV = asStrings(Hier2VecUtils.svdPCA(pvs, 2));
//    String hvCSV = asStrings(Hier2VecUtils.svdPCA(hvs, 2));

    // output comparison
    for (Map.Entry<String, String[]> c : comparison.entrySet()) {
      String[] values = c.getValue();
      if (!values[0].equals(values[1])) {
        System.out.println(c.getKey() + ": " + Arrays.toString(values));
      }
    }

    // measure similarity indexes
    double[] intraDocumentSimilarity = getIntraDocumentSimilarity(comparison);
    System.out.println("ids:" + Arrays.toString(intraDocumentSimilarity));
    double[] depthSimilarity = getDepthSimilarity(comparison);
    System.out.println("ds:" + Arrays.toString(depthSimilarity));

    // TODO : add section classification accuracy measures
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
      builder.append(entry.getKey()).append(", ").append(Arrays.toString(entry.getValue().data().asDouble()));
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

  @Ignore
  @Test
  public void testDataSplit() throws Exception {
    String regex = "\\n\\d(\\.\\d)*\\.?\\u0020[\\w|\\-|\\|\\–:]+(\\u0020[\\w|\\-|\\:|\\–]+){0,10}\\n";
    String prefix = "/path/to/h2v/";
    Pattern pattern = Pattern.compile(regex);
    Path path = Paths.get(getClass().getResource("/papers/raw/").getFile());
    File file = path.toFile();
    if (file.exists() && file.list() != null) {
      for (File doc : file.listFiles()) {
        String s = IOUtils.toString(new FileInputStream(doc));
        String docName = doc.getName();
        File fileDir = new File(prefix + docName);
        assert fileDir.mkdir();
        Matcher matcher = pattern.matcher(s);
        int start = 0;
        String sectionName = "abstract";
        while (matcher.find(start)) {
          String string = matcher.group(0);
          if (isValid(string)) {

            String content;

            if (start == 0) {
              // abstract
              content = s.substring(0, matcher.start());
            } else {
              content = s.substring(start, matcher.start());
            }

            File f = new File(prefix + docName + "/" + docName + "_" + sectionName);
            assert f.createNewFile() : "could not create file" + f.getAbsolutePath();
            FileOutputStream outputStream = new FileOutputStream(f);
            IOUtils.write(content, outputStream);

            start = matcher.end();
            sectionName = string.replaceAll("\n", "").trim();
          } else {
            start = matcher.end();
          }
        }
        // remaining
        File f = new File(prefix + docName + "/" + docName + "_" + sectionName);
        assert f.createNewFile();
        FileOutputStream outputStream = new FileOutputStream(f);

        IOUtils.write(s.substring(start), outputStream);
      }
    }
  }

  private boolean isValid(String string) {
    boolean result = false;
    char[] chars = string.toCharArray();
    for (char aChar : chars) {
      if (result = Character.isLetter(aChar)) {
        break;
      }
    }
    return result;
  }


}
