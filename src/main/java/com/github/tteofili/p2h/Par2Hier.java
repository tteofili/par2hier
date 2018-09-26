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

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;

import com.google.common.collect.Lists;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.apache.commons.lang.ArrayUtils;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.SequenceLearningAlgorithm;
import org.deeplearning4j.models.embeddings.learning.impl.sequence.DM;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Basic Par2Hier implementation for DL4j, as wrapper over SequenceVectors
 *
 */
public class Par2Hier extends ParagraphVectors {
  @Getter
  protected LabelsSource labelsSource;
  @Getter
  @Setter
  protected transient LabelAwareIterator labelAwareIterator;
  protected INDArray labelsMatrix;
  protected List<VocabWord> labelsList = new ArrayList<>();
  protected boolean normalizedLabels = false;
  private Par2HierUtils.Method smoothing;
  private Integer k;

  Par2Hier(ParagraphVectors paragraphVectors, Par2HierUtils.Method smoothing, int k) {
    this.smoothing = smoothing;
    this.k = k;

    this.labelsSource = paragraphVectors.getLabelsSource();
    this.labelAwareIterator = paragraphVectors.getLabelAwareIterator();
    this.vocab = paragraphVectors.getVocab();

    this.lookupTable = rebuildLookupTable(paragraphVectors.lookupTable(), this.vocab);

    this.tokenizerFactory = paragraphVectors.getTokenizerFactory();
    this.modelUtils = paragraphVectors.getModelUtils();

  }

  private static WeightLookupTable<VocabWord> rebuildLookupTable(WeightLookupTable parLT, VocabCache<VocabWord> vocabCache) {
    WeightLookupTable<VocabWord> lookupTable = new InMemoryLookupTable.Builder<VocabWord>()
        .vectorLength(parLT.layerSize()).cache(vocabCache).build();
    lookupTable.resetWeights();

    for (String w : vocabCache.words()) {
      INDArray vector = parLT.vector(w);
      lookupTable.putVector(w, vector);
    }
    return lookupTable;
  }


  @Override
  public String toString() {
    return "Par2Hier{" +
        "smoothing=" + smoothing +
        ", k=" + k +
        '}';
  }

  public void extractLabels() {
    Collection<VocabWord> vocabWordCollection = vocab.vocabWords();
    List<VocabWord> vocabWordList = new ArrayList<>();
    int[] indexArray;

    //INDArray pulledArray;
    //Check if word has label and build a list out of the collection
    for (VocabWord vWord : vocabWordCollection) {
      if (vWord.isLabel()) {
        vocabWordList.add(vWord);
      }
    }
    //Build array of indexes in the order of the vocablist
    indexArray = new int[vocabWordList.size()];
    int i = 0;
    for (VocabWord vWord : vocabWordList) {
      indexArray[i] = vWord.getIndex();
      i++;
    }
    //pull the label rows and create new matrix
    if (i > 0) {
      labelsMatrix = Nd4j.pullRows(lookupTable.getWeights(), 1, indexArray);
      labelsList = vocabWordList;
    }
  }

  /**
   * This method calculates inferred vector for given text
   *
   */
  public INDArray inferVector(String text, double learningRate, double minLearningRate, int iterations) {
    if (tokenizerFactory == null) {
      throw new IllegalStateException("TokenizerFactory should be defined, prior to predict() call");
    }

    List<String> tokens = tokenizerFactory.create(text).getTokens();
    List<VocabWord> document = new ArrayList<>();
    for (String token : tokens) {
      if (vocab.containsWord(token)) {
        document.add(vocab.wordFor(token));
      }
    }

    return inferVector(document, learningRate, minLearningRate, iterations);
  }

  /**
   * This method calculates inferred vector for given document
   * @param document the document to extract the vector for
   * @param iterations the no. of iterations used to infer the sequence
   * @param learningRate the start learning rate
   * @param minLearningRate the minimum learning rate
   * @return a vector representation of the document
   */
  public INDArray inferVector(LabelledDocument document, double learningRate, double minLearningRate, int iterations) {
    if (document.getReferencedContent() != null) {
      return inferVector(document.getReferencedContent(), learningRate, minLearningRate, iterations);
    } else {
      return inferVector(document.getContent(), learningRate, minLearningRate, iterations);
    }
  }

  /**
   * This method calculates inferred vector for given document represented as a list of words
   *
   * @param document the document to extract the vector for
   * @param iterations the no. of iterations used to infer the sequence
   * @param learningRate the start learning rate
   * @param minLearningRate the minimum learning rate
   * @return a vector representation of the document
   */
  public INDArray inferVector(List<VocabWord> document, double learningRate, double minLearningRate, int iterations) {
    SequenceLearningAlgorithm<VocabWord> learner = sequenceLearningAlgorithm;

    if (learner == null) {
      log.info("Creating new PV-DM learner...");
      learner = new DM<>();
      learner.configure(vocab, lookupTable, configuration);
    }

    Sequence<VocabWord> sequence = new Sequence<>();
    sequence.addElements(document);
    sequence.setSequenceLabel(new VocabWord(1.0, String.valueOf(new Random().nextInt())));

        /*
        for (int i = 0; i < iterations; i++) {
            sequenceLearningAlgorithm.learnSequence(sequence, new AtomicLong(0), learningRate);
        }*/

    initLearners();

    return learner.inferSequence(sequence, 119, learningRate, minLearningRate, iterations);
  }

  /**
   * This method calculates inferred vector for given text, with default parameters for learning rate and iterations
   *
   * @param text the text to extract the vector for
   * @return a vector representation for the given text
   */
  public INDArray inferVector(String text) {
    return inferVector(text, this.learningRate.get(), this.minLearningRate, this.numEpochs * this.numIterations);
  }

  /**
   * This method calculates inferred vector for given document, with default parameters for learning rate and iterations
   *
   * @param document a document to extract the vector for
   * @return a vector representation of the given document
   */
  public INDArray inferVector(LabelledDocument document) {
    return inferVector(document, this.learningRate.get(), this.minLearningRate, this.numEpochs * this.numIterations);
  }

  /**
   * This method calculates inferred vector for given list of words, with default parameters for learning rate and iterations
   *
   * @param document a document to extract the vector for
   * @return a vector representation of the given document
   */
  public INDArray inferVector(List<VocabWord> document) {
    return inferVector(document, this.learningRate.get(), this.minLearningRate, this.numEpochs * this.numIterations);
  }


  /**
   * This method returns top N labels nearest to specified document
   *
   * @param document a document
   * @param topN no. of nearest labels to find
   * @return a {@code Collection} of the nearest labels to the given document
   */
  public Collection<String> nearestLabels(LabelledDocument document, int topN) {
    if (document.getReferencedContent() != null) {
      return nearestLabels(document.getReferencedContent(), topN);
    } else {
      return nearestLabels(document.getContent(), topN);
    }
  }

  /**
   * This method returns top N labels nearest to specified text
   *
   * @param rawText a raw text
   * @param topN no. of nearest labels to find
   * @return a {@code Collection} of the nearest labels to the given text
   */
  public Collection<String> nearestLabels(String rawText, int topN) {
    List<String> tokens = tokenizerFactory.create(rawText).getTokens();
    List<VocabWord> document = new ArrayList<>();
    for (String token : tokens) {
      if (vocab.containsWord(token)) {
        document.add(vocab.wordFor(token));
      }
    }
    return nearestLabels(document, topN);
  }

  /**
   * This method returns top N labels nearest to specified set of vocab words
   *
   * @param document a document
   * @param topN no. of nearest labels to find
   * @return a {@code Collection} of the nearest labels to the given document
   */
  public Collection<String> nearestLabels(Collection<VocabWord> document, int topN) {
    INDArray vector = inferVector(new ArrayList<>(document));
    return nearestLabels(vector, topN);
  }

  /**
   * This method returns top N labels nearest to specified features vector
   *
   * @param labelVector a vector for the source label
   * @param topN no. of nearest labels to find
   * @return a {@code Collection} of the nearest labels to the given document
   */
  public Collection<String> nearestLabels(INDArray labelVector, int topN) {
    List<BasicModelUtils.WordSimilarity> result = new ArrayList<>();

    // if list still empty - return empty collection
    if (labelsMatrix == null || labelsList == null || labelsList.isEmpty()) {
      log.warn("Labels list is empty!");
      return new ArrayList<>();
    }

    if (!normalizedLabels) {
      synchronized (this) {
        if (!normalizedLabels) {
          labelsMatrix.diviColumnVector(labelsMatrix.norm1(1));
          normalizedLabels = true;
        }
      }
    }

    INDArray similarity = Transforms.unitVec(labelVector).mmul(labelsMatrix.transpose());
    List<Double> highToLowSimList = getTopN(similarity, topN + 20);

    for (Double aHighToLowSimList : highToLowSimList) {
      String word = labelsList.get(aHighToLowSimList.intValue()).getLabel();
      if (word != null && !word.equals("UNK") && !word.equals("STOP")) {
        INDArray otherVec = lookupTable.vector(word);
        double sim = Transforms.cosineSim(labelVector, otherVec);

        result.add(new BasicModelUtils.WordSimilarity(word, sim));
      }
    }

    result.sort(new BasicModelUtils.SimilarityComparator());

    return BasicModelUtils.getLabels(result, topN);
  }

  /**
   * Get top N elements
   *
   * @param vec the vec to extract the top elements from
   * @param N the number of elements to extract
   * @return the indices and the sorted top N elements
   */
  private List<Double> getTopN(INDArray vec, int N) {
    BasicModelUtils.ArrayComparator comparator = new BasicModelUtils.ArrayComparator();
    PriorityQueue<Double[]> queue = new PriorityQueue<>(vec.rows(), comparator);

    for (int j = 0; j < vec.length(); j++) {
      final Double[] pair = new Double[] {vec.getDouble(j), (double) j};
      if (queue.size() < N) {
        queue.add(pair);
      } else {
        Double[] head = queue.peek();
        if (comparator.compare(pair, head) > 0) {
          queue.poll();
          queue.add(pair);
        }
      }
    }

    List<Double> lowToHighSimLst = new ArrayList<>();

    while (!queue.isEmpty()) {
      double ind = queue.poll()[1];
      lowToHighSimLst.add(ind);
    }
    return Lists.reverse(lowToHighSimLst);
  }

  @Override
  public void fit() {
    if (lookupTable == null) {
      super.fit();
    }

    Map<String, INDArray> hvs = Par2HierUtils.getPar2Hier(labelAwareIterator, lookupTable, labelsSource.getLabels(), k, smoothing);
    for (Map.Entry<String, INDArray> entry : hvs.entrySet()) {
      lookupTable.putVector(entry.getKey(), entry.getValue());
    }

    extractLabels();
  }

  @Override
  public INDArray getWordVectorsMean(Collection<String> labels) {
    INDArray array = getWordVectors(labels);
    return array.mean(0);
  }

  @Override
  public INDArray getWordVectors(@NonNull Collection<String> labels) {
    int indexes[] = new int[labels.size()];
    int cnt = 0;
    for (String label : labels) {
      if (vocab.containsWord(label)) {
        indexes[cnt] = vocab.indexOf(label);
      } else
        indexes[cnt] = -1;
      cnt++;
    }

    while (ArrayUtils.contains(indexes, -1)) {
      indexes = ArrayUtils.removeElement(indexes, -1);
    }

    return Nd4j.pullRows(lookupTable.getWeights(), 1, indexes);
  }

  @Override
  public Collection<String> wordsNearest(INDArray words, int top) {
    return modelUtils.wordsNearest(words, top);
  }

}
