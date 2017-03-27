package com.github.tteofili.p2h;

import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import com.github.tteofili.p2h.tools.LabelSeeker;
import com.github.tteofili.p2h.tools.MeansBuilder;
import org.apache.commons.collections4.trie.PatriciaTrie;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Tests for classification based on {@link Par2Hier}
 */
@Ignore("for this to be meaningful we need to encode more info about the class hierarchy in our p2h impl")
@RunWith(Parameterized.class)
public class Par2HierClassificationTest {

  private final Par2HierUtils.Method method;
  private int k;

  public Par2HierClassificationTest(Par2HierUtils.Method method, int k) {
    this.method = method;
    this.k = k;
  }

  @Parameterized.Parameters
  public static Collection<Object[]> data() {
    return Arrays.asList(new Object[][] {
        {Par2HierUtils.Method.CLUSTER, 4},
        {Par2HierUtils.Method.CLUSTER, 3},
        {Par2HierUtils.Method.CLUSTER, 2},
        {Par2HierUtils.Method.SUM, 4},
        {Par2HierUtils.Method.SUM, 3},
        {Par2HierUtils.Method.SUM, 2},
    });
  }

  @Test
  public void testClassification() throws Exception {
    ParagraphVectors paragraphVectors;
    LabelAwareIterator iterator;
    TokenizerFactory tokenizerFactory;

    ClassPathResource resource = new ClassPathResource("papers/labeled");

    // build a iterator for our dataset
    iterator = new FileLabelAwareIterator.Builder()
        .addSourceFolder(resource.getFile())
        .build();

    tokenizerFactory = new DefaultTokenizerFactory();
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

    // ParagraphVectors training configuration
    double learningRate = 0.05;
    int iterations = 1;
    int windowSize = 5;
    int layerSize = 30;
    int numEpochs = 1;
    int minWordFrequency = 1;
    double minLearningRate = 0.001;
    int batchSize = 1;

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

    Par2Hier par2Hier = new Par2Hier(paragraphVectors, method, k);

    // fit model
    par2Hier.fit();

    checkUnlabelledData(paragraphVectors, iterator, tokenizerFactory);

    checkUnlabelledData(par2Hier, iterator, tokenizerFactory);

  }

  private void checkUnlabelledData(Word2Vec paragraphVectors, LabelAwareIterator iterator, TokenizerFactory tokenizerFactory) throws FileNotFoundException {
    ClassPathResource unClassifiedResource = new ClassPathResource("papers/unlabeled");
    FileLabelAwareIterator unClassifiedIterator = new FileLabelAwareIterator.Builder()
        .addSourceFolder(unClassifiedResource.getFile())
        .build();

    MeansBuilder meansBuilder = new MeansBuilder(
        (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(),
        tokenizerFactory);
    LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
        (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

    System.out.println(paragraphVectors + " classification results");
    double cc = 0;
    double size = 0;
    while (unClassifiedIterator.hasNextDocument()) {
      LabelledDocument document = unClassifiedIterator.nextDocument();
      INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
      List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

      double max = -Integer.MAX_VALUE;
      String cat = null;
      for (Pair<String, Double> p : scores) {
        if (p.getSecond() > max) {
          max = p.getSecond();
          cat = p.getFirst();
        }
      }
      if (document.getLabels().contains(cat)) {
        cc++;
      }
      size++;

    }
    System.out.println("acc:" + (cc / size));

  }
}
