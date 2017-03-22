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

import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Tests for {@link Par2HierUtils}
 */
public class Par2HierUtilsTest {

  private double[][] getDoubles() {
    Random random = new Random();
    // 10 word vectors with dimension 30
    double[][] data = new double[10][30];
    for (int i = 0; i < data.length; i++) {
      for (int j = 0; j < data[i].length; j++) {
        data[i][j] = random.nextDouble();
      }
    }
    return data;
  }

  @Test
  public void testTruncatedSVD() throws Exception {
    double[][] data = getDoubles();

    double[][] truncatedSVD = Par2HierUtils.getTruncatedSVD(data, 3);

    assertEquals(data.length, truncatedSVD.length);
    assertEquals(data[0].length, truncatedSVD[0].length);
  }

  @Test
  public void testSVDPCA() throws Exception {
    double[][] data = getDoubles();

    Map<String, INDArray> weightTable = new TreeMap<>();
    Random r = new Random();
    for (double[] d : data) {
      byte[] bytes = new byte[10];
      r.nextBytes(bytes);
      weightTable.put(new String(bytes), Nd4j.create(d));
    }
    Map<String, INDArray> svdPCA = Par2HierUtils.svdPCA(weightTable, 2);
    assertEquals(weightTable.size(), svdPCA.size());
    for (Map.Entry<String, INDArray> e : svdPCA.entrySet()) {
      assertEquals(2, e.getValue().columns());
      assertNotNull(weightTable.get(e.getKey()));
    }
  }

  @Test
  public void testTruncatedUT() throws Exception {
    double[][] data = getDoubles();

    double[][] truncatedUT = Par2HierUtils.getTruncatedUT(Nd4j.create(data), 3);

    assertEquals(data.length, truncatedUT.length);
    assertEquals(3, truncatedUT[0].length);
  }

  @Test
  public void testTruncatedVT() throws Exception {
    double[][] data = getDoubles();

    double[][] truncatedVT = Par2HierUtils.getTruncatedVT(Nd4j.create(data), 3);

    assertEquals(data[0].length, truncatedVT[0].length);
    assertEquals(3, truncatedVT.length);
  }

}