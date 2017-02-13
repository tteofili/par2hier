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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.io.IOUtils;
import org.junit.Ignore;
import org.junit.Test;

/**
 * Tests for data splitting tasks to prepare data to be processed
 */
public class DataSplitTest {

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
