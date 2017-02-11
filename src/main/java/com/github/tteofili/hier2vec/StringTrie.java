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
import java.util.LinkedList;
import java.util.TreeSet;

/**
 * A simple Trie implementation based on nodes whose keys
 * are {@link Character}s and elements are Strings.
 * <p/>
 * This is a simplified Trie version which holds keys in nodes.
 */
class StringTrie {

  private static final char ROOT_CHAR = '_';
  private final CharTrieNode root;

  StringTrie() {
    this.root = new CharTrieNode(ROOT_CHAR);
  }

  public void add(String element) {
    char[] characters = element.toCharArray();
    add(characters, root);
  }

  void addAll(String... elements) {
    for (String element : elements) {
      char[] characters = element.toCharArray();
      add(characters, root);
    }
  }

  private void add(char[] characters, CharTrieNode root) {
    if (characters != null && characters.length > 0) {
      CharTrieNode affectedNode = getAffectedNode(root, characters[0], true);
      add(createSubArray(characters), affectedNode);
    } else {
      root.setWord(true);
    }
  }

  private char[] createSubArray(char[] characters) {
    char[] newar = new char[characters.length - 1];
    System.arraycopy(characters, 1, newar, 0, characters.length - 1);
    return newar;
  }

  private CharTrieNode getAffectedNode(CharTrieNode root, char c, Boolean createIfNotExisting) {
    boolean found = false;
    CharTrieNode resNode = null;
    for (CharTrieNode node : root.getChildren()) {
      if (node.getKey() == c) {
        found = true;
        resNode = node;
        break;
      }
    }
    if (!found && createIfNotExisting) {
      resNode = new CharTrieNode(c);
      root.getChildren().add(resNode);
    }
    return resNode;
  }

  public Collection<String> search(String prefix) {
    char[] chars = prefix.toCharArray();
    CharTrieNode foundSubTree = search(chars, root);
    return foundSubTree != null ? toStrings(prefix, foundSubTree) : new LinkedList<String>();
  }

  /**
   * transforming a node and all its descendants to a collection of elements
   *
   * @param prefix       the search prefix
   * @param foundSubTree the sub tree to transform represented by its root node
   * @return a collection of single elements in the sub tree
   */
  private Collection<String> toStrings(String prefix, CharTrieNode foundSubTree) {
    Collection<String> results = new LinkedList<String>();
    if (foundSubTree.isWord()) {
      results.add(prefix);
    }
    for (CharTrieNode child : foundSubTree.getChildren()) {
      String currentPrefix = prefix + child.getKey();
      results.addAll(toStrings(currentPrefix, child));
    }
    return results;
  }

  private CharTrieNode search(char[] chars, CharTrieNode currentRoot) {
    CharTrieNode subTree = currentRoot;
    // still some chars have to be consumed
    if (chars != null && chars.length > 0) {
      if (!currentRoot.getChildren().isEmpty()) {
        for (CharTrieNode node : currentRoot.getChildren()) {
          if (node.getKey().equals(chars[0])) {
            subTree = search(createSubArray(chars), node);
            break;
          }
        }
      } else {
        // this happens when I have still a prefix and no more nodes to inspect
        subTree = null;
      }

    }
    return subTree;
  }

  class CharTrieNode implements Comparable<CharTrieNode> {

    private final Character key;
    private final Collection<CharTrieNode> children;
    private Boolean isWord = false;

    CharTrieNode(Character key) {
      this(key, new TreeSet<>());
    }

    CharTrieNode(Character key, Collection<CharTrieNode> children) {
      this.key = key;
      this.children = children;
    }

    Character getKey() {
      return key;
    }

    Collection<CharTrieNode> getChildren() {
      return children;
    }

    Boolean isWord() {
      return isWord;
    }

    public int compareTo(CharTrieNode other) {
      return key.compareTo(other.getKey());
    }

    public void setWord(Boolean isWord) {
      this.isWord = isWord;
    }
  }

}