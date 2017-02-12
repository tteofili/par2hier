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
public class StringTrie {

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

  public String remove(String element) {
    char[] characters = element.toCharArray();
    StringBuilder stringBuilder = remove(characters, root);
    return stringBuilder != null ? stringBuilder.toString() : null;
  }

  private StringBuilder remove(char[] characters, CharTrieNode root) {
    StringBuilder elementBuilder = new StringBuilder();
    if (characters != null && characters.length > 1) {
      CharTrieNode affectedNode = getAffectedNode(root, characters[0], false);
      if (affectedNode != null) {
        elementBuilder.append(affectedNode.getKey());
        StringBuilder removeBuilder = remove(createSubArray(characters), affectedNode);
        if (removeBuilder != null) {
          elementBuilder.append(removeBuilder);
        } else {
          elementBuilder = null;
        }
      }
    } else if (characters != null && characters.length == 1) {
      CharTrieNode affectedNode = getAffectedNode(root, characters[0], false);
      if (affectedNode != null && affectedNode.isWord()) {
        root.getChildren().remove(affectedNode);
        elementBuilder.append(characters[0]);
        // TODO : back compress the trie till a sibling node is found to avoid having orphan nodes
      } else {
        elementBuilder = null;
      }
    } else {
      elementBuilder = null;
    }
    return elementBuilder;
  }

  /**
   * transforming the whole trie into a collection of elements
   *
   * @return a collection of single elements in the sub tree
   */
  public Collection<String> toStrings() {
    return toStrings("", root);
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

    public CharTrieNode(Character key) {
      this(key, new TreeSet<>());
    }

    public CharTrieNode(Character key, Collection<CharTrieNode> children) {
      this.key = key;
      this.children = children;
    }

    public Character getKey() {
      return key;
    }

    public Collection<CharTrieNode> getChildren() {
      return children;
    }

    public Boolean isWord() {
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