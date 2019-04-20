"""
208. Implement Trie (Prefix Tree)

Implement a trie with insert, search, and startsWith methods.

Example:
Trie trie = new Trie();

trie.insert("apple");
trie.search("apple");   // returns true
trie.search("app");     // returns false
trie.startsWith("app"); // returns true
trie.insert("app");   
trie.search("app");     // returns true

Note:
You may assume that all inputs are consist of lowercase letters a-z.
All inputs are guaranteed to be non-empty strings.
"""

class TrieNode:
    def __init__(self):
        self.children = [None] * 26
        self.is_end = False

class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        curr = self.root
        for c in word:
            i = ord(c) - ord("a")
            if not curr.children[i]:
                curr.children[i] = TrieNode()
            curr = curr.children[i]
        curr.is_end = True
        
    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        curr = self.root
        for c in word:
            i = ord(c) - ord("a")
            if not curr.children[i]:
                return False
            curr = curr.children[i]
        return curr.is_end        

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        curr = self.root
        for c in prefix:
            i = ord(c) - ord("a")
            if not curr.children[i]:
                return False
            curr = curr.children[i]
        return True         

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

"""
211. Add and Search Word - Data structure design

Design a data structure that supports the following two operations:

void addWord(word)
bool search(word)
search(word) can search a literal word or a regular expression string containing only letters a-z or ".". A
"." means it can represent any one letter.

Example:
addWord("bad")
addWord("dad")
addWord("mad")
search("pad") -> false
search("bad") -> true
search(".ad") -> true
search("b..") -> true

Note:
You may assume that all words are consist of lowercase letters a-z.
"""

class TrieNode(object):
    def __init__(self):
        self.children = [None for _ in range(26)]
        self.is_word = False

class WordDictionary(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: None
        """
        cursor = self.root
        
        for ch in word:
            idx = ord(ch) - ord("a")
            if cursor.children[idx] is None:
                cursor.children[idx] = TrieNode()
            cursor = cursor.children[idx]
        
        cursor.is_word = True

    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent
        any one letter.
        :type word: str
        :rtype: bool
        """
        return self.dfs(word, self.root)
    
    def dfs(self, word, cursor):
        if not word:
            return cursor.is_word
        
        if word[0] != ".":
            idx = ord(word[0]) - ord("a")
            if cursor.children[idx] is None:
                return False
            return self.dfs(word[1:], cursor.children[idx])
        
        for trie_node in cursor.children:
            if trie_node is not None:
                if self.dfs(word[1:], trie_node):
                    return True
        
        return False