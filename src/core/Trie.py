class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, tokens):
        cur_node = self.root

        for token in tokens:
            if token not in cur_node.children:
                cur_node.children[token] = TrieNode()
            cur_node = cur_node.children[token]

        cur_node.is_end = True
    
    def get_allowed_next_tokens(self, node):
        return list(node.children.keys())

    def get_node(self, prefix, cur_node):
        return cur_node.children[prefix]