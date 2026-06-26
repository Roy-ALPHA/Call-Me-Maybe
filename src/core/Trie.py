class TrieNode:
    """Represent a node in a token trie."""

    def __init__(self) -> None:
        """Create an empty trie node."""
        self.children: dict[int, TrieNode] = {}
        self.is_end: bool = False


class Trie:
    """Store token sequences and query valid next tokens."""

    def __init__(self) -> None:
        """Create an empty trie with a single root node."""
        self.root: TrieNode = TrieNode()

    def insert(self, tokens: list[int]) -> None:
        """Insert a token sequence into the trie.

        Args:
            tokens: The token ids that make up the sequence.
        """
        cur_node = self.root

        for token in tokens:
            if token not in cur_node.children:
                cur_node.children[token] = TrieNode()
            cur_node = cur_node.children[token]

        cur_node.is_end = True

    def get_allowed_next_tokens(self, node: TrieNode) -> list[int]:
        """Return the token ids allowed after the current node.

        Args:
            node: The trie node representing the current prefix.

        Returns:
            A list of valid next-token ids.
        """
        return list(node.children.keys())

    def get_node(self, prefix: int, cur_node: TrieNode) -> TrieNode:
        """Return the child node reached by following a token id.

        Args:
            prefix: The token id to follow.
            cur_node: The current trie node.

        Returns:
            The child node associated with the given token id.
        """
        return cur_node.children[prefix]
