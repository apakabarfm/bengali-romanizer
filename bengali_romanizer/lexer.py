class Lexer:
    """Universal text lexer with convenient navigation methods"""

    def __init__(self, text: str, word_boundaries: str = ".,!?;:"):
        self.text = text
        self.pos = 0
        self.word_boundaries = word_boundaries

    def peek(self, offset: int = 0) -> str | None:
        """Look at character at current position + offset without consuming"""
        pos = self.pos + offset
        return self.text[pos] if 0 <= pos < len(self.text) else None

    def eat(self, count: int = 1) -> str:
        """Consume and return next count characters"""
        start = self.pos
        self.pos = min(self.pos + count, len(self.text))
        return self.text[start : self.pos]

    def eat_while(self, condition) -> str:
        """Consume characters while condition is true"""
        start = self.pos
        while self.pos < len(self.text) and condition(self.text[self.pos]):
            self.pos += 1
        return self.text[start : self.pos]

    def look_ahead(self, pattern: str) -> bool:
        """Check if text starting at current position matches pattern"""
        return self.text[self.pos :].startswith(pattern)

    def at_end(self) -> bool:
        """Check if at end of text"""
        return self.pos >= len(self.text)

    def remaining(self) -> str:
        """Get remaining text from current position"""
        return self.text[self.pos :]

    def peek_any(self, candidates: set | dict, offset: int = 0):
        """Check if character at position matches any from candidates set/dict keys"""
        char = self.peek(offset)
        return char if char in candidates else None

    def match_sequence(self, *patterns) -> bool:
        """Check if upcoming characters match sequence of patterns"""
        for i, pattern in enumerate(patterns):
            char = self.peek(i)
            if isinstance(pattern, str):
                if char != pattern:
                    return False
            elif isinstance(pattern, (set, dict)):
                if char not in pattern:
                    return False
        return True

    def extract_while(self, condition) -> str:
        """Extract and consume characters while condition is true"""
        chars = []
        while not self.at_end() and condition(self.peek()):
            chars.append(self.eat())
        return "".join(chars)

    def is_word_boundary(self) -> bool:
        """Check if at word boundary (whitespace or punctuation)"""
        char = self.peek()
        return char is None or char.isspace() or char in self.word_boundaries
