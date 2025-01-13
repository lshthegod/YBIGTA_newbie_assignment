from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable


"""
TODO:
- Trie.push 구현하기
- (필요할 경우) Trie에 추가 method 구현하기
"""


T = TypeVar("T")


@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: list[int] = field(default_factory=lambda: [])
    is_end: bool = False


class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable[T]) -> None:
        """
        trie에 seq을 저장한다.

        Args:
            - seq: T의 열 (list[int]일 수도 있고 str일 수도 있고 등등...)

        Returns:
            - None : trie가 생성되며 반환값이 없다.
        """
        pointer = 0
        for char in seq:
            found = False
            for child in self[pointer].children:
                if self[child].body == char:
                    pointer = child
                    found = True
                    break
            if not found:
                new_node = TrieNode(body=char)
                self.append(new_node)
                self[pointer].children.append(len(self) - 1)
                pointer = len(self) - 1
                
        self[pointer].is_end = True


import sys


"""
TODO:
- 일단 Trie부터 구현하기
- main 구현하기

힌트: 한 글자짜리 자료에도 그냥 str을 쓰기에는 메모리가 아깝다...
"""


def main() -> None:
    # 구현하세요!
    pass


if __name__ == "__main__":
    main()