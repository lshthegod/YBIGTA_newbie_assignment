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
- count 구현하기
- main 구현하기
"""


def count(trie: Trie, query_seq: str) -> int:
    """
    string을 찾기 위한 입력 수를 return한다.
    
    Args:
        - trie : 이름 그대로 trie
        - query_seq : 단어 ("hello", "goodbye", "structures" 등)

    returns:
        - int : query_seq의 단어를 입력하기 위해 버튼을 눌러야 하는 횟수
    """
    pointer = 0
    cnt = 0

    for element in query_seq:
        if len(trie[pointer].children) > 1 or trie[pointer].is_end:
            cnt += 1

        for child_index in trie[pointer].children:
            if trie[child_index].body == element:
                pointer = child_index
                break

    return cnt + int(len(trie[0].children) == 1)


def main() -> None:
    while 1:
        try: n = int(sys.stdin.readline())
        except: break
        trie: Trie = Trie()
        str_list = []

        for _ in range(n):
            str_tmp = sys.stdin.readline().rstrip()
            trie.push(str_tmp)
            str_list.append(str_tmp)

        sum = 0
        for word in str_list:
            sum += count(trie,word)
        print("%.2f" % (sum/n))


if __name__ == "__main__":
    main()