from lib import Trie
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