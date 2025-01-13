from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    input = sys.stdin.readline
    T = int(input())
    N = 1_000_000 + 1
    arr = [0] * N
    seg_tree = [0] * (N * 4)
    tree: SegmentTree[list[int], list[int]] = SegmentTree(arr, seg_tree)
    for _ in range(T):
        li = list(map(int, input().split()))
        if li[0] == 1:
            b = li[1]
            result = tree.rank_node(1,1,N,b)
            print(result)
            tree.relative_update(1,1,N,result,-1)
        elif li[0] == 2:
            b, c = li[1], li[2]
            tree.relative_update(1,1,N,b,c)
    pass


if __name__ == "__main__":
    main()