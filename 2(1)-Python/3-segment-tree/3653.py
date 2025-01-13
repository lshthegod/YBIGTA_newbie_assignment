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
    for _ in range(T):
        n, m = map(int, input().split())
        arr = [1] * (n + 1) + [0] * m
        dvd_idx = {n + 1 - i: i for i in range(1, n + 1)}
        seg_tree = [0] * ((n+m) * 4)
        tree: SegmentTree[list[int], list[int]]  = SegmentTree(arr, seg_tree)
        tree.init_set(1,1,n + m)
        new_idx = n

        for dvd_num in map(int, input().split()):
            idx = dvd_idx[dvd_num]
            print(tree.query(1, 1, n + m, idx + 1, new_idx), end=' ')
            tree.update(1, 1, n + m, idx, 0)
            new_idx += 1
            tree.update(1, 1, n + m, new_idx, 1)
            dvd_idx[dvd_num] = new_idx
        print()


if __name__ == "__main__":
    main()