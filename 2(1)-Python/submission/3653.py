from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable


"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")
U = TypeVar("U")


class SegmentTree(Generic[T, U]):
    def __init__(self, arr:list[int], seg_tree:list[int]) -> None:
        """
        segment tree의 method를 사용할 수 있도록 한다.

        Args:
            - arr : 원본 배열이다.
            - seg_tree : 크기가 N의 4배인 list이다.

        Returns:
            - None : segmentree를 사용하게 되며 반환값이 없다.
        """
        self.arr: list[int] = arr
        self.tree: list[int] = seg_tree
        
    def init_set(self, node:int, start:int, end:int) -> int:
        """
        3653번 문제에 대해, 갖고 있는 DVD index를 표현한다.

        Args:
            - node : tree의 index다.
            - start : tree의 왼쪽 끝이다.
            - end : tree의 오른쪽 끝이다.

        Returns:
            - int : 재귀문을 위해 사용하며 합을 표현하도록 한다.
        """
        if start == end:
            self.tree[node] = self.arr[start]
        else:
            mid = (start + end) // 2
            left = self.init_set(node*2,start,mid)
            right = self.init_set(node*2+1,mid+1,end)
            self.tree[node] = left + right
        return self.tree[node]

    def rank_node(self, node:int, start:int, end:int, rank:int) -> int:
        """
        2243번 문제에 대해, 원하는 rank 값을 가져온다.

        Args:
            - node : tree의 index다.
            - start : tree의 왼쪽 끝이다.
            - end : tree의 오른쪽 끝이다.
            - rank : 가져오고자 하는 index이다.

        Returns:
            - int : rank 값을 return한다.
        """
        if start == end:
            return start
        
        mid = (start + end) // 2
        left = self.tree[node * 2]
        if left >= rank:
            return self.rank_node(node*2, start, mid, rank)
        return self.rank_node(node*2+1, mid+1, end, rank-left)
    
    def query(self, node:int, start:int, end:int, left:int, right:int) -> int:
        """
        segment tree를 이용해 부분합을 구한다.

        Args:
            - node : tree의 index다.
            - start : tree의 왼쪽 끝이다.
            - end : tree의 오른쪽 끝이다.
            - left : 부분합의 왼쪽 끝이다.
            - right : 부분합의 오른쪽 끝이다.

        Returns:
            - int : 부분합을 return한다.
        """
        if right < start or end < left:
            return 0
        if left <= start and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        tmp1 = self.query(node * 2, start, mid, left, right)
        tmp2 = self.query(node * 2 + 1, mid+1, end, left, right)
        return tmp1 + tmp2

    def relative_update(self, node:int, start:int, end:int, idx:int, diff:int) -> int:
        """
        기존에 있던 값을 기준으로 update를 진행한다.

        Args:
            - node : tree의 index다.
            - start : tree의 왼쪽 끝이다.
            - end : tree의 오른쪽 끝이다.
            - idx : 해당 index까지 update를 한다.
            - diff : 기존의 값과의 차이이다.

        Returns:
            - int : 재귀문을 위해 사용하며 update된 값을 윗쪽에 반영한다.
        """
        
        if end < idx or idx < start:
            pass
        elif start == end:
            self.tree[node] += diff
        else:
            mid = (start + end) // 2
            tmp1 = self.relative_update(node * 2, start, mid, idx, diff)
            tmp2 = self.relative_update(node * 2 + 1, mid + 1, end, idx, diff)        
            self.tree[node] = tmp1 + tmp2
        return self.tree[node]
    
    def update(self, node:int, start:int, end:int, idx:int, val:int) -> int:
        """
        기존에 있던 값이랑 관계없이 update를 진행한다.
        
        Args:
            - node : tree의 index다.
            - start : tree의 왼쪽 끝이다.
            - end : tree의 오른쪽 끝이다.
            - idx : 해당 index까지 update를 한다.
            - val : 새롭게 입력하고자 하는 값이다.

        Returns:
            - int : 재귀문을 위해 사용하며 update된 값을 윗쪽에 반영한다.
        """
        if end < idx or idx < start:
            pass
        elif start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            tmp1 = self.update(node * 2, start, mid, idx, val)
            tmp2 = self.update(node * 2 + 1, mid + 1, end, idx, val)        
            self.tree[node] = tmp1 + tmp2
        return self.tree[node]


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