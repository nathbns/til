A segment tree is a binary tree data structure used for efficient range queries and updates on arrays, such as finding the sum or minimum in a subarray. It is especially useful when you need to perform multiple queries and updates on an array.

**When to use a segment tree?**
- When you need to answer range queries (like sum, min, max) and also update elements efficiently.
- Examples: Range sum queries, range minimum/maximum queries, dynamic interval problems.

**Python Implementation Example:**



```python
from __future__ import annotations
from typing import Callable, Generic, List, TypeVar
import math

T = TypeVar("T")


class SegmentTree(Generic[T]):
    """Segment tree itératif générique.

    - Indices 0-based
    - `query(left, right)` utilise des bornes inclusives [left, right]
    - `operation` est associative (ex: sum, min, max)
    - `identity` est l'élément neutre pour `operation` (ex: 0 pour sum, +inf pour min)
    """

    def __init__(
        self,
        data: List[T],
        operation: Callable[[T, T], T] = lambda a, b: a + b,
        identity: T = 0,  # 0 pour la somme
    ) -> None:
        if not data:
            raise ValueError("'data' ne doit pas être vide")
        self.operation = operation
        self.identity = identity

        # Taille = puissance de 2 >= n
        n = len(data)
        size = 1
        while size < n:
            size <<= 1
        self._size = size

        # Arbre à 1-based indexing logique (racine à 1)
        self._tree: List[T] = [identity] * (2 * size)

        # Build: placer les feuilles puis remonter
        self._tree[size : size + n] = data
        for i in range(size - 1, 0, -1):
            self._tree[i] = self.operation(self._tree[2 * i], self._tree[2 * i + 1])

    def update(self, index: int, value: T) -> None:
        """Met à jour la valeur à `index` (point update)."""
        if index < 0:
            raise IndexError("index < 0")
        pos = index + self._size
        if pos >= len(self._tree):
            raise IndexError("index hors limites")
        self._tree[pos] = value
        pos //= 2
        while pos >= 1:
            self._tree[pos] = self.operation(self._tree[2 * pos], self._tree[2 * pos + 1])
            pos //= 2

    def query(self, left: int, right: int) -> T:
        """Retourne l'agrégat sur l'intervalle inclusif [left, right]."""
        if left < 0 or right < 0:
            raise IndexError("indices négatifs interdits")
        if right < left:
            raise ValueError("right doit être >= left")

        l = left + self._size
        r = right + self._size
        if l >= len(self._tree) or r >= len(self._tree):
            raise IndexError("intervalle hors limites")

        res_left: T = self.identity
        res_right: T = self.identity
        while l <= r:
            if (l & 1) == 1:  # l est un fils droit
                res_left = self.operation(res_left, self._tree[l])
                l += 1
            if (r & 1) == 0:  # r est un fils gauche
                res_right = self.operation(self._tree[r], res_right)
                r -= 1
            l //= 2
            r //= 2
        return self.operation(res_left, res_right)


if __name__ == "__main__":
    # Exemple d'utilisation (somme par défaut)
    st = SegmentTree([1, 3, 5, 7, 9, 11])
    print(st.query(1, 3))  # 3 + 5 + 7 = 15
    st.update(1, 10)
    print(st.query(1, 3))  # 10 + 5 + 7 = 22

    # Variante: minimum (identité = +inf)
    st_min = SegmentTree([4, 2, 6, 1, 5], operation=min, identity=math.inf)
    print(st_min.query(0, 4))  # 1
```

**Complexité**

- Build: O(n)
- **query** (intervalle): O(log n)
- **update** (point): O(log n)

Notes:

- Les indices sont 0-based et les intervalles sont inclusifs.
- Pour un segment tree max, utilisez `operation=max` et `identity=-math.inf`.

