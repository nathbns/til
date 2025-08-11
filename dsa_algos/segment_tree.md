A segment tree is a binary tree data structure used for efficient range queries and updates on arrays, such as finding the sum or minimum in a subarray. It is especially useful when you need to perform multiple queries and updates on an array.

**When to use a segment tree?**
- When you need to answer range queries (like sum, min, max) and also update elements efficiently.
- Examples: Range sum queries, range minimum/maximum queries, dynamic interval problems.

**Python Implementation Example on Leetcode 307. Range Sum Query - Mutable:**



```python
class NumArray:
    def __init__(self, nums: List[int], L: int = 0, R: int | None = None):
        if R is None: 
            R = len(nums) - 1
        self.L, self.R = L, R

        if L == R:    
            self.sum   = nums[L]
            self.left  = None
            self.right = None
        else:          
            M = (L + R) // 2
            self.left  = NumArray(nums, L, M)
            self.right = NumArray(nums, M + 1, R)
            self.sum   = self.left.sum + self.right.sum

    def update(self, index: int, val: int) -> None:
        if self.L == self.R:
            self.sum = val
            return
        M = (self.L + self.R) // 2
        if index > M:
            self.right.update(index, val)
        else:
            self.left.update(index, val)
        self.sum = self.left.sum + self.right.sum

    def sumRange(self, left: int, right: int) -> int:
        if self.L == left and self.R == right:
            return self.sum
        M = (self.L + self.R) // 2
        if left > M:
            return self.right.sumRange(left, right)
        elif right <= M:
            return self.left.sumRange(left, right)
        else: 
            return (self.left.sumRange(left, M) + self.right.sumRange(M + 1, right))

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# obj.update(index,val)
# param_2 = obj.sumRange(left,right)
```

**Complexité**

- Build: O(n)
- **query** (interval): O(log n)
- **update** (point): O(log n)
