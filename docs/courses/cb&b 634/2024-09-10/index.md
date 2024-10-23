---
layout: default
parent: CB&B 634 Computational Methods in Informatics
grand_parent: Courses at Yale
title: "2024-09-10 Data Structures and Algorithms"
nav_order: 22
discuss: true
math: katex
---

# Data Structures and Algorithms

## 1. Basic Data Concepts

### Data Types
- **Conceptual Types**:
  - Numbers
  - Text
  - Time series
  - Images

### Data Organization
- **Structures**:
  - Flat
  - Hierarchical tree
  - Web/graph
- **Access Order**:
  - Arbitrary
  - Sequential
  - Sequential by importance

### Key Considerations
- Memory usage
- Time efficiency
- Disk storage
- Network performance

## 2. Python Data Types

### `PyObject`: metadata before data
The `PyObject` structure is the core data type in Python.
```c
typedef struct _object {
    Py_ssize_t ob_refcnt;
    struct _typeobject *ob_type;
} PyObject;
```

#### `sys.getsizeof()`
We can use `sys.getsizeof()` to get the size of an object in bytes.
```python
>>> import sys
>>> sys.getsizeof(None)
16
>>> sys.getsizeof(0)
24
>>> sys.getsizeof("doxycycline")
60
```

### Primitive Types
1. **Integer (int)**
![alt text](image.png)
   - Variable size based on value
   - Uses IEEE 754 standard
   - Memory usage increases with number size:
     - For $n \in \mathbb{N}$, if $2^{n-1} \leq x < 2^n$, 
     then Python will use $24 + 4n$ bytes to store the integer.

   Example calculation:
   ```python
   import sys

   def calculate_int_size(x):
       n = x.bit_length()
       return 24 + 4 * ((n - 1) // 30 + 1)

   # Example usage
   x = 1073741824  # 2^30
   calculated_size = calculate_int_size(x)
   actual_size = sys.getsizeof(x)

   print(f"For integer {x}:")
   print(f"Calculated size: {calculated_size} bytes")
   print(f"Actual size: {actual_size} bytes")
   ```

2. **Float**
![alt text](image-1.png)
   - 24 bytes fixed size
   - Uses IEEE 754 double precision
   - Non-uniform distribution of values
   ![alt text](image-2.png)
   - Note: Floating point math isn't exact
     ```python
     >>> 1e-20 > 0
     True
     >>> 1 + 1e-20 > 1
     False
     >>> 1 + 1e-20 == 1
     True
     ```

3. **String (str)**
   - *Immutable* in Python
   - Uses Unicode standard (UTF-8)
   - Variable memory usage based on encoding:
     - Latin-1: 49 + 1 byte/character
     - Most other languages: 74 + 2 bytes/character
     - Emoji: 80 + 4 bytes/character

### Container Types
1. **List**
   - Random access collection
   - 56 bytes + 8 bytes/item
   ![alt text](image-3.png)
   - Mutable structure
     - Adjusts only memory pointers, not items
     - `sys.getsizeof()` reports list structure size, not item storage
   - Linear search time O(n)

2. **Arrays**
   - Contiguous memory
   - Fixed data type
   - Types:
     - `array.array`: 64 bytes + 8 bytes per value
     - `numpy.array`: 96 bytes + 8 bytes per value
     - `bitarray`: 96 bytes + 1 byte per 8 bits
   - Fast read/write access, slow add/remove

3. **Set**
   - Unordered collection
   - No duplicates
   - Constant-time lookups
   - Hash-based implementation

  > **Hash Function**:
  > A hash function maps data to a number within a known range, ensuring:
  > - Equivalent data produces the same hash
  > - Different data may share a hash
  > - Ideally, similar data yields distinct hashes

4. **Dictionary (dict)**
   - Key-value pairs
   - Hash-based lookup (on keys only)
   - Constant-time access
   - Good for hierarchical data

## 3. Advanced Data Structures

### Queue
- First-in, First-out (FIFO)
- Use cases:
  - Parallel parameter sweeps
  - Request handling
  - Breadth-first search

### Stack
- Last-in, First-out (LIFO)
- Applications:
  - Recursion
  - Depth-first search

### Trees
1. **Basic Tree Properties**
   - Root node
   - Parent-child relationships
   - No cycles

2. **Binary Trees**
   - 0-2 children per node
   - Applications:
     - Decision trees
     - Priority queues
     - Parsing XML/HTML

3. **Binary Search Tree**
   - Ordered structure
   - O(log n) search time when balanced
   - Used for sorting and efficient lookups

### Graphs
- Collection of vertices and edges
- Applications:
  - Protein interaction networks
  - Social networks
  - Brain connectivity
  - Language processing

## 4. Complexity Analysis

### Big O Notation
- Measures upper bound of growth rate
- Common complexities (from fastest to slowest):
  - O(1): Constant time
  - O($\log n$): Logarithmic
  - O($n$): Linear
  - O($n \log n$): Log-linear
  - O($n^2$): Quadratic
  - O($2^n$): Exponential

### Practical Considerations
- Memory usage vs. time tradeoffs
- Implementation overhead
- Data size impact
- System constraints

### Common Operation Complexities
- List/Dictionary index lookup: O(1)
- List search: O($n$)
- Set membership test: O(1)
- Sorting algorithms:
  - Bubble Sort: O($n^2$)
  - Merge Sort: O($n \log n$)
