# Parallel Shortest Path Solver for Large-Scale Road Networks

> **COMP-460 Parallel and Distributed Computing** — Course Project  
> Pak-Austria Fachhochschule: Institute of Applied Sciences and Technology, Haripur  
> Submitted: December 2025

## Overview

This project implements and benchmarks two parallel shortest-path (SSSP) algorithms on large road networks with up to **1 million nodes** and **8 million edges**, using **C++17** and **OpenMP** for shared-memory parallelism.

| Algorithm | Approach | Best For |
|---|---|---|
| **Parallel Dijkstra** | Serial extract-min + parallel neighbor relaxation via atomics | Small graphs, correctness reference |
| **Delta-Stepping** | Bucket-based SSSP with parallel light/heavy edge relaxation | Large-scale road networks ✅ |

---

## Algorithms

### 1. Parallel Dijkstra (`dijkstra_pq_parallel.cpp`)

- Uses a **shared priority queue** protected by a `std::mutex`
- Distance array uses `std::atomic<long long>` for lock-free reads
- Neighbor relaxation is parallelized with `#pragma omp parallel for`
- CAS (`compare_exchange_strong`) loop prevents race conditions on distance updates
- Also includes a **sequential baseline** with path reconstruction for correctness comparison

**Limitation:** Priority queue contention is a bottleneck — performance degrades beyond 2 threads on large graphs.

---

### 2. Delta-Stepping (`delta_stepping.cpp`)

- Partitions edges into **light** (weight ≤ δ) and **heavy** (weight > δ)
- Maintains distance **buckets** — each bucket covers a range of width δ
- Processes buckets in ascending order; repeatedly relaxes light edges within the current bucket, then relaxes heavy edges once the bucket is settled
- Uses atomic CAS for distance updates; only `#pragma omp critical` when inserting into buckets
- Scales well — tested up to **8 threads** on 1M-node graphs

**Optimal parameters:** `delta = 100–200`, `threads = 8`

---

## Graph Format

Both programs accept a plain-text graph file:

```
n m
u1 v1 w1
u2 v2 w2
...
```

- First line: number of nodes `n` and edges `m`
- Each following line: directed edge from `u` to `v` with integer weight `w`
- Nodes are **0-indexed**

Two sample graphs are included:

| File | Description |
|---|---|
| `small_graph.txt` | Tiny graph for quick testing and debugging |
| `road_network_1M.txt` | Synthetic road network — ~1M nodes, ~8M edges |

---

## Build

Requires a C++17-compatible compiler with OpenMP support (GCC 7+ recommended).

```bash
# Parallel Dijkstra
g++ dijkstra_pq_parallel.cpp -fopenmp -O2 -std=c++17 -o dijkstra_pq_parallel

# Delta-Stepping
g++ delta_stepping.cpp -fopenmp -O2 -std=c++17 -o delta_stepping
```

---

## Usage

### Parallel Dijkstra

```bash
./dijkstra_pq_parallel <graph_file> <num_threads>
```

```bash
# Example
./dijkstra_pq_parallel small_graph.txt 4
# Enter source node and destination node when prompted
```

Prints both **sequential** and **parallel** times, the shortest distance, and the reconstructed path (from the sequential parent array).

---

### Delta-Stepping

```bash
./delta_stepping <graph_file> <delta> <num_threads>
```

```bash
# Example on large graph
./delta_stepping road_network_1M.txt 100 8
# Enter source node and destination node when prompted
```

Prints the shortest distance and wall-clock time.

---

## Performance Results

### Observed Scaling (1M-node road network)

| Algorithm | Small Graph | Large Graph | Scaling |
|---|---|---|---|
| Parallel Dijkstra | Slower than seq | Much slower than seq | ❌ Poor |
| Delta-Stepping | Fast | Faster than seq | ✅ Good |

### Bottleneck Analysis

**Parallel Dijkstra:**
- Shared priority queue causes heavy mutex contention
- Atomic CAS overhead increases with thread count
- Poor cache locality from random memory access patterns
- Performance worsens beyond 2 threads

**Delta-Stepping:**
- Bucket management has low synchronization cost
- Minor load imbalance when edge-weight distribution is skewed
- Scales well up to 8 threads; practical for large road networks

---

## Key Takeaways

- Naively parallelizing Dijkstra's algorithm is **ineffective** at scale — contention dominates
- Delta-Stepping's bucket partitioning **decouples** light and heavy edge processing, enabling real thread-level speedup
- For large-scale road network SSSP, **Delta-Stepping is the preferred approach**

---

## Project Structure

```
.
├── dijkstra_pq_parallel.cpp   # Parallel Dijkstra with sequential baseline
├── delta_stepping.cpp         # Delta-Stepping parallel SSSP
├── small_graph.txt            # Small test graph
├── road_network_1M.txt        # Large road network (1M nodes)
└── README.md
```

---

## Authors

- **Azeem Mohamed Husain**
- Iqbal Mohamed Askee

**Instructor:** Mr. Shoaib Khan  
**Course:** COMP-460 Parallel and Distributed Computing  
**Institution:** Pak-Austria Fachhochschule, Haripur, Pakistan
