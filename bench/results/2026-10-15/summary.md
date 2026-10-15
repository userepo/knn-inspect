**SIFT (1,000,000 x 128, l2)**

| Measurement | knn-inspect | Faiss Python |
| --- | --- | --- |
| Open and parse, end to end (ms) | 2.2 | 352 (Python import alone: 106) |
| Peak memory after open (MiB) | 7 | 674 (Python import alone: 40) |
| Peak memory after touching every page (MiB) | 629 | 674 (already loaded) |
| Recall@10, 10,000 queries, efSearch 100 | 0.97802 | 0.97802 |
| HNSW search time, in process (ms) | 533 | 204 |
| Ground-truth time, in process (ms) | 51,801 | 10,058 |

**GloVe (1,183,514 x 100, inner_product)**

| Measurement | knn-inspect | Faiss Python |
| --- | --- | --- |
| Open and parse, end to end (ms) | 2.4 | 346 (Python import alone: 99) |
| Peak memory after open (MiB) | 7 | 664 (Python import alone: 40) |
| Peak memory after touching every page (MiB) | 617 | 664 (already loaded) |
| Recall@10, 10,000 queries, efSearch 100 | 0.78727 | 0.78727 |
| HNSW search time, in process (ms) | 618 | 207 |
| Ground-truth time, in process (ms) | 43,719 | 7,914 |

Machine and versions:

```
date_utc: 2026-10-15T00:39:48Z
cpu: Intel(R) Core(TM) i9-14900HX
logical_cpus: 32
memory: 15Gi
kernel: Linux 6.18.33.2-microsoft-standard-WSL2
os: Ubuntu 24.04.4 LTS
threads: 16
runs: 5
knn_inspect: knn-inspect 0.2.0, commit ca8fde0
python: Python 3.12.3
faiss: 1.15.1
hyperfine: hyperfine 2.0.0
```
