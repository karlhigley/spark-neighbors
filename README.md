# spark-neighbors

Spark-based approximate nearest neighbors (ANN) using locality-sensitive hashing (LSH)

### Motivation

Spark itself doesn't yet support locality-sensitive hashing or nearest neighbor search. While there are LSH Spark packages available for a variety of distance measures, it's an open question how to support multiple distance measures and hashing schemes behind a unified interface.

### Features
- Batch computation of the nearest neighbors for each point in a dataset
- Hamming distance via bit sampling LSH
- Cosine distance via sign-random-projection LSH
- Euclidean distance via scalar-random-projection LSH
- Jaccard distance via Minhash LSH

### Usage

Hamming distance:

```scala
val ann =
  new ANN(dimensions, "hamming")
    .setTables(4)
    .setSignatureLength(64)

val model = ann.train(vectors)
val neighbors = model.neighbors(10)
```

Cosine distance:

```scala
val ann =
  new ANN(dimensions, "cosine")
    .setTables(4)
    .setSignatureLength(64)

val model = ann.train(vectors)
val neighbors = model.neighbors(10)
```

Euclidean distance:
```scala
val ann =
  new ANN(dimensions, "euclidean")
    .setTables(4)
    .setSignatureLength(32)
    .setBucketWidth(5)

val model = ann.train(vectors)
val neighbors = model.neighbors(10)
```

Jaccard distance:
```scala
val ann =
  new ANN(dimensions, "jaccard")
    .setTables(4)
    .setSignatureLength(128)
    .setBands(16)
    .setPrimeModulus(739)

val model = ann.train(vectors)
val neighbors = model.neighbors(10)
```

### Future Possibilities

Would be nice to add:

- Queries against pre-trained models. The existing code only supports computing the neighbors of all points as a batch, but it would be great to be able to quickly get neighbors for a single point or a small set of points (e.g. for streaming applications).

- Dense vector support. Currently, only sparse vectors are supported, since high dimensional data (e.g. TFIDF vectors) are likely to be sparse. It could be handy to also support dense vectors, but that will likely require optimized variants of some code paths for both dense and sparse vectors.

- Distributed random projections. Currently, projection matrices must fit in worker memory. There's an open JIRA ticket to add distributed random projection to Spark's MLlib ([SPARK-7334](https://issues.apache.org/jira/browse/SPARK-7334)); in the mean time, they could also be implemented here as an alternate subclass of RandomProjection. I've previously [implemented](https://github.com/karlhigley/lexrank-summarizer/blob/master/src/main/scala/io/github/karlhigley/lexrank/SignRandomProjectionLSH.scala) arbitrarily large projection matrices via the [pooling trick](http://personal.denison.edu/~lalla/papers/online-lsh.pdf), which could also be included as an option in the future.

## References

Sign random projection:
- Charikar, M. "[Similarity Estimation Techniques from Rounding Algorithms.](http://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/CharikarEstim.pdf)" STOC, 2002.

Scalar random projection:
- Datar, Immorlica, Indyk, and Mirrokni. "[Locality-sensitive hashing scheme based on p-stable distributions.](http://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/p253-datar.pdf)" SCG, 2004.

Minhash:
- Broder, A. "[On the resemblance and containment of documents.](http://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/broder97resemblance.pdf)" Compression and Complexity of Sequences: Proceedings, 1997.
- Broder, et al. "[Min-wise independent permutations.](http://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/BroderCFM-minwise.pdf)" STOC, 1998.

Survey papers:
- Loic Pauleve, Herve Jegou, Laurent Amsaleg. "[Locality sensitive hashing: a comparison of hash function types and querying mechanisms.](https://hal.inria.fr/file/index/docid/567191/filename/paper.pdf)" Pattern Recognition Letters, 2010.
- Jingdong Wang, Heng Tao Shen, Jingkuan Song, and Jianqiu Ji. "[Hashing for similarity search: A survey.](http://arxiv.org/pdf/1408.2927.pdf)" CoRR, abs/1408.2927, 2014.
