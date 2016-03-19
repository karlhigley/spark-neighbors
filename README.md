# spark-neighbors [![GitHub version](https://badge.fury.io/gh/karlhigley%2Fspark-neighbors.svg)](https://badge.fury.io/gh/karlhigley%2Fspark-neighbors) [![Build Status](https://travis-ci.org/karlhigley/spark-neighbors.svg?branch=master)](https://travis-ci.org/karlhigley/spark-neighbors)

Spark-based approximate nearest neighbors (ANN) using locality-sensitive hashing (LSH)

### Motivation

Spark's MLlib doesn't yet support locality-sensitive hashing or nearest neighbor search. While there are LSH Spark packages available for a variety of distance measures, it has been open question how to support multiple distance measures and hashing schemes behind a unified interface. This library presents one possible way to do that.

### Features

- Batch computation of the K nearest neighbors for each point in a dataset
- Support for four distance measures:
    - Hamming distance via bit sampling LSH
    - Cosine distance via sign-random-projection LSH
    - Euclidean distance via scalar-random-projection LSH
    - Jaccard distance via Minhash LSH

### Linking

You can link against this library (for Spark 1.6+) in your program at the following coordinates:

Using SBT:

```
libraryDependencies += "com.github.karlhigley" %% "spark-neighbors" % "0.1.0"
```

Using Maven:

```xml
<dependency>
    <groupId>com.github.karlhigley</groupId>
    <artifactId>spark-neighbors_2.10</artifactId>
    <version>0.1.0</version>
</dependency>
```

This library can also be added to Spark jobs launched through spark-shell or spark-submit by using the --packages command line option. For example, to include it when starting the spark shell:

```
$ bin/spark-shell --packages com.github.karlhigley:spark-neighbors_2.10:0.1.0
```

Unlike using --jars, using --packages ensures that this library and its dependencies will be added to the classpath. The --packages argument can also be used with bin/spark-submit.

### Usage

ANN models are created using the builder pattern:

```scala
import com.github.karlhigley.spark.neighbors.ANN

val annModel =
  new ANN(dimensions = 1000, measure = "hamming")
    .setTables(4)
    .setSignatureLength(64)
    .train(points)
```

An ANN model can compute a variable number of approximate nearest neighbors:

```scala
val neighbors = model.neighbors(10)
```

#### Distance Measures and Parameters

The supported distance measures are "hamming", "cosine", "euclidean", "manhattan", and "jaccard". All distance measures allow the number of hash tables and the length of the computed hash signatures to be configured as above. The hashing schemes for Euclidean, Manhattan, and Jaccard distances have some additional configurable parameters:

##### Euclidean and Manhattan Distances

This hash function depends on a bucket or quantization width. Higher widths lead to signatures that are more similar:

```scala
val annModel =
  new ANN(dimensions = 1000, measure = "euclidean")
    .setTables(4)
    .setSignatureLength(64)
    .setBucketWidth(5)
    .train(points)
```

##### Jaccard Distance

Minhashing requires two additional parameters: a prime larger than the number of input dimensions and the number of minhash bands. The prime is used in the permutation functions that generate minhash signatures, and the number of bands is used in the process of generating candidate pairs from the signatures.

```scala
val annModel =
  new ANN(dimensions = 1000, measure = "jaccard")
    .setTables(4)
    .setSignatureLength(128)
    .setPrimeModulus(739)
    .setBands(16)
    .train(points)
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

Performance evaluation and tuning:
- Wei Dong, Zhe Wang, William Josephson, Moses Charikar, and Kai Li. "[Modeling LSH for performance tuning](http://www.cs.princeton.edu/cass/papers/cikm08.pdf)" CIKM, 2008.
