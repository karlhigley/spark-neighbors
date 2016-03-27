package com.github.karlhigley.spark.neighbors

import org.scalatest.FunSuite

import com.github.karlhigley.spark.neighbors.lsh.HashTableEntry

class ANNSuite extends FunSuite with TestSparkContext {
  import ANNSuite._

  val numPoints = 1000
  val dimensions = 100
  val density = 0.5

  val localPoints = TestHelpers.generateRandomPoints(numPoints, dimensions, density)

  test("compute hamming nearest neighbors as a batch") {
    val points = sc.parallelize(localPoints.zipWithIndex.map(_.swap))

    val ann =
      new ANN(dimensions, "hamming")
        .setTables(1)
        .setSignatureLength(16)

    val model = ann.train(points)
    val neighbors = model.neighbors(10)

    val localHashTables = model.hashTables.collect()
    val localNeighbors = neighbors.collect()

    runAssertions(localHashTables, localNeighbors)
  }

  test("compute cosine nearest neighbors as a batch") {
    val points = sc.parallelize(localPoints.zipWithIndex.map(_.swap))

    val ann =
      new ANN(dimensions, "cosine")
        .setTables(1)
        .setSignatureLength(4)

    val model = ann.train(points)
    val neighbors = model.neighbors(10)

    val localHashTables = model.hashTables.collect()
    val localNeighbors = neighbors.collect()

    runAssertions(localHashTables, localNeighbors)
  }

  test("compute euclidean nearest neighbors as a batch") {
    val points = sc.parallelize(localPoints.zipWithIndex.map(_.swap))

    val ann =
      new ANN(dimensions, "euclidean")
        .setTables(1)
        .setSignatureLength(4)
        .setBucketWidth(2)

    val model = ann.train(points)
    val neighbors = model.neighbors(10)

    val localHashTables = model.hashTables.collect()
    val localNeighbors = neighbors.collect()

    runAssertions(localHashTables, localNeighbors)
  }

  test("compute manhattan nearest neighbors as a batch") {
    val points = sc.parallelize(localPoints.zipWithIndex.map(_.swap))

    val ann =
      new ANN(dimensions, "manhattan")
        .setTables(1)
        .setSignatureLength(4)
        .setBucketWidth(25)

    val model = ann.train(points)
    val neighbors = model.neighbors(10)

    val localHashTables = model.hashTables.collect()
    val localNeighbors = neighbors.collect()

    runAssertions(localHashTables, localNeighbors)
  }

  test("compute jaccard nearest neighbors as a batch") {
    val points = sc.parallelize(localPoints.zipWithIndex.map(_.swap))

    val ann =
      new ANN(dimensions, "jaccard")
        .setTables(1)
        .setSignatureLength(8)
        .setBands(4)
        .setPrimeModulus(739)

    val model = ann.train(points)
    val neighbors = model.neighbors(10)

    val localHashTables = model.hashTables.collect()
    val localNeighbors = neighbors.collect()

    runAssertions(localHashTables, localNeighbors)
  }

  test("with multiple hash tables neighbors don't contain duplicates") {
    val withDuplicates = localPoints ++ localPoints
    val points = sc.parallelize(withDuplicates.zipWithIndex.map(_.swap))

    val ann =
      new ANN(dimensions, "hamming")
        .setTables(4)
        .setSignatureLength(16)

    val model = ann.train(points)
    val neighbors = model.neighbors(10)

    val localNeighbors = neighbors.collect()

    localNeighbors.foreach {
      case (id1, distances) => {
        val neighborSet = distances.map {
          case (id2, distance) => id2
        }.toSet

        assert(neighborSet.size == distances.size)
      }
    }
  }

  test("find neighbors for a set of query points") {
    val points = sc.parallelize(localPoints.zipWithIndex.map(_.swap))

    val localTestPoints = TestHelpers.generateRandomPoints(100, dimensions, density)
    val testPoints = sc.parallelize(localTestPoints.zipWithIndex.map(_.swap))

    val ann =
      new ANN(dimensions, "hamming")
        .setTables(1)
        .setSignatureLength(16)

    val model = ann.train(points)
    val neighbors = model.neighbors(testPoints, 10)
    val neighborIds = neighbors.map(_._1)
  }
}

object ANNSuite {
  def runAssertions(
    hashTables: Array[_ <: HashTableEntry[_]],
    neighbors: Array[(Int, Array[(Int, Double)])]
  ) = {

    // At least some neighbors are found
    assert(neighbors.size > 0)

    neighbors.map {
      case (id1, distances) => {
        var maxDist = 0.0
        distances.map {
          case (id2, distance) => {
            // No neighbor pair contains the same ID twice
            assert(id1 != id2)

            // The neighbors are sorted in ascending order of distance
            assert(distance >= maxDist)
            maxDist = distance
          }
        }
      }
    }
  }
}
