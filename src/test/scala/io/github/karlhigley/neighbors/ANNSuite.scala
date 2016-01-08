package io.github.karlhigley.neighbors

import scala.util.Random

import org.apache.spark.mllib.linalg.SparseVector
import org.scalatest.FunSuite

import io.github.karlhigley.neighbors.lsh.HashTableEntry

class ANNSuite extends FunSuite with TestSparkContext {
  import ANNSuite._

  val numVectors = 1000
  val dimensions = 100
  val density = 0.5

  val localVectors = generateRandomVectors(numVectors, dimensions, density)

  test("compute cosine nearest neighbors as a batch") {
    val vectors = sc.parallelize(localVectors.zipWithIndex.map(_.swap))

    val ann =
      new ANN(dimensions, "cosine")
        .setTables(1)
        .setSignatureLength(4)

    val model = ann.train(vectors)
    val neighbors = model.neighbors(10)

    val localHashTables = model.hashTables.collect()
    val localNeighbors = neighbors.collect()

    runAssertions(localHashTables, localNeighbors)
  }

  test("compute euclidean nearest neighbors as a batch") {
    val vectors = sc.parallelize(localVectors.zipWithIndex.map(_.swap))

    val ann =
      new ANN(dimensions, "euclidean")
        .setTables(1)
        .setSignatureLength(4)
        .setBucketWidth(2)

    val model = ann.train(vectors)
    val neighbors = model.neighbors(10)

    val localHashTables = model.hashTables.collect()
    val localNeighbors = neighbors.collect()

    runAssertions(localHashTables, localNeighbors)
  }

  test("compute jaccard nearest neighbors as a batch") {
    val vectors = sc.parallelize(localVectors.zipWithIndex.map(_.swap))

    val ann =
      new ANN(dimensions, "jaccard")
        .setTables(1)
        .setSignatureLength(8)
        .setBands(4)
        .setPrimeModulus(739)

    val model = ann.train(vectors)
    val neighbors = model.neighbors(10)

    val localHashTables = model.hashTables.collect()
    val localNeighbors = neighbors.collect()

    runAssertions(localHashTables, localNeighbors)
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

  def generateRandomVectors(quantity: Int, dimensions: Int, density: Double) = {
    val numElements = math.floor(dimensions * density).toInt
    val vectors = new Array[SparseVector](quantity)
    var i = 0
    while (i < quantity) {
      val indices = generateIndices(numElements, dimensions)
      val values = generateValues(numElements)
      vectors(i) = new SparseVector(dimensions, indices, values)
      i += 1
    }
    vectors
  }

  def generateIndices(quantity: Int, dimensions: Int) = {
    val indices = new Array[Int](quantity)
    var i = 0
    while (i < quantity) {
      val possible = Random.nextInt(dimensions)
      if (!indices.contains(possible)) {
        indices(i) = possible
        i += 1
      }
    }
    indices
  }

  def generateValues(quantity: Int) = {
    val values = new Array[Double](quantity)
    var i = 0
    while (i < quantity) {
      values(i) = Random.nextGaussian()
      i += 1
    }
    values
  }
}