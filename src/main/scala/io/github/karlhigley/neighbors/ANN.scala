package io.github.karlhigley.neighbors

import java.util.{ Random => JavaRandom }
import scala.util.Random

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.storage.StorageLevel

import io.github.karlhigley.neighbors.candidates.{ CandidateStrategy, SimpleCandidateStrategy }
import io.github.karlhigley.neighbors.linalg.{ CosineDistance, DistanceMeasure, EuclideanDistance }
import io.github.karlhigley.neighbors.lsh.LSHFunction
import io.github.karlhigley.neighbors.lsh.ScalarRandomProjectionFunction
import io.github.karlhigley.neighbors.lsh.SignRandomProjectionFunction

/**
 * Approximate Nearest Neighbors (ANN) using locality-sensitive hashing (LSH)
 *
 * @see [[https://en.wikipedia.org/wiki/Nearest_neighbor_search Nearest neighbor search
 *       (Wikipedia)]]
 */
class ANN private (
    private var measureName: String,
    private var origDimension: Int,
    private var numTables: Int,
    private var signatureLength: Int,
    private var bucketWidth: Double,
    private var randomSeed: Int
) {

  /**
   * Constructs an ANN instance with default parameters.
   */
  def this(dimensions: Int, measure: String) = {
    this(
      origDimension = dimensions,
      measureName = measure,
      numTables = 1,
      signatureLength = 16,
      bucketWidth = 0.0,
      randomSeed = Random.nextInt()
    )
  }

  /**
   * Number of hash tables to compute
   */
  def getTables(): Int = {
    numTables
  }

  /**
   * Number of hash tables to compute
   */
  def setTables(tables: Int): this.type = {
    numTables = tables
    this
  }

  /**
   * Number of elements in each signature (e.g. # signature bits for sign-random-projection)
   */
  def getSignatureLength(): Int = {
    signatureLength
  }

  /**
   * Number of elements in each signature (e.g. # signature bits for sign-random-projection)
   */
  def setSignatureLength(length: Int): this.type = {
    signatureLength = length
    this
  }

  /**
   * Bucket width (commonly named "W") used by scalar-random-projection hash functions.
   */
  def getBucketWidth(): Double = {
    bucketWidth
  }

  /**
   * Bucket width (commonly named "W") used by scalar-random-projection hash functions.
   */
  def setBucketWidth(width: Double): this.type = {
    require(
      measureName == "euclidean",
      "Bucket width only applies when distance measure is euclidean."
    )
    bucketWidth = width
    this
  }

  /**
   * Random seed used to generate hash functions
   */
  def getRandomSeed(): Int = {
    randomSeed
  }

  /**
   * Random seed used to generate hash functions
   */
  def setRandomSeed(seed: Int): this.type = {
    randomSeed = seed
    this
  }

  /**
   * Build an ANN model using the given dataset.
   *
   * @param vectors    RDD of vectors paired with IDs.
   *                   IDs must be unique and >= 0.
   * @return ANNModel containing computed hash tables
   */
  def train(
    vectors: RDD[(Int, SparseVector)],
    persistenceLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK
  ): ANNModel = {
    var hashFunctions: Array[LSHFunction[_]] = Array()
    var candidateStrategy: CandidateStrategy = new SimpleCandidateStrategy(persistenceLevel)
    var distanceMeasure: DistanceMeasure = CosineDistance
    val random = new JavaRandom(randomSeed)

    measureName.toLowerCase match {
      case "cosine" => {
        hashFunctions = (1 to numTables).map(i =>
          SignRandomProjectionFunction.generate(origDimension, signatureLength, random)).toArray
      }
      case "euclidean" => {
        require(bucketWidth > 0.0, "Bucket width must be greater than zero.")

        distanceMeasure = EuclideanDistance
        hashFunctions = (1 to numTables).map(i =>
          ScalarRandomProjectionFunction.generate(
            origDimension,
            signatureLength,
            bucketWidth,
            random
          )).toArray
      }
      case other: Any =>
        throw new IllegalArgumentException(
          s"Only cosine and euclidean distances are supported but got $other."
        )
    }

    ANNModel.train(
      vectors,
      hashFunctions,
      candidateStrategy,
      distanceMeasure,
      persistenceLevel
    )
  }
}
