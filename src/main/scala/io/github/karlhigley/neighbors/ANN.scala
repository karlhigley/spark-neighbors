package io.github.karlhigley.neighbors

import java.util.{ Random => JavaRandom }
import scala.util.Random

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.storage.StorageLevel

import io.github.karlhigley.neighbors.candidates.{ BandingCandidateStrategy, CandidateStrategy, SimpleCandidateStrategy }
import io.github.karlhigley.neighbors.linalg.{ CosineDistance, DistanceMeasure, EuclideanDistance, HammingDistance, JaccardDistance }
import io.github.karlhigley.neighbors.lsh.LSHFunction
import io.github.karlhigley.neighbors.lsh.BitSamplingFunction
import io.github.karlhigley.neighbors.lsh.MinhashFunction
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
    private var primeModulus: Int,
    private var numBands: Int,
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
      primeModulus = 0,
      numBands = 0,
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
   * Common prime modulus used by minhash functions.
   */
  def getPrimeModulus(): Int = {
    primeModulus
  }

  /**
   * Common prime modulus used by minhash functions.
   *
   * Should be larger than the number of dimensions.
   */
  def setPrimeModulus(prime: Int): this.type = {
    require(
      measureName == "jaccard",
      "Prime modulus only applies when distance measure is jaccard."
    )
    primeModulus = prime
    this
  }

  /**
   * Number of bands to use for minhash candidate pair generation
   */
  def getBands(): Int = {
    numBands
  }

  /**
   * Number of bands to use for minhash candidate pair generation
   */
  def setBands(bands: Int): this.type = {
    require(
      measureName == "jaccard",
      "Number of bands only applies when distance measure is jaccard."
    )
    numBands = bands
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
    var distanceMeasure: DistanceMeasure = HammingDistance
    val random = new JavaRandom(randomSeed)

    measureName.toLowerCase match {
      case "hamming" => {
        hashFunctions = (1 to numTables).map(i =>
          BitSamplingFunction.generate(origDimension, signatureLength, random)).toArray
      }
      case "cosine" => {
        distanceMeasure = CosineDistance
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
      case "jaccard" => {
        require(primeModulus > 0, "Prime modulus must be greater than zero.")
        require(numBands > 0, "Number of bands must be greater than zero.")
        require(
          signatureLength % numBands == 0,
          "Number of bands must evenly divide signature length."
        )

        distanceMeasure = JaccardDistance
        hashFunctions = (1 to numTables).map(i =>
          MinhashFunction.generate(origDimension, signatureLength, primeModulus, random)).toArray
        candidateStrategy = new BandingCandidateStrategy(10, persistenceLevel)
      }
      case other: Any =>
        throw new IllegalArgumentException(
          s"Only cosine, euclidean, and jaccard distances are supported but got $other."
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
