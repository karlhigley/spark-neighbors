package io.github.karlhigley.neighbors.lsh

import java.util.Random

import org.apache.spark.mllib.linalg.{ SparseVector, Vectors }

import io.github.karlhigley.neighbors.linalg.RandomProjection

/**
 *
 * References:
 *  - Datar, Immorlica, Indyk, and Mirrokni. "Locality-sensitive hashing scheme
 *      based on p-stable distributions." SCG, 2004.
 *
 * @see [[https://en.wikipedia.org/wiki/Locality-sensitive_hashing#Stable_distributions
 *          Stable distributions (Wikipedia)]]
 */
private[neighbors] class ScalarRandomProjectionFunction(
    private[this] val projection: RandomProjection,
    private[this] val b: Array[Double],
    val bucketWidth: Double
) extends LSHFunction[IntSignature] with Serializable {

  /**
   * Compute the hash signature of the supplied vector
   */
  def signature(vector: SparseVector): IntSignature = {
    val ax = projection.project(vector)
    val sig = new Array[Int](ax.size)

    ax.foreachActive((i, v) => {
      sig(i) = math.floor((ax(i) + b(i)) / bucketWidth).toInt
    })

    new IntSignature(sig)
  }

  /**
   * Build a hash table entry for the supplied vector
   */
  def hashTableEntry(id: Int, table: Int, v: SparseVector): IntHashTableEntry = {
    IntHashTableEntry(id, table, signature(v))
  }
}

private[neighbors] object ScalarRandomProjectionFunction {
  /**
   * Build a random hash function, given the vector dimension,
   * signature length, and bucket width.
   *
   * @param originalDim dimensionality of the vectors to be hashed
   * @param signatureLength the number of integers in each hash signature
   * @param bucketWidth the width to use when truncating hash values to integers
   * @return randomly selected hash function from scalar RP family
   */
  def generate(
    originalDim: Int,
    signatureLength: Int,
    bucketWidth: Double,
    random: Random = new Random
  ): ScalarRandomProjectionFunction = {

    val projection = RandomProjection.generate(originalDim, signatureLength, random)
    val offsets = generateOffsets(signatureLength, bucketWidth, random)
    new ScalarRandomProjectionFunction(projection, offsets, bucketWidth)
  }

  /**
   * Generate a set of offsets (individually referred to as "b")
   * to use in component hash functions
   */
  private def generateOffsets(quantity: Int, width: Double, random: Random): Array[Double] = {
    val offsets = new Array[Double](quantity)
    var i = 0
    while (i < quantity) {
      offsets(i) = random.nextDouble() * width
      i += 1
    }
    offsets
  }
}
