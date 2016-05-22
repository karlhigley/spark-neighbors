package com.github.karlhigley.spark.neighbors.lsh

import java.util.Random
import scala.collection.immutable.BitSet

import org.apache.spark.mllib.linalg.SparseVector

/**
 *
 * References:
 *  - Gionis, Indyk, Motwanit. "Similarity Search in High Dimensions via Hashing."
 *    Very Large Data Bases, 1999.
 *
 * @see [[https://en.wikipedia.org/wiki/Locality-sensitive_hashing#Bit_sampling_for_Hamming_distance
 *          Bit sampling for Hamming Distance (Wikipedia)]]
 */
private[neighbors] class BitSamplingFunction(
    private[this] val sampledBits: Array[Int]
) extends LSHFunction[BitSignature] with Serializable {

  /**
   * Compute the hash signature of the supplied vector
   */
  def signature(vector: SparseVector): BitSignature = {
    val sampled = vector.indices.intersect(sampledBits)
    new BitSignature(BitSet(sampled: _*))
  }

  /**
   * Build a hash table entry for the supplied vector
   */
  def hashTableEntry(id: Long, table: Int, v: SparseVector): BitHashTableEntry = {
    BitHashTableEntry(id, table, signature(v), v)
  }
}

private[neighbors] object BitSamplingFunction {
  /**
   * Build a random hash function, given the vector dimension
   * and signature length
   *
   * @param originalDim dimensionality of the vectors to be hashed
   * @param signatureLength the number of bits in each hash signature
   * @return randomly selected hash function from bit sampling family
   */
  def generate(
    originalDim: Int,
    signatureLength: Int,
    random: Random = new Random
  ): BitSamplingFunction = {
    val indices = Array.fill(signatureLength) {
      random.nextInt(originalDim)
    }

    new BitSamplingFunction(indices)
  }
}
