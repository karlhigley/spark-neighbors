package com.github.karlhigley.spark.neighbors.lsh

import java.util.Random
import scala.collection.immutable.BitSet
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.mllib.linalg.{ SparseVector, Vectors }

import com.github.karlhigley.spark.neighbors.linalg.RandomProjection

/**
 *
 * References:
 *  - Charikar, M. "Similarity Estimation Techniques from Rounding Algorithms." STOC, 2002.
 *
 * @see [[https://en.wikipedia.org/wiki/Locality-sensitive_hashing#Random_projection
 *          Random projection (Wikipedia)]]
 */
private[neighbors] class SignRandomProjectionFunction(
    private[this] val projection: RandomProjection,
    signatureLength: Int
) extends LSHFunction[BitSignature] with Serializable {

  /**
   * Compute the hash signature of the supplied vector
   */
  def signature(vector: SparseVector): BitSignature = {
    val projected = projection.project(vector)
    val bits = new ArrayBuffer[Int]

    projected.foreachActive((i, v) => {
      if (v > 0.0) { bits += i }
    })
    new BitSignature(BitSet(bits.toArray: _*))
  }

  /**
   * Build a hash table entry for the supplied vector
   */
  def hashTableEntry(id: Long, table: Int, v: SparseVector): BitHashTableEntry = {
    BitHashTableEntry(id, table, signature(v), v)
  }
}

private[neighbors] object SignRandomProjectionFunction {
  /**
   * Build a random hash function, given the vector dimension
   * and signature length
   *
   * @param originalDim dimensionality of the vectors to be hashed
   * @param signatureLength the number of bits in each hash signature
   * @return randomly selected hash function from sign RP family
   */
  def generate(
    originalDim: Int,
    signatureLength: Int,
    random: Random = new Random
  ): SignRandomProjectionFunction = {

    val projection = RandomProjection.generateGaussian(originalDim, signatureLength, random)
    new SignRandomProjectionFunction(projection, signatureLength)
  }
}
