package com.github.karlhigley.spark.neighbors.lsh

import java.util.Random

import org.apache.spark.mllib.linalg.SparseVector

/**
 *
 * References:
 *  - Broder, A. "On the resemblance and containment of documents."
 *    Compression and Complexity of Sequences: Proceedings, 1997.
 *
 * @see [[https://en.wikipedia.org/wiki/MinHash MinHash (Wikipedia)]]
 */
private[neighbors] class MinhashFunction(
    private[this] val permutations: Array[PermutationFunction]
) extends LSHFunction[IntSignature] with Serializable {

  /**
   * Compute minhash signature for a vector.
   *
   * Since Spark doesn't support binary vectors, this uses
   * SparseVectors and treats any active element of the vector
   * as a member of the set. Note that "active" includes explicit
   * zeros, which should not (but still might) be present in SparseVectors.
   */
  def signature(vector: SparseVector): IntSignature = {
    val sig = permutations.map(p => {
      vector.indices.map(p.apply).min
    })

    new IntSignature(sig)
  }

  /**
   * Build a hash table entry for the supplied vector
   */
  def hashTableEntry(id: Long, table: Int, v: SparseVector): IntHashTableEntry = {
    IntHashTableEntry(id, table, signature(v), v)
  }
}

private[neighbors] object MinhashFunction {

  /**
   * Randomly generate a new minhash function
   */
  def generate(
    dimensions: Int,
    signatureLength: Int,
    prime: Int,
    random: Random = new Random
  ): MinhashFunction = {

    val perms = new Array[PermutationFunction](signatureLength)
    var i = 0
    while (i < signatureLength) {
      perms(i) = PermutationFunction.random(dimensions, prime, random)
      i += 1
    }

    new MinhashFunction(perms)
  }
}
