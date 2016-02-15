package com.github.karlhigley.spark.neighbors.lsh

import java.util.Random

/**
 * Hash function used to permute vector rows for
 * minhash functions. The supplied prime should
 * be greater than the number of dimensions.
 *
 * References:
 *  - Broder, et al. "Min-wise independent permutations." STOC, 1998.
 *
 * @see [[https://en.wikipedia.org/wiki/MinHash#Min-wise_independent_permutations
 *          Min-wise independent permutations (Wikipedia)]]
 */
private[lsh] class PermutationFunction(
    a: Int,
    b: Int,
    prime: Int,
    dimensions: Int
) extends Serializable {

  implicit class LongWithMod(x: Long) {
    def mod(y: Long): Long = x % y + (if (x < 0) y else 0)
  }

  /**
   * Permute a dimension index, returning a new index
   */
  def apply(x: Int): Int = {
    ((((a.longValue * x) + b) mod prime) mod dimensions).toInt
  }
}

private[lsh] object PermutationFunction {

  /**
   * Randomly generate a new permutation function
   */
  def random(dimensions: Int, prime: Int, random: Random): PermutationFunction = {
    val a = nonZeroRandomInt(random)
    val b = random.nextInt()
    new PermutationFunction(a, b, prime, dimensions)
  }

  /**
   * Use recursion to randomly generate a non-zero integer
   */
  private def nonZeroRandomInt(random: Random): Int = {
    val randomInt = random.nextInt()
    if (randomInt == 0) {
      nonZeroRandomInt(random)
    } else {
      randomInt
    }
  }
}
