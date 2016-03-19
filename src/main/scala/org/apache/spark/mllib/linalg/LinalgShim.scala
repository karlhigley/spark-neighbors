package org.apache.spark.mllib.linalg

import breeze.linalg.{ SparseVector => BSV, Vector => BV }

/**
 * This shim reaches into Spark's private linear algebra
 * code, in order to take advantage of optimized dot products.
 * While the dot product implementation in question is part of
 * MLlib's BLAS module, BLAS itself only supports dot products
 * between dense vectors, and MLlib implements sparse vector
 * dot products. Using a shim here avoids copy/pasting that
 * implementation.
 */
object LinalgShim {

  /**
   * Compute the dot product between two vectors
   *
   * Under the hood, Spark's BLAS module calls a BLAS routine
   * from netlib-java for the case of two dense vectors, or an
   * optimized Scala implementation in the case of sparse vectors.
   */
  def dot(x: Vector, y: Vector): Double = {
    BLAS.dot(x, y)
  }

  /**
   * Convert a Spark vector to a Breeze vector to access
   * vector operations that Spark doesn't provide.
   */
  def toBreeze(x: Vector): BV[Double] = {
    x.toBreeze
  }
}
