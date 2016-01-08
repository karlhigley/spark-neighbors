package io.github.karlhigley.neighbors.linalg

import org.apache.spark.mllib.linalg.{ SparseVector, Vectors }

import org.apache.spark.mllib.linalg.LinalgShim

/**
 * This abstract base class provides the interface for
 * distance measures to be used in computing the actual
 * distances between candidate pairs.
 *
 * It's framed in terms of distance rather than similarity
 * to provide a common interface that works for Euclidean
 * distance along with other distances. (Cosine distance is
 * admittedly not a proper distance measure, but is computed
 * similarly nonetheless.)
 */
private[neighbors] sealed abstract class DistanceMeasure extends Serializable {
  def compute(v1: SparseVector, v2: SparseVector): Double
}

private[neighbors] final object CosineDistance extends DistanceMeasure {

  /**
   * Compute cosine distance between vectors
   *
   * LinalgShim reaches into Spark's private linear algebra
   * code to use a BLAS dot product. Could probably be
   * replaced with a direct invocation of the appropriate
   * BLAS method.
   */
  def compute(v1: SparseVector, v2: SparseVector): Double = {
    val dotProduct = LinalgShim.dot(v1, v2)
    val norms = Vectors.norm(v1, 2) * Vectors.norm(v2, 2)
    1.0 - (math.abs(dotProduct) / norms)
  }
}
