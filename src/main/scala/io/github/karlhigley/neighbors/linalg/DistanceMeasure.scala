package io.github.karlhigley.neighbors.linalg

import org.apache.spark.mllib.linalg.SparseVector

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
