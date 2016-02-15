package com.github.karlhigley.spark.neighbors.linalg

import java.util.Random

import org.apache.spark.mllib.linalg.{ DenseMatrix, Matrices }
import org.apache.spark.mllib.linalg.{ DenseVector, Vector }

/**
 * A simple random projection based on Spark's existing
 * random generation and multiplication of dense matrices.
 */
private[neighbors] class RandomProjection(
    private[this] val matrix: DenseMatrix
) extends Serializable {

  /**
   * Apply the projection to supplied vector
   */
  def project(vector: Vector): DenseVector = {
    matrix.multiply(vector)
  }
}

private[neighbors] object RandomProjection {

  /**
   * Generate a random projection based on the input and output
   * dimensions
   */
  def generate(originalDim: Int, projectedDim: Int, random: Random): RandomProjection = {
    val localMatrix = DenseMatrix.randn(projectedDim, originalDim, random)
    new RandomProjection(localMatrix)
  }
}
