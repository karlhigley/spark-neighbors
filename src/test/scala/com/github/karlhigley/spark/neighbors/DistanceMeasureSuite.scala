package com.github.karlhigley.spark.neighbors

import org.scalatest.FunSuite

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.SparseVector

import com.github.karlhigley.spark.neighbors.linalg._

class DistanceMeasureSuite extends FunSuite with TestSparkContext {
  import org.scalactic.Tolerance._

  test("Jaccard distance") {
    val values = Array(1.0, 1.0, 1.0, 1.0)

    val v1 = new SparseVector(10, Array(0, 3, 6, 8), values)
    val v2 = new SparseVector(10, Array(1, 4, 7, 9), values)
    val v3 = new SparseVector(10, Array(2, 5, 7, 9), values)

    assert(JaccardDistance.compute(v1, v1) == 0.0)
    assert(JaccardDistance.compute(v1, v2) == 1.0)
    assert(JaccardDistance.compute(v2, v3) === 0.67 +- 0.01)
  }
}
