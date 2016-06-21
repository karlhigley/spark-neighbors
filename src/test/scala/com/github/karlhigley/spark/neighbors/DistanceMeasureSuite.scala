package com.github.karlhigley.spark.neighbors

import org.scalatest.FunSuite

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.SparseVector

import com.github.karlhigley.spark.neighbors.linalg._

class DistanceMeasureSuite extends FunSuite with TestSparkContext {
  import org.scalactic.Tolerance._

  val values = Array(1.0, 1.0, 1.0, 1.0)

  val v1 = new SparseVector(10, Array(0, 3, 6, 8), values)
  val v2 = new SparseVector(10, Array(1, 4, 7, 9), values)
  val v3 = new SparseVector(10, Array(2, 5, 7, 9), values)

  test("Cosine distance") {
    assert(CosineDistance.compute(v1, v1) === 0.0)
    assert(CosineDistance.compute(v1, v2) === 1.0)
    assert(CosineDistance.compute(v2, v3) === 0.5)
  }

  test("Euclidean distance") {
    assert(EuclideanDistance.compute(v1, v1) === 0.0)
    assert(EuclideanDistance.compute(v1, v2) === 2.83 +- 0.01)
    assert(EuclideanDistance.compute(v2, v3) === 2.0)
  }

  test("Manhattan distance") {
    assert(ManhattanDistance.compute(v1, v1) === 0.0)
    assert(ManhattanDistance.compute(v1, v2) === 8.0)
    assert(ManhattanDistance.compute(v2, v3) === 4.0)
  }

  test("Hamming distance") {
    assert(HammingDistance.compute(v1, v1) === 0.0)
    assert(HammingDistance.compute(v1, v2) === 8.0)
    assert(HammingDistance.compute(v2, v3) === 4.0)
  }

  test("Jaccard distance") {
    assert(JaccardDistance.compute(v1, v1) === 0.0)
    assert(JaccardDistance.compute(v1, v2) === 1.0)
    assert(JaccardDistance.compute(v2, v3) === 0.67 +- 0.01)
  }
}
