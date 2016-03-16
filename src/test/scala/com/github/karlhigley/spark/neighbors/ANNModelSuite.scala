package com.github.karlhigley.spark.neighbors

import org.scalatest.FunSuite

import com.github.karlhigley.spark.neighbors.lsh.HashTableEntry

class ANNModelSuite extends FunSuite with TestSparkContext {
  val numPoints = 1000
  val dimensions = 100
  val density = 0.5

  val localPoints = TestHelpers.generateRandomPoints(numPoints, dimensions, density)

  test("average selectivity is between zero and one") {
    val points = sc.parallelize(localPoints.zipWithIndex.map(_.swap))

    val ann =
      new ANN(dimensions, "cosine")
        .setTables(1)
        .setSignatureLength(16)

    val model = ann.train(points)
    val selectivity = model.avgSelectivity()

    assert(selectivity > 0.0)
    assert(selectivity < 1.0)
  }

  test("average selectivity increases with more tables") {
    val points = sc.parallelize(localPoints.zipWithIndex.map(_.swap))

    val ann =
      new ANN(dimensions, "cosine")
        .setTables(1)
        .setSignatureLength(16)

    val model1 = ann.train(points)

    ann.setTables(2)
    val model2 = ann.train(points)

    assert(model1.avgSelectivity() < model2.avgSelectivity())
  }

  test("average selectivity decreases with signature length") {
    val points = sc.parallelize(localPoints.zipWithIndex.map(_.swap))

    val ann =
      new ANN(dimensions, "cosine")
        .setTables(1)
        .setSignatureLength(4)

    val model4 = ann.train(points)

    ann.setSignatureLength(8)
    val model8 = ann.train(points)

    assert(model4.avgSelectivity() > model8.avgSelectivity())
  }

}
