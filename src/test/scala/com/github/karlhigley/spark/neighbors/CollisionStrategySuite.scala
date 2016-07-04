package com.github.karlhigley.spark.neighbors

import org.scalatest.FunSuite

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.SparseVector

class CollisionStrategySuite extends FunSuite with TestSparkContext {
  val numPoints = 1000
  val dimensions = 100
  val density = 0.5

  var points: RDD[(Long, SparseVector)] = _

  override def beforeAll() {
    super.beforeAll()
    val localPoints = TestHelpers.generateRandomPoints(numPoints, dimensions, density)
    points = sc.parallelize(localPoints).zipWithIndex.map(_.swap)
  }

  test("SimpleCollisionStrategy produces the correct number of tuples") {
    val ann =
      new ANN(dimensions, "cosine")
        .setTables(1)
        .setSignatureLength(8)

    val model = ann.train(points)

    val hashTables = model.hashTables
    val collidable = model.collisionStrategy(hashTables)

    assert(collidable.count() == numPoints)
  }

  test("BandingCollisionStrategy produces the correct number of tuples") {
    val numBands = 4

    val ann =
      new ANN(dimensions, "jaccard")
        .setTables(1)
        .setSignatureLength(8)
        .setBands(numBands)
        .setPrimeModulus(739)

    val model = ann.train(points)

    val hashTables = model.hashTables
    val collidable = model.collisionStrategy(hashTables)

    assert(collidable.count() == numPoints * numBands)
  }

}
