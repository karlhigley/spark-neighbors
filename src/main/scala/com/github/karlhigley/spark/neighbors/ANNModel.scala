package com.github.karlhigley.spark.neighbors

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._
import org.apache.spark.storage.StorageLevel

import com.github.karlhigley.spark.neighbors.collision.CollisionStrategy
import com.github.karlhigley.spark.neighbors.linalg.DistanceMeasure
import com.github.karlhigley.spark.neighbors.lsh.{ HashTableEntry, LSHFunction, Signature }

/**
 * Model containing hash tables produced by computing signatures
 * for each supplied vector.
 */
class ANNModel private[neighbors] (
    private[neighbors] val hashTables: RDD[_ <: HashTableEntry[_]],
    private[neighbors] val hashFunctions: Array[_ <: LSHFunction[_]],
    private[neighbors] val collisionStrategy: CollisionStrategy,
    private[neighbors] val measure: DistanceMeasure,
    private[neighbors] val numPoints: Int
) extends Serializable {

  type Point = (Int, SparseVector)
  type CandidateGroup = Iterable[Point]

  /**
   * Identify pairs of nearest neighbors by applying a
   * collision strategy to the hash tables and then computing
   * the actual distance between candidate pairs.
   */
  def neighbors(quantity: Int): RDD[(Int, Array[(Int, Double)])] = {
    val candidates = collisionStrategy.apply(hashTables).groupByKey(hashTables.getNumPartitions).values
    val neighbors = computeDistances(candidates)
    neighbors.topByKey(quantity)(ANNModel.ordering)
  }

  /**
   * Identify the nearest neighbors of a collection of new points
   * by computing their signatures, filtering the hash tables to
   * only potential matches, cogrouping the two RDDs, and
   * computing candidate distances in the "normal" fashion.
   */
  def neighbors(queryPoints: RDD[Point], quantity: Int): RDD[(Int, Array[(Int, Double)])] = {
    val modelEntries = collisionStrategy.apply(hashTables)

    val queryHashTables = ANNModel.generateHashTable(queryPoints, hashFunctions)
    val queryEntries = collisionStrategy.apply(queryHashTables)

    val candidateGroups = modelEntries.cogroup(queryEntries).values
    val neighbors = computeBipartiteDistances(candidateGroups)
    neighbors.topByKey(quantity)(ANNModel.ordering)
  }

  /**
   * Compute the average selectivity of the points in the
   * dataset. (See "Modeling LSH for Performance Tuning" in CIKM '08.)
   */
  def avgSelectivity(): Double = {
    val candidates = collisionStrategy.apply(hashTables).groupByKey(hashTables.getNumPartitions).values

    val candidateCounts =
      candidates
        .flatMap {
          case candidates => {
            for (
              (id1, vector1) <- candidates.iterator;
              (id2, vector2) <- candidates.iterator
            ) yield (id1, id2)
          }
        }
        .distinct()
        .countByKey()
        .values

    candidateCounts.map(_.toDouble / numPoints).reduce(_ + _) / numPoints
  }

  /**
   * Compute the actual distance between candidate pairs
   * using the supplied distance measure.
   */
  private def computeDistances(candidates: RDD[CandidateGroup]): RDD[(Int, (Int, Double))] = {
    candidates
      .flatMap {
        case group => {
          for (
            (id1, vector1) <- group.iterator;
            (id2, vector2) <- group.iterator;
            if id1 < id2
          ) yield ((id1, id2), measure.compute(vector1, vector2))
        }
      }
      .reduceByKey((a, b) => a)
      .flatMap {
        case ((id1, id2), dist) => Array((id1, (id2, dist)), (id2, (id1, dist)))
      }
  }

  /**
   * Compute the actual distance between candidate pairs
   * using the supplied distance measure.
   */
  private def computeBipartiteDistances(candidates: RDD[(CandidateGroup, CandidateGroup)]): RDD[(Int, (Int, Double))] = {
    candidates
      .flatMap {
        case (groupA, groupB) => {
          for (
            (id1, vector1) <- groupA.iterator;
            (id2, vector2) <- groupB.iterator
          ) yield ((id1, id2), measure.compute(vector1, vector2))
        }
      }
      .reduceByKey((a, b) => a)
      .map {
        case ((id1, id2), dist) => (id1, (id2, dist))
      }
  }
}

object ANNModel {
  private val ordering = Ordering[Double].on[(Int, Double)](_._2).reverse

  /**
   * Train a model by computing signatures for the supplied
   * points
   */
  def train(
    points: RDD[(Int, SparseVector)],
    hashFunctions: Array[_ <: LSHFunction[_]],
    collisionStrategy: CollisionStrategy,
    measure: DistanceMeasure,
    persistenceLevel: StorageLevel
  ): ANNModel = {
    val hashTables: RDD[_ <: HashTableEntry[_]] = generateHashTable(points, hashFunctions)
    hashTables.persist(persistenceLevel)
    new ANNModel(
      hashTables,
      hashFunctions,
      collisionStrategy,
      measure,
      points.count().toInt
    )
  }

  def generateHashTable(
    points: RDD[(Int, SparseVector)],
    hashFunctions: Array[_ <: LSHFunction[_]]
  ): RDD[_ <: HashTableEntry[_]] = {
    val indHashFunctions: Array[(_ <: LSHFunction[_], Int)] = hashFunctions.zipWithIndex
    points.flatMap {
      case (id, vector) =>
        indHashFunctions.map {
          case (hashFunc, table) =>
            hashFunc.hashTableEntry(id, table, vector)
        }
    }
  }
}
