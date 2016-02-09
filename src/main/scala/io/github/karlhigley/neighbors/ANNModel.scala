package io.github.karlhigley.neighbors

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._
import org.apache.spark.storage.StorageLevel

import io.github.karlhigley.neighbors.candidates.CandidateStrategy
import io.github.karlhigley.neighbors.linalg.DistanceMeasure
import io.github.karlhigley.neighbors.lsh.{ HashTableEntry, LSHFunction, Signature }

/**
 * Model containing hash tables produced by computing signatures
 * for each supplied vector.
 */
class ANNModel private[neighbors] (
    val points: RDD[(Int, SparseVector)],
    val numTables: Int,
    private[neighbors] val hashTables: RDD[_ <: HashTableEntry[_]],
    private[neighbors] val candidateStrategy: CandidateStrategy,
    private[neighbors] val measure: DistanceMeasure,
    val persistenceLevel: StorageLevel
) extends Serializable {

  /**
   * Identify pairs of nearest neighbors by applying a
   * candidate strategy to the hash tables and then computing
   * the actual distance between candidate pairs.
   */
  def neighbors(quantity: Int): RDD[(Int, Array[(Int, Double)])] = {
    val candidates = candidateStrategy.identify(hashTables).distinct()
    val neighbors = computeDistances(candidates)
    neighbors.topByKey(quantity)(ANNModel.ordering)
  }

  /**
   * Compute the actual distance between candidate pairs
   * using the supplied distance measure.
   */
  private def computeDistances(candidates: RDD[(Int, Int)]): RDD[(Int, (Int, Double))] = {
    points.persist(persistenceLevel)
    candidates
      .join(points)
      .map {
        case (id1, (id2, vector1)) => (id2, (id1, vector1))
      }
      .join(points)
      .flatMap {
        case (id2, ((id1, vector1), vector2)) =>
          val distance = measure.compute(vector1, vector2)
          Array((id1, (id2, distance)), (id2, (id1, distance)))
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
    CandidateStrategy: CandidateStrategy,
    measure: DistanceMeasure,
    persistenceLevel: StorageLevel
  ): ANNModel = {

    val numTables = hashFunctions.size
    val indHashFunctions = hashFunctions.zipWithIndex
    val hashTables: RDD[_ <: HashTableEntry[_]] = points.flatMap {
      case (id, vector) =>
        indHashFunctions.map {
          case (hashFunc, table) =>
            hashFunc.hashTableEntry(id, table, vector)
        }
    }
    new ANNModel(
      points,
      numTables,
      hashTables,
      CandidateStrategy,
      measure,
      persistenceLevel
    )
  }
}
