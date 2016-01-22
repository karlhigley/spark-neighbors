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
    val vectors: RDD[(Int, SparseVector)],
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
    val candidates = candidateStrategy.identify(hashTables)
    val neighbors = computeDistances(candidates)
    neighbors.topByKey(quantity)(ANNModel.ordering)
  }

  /**
   * Compute the actual distance between candidate pairs
   * using the supplied distance measure.
   */
  private def computeDistances(candidates: RDD[(Int, Int)]): RDD[(Int, (Int, Double))] = {
    vectors.persist(persistenceLevel)
    candidates
      .join(vectors)
      .map {
        case (id1, (id2, vector1)) => (id2, (id1, vector1))
      }
      .join(vectors)
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
   * vectors
   */
  def train(
    vectors: RDD[(Int, SparseVector)],
    hashFunctions: Array[_ <: LSHFunction[_]],
    CandidateStrategy: CandidateStrategy,
    measure: DistanceMeasure,
    persistenceLevel: StorageLevel
  ): ANNModel = {

    val numTables = hashFunctions.size
    val indHashFunctions = hashFunctions.zipWithIndex
    val hashTables: RDD[_ <: HashTableEntry[_]] = vectors.flatMap {
      case (id, vector) =>
        indHashFunctions.map {
          case (hashFunc, table) =>
            hashFunc.hashTableEntry(id, table, vector)
        }
    }
    new ANNModel(
      vectors,
      numTables,
      hashTables,
      CandidateStrategy,
      measure,
      persistenceLevel
    )
  }
}
