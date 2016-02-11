package io.github.karlhigley.neighbors.candidates

import scala.util.hashing.MurmurHash3

import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import io.github.karlhigley.neighbors.lsh.{ BitSignature, HashTableEntry, IntSignature }

/**
 * A banding candidate identification strategy for Minhash
 *
 * (See Mining Massive Datasets, Ch. 3)
 */
private[neighbors] class BandingCandidateStrategy(
    bands: Int,
    persistenceLevel: StorageLevel
) extends CandidateStrategy with Serializable {

  /**
   * Identify candidates by finding a signature match
   * in any band of any hash table.
   */
  def identify(hashTables: RDD[_ <: HashTableEntry[_]]): RDD[((Int, SparseVector), (Int, SparseVector))] = {
    val bandEntries = hashTables.flatMap(entry => {
      val sigElements = entry.signature match {
        case BitSignature(values) => values.toArray
        case IntSignature(values) => values
      }

      val banded = sigElements.grouped(bands).zipWithIndex
      banded.map {
        case (bandSig, band) => {
          // Arrays are mutable and can't be used in RDD keys
          // Use a hash value (i.e. an int) as a substitute
          val bandSigHash = MurmurHash3.arrayHash(bandSig)
          ((entry.table, band, bandSigHash), (entry.id, entry.point))
        }
      }
    })

    bandEntries.persist(persistenceLevel)
    bandEntries.join(bandEntries).flatMap {
      case (_, ((id1, point1), (id2, point2))) if (id1 < id2) => Some(((id1, point1), (id2, point2)))
      case _ => None
    }
  }
}
