package com.github.karlhigley.spark.neighbors.candidates

import scala.util.hashing.MurmurHash3

import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.github.karlhigley.spark.neighbors.lsh.{ BitSignature, HashTableEntry, IntSignature }

/**
 * A banding candidate identification strategy for Minhash
 *
 * (See Mining Massive Datasets, Ch. 3)
 */
private[neighbors] class BandingCandidateStrategy(
    bands: Int
) extends CandidateStrategy with Serializable {

  /**
   * Identify candidates by finding a signature match
   * in any band of any hash table.
   */
  def identify(hashTables: RDD[_ <: HashTableEntry[_]]): RDD[CandidateGroup] = {
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

    bandEntries.cogroup(bandEntries).map(_._2)
  }
}
