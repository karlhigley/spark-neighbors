package com.github.karlhigley.spark.neighbors.candidates

import scala.util.hashing.MurmurHash3

import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.github.karlhigley.spark.neighbors.lsh.{ BitSignature, HashTableEntry, IntSignature }

/**
 * A very simple candidate identification strategy based on
 * an OR-construction between hash functions/tables
 *
 * (See Mining Massive Datasets, Ch. 3)
 */
private[neighbors] class SimpleCandidateStrategy extends CandidateStrategy with Serializable {

  /**
   * Identify candidates by finding a signature match
   * in any hash table.
   */
  def identify(hashTables: RDD[_ <: HashTableEntry[_]]): RDD[(Product, Point)] = {
    val entries = hashTables.map(entry => {
      // Arrays are mutable and can't be used in RDD keys
      // Use a hash value (i.e. an int) as a substitute
      val key = (entry.table, MurmurHash3.arrayHash(entry.sigElements)).asInstanceOf[Product]
      (key, (entry.id, entry.point))
    })

    entries
  }
}
