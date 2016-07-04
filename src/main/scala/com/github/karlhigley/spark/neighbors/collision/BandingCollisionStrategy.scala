package com.github.karlhigley.spark.neighbors.collision

import scala.util.hashing.MurmurHash3

import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.github.karlhigley.spark.neighbors.lsh.{ BitSignature, HashTableEntry, IntSignature }

/**
 * A banding collision strategy for candidate identification with Minhash
 */
private[neighbors] class BandingCollisionStrategy(
    bands: Int
) extends CollisionStrategy with Serializable {

  /**
   * Convert hash tables into an RDD that is "collidable" using groupByKey.
   * The new keys contain the hash table id, the band id, and a hashed version
   * of the banded signature.
   */
  def apply(hashTables: RDD[_ <: HashTableEntry[_]]): RDD[(Product, Point)] = {
    val bandEntries = hashTables.flatMap(entry => {
      val elements = entry.sigElements
      val banded = elements.grouped(elements.size / bands).zipWithIndex
      banded.map {
        case (bandSig, bandNum) => {
          // Arrays are mutable and can't be used in RDD keys
          // Use a hash value (i.e. an int) as a substitute
          val bandSigHash = MurmurHash3.arrayHash(bandSig)
          val key = (entry.table, bandNum, bandSigHash).asInstanceOf[Product]
          (key, (entry.id, entry.point))
        }
      }
    })

    bandEntries
  }
}
