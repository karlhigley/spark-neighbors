package com.github.karlhigley.spark.neighbors.collision

import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.rdd.RDD

import com.github.karlhigley.spark.neighbors.lsh.HashTableEntry

/**
 * Abstract base class for approaches to identifying collisions from
 * the pre-computed hash tables. This should be sufficiently
 * general to support a variety of collision and candidate identification
 * strategies, including multi-probe (for scalar-random-projection LSH),
 * and banding (for minhash LSH).
 */
private[neighbors] abstract class CollisionStrategy {
  type Point = (Long, SparseVector)

  def apply(hashTables: RDD[_ <: HashTableEntry[_]]): RDD[(Product, Point)]
}
