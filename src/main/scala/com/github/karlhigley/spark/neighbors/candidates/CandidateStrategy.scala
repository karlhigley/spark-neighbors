package com.github.karlhigley.spark.neighbors.candidates

import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.rdd.RDD

import com.github.karlhigley.spark.neighbors.lsh.HashTableEntry

/**
 * Abstract base class for approaches to identifying candidate
 * pairs from pre-computed hash tables. This should be sufficiently
 * general to support a variety of candidate identification strategies,
 * including multi-probe (for scalar-random-projection LSH), and
 * banding (for minhash LSH).
 */
private[neighbors] abstract class CandidateStrategy {
  type Point = (Int, SparseVector)

  def identify(hashTables: RDD[_ <: HashTableEntry[_]]): RDD[(Product, Point)]
}
