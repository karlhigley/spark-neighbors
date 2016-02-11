lazy val root = (project in file(".")).
  settings(
    spName := "karlhigley/spark-neighbors",
    version := "0.1.0",
    scalaVersion := "2.10.5",
    sparkVersion := "1.6.0",
    sparkComponents += "mllib",

    spShortDescription := "Approximate nearest neighbor search using locality-sensitive hashing",
	spDescription := """Batch computation of the nearest neighbors for each point in a dataset using:
	                    | - Hamming distance via bit sampling LSH
	                    | - Cosine distance via sign-random-projection LSH
	                    | - Euclidean distance via scalar-random-projection LSH
	                    | - Jaccard distance via Minhash LSH""".stripMargin
  )

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "2.2.4" % "test",
  "com.typesafe.akka" %% "akka-actor" % "2.3.4" % "test"
)

parallelExecution in Test := false

credentials += Credentials(Path.userHome / ".ivy2" / ".spark-package-credentials")
