name := "spark-neighbors"

organization := "com.github.karlhigley"

description := "Spark-based approximate nearest neighbor search using locality-sensitive hashing"

version := "0.2.0"

scalaVersion := "2.10.5"

spName := "karlhigley/spark-neighbors"

sparkVersion := "1.6.0"

sparkComponents := Seq("core", "mllib")

val testSparkVersion = settingKey[String]("The version of Spark to test against.")

testSparkVersion := sys.props.get("spark.testVersion").getOrElse(sparkVersion.value)

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.12",
  "org.scalatest" %% "scalatest" % "2.2.4" % "test"
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % testSparkVersion.value % "test" force(),
  "org.apache.spark" %% "spark-mllib" % testSparkVersion.value % "test" force(),
  "org.scala-lang" % "scala-library" % scalaVersion.value % "compile"
)

// This is necessary because of how we explicitly specify Spark dependencies
// for tests rather than using the sbt-spark-package plugin to provide them.
spIgnoreProvided := true

parallelExecution in Test := false

publishArtifact in Test := false

publishMavenStyle := true

spIncludeMaven := true

licenses += ("MIT", url("http://opensource.org/licenses/MIT"))

credentials += Credentials(Path.userHome / ".ivy2" / ".spark-package-credentials")

publishTo := {
  val nexus = "https://oss.sonatype.org/"
  if (version.value.endsWith("SNAPSHOT"))
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases"  at nexus + "service/local/staging/deploy/maven2")
}

pomExtra := (
  <url>https://github.com/karlhigley/spark-neighbors</url>
  <licenses>
    <license>
      <name>The MIT License (MIT)</name>
	  <url>http://opensource.org/licenses/MIT</url>
	  <distribution>repo</distribution>
    </license>
  </licenses>
  <scm>
    <url>git@github.com:karlhigley/spark-neighbors.git</url>
    <connection>scm:git:git@github.com:karlhigley/spark-neighbors.git</connection>
  </scm>
  <developers>
    <developer>
      <id>karlhigley</id>
      <name>Karl Higley</name>
      <url>https://github.com/karlhigley</url>
    </developer>
  </developers>)

pomIncludeRepository := { _ => false }

useGpg := true
