# Spark

## Spark introduction - Scala
http://spark.apache.org/docs/0.9.0/scala-programming-guide.html
Resilient distributed datasets (RDDs) are fault tolerant, distributed collections that support parallel operation, and can be persisted in memory.

There are special shared variables that can be created that will be available at each node - these can be read only or accumulators. Read only broadcast variables can be used to give every node a copy of a large input dataset efficiently.

A SparkContext must be initialized for a spark job, which has configuration for how to access the cluster.

##RDDs
There are two types: parallelized collections, which are created by calling .parallelize() on a Scala collection, and Hadoop datasets, which are read from HDFS/Hbase/etc files.

When operating on thes, one can specify the number of slices the dataset is cut into, resulting in one task per slice. Typically, have 2-4 slices per CPU in the cluster.

Two operation types: transformations, which creates a new dataset from the existing dataset, and actions, which return a value to the job driver.

All transformations are lazy. Transformations are only performed when an action requires a result that depends on it.
Note! Transformations are recomputed for each action by default - persist it using the cache() method to keep it in memory. There are also ways to persist on disk or replicated in the cluster or various levels inbetween - yielding tradeoffs between cpu efficieny and memory usage. Memory is ideal if it can fit.
