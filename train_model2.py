from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.evaluation import ClusteringEvaluator
import numpy as np
import os

# Initialize Spark Session
spark = SparkSession.builder.appName("AnimeRecommendationClustering").getOrCreate()

# Define schema
schema = StructType([
    StructField("username", StringType(), True),
    StructField("anime_id", IntegerType(), True),
    StructField("my_score", IntegerType(), True),
    StructField("user_id", IntegerType(), True),
    StructField("gender", StringType(), True),
    StructField("title", StringType(), True),
    StructField("type", StringType(), True),
    StructField("source", StringType(), True),
    StructField("score", FloatType(), True),
    StructField("scored_by", IntegerType(), True),
    StructField("rank", IntegerType(), True),
    StructField("popularity", IntegerType(), True),
    StructField("genre", StringType(), True)
])

# Load batch files
file_paths = [
    "batches/batch_1_1731428870.csv",
    "batches/batch_2_1731428875.csv",
    "batches/batch_3_1731428881.csv"
]

# Preprocess the data
def preprocess_data(file_path):
    df = spark.read.csv(file_path, schema=schema, header=False)

    # Check for null values
    if df.filter(df["my_score"].isNull() | df["score"].isNull() | df["popularity"].isNull()).count() > 0:
        print("Warning: Null values found in features. Filling with defaults.")
        df = df.na.fill({"my_score": 0, "score": 0.0, "popularity": 0})  # Fill nulls

    # Select features for clustering, e.g., 'my_score', 'score', 'popularity'
    feature_columns = ["my_score", "score", "popularity"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)

    # Standardize features
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    return df

# Model 1: K-Means Clustering
def kmeans_clustering(df):
    kmeans = KMeans(featuresCol="scaled_features", predictionCol="kmeans_cluster", k=5)  # Adjust k as needed
    model = kmeans.fit(df)
    
    # Save the KMeans model
    model.save("models/kmeans_model")
    
    return model.transform(df), model

# Model 2: Gaussian Mixture Model Clustering
def gaussian_clustering(df):
    gmm = GaussianMixture(featuresCol="scaled_features", predictionCol="gaussian_cluster", k=5)  # Adjust k as needed
    model = gmm.fit(df)
    
    # Save the Gaussian Mixture model
    model.save("models/gaussian_mixture_model")
    
    return model.transform(df), model

# Model 3: Simplified CURE Clustering (Custom Implementation Approximation)
def cure_clustering(df, num_clusters=5, num_representatives=5, shrink_factor=0.5):
    # Step 1: Apply K-Means for initial clustering to create rough clusters
    kmeans = KMeans(featuresCol="scaled_features", predictionCol="initial_cluster", k=num_clusters)
    initial_model = kmeans.fit(df)
    clustered_df = initial_model.transform(df)

    # Save the initial KMeans model for CURE
    initial_model.save("models/cure_initial_kmeans_model")

    # Step 2: For each cluster, pick representative points and apply shrink factor
    def get_representatives(cluster_points):
        # Convert to list
        cluster_points_list = list(cluster_points)

        # Ensure all points are in the correct format
        feature_vectors = [point['scaled_features'].toArray() for point in cluster_points_list]

        # Check if we have enough points to sample from
        if len(feature_vectors) < num_representatives:
            return feature_vectors  # Return all if not enough points

        # Randomly sample representatives
        representative_indices = np.random.choice(len(feature_vectors), size=num_representatives, replace=False)
        representative_points = [feature_vectors[i] for i in representative_indices]

        centroid = np.mean(representative_points, axis=0)
        shrunk_points = [(1 - shrink_factor) * centroid + shrink_factor * point for point in representative_points]

        return shrunk_points

    # Convert to RDD for custom aggregation
    rdd = clustered_df.rdd.map(lambda row: (row['initial_cluster'], row))
    representatives = rdd.groupByKey().mapValues(get_representatives).collect()

    # Create a list of tuples for the new DataFrame
    representative_rows = []
    for cluster_id, reps in representatives:
        for rep in reps:
            # Convert rep to a vector using Vectors.dense
            representative_rows.append((cluster_id, Vectors.dense(rep)))

    # Define the schema explicitly
    schema = StructType([
        StructField("cluster_id", IntegerType(), True),
        StructField("representative_features", VectorUDT(), True)  # Use VectorUDT for the vector type
    ])

    # Create a new DataFrame from the representatives with the specified schema
    representatives_df = spark.createDataFrame(representative_rows, schema=schema)

    return representatives_df  # Return the DataFrame with representatives

# Execute each model on respective batches
batch1_df = preprocess_data(file_paths[0])
batch2_df = preprocess_data(file_paths[1])
batch3_df = preprocess_data(file_paths[2])

# Apply models
kmeans_result, kmeans_model = kmeans_clustering(batch1_df)
gaussian_result, gmm_model = gaussian_clustering(batch2_df)
cure_result = cure_clustering(batch3_df)

# Evaluate and show accuracy using Silhouette score
evaluator = ClusteringEvaluator(predictionCol="kmeans_cluster")  # Specify the correct prediction column

kmeans_silhouette = evaluator.evaluate(kmeans_result)
print(f"KMeans Silhouette Score: {kmeans_silhouette}")

gmm_evaluator = ClusteringEvaluator(predictionCol="gaussian_cluster")  # Specify the correct prediction column for GMM


gmm_silhouette = gmm_evaluator.evaluate(gaussian_result)

print(f"Gaussian Mixture Silhouette Score: {gmm_silhouette}")

# Show sample results for each model
print("KMeans Result:")
kmeans_result.show()
print("Gaussian Mixture Result:")
gaussian_result.show()
print("CURE Result (Simplified Approximation):")
cure_result.show()

# Stop Spark session
spark.stop()
