# Import libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel, GBTClassificationModel
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import StringType, StructType, StructField

# Initialize Spark session
spark = SparkSession.builder \
    .appName('Sentiment Analysis Prediction') \
    .getOrCreate()

# Load the saved models
rf_model = RandomForestClassificationModel.load("/home/ika/Documents/big-data/project-kafka/model/random_forest_model")
gbt_model = GBTClassificationModel.load("/home/ika/Documents/big-data/project-kafka/model/gbt_model")

# Function to preprocess and make predictions
def predict_sentiment(input_text):
    # Create a DataFrame from input text
    schema = StructType([StructField("text", StringType(), True)])
    input_df = spark.createDataFrame([(input_text,)], schema)

    # Text preprocessing
    # Tokenization
    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    tokenized_df = tokenizer.transform(input_df)

    # Remove stopwords
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    no_stopwords_df = remover.transform(tokenized_df)

    # Feature extraction using HashingTF and IDF
    hashing_tf = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=10000)
    featurized_df = hashing_tf.transform(no_stopwords_df)
    
    idf = IDF(inputCol="raw_features", outputCol="features")
    idf_model = idf.fit(featurized_df)
    processed_df = idf_model.transform(featurized_df)

    # Assemble features for final prediction
    final_assembler = VectorAssembler(inputCols=['features'], outputCol='final_features')
    final_df = final_assembler.transform(processed_df).select("final_features")

    # Predict using each model
    rf_prediction = rf_model.transform(final_df).select("prediction").collect()[0][0]
    gbt_prediction = gbt_model.transform(final_df).select("prediction").collect()[0][0]

    # Output results
    predictions = {
        "Random Forest Prediction": "Depressed" if rf_prediction == 1.0 else "Not Depressed",
        "Gradient Boosted Tree Prediction": "Depressed" if gbt_prediction == 1.0 else "Not Depressed"
    }
    
    return predictions

# Example input
input_text = "Happy"

# Run prediction
results = predict_sentiment(input_text)
print("Sentiment Predictions:")
for model, prediction in results.items():
    print(f"{model}: {prediction}")
