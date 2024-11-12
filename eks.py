# Import libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel, DecisionTreeClassificationModel, GBTClassificationModel
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import StringType, StructType, StructField

# Initialize Spark session
spark = SparkSession.builder \
    .appName('Sentiment Analysis Prediction') \
    .getOrCreate()

# Load the saved models
rf_model = RandomForestClassificationModel.load("/home/ika/Documents/big-data/project-kafka/model/random_forest_model")
dt_model = DecisionTreeClassificationModel.load("/home/ika/Documents/big-data/project-kafka/model/decision_tree_model")
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
    dt_prediction = dt_model.transform(final_df).select("prediction").collect()[0][0]
    gbt_prediction = gbt_model.transform(final_df).select("prediction").collect()[0][0]

    print(rf_prediction)
    print(dt_prediction)
    print(gbt_prediction)

    # Output results
    predictions = {
        "Random Forest Prediction": "Depressed" if rf_prediction == 1.0 else "Not Depressed",
        "Decision Tree Prediction": "Depressed" if dt_prediction == 1.0 else "Not Depressed",
        "Gradient Boosted Tree Prediction": "Depressed" if gbt_prediction == 1.0 else "Not Depressed"
    }
    
    return predictions

# Example input
input_text = "I tried running away from it all. I thought that if I left I wouldn't feel this way anymore. Get rid of the cause and the symptoms will follow right? But here I am, as far away from home as humanly possible, and every second of every day all I can think about is ways to die. Every time I go catch the bus I think 'I could just step forward, right now, and it would all be over'. I've even taken that step forward, only to be unsuccessful. I don't want to go home, and I don't want to be here. I feel like there's no place on earth that could make this all go away. I can't see myself making it to summer, I can't even see myself making it through the weekend. "

# Run prediction
results = predict_sentiment(input_text)
print("Sentiment Predictions:")
for model, prediction in results.items():
    print(f"{model}: {prediction}")
