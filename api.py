from flask import Flask, request, jsonify
from pyspark.ml.classification import RandomForestClassificationModel, DecisionTreeClassificationModel, GBTClassificationModel
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# Initialize Spark session
spark = SparkSession.builder.appName("API_Model").getOrCreate()

# Load trained models
rf_model = RandomForestClassificationModel.load("/home/ika/Documents/big-data/project-kafka/model/random_forest_model")
dt_model = DecisionTreeClassificationModel.load("/home/ika/Documents/big-data/project-kafka/model/decision_tree_model")
gbt_model = GBTClassificationModel.load("/home/ika/Documents/big-data/project-kafka/model/gbt_model")

# Initialize Flask app
app = Flask(__name__)

# Preprocessing pipeline components
tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
hashing_tf = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=10000)

assembler = VectorAssembler(inputCols=["upvotes", "num_comments", "features"], outputCol="final_features")

def preprocess(data):
    """
    Preprocess the input JSON data to match the format expected by the model.
    """
    # Remove unnecessary fields for prediction
    data = {key: value for key, value in data.items() if key in ["title", "body", "upvotes", "num_comments"]}

    # Convert JSON data to a Spark DataFrame
    df = spark.createDataFrame([data])

    # Combine text fields
    df = df.withColumn("text", F.concat(F.col("title"), F.lit(" "), F.col("body")))

    # Tokenize, remove stop words, and apply TF-IDF
    df = tokenizer.transform(df)
    df = remover.transform(df)
    df = hashing_tf.transform(df)

    # Dynamically fit the IDF model on the incoming data
    idf_model = IDF(inputCol="raw_features", outputCol="features").fit(df)
    df = idf_model.transform(df)

    # Assemble features for the model
    df = assembler.transform(df)
    return df.select("final_features")

# API endpoints
@app.route("/predict_rf", methods=["POST"])
def predict_rf():
    data = request.get_json()  # Get JSON data from user
    df = preprocess(data)  # Preprocess the input data
    predictions = rf_model.transform(df)  # Get predictions from the model
    label = predictions.select("prediction").first()[0]  # Extract predicted label
    return jsonify({"label": int(label)})  # Return the label as JSON

@app.route("/predict_dt", methods=["POST"])
def predict_dt():
    data = request.get_json()
    df = preprocess(data)
    predictions = dt_model.transform(df)
    label = predictions.select("prediction").first()[0]
    return jsonify({"label": int(label)})

@app.route("/predict_gbt", methods=["POST"])
def predict_gbt():
    data = request.get_json()
    df = preprocess(data)
    predictions = gbt_model.transform(df)
    label = predictions.select("prediction").first()[0]
    return jsonify({"label": int(label)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
