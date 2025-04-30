package com.wine;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

public class WinePredictor {

    public static void main(String[] args) throws Exception {
        // Set up Spark Configuration and Context
        SparkConf conf = new SparkConf()
                .setAppName("Wine Quality Predictor")
                .setMaster("local"); // Important: local master inside Docker
        JavaSparkContext sc = new JavaSparkContext(conf);

        // === PATHS ===
        String modelPath = "/app/wine-model"; // Correct model path inside container
        String validationDataPath = args.length > 0 ? args[0] : "/app/ValidationDataset.libsvm";

        System.out.println("Loading trained model from " + modelPath);

        // Load trained model
        RandomForestModel model = RandomForestModel.load(sc.sc(), modelPath); // use sc.sc() here!!

        // Load validation dataset
        JavaRDD<LabeledPoint> validationData = MLUtils.loadLibSVMFile(sc.sc(), validationDataPath).toJavaRDD(); // also sc.sc() here

        // Predict and compare
        JavaRDD<Double> predictions = validationData.map(point -> model.predict(point.features()));

        // Calculate accuracy
        long correct = validationData.zip(predictions)
                .filter(pair -> pair._1().label() == pair._2())
                .count();
        long total = validationData.count();
        double accuracy = (double) correct / total;

        System.out.println("Prediction Accuracy: " + (accuracy * 100) + "%");

        // Stop Spark Context
        sc.close();
    }
}
