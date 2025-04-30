package com.wine;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.SparkConf;

public class WineTrainer {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("Wine Quality Trainer").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);

        String trainingDataPath = args[0];
        JavaRDD<LabeledPoint> trainingData = MLUtils.loadLibSVMFile(sc.sc(), trainingDataPath).toJavaRDD();

        int numClasses = 10;
        java.util.HashMap<Integer, Integer> categoricalFeaturesInfo = new java.util.HashMap<>();
        int numTrees = 10;
        String featureSubsetStrategy = "auto";
        String impurity = "gini";
        int maxDepth = 5;
        int maxBins = 32;

        RandomForestModel model = RandomForest.trainClassifier(
            trainingData, numClasses, categoricalFeaturesInfo, numTrees,
            featureSubsetStrategy, impurity, maxDepth, maxBins, 12345
        );

        // ✅ Correct model save
        model.save(sc.sc(), "file:///home/ubuntu/wine-predictor/wine-model");

        sc.close();
    }
}
