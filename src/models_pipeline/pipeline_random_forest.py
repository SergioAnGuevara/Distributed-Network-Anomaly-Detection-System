
"""
pipeline_random_forest.py

PURPOSE:
    Runs in sequence the three blocks of the supervised Random Forest
    anomaly detection pipeline over the CIC-IDS2017 dataset:

    BLOCK 1 — TRAINING :
        Trains the base RF classifier (model_v1) on the processed
        Parquet, using per-class weights to handle class imbalance.
        Saves the model and its training metrics.

    BLOCK 2 — BATCH EVALUATION :
        Evaluates model_v1 on the test set (Tuesday + Thursday).
        Reports multiclass, binary metrics, confusion matrix,
        analysis of UNSEEN classes and distributed system metrics.

    BLOCK 3 — OPTIMIZATION :
        Searches for the best hyperparameters via CrossValidator on
        a 30% stratified sample, retrains the winning model on the
        full training set, evaluates it on test and generates
        a base vs. optimized comparative report.

INPUTS (S3 input paths):
    - s3://proyecto-anomalias-emr/data/cic-ids2017/train/processed/
    - s3://proyecto-anomalias-emr/data/cic-ids2017/test/processed/

OUTPUTS (S3 output paths):
    - .../models/random_forest_pipeline/model_v1/
    - .../models/random_forest_pipeline/model_optimized/
    - .../results/batch_evaluation/train_metrics_rf.json
    - .../results/batch_evaluation/test_metrics_rf.json
    - .../results/batch_evaluation/predictions_rf/          (Parquet)
    - .../results/batch_evaluation/optimization_report_rf.json



"""


import time                                          # time measurement
import json                                          # metrics serialization
import logging                                       # structured logging
from urllib.parse import urlparse                    # parse s3:// paths

import boto3                                         # JSON I/O with S3
import numpy as np                                   # argmax over CV metrics

from pyspark.sql import SparkSession                 # Spark entry point
from pyspark.sql import functions as F               # SQL functions
from pyspark.sql.functions import udf                 # UDF decorator
from pyspark.sql.types import DoubleType            # UDF type
from pyspark.ml import Pipeline, PipelineModel       # MLlib pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexerModel    # class names
from pyspark.ml.linalg import Vectors, VectorUDT     # vector for AUC
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)
from pyspark.mllib.evaluation import MulticlassMetrics  # multiclass metrics (RDD)


#  LOGGING 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("PipelineRandomForest")


# 
#  GLOBAL CONFIGURATION 
# 
BUCKET        = "s3://project-anomalies-emr"                   # root bucket
TRAIN_RAW     = f"{BUCKET}/data/cic-ids2017/train/"             # training
TEST_RAW      = f"{BUCKET}/data/cic-ids2017/test/"              # test
MODELS_PATH   = f"{BUCKET}/models/random_forest_pipeline/"     # models
RESULTS_PATH  = f"{BUCKET}/results/batch_evaluation/"          # results

# Derived paths.
TRAIN_PROCESSED  = f"{TRAIN_RAW}processed/"                    # training Parquet
TEST_PROCESSED   = f"{TEST_RAW}processed/"                     # test Parquet
INDEXER_PATH     = f"{MODELS_PATH}preprocessors/indexer/"     # fitted StringIndexer
MODEL_V1_OUT     = f"{MODELS_PATH}model_v1/"                   # base model
MODEL_OPT_OUT    = f"{MODELS_PATH}model_optimized/"            # optimized model
TRAIN_METRICS    = f"{RESULTS_PATH}train_metrics_rf.json"      # training metrics
TEST_METRICS_OUT = f"{RESULTS_PATH}test_metrics_rf.json"       # test metrics
PREDICTIONS_OUT  = f"{RESULTS_PATH}predictions_rf/"            # Parquet predictions
OPT_REPORT_OUT   = f"{RESULTS_PATH}optimization_report_rf.json"  # comparative report

# --- Block 1 hyperparameters (base model) ---
NUM_TREES = 100                                                # number of trees
MAX_DEPTH = 10                                                 # maximum depth
FSS       = "sqrt"                                             # featureSubsetStrategy
SEED      = 42                                                 # seed

# --- Block 3 parameters (optimization) ---
SAMPLE_FRACTION = 0.30                                         # fraction for CrossValidator
NUM_FOLDS       = 3                                            # CrossValidator folds

# --- Cluster parameters ---
EXECUTOR_INSTANCES = 2                                          # number of executors
EXECUTOR_CORES     = 2                                          # cores per executor
SHUFFLE_PARTITIONS = 16                                         # shuffle partitions


# 
#  SHARED HELPERS 
# 
def _parse_s3(s3_path):
    """Splits s3://bucket/key into (bucket, key)."""
    p = urlparse(s3_path)
    return p.netloc, p.path.lstrip("/")


def write_json_s3(obj, s3_path):
    """Writes a dict as JSON to S3 using boto3."""
    bucket, key = _parse_s3(s3_path)
    body = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=body)


def read_json_s3(s3_path):
    """Reads a JSON from S3 using boto3. Returns None if it fails."""
    try:
        bucket, key = _parse_s3(s3_path)
        resp = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception as e:
        logger.warning("Could not read %s (%s). Continuing without base metrics.", s3_path, e)
        return None


def feature_names(df, col_vec):
    """Recovers the feature names from the ML vector metadata.
    Returns the list in the order of the vector indices."""
    meta = df.schema[col_vec].metadata                         # metadata of the vector column
    groups = meta.get("ml_attr", {}).get("attrs", {})          # attributes (numeric/binary/nominal)
    idx_to_name = {}                                           # idx -> name map
    for group in groups.values():                              # iterate each attribute group
        for attr in group:
            idx_to_name[attr["idx"]] = attr["name"]
    if not idx_to_name:                                        # fallback if no metadata
        return None
    return [idx_to_name[i] for i in sorted(idx_to_name)]


def build_spark():
    """SparkSession with ALL cluster parameters via .config()."""
    spark = (
        SparkSession.builder
        .appName("ProjectAnomalies-RandomForest")                                  # exact required name
        .config("spark.executor.cores", "2")
        .config("spark.executor.memory", "6g")
        .config("spark.executor.memoryOverhead", "1024")
        .config("spark.driver.cores", "2")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.instances", str(EXECUTOR_INSTANCES))
        .config("spark.default.parallelism", "16")
        .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "256m")
        .config("spark.kryo.registrationRequired", "false")
        .config("spark.memory.fraction", "0.75")
        .config("spark.memory.storageFraction", "0.40")
        .config("spark.task.maxFailures", "4")
        .config("spark.speculation", "true")
        .config("spark.speculation.multiplier", "1.5")
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# 
#  BLOCK 1: BASE MODEL TRAINING 
# (source: 02_train_rf.py)
# 
def block1_train_rf(spark):
    """Trains the base Random Forest (model_v1) on the processed
    training set. Saves the model and its metrics to S3."""
    logger.info("=" * 60)
    logger.info("BLOCK 1 — RF TRAINING (model_v1)")
    logger.info("=" * 60)

    # (c) LOADING THE PROCESSED TRAINING DATAFRAME
    try:
        df_train = spark.read.parquet(TRAIN_PROCESSED)          # read Parquet
        logger.info("Training Parquet loaded from %s", TRAIN_PROCESSED)
    except Exception as e:
        logger.error("ERROR reading the training Parquet (%s): %s", TRAIN_PROCESSED, e)
        raise

    num_records = df_train.count()                              # number of training records
    logger.info("Training records: %d", num_records)

    # (d) CLASSIFIER DEFINITION
    rf = RandomForestClassifier(
        numTrees=NUM_TREES,                     # 100 trees
        maxDepth=MAX_DEPTH,                      # maximum depth 10
        featureSubsetStrategy=FSS,              # "sqrt"
        seed=SEED,                              # reproducibility
        labelCol="label_index",                 # numeric label
        featuresCol="scaled_features",          # scaled features
        weightCol="class_weight",               # weights for imbalance
    )

    # (e) PIPELINE with the classifier as the only stage
    pipeline = Pipeline(stages=[rf])                            # preprocessing already done in gestion_datos.py

    # (f) DISTRIBUTED TRAINING WITH TIME MEASUREMENT
    logger.info("Starting RF training (numTrees=%d, maxDepth=%d, fss=%s)...",
                NUM_TREES, MAX_DEPTH, FSS)
    t0 = time.time()                                            # start mark
    model = pipeline.fit(df_train)                              # training
    t1 = time.time()                                            # end mark
    training_time = t1 - t0                                     # total time (s)
    logger.info("Training completed in %.2f s", training_time)

    rf_model = model.stages[-1]                                 # RandomForestClassificationModel

    # (g) MODEL SAVING
    try:
        model.write().overwrite().save(MODEL_V1_OUT)            # serialize PipelineModel
        logger.info("RF model saved to %s", MODEL_V1_OUT)
    except Exception as e:
        logger.error("ERROR saving the model (%s): %s", MODEL_V1_OUT, e)
        raise

    # (h) SORTED FEATURE IMPORTANCES WITH NAMES
    importances = rf_model.featureImportances.toArray()        # importance vector
    names = feature_names(df_train, "scaled_features")          # names from metadata
    if names is None:                                           # fallback if no metadata
        names = [f"f{i}" for i in range(len(importances))]
    pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)  # descending order
    logger.info("Top 15 features by importance:")
    for name, imp in pairs[:15]:                                # print the top 15
        logger.info("   %-32s %.6f", name, imp)
    top_importances = [{"feature": n, "importance": float(v)} for n, v in pairs]  # for the JSON

    # (i) SAVING TRAINING METRICS
    metrics = {
        "training_time_seconds": round(training_time, 4),       # training time
        "num_trees": NUM_TREES,                                 # number of trees
        "max_depth": MAX_DEPTH,                                 # maximum depth
        "feature_subset_strategy": FSS,                         # subset strategy
        "num_partitions": SHUFFLE_PARTITIONS,                   # shuffle partitions
        "num_training_records": int(num_records),               # training records
        "executor_instances": EXECUTOR_INSTANCES,               # number of executors
        "cluster_cores_total": EXECUTOR_INSTANCES * EXECUTOR_CORES,  # total cores
        "feature_importances": top_importances,                 # sorted importances
    }
    try:
        write_json_s3(metrics, TRAIN_METRICS)                   # write JSON to S3
        logger.info("Training metrics saved to %s", TRAIN_METRICS)
    except Exception as e:
        logger.error("ERROR writing metrics (%s): %s", TRAIN_METRICS, e)
        raise

    logger.info("BLOCK 1 completed successfully.")



#  BLOCK 2: BATCH EVALUATION OF THE BASE MODEL 


def block2_evaluate_rf(spark):
    """Evaluates model_v1 on the test set. Reports multiclass,
    binary metrics, confusion matrix and system metrics."""
    logger.info("=" * 60)
    logger.info("BLOCK 2 — RF BATCH EVALUATION (model_v1 on test)")
    logger.info("=" * 60)

    # (b) LOADING THE MODEL, TEST DATA AND INDEXER
    try:
        model = PipelineModel.load(MODEL_V1_OUT)                # RF model
        logger.info("RF model loaded from %s", MODEL_V1_OUT)
    except Exception as e:
        logger.error("ERROR loading the model (%s): %s", MODEL_V1_OUT, e)
        raise

    try:
        df_test = spark.read.parquet(TEST_PROCESSED)            # test Parquet
        logger.info("Test Parquet loaded from %s", TEST_PROCESSED)
    except Exception as e:
        logger.error("ERROR reading the test Parquet (%s): %s", TEST_PROCESSED, e)
        raise

    try:
        indexer_model = StringIndexerModel.load(INDEXER_PATH)   # names of known classes
        labels = list(indexer_model.labels)                     # labels[i] = name of index i
        logger.info("StringIndexer loaded: %d known classes.", len(labels))
    except Exception as e:
        logger.error("ERROR loading the StringIndexer (%s): %s", INDEXER_PATH, e)
        raise

    # (c) DISTRIBUTED INFERENCE WITH TIME MEASUREMENT
    num_partitions = df_test.rdd.getNumPartitions()            # DataFrame partitions
    logger.info("Starting batch inference (partitions=%d)...", num_partitions)

    t0 = time.time()                                           # start mark
    df_pred = model.transform(df_test)                         # adds prediction/probability
    df_pred = df_pred.cache()                                  # cache for reuse in metrics
    num_records = df_pred.count()                              # action that forces the inference
    t1 = time.time()                                          # end mark

    inference_time = t1 - t0                                   # total time (s)
    throughput = num_records / inference_time if inference_time > 0 else 0.0
    latency_us = (inference_time / num_records * 1_000_000) if num_records > 0 else 0.0
    logger.info("Inference: %d records in %.2f s | throughput=%.1f rec/s | latency=%.2f µs/rec",
                num_records, inference_time, throughput, latency_us)

    # (d) MULTICLASS AND BINARY METRICS
    # MulticlassMetrics requires an RDD of (prediction, label) as floats.
    rdd_pred = df_pred.select("prediction", "label_index").rdd.map(
        lambda r: (float(r["prediction"]), float(r["label_index"])))
    mm = MulticlassMetrics(rdd_pred)                           # multiclass metrics

    accuracy = float(mm.accuracy)                              # overall accuracy
    weighted_precision = float(mm.weightedPrecision)           # weighted precision
    weighted_recall = float(mm.weightedRecall)                 # weighted recall
    weighted_f1 = float(mm.weightedFMeasure())                 # weighted F1

    # Macro = simple average of the metric per class (not weighted by support).
    present_labels = sorted(
        [float(r["label_index"]) for r in df_pred.select("label_index").distinct().collect()])
    macro_precision = sum(mm.precision(l) for l in present_labels) / len(present_labels)
    macro_recall    = sum(mm.recall(l) for l in present_labels) / len(present_labels)
    macro_f1        = sum(mm.fMeasure(l) for l in present_labels) / len(present_labels)

    logger.info("MULTICLASS -> Acc=%.4f | F1(w)=%.4f F1(macro)=%.4f | P(w)=%.4f P(macro)=%.4f | R(w)=%.4f R(macro)=%.4f",
                accuracy, weighted_f1, macro_f1, weighted_precision, macro_precision,
                weighted_recall, macro_recall)

    
    df_pred = df_pred.withColumn(                              # real binary label
        "label_binaria", F.when(F.col("label_index") == 0.0, 0).otherwise(1))
    df_pred = df_pred.withColumn(                              # binary prediction
        "prediccion_binaria", F.when(F.col("prediction") == 0.0, 0).otherwise(1))

    # AUC-ROC: score = attack probability = 1 - P(benign) (index 0).
    @udf(VectorUDT())
    def attack_prob_vec(prob):
        """Returns [P(benign), P(attack)] from the probability vector."""
        p_benign = float(prob[0])                              # probability of class 0
        return Vectors.dense([p_benign, 1.0 - p_benign])       # [neg, pos]

    df_pred = df_pred.withColumn("score_vec", attack_prob_vec(F.col("probability")))
    auc_roc = float(BinaryClassificationEvaluator(
        rawPredictionCol="score_vec", labelCol="label_binaria",
        metricName="areaUnderROC").evaluate(df_pred))
    logger.info("AUC-ROC (benign vs attack): %.4f", auc_roc)

    # Global FPR (binary): false positives over the total of benign samples.
    cm_bin = df_pred.agg(
        F.sum(F.when((F.col("prediccion_binaria") == 1) & (F.col("label_binaria") == 0), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("prediccion_binaria") == 0) & (F.col("label_binaria") == 0), 1).otherwise(0)).alias("tn"),
    ).first()
    fp_g, tn_g = int(cm_bin["fp"]), int(cm_bin["tn"])
    fpr_global = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0.0
    logger.info("Global False Positive Rate: %.4f", fpr_global)

    # (e) CONFUSION MATRIX PER CLASS (with original names)
    # Class name: for known classes use the indexer labels; for any
    # out-of-range index (unseen labels grouped by handleInvalid='keep')
    # use a generic marker.
    def index_name(idx):
        i = int(idx)
        return labels[i] if 0 <= i < len(labels) else "__unseen__"

    cm_rows = (df_pred.groupBy("label_index", "prediction").count()
               .orderBy("label_index", "prediction").collect())
    matrix = []                                                # matrix for the JSON
    for r in cm_rows:
        matrix.append({
            "true_class": index_name(r["label_index"]),
            "label_index": int(r["label_index"]),
            "predicted_class": index_name(r["prediction"]),
            "prediction": int(r["prediction"]),
            "count": int(r["count"]),
        })
    logger.info("Confusion matrix computed (%d non-empty cells).", len(matrix))

    # Per-class metrics (precision/recall/F1) using MulticlassMetrics.
    per_class = []
    logger.info("%-28s | %-9s | %-9s | %-9s", "class", "precision", "recall", "f1")
    for l in present_labels:
        name = index_name(l)
        p, r_, f = mm.precision(l), mm.recall(l), mm.fMeasure(l)
        per_class.append({"class": name, "label_index": int(l),
                          "precision": p, "recall": r_, "f1": f})
        logger.info("%-28s | %-9.4f | %-9.4f | %-9.4f", name, p, r_, f)

    # (f) ANALYSIS OF UNSEEN CLASSES (Web attacks + Infiltration - Thursday)
    # These classes do NOT appear in training (Mon/Wed/Fri). They are
    # identified by the ORIGINAL 'label' column (string), always available
    # in the test Parquet, since the indexer fitted on train does not know them.
    df_unseen = df_pred.filter(
        (F.lower(F.col("label")).contains("web attack")) |
        (F.lower(F.col("label")).contains("infiltration")))
    unseen_analysis = []
    unseen_groups = (df_unseen.groupBy("label")
                     .agg(F.count("*").alias("n"),
                          F.sum(F.col("prediccion_binaria")).alias("detected"))
                     .collect())
    for g in unseen_groups:
        n = int(g["n"])
        detected = int(g["detected"])                          # number predicted as attack
        class_recall = detected / n if n > 0 else 0.0          # recall = detected / total
        unseen_analysis.append({
            "class": g["label"], "num_records": n,
            "detected_as_attack": detected,
            "detection_recall": class_recall,
        })
        logger.info("[UNSEEN] %-28s | n=%-8d | detection_recall=%.4f",
                    g["label"], n, class_recall)

    # (g) DISTRIBUTED SYSTEM METRICS
    system_metrics = {
        "throughput_records_per_second": round(throughput, 2),
        "average_latency_us": round(latency_us, 4),
        "num_partitions": num_partitions,
        "inference_time_seconds": round(inference_time, 4),
    }
    logger.info("System metrics: %s", system_metrics)

    # (h) SAVING METRICS AND PREDICTIONS
    output = {
        "multiclass": {
            "accuracy": accuracy,
            "f1_weighted": weighted_f1, "f1_macro": macro_f1,
            "precision_weighted": weighted_precision, "precision_macro": macro_precision,
            "recall_weighted": weighted_recall, "recall_macro": macro_recall,
        },
        "binary": {
            "auc_roc": auc_roc,
            "fpr_global": fpr_global,
        },
        "confusion_matrix": matrix,
        "metrics_per_class": per_class,
        "unseen_classes_analysis": unseen_analysis,
        "system_metrics": system_metrics,
    }
    try:
        write_json_s3(output, TEST_METRICS_OUT)                 # metrics JSON
        logger.info("Test metrics saved to %s", TEST_METRICS_OUT)
    except Exception as e:
        logger.error("ERROR writing metrics (%s): %s", TEST_METRICS_OUT, e)
        raise

    try:
        (df_pred
         .select("label", "label_index", "prediction", "prediccion_binaria",
                 "label_binaria", "probability")
         .write.mode("overwrite").parquet(PREDICTIONS_OUT))     # Parquet predictions
        logger.info("Predictions saved to %s", PREDICTIONS_OUT)
    except Exception as e:
        logger.error("ERROR writing predictions (%s): %s", PREDICTIONS_OUT, e)
        raise

    df_pred.unpersist()
    logger.info("BLOCK 2 completed successfully.")


# 
#  BLOCK 3: HYPERPARAMETER OPTIMIZATION 

def block3_optimize_rf(spark):
    """Searches for the best hyperparameters with CrossValidator on a
    stratified sample, retrains on the full dataset and generates
    a comparative report against the base model (model_v1)."""
    logger.info("=" * 60)
    logger.info("BLOCK 3 — RF OPTIMIZATION (CrossValidator 30% sample)")
    logger.info("=" * 60)

  
    # LOADING THE FULL TRAINING SET
    try:
        df_train = spark.read.parquet(TRAIN_PROCESSED)
        logger.info("Training Parquet loaded from %s", TRAIN_PROCESSED)
    except Exception as e:
        logger.error("ERROR reading the training Parquet (%s): %s", TRAIN_PROCESSED, e)
        raise

    # (d) 30% STRATIFIED SAMPLE (sampleBy on label_index)
    indices = [r["label_index"] for r in df_train.select("label_index").distinct().collect()]
    fractions = {idx: SAMPLE_FRACTION for idx in indices}       # same fraction per class
    df_sample = df_train.sampleBy("label_index", fractions=fractions, seed=SEED).cache()
    n_sample = df_sample.count()
    logger.info("Stratified sample (30%%): %d records over %d classes.", n_sample, len(indices))

    # (b)(c) GRID + CROSSVALIDATOR
    rf = RandomForestClassifier(
        labelCol="label_index", featuresCol="scaled_features",
        weightCol="class_weight", seed=SEED)
    pipeline = Pipeline(stages=[rf])                            # CV estimator

    grid = (ParamGridBuilder()
            .addGrid(rf.numTrees, [50, 100, 200])               # 3 values
            .addGrid(rf.maxDepth, [5, 10, 15])                  # 3 values
            .addGrid(rf.featureSubsetStrategy, ["sqrt", "log2"])  # 2 values
            .build())                                          # 3×3×2 = 18 combinations
    logger.info("Grid: %d combinations × %d folds = %d trainings.",
                len(grid), NUM_FOLDS, len(grid) * NUM_FOLDS)
  

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label_index", predictionCol="prediction", metricName="f1")

    cv = CrossValidator(
        estimator=pipeline, estimatorParamMaps=grid,
        evaluator=evaluator, numFolds=NUM_FOLDS,
        parallelism=2, seed=SEED)                              # parallelism bounded to the base cluster

    # FITTING THE CROSSVALIDATOR ON THE SAMPLE
    logger.info("Running CrossValidator on the sample...")
    t0 = time.time()
    cv_model = cv.fit(df_sample)
    t_cv = time.time() - t0
    logger.info("CrossValidator completed in %.2f s", t_cv)

    # (e) BEST HYPERPARAMETERS
    best_idx = int(np.argmax(cv_model.avgMetrics))             # index of the best average F1
    best_map = cv_model.getEstimatorParamMaps()[best_idx]      # winning ParamMap
    best = {}                                                  # extract by parameter name
    for param, value in best_map.items():
        best[param.name] = value
    logger.info("Best hyperparameters: numTrees=%s, maxDepth=%s, fss=%s (F1_cv=%.4f)",
                best.get("numTrees"), best.get("maxDepth"),
                best.get("featureSubsetStrategy"), cv_model.avgMetrics[best_idx])

    # (f) RETRAINING ON THE FULL TRAINING SET
    rf_opt = RandomForestClassifier(
        labelCol="label_index", featuresCol="scaled_features", weightCol="class_weight",
        seed=SEED,
        numTrees=int(best.get("numTrees", 100)),
        maxDepth=int(best.get("maxDepth", 10)),
        featureSubsetStrategy=str(best.get("featureSubsetStrategy", "sqrt")))
    pipeline_opt = Pipeline(stages=[rf_opt])

    logger.info("Retraining the best model on the FULL training set...")
    t0 = time.time()
    model_opt = pipeline_opt.fit(df_train)
    t_opt = time.time() - t0
    logger.info("Retraining completed in %.2f s", t_opt)

    # (g) RE-EVALUATION ON THE TEST SET
    try:
        df_test = spark.read.parquet(TEST_PROCESSED)
        logger.info("Test Parquet loaded from %s", TEST_PROCESSED)
    except Exception as e:
        logger.error("ERROR reading the test Parquet (%s): %s", TEST_PROCESSED, e)
        raise

    df_pred_opt = model_opt.transform(df_test)
    ev = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction")
    f1_opt = float(ev.setMetricName("f1").evaluate(df_pred_opt))
    acc_opt = float(ev.setMetricName("accuracy").evaluate(df_pred_opt))
    logger.info("Optimized TEST -> F1_weighted=%.4f | Accuracy=%.4f", f1_opt, acc_opt)

    # (h) SAVING THE BEST MODEL + COMPARATIVE REPORT
    try:
        model_opt.write().overwrite().save(MODEL_OPT_OUT)       # serialize optimized model
        logger.info("Optimized model saved to %s", MODEL_OPT_OUT)
    except Exception as e:
        logger.error("ERROR saving the optimized model (%s): %s", MODEL_OPT_OUT, e)
        raise

    # Metrics of the base model (if they exist).
    train_base = read_json_s3(TRAIN_METRICS) or {}
    test_base  = read_json_s3(TEST_METRICS_OUT) or {}
    f1_base  = test_base.get("multiclass", {}).get("f1_weighted")
    acc_base = test_base.get("multiclass", {}).get("accuracy")
    t_base   = train_base.get("training_time_seconds")

    report = {
        "base_model": {
            "hyperparameters": {
                "numTrees": train_base.get("num_trees", 100),
                "maxDepth": train_base.get("max_depth", 10),
                "featureSubsetStrategy": train_base.get("feature_subset_strategy", "sqrt"),
            },
            "f1_weighted": f1_base,
            "accuracy": acc_base,
            "training_time_s": t_base,
        },
        "optimized_model": {
            "hyperparameters": {
                "numTrees": int(best.get("numTrees", 100)),
                "maxDepth": int(best.get("maxDepth", 10)),
                "featureSubsetStrategy": str(best.get("featureSubsetStrategy", "sqrt")),
            },
            "f1_weighted": f1_opt,
            "accuracy": acc_opt,
            "training_time_s": round(t_opt, 4),
            "avg_cv_f1": float(cv_model.avgMetrics[best_idx]),
            "crossvalidator_time_s": round(t_cv, 4),
        },
        "search_config": {
            "numTrees": [50, 100, 200], "maxDepth": [5, 10, 15],
            "featureSubsetStrategy": ["sqrt", "log2"],
            "combinations": len(grid), "folds": NUM_FOLDS,
            "sample_fraction": SAMPLE_FRACTION,
        },
    }
    try:
        write_json_s3(report, OPT_REPORT_OUT)
        logger.info("Optimization report saved to %s", OPT_REPORT_OUT)
    except Exception as e:
        logger.error("ERROR writing the report (%s): %s", OPT_REPORT_OUT, e)
        raise

    # (i) COMPARATIVE TABLE IN CONSOLE
    def _fmt(x):
        """Formats a number or returns 'N/D' if it is None."""
        return f"{x:.3f}" if isinstance(x, (int, float)) else "N/D"

    logger.info("| Configuration     | F1 weighted | Accuracy | Training time    |")
    logger.info("|-------------------|-------------|----------|------------------|")
    logger.info("| Base model        |    %-6s   |  %-6s  |     %8s s     |",
                _fmt(f1_base), _fmt(acc_base), _fmt(t_base))
    logger.info("| Optimized model   |    %-6s   |  %-6s  |     %8s s     |",
                _fmt(f1_opt), _fmt(acc_opt), _fmt(round(t_opt, 3)))

    df_sample.unpersist()
    logger.info("BLOCK 3 completed successfully.")



def main():
    """Runs in sequence the three blocks of the RF pipeline with
    a single shared SparkSession."""
    spark = build_spark()
    logger.info("SparkSession 'ProjectAnomalies-RandomForest' started.")
    logger.info("Starting the complete Random Forest pipeline (3 blocks).")

    try:
        block1_train_rf(spark)          # Base model training
        block2_evaluate_rf(spark)       # Batch evaluation on test
        block3_optimize_rf(spark)       # Hyperparameter optimization
    except Exception as e:
        logger.error("ERROR in the RF pipeline: %s", e)
        spark.stop()
        raise

    spark.stop()
    logger.info("Random Forest pipeline finished successfully (3/3 blocks).")


if __name__ == "__main__":
    main()