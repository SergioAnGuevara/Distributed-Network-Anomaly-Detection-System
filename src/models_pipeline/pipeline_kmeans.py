
"""
pipeline_kmeans.py

PURPOSE:
    Runs in sequence the four blocks of the unsupervised KMeans
    anomaly detection pipeline over the CIC-IDS2017 dataset:

    BLOCK 1 — TRAINING :
        Trains KMeans ONLY on benign traffic (Monday). The model
        learns the structure of normal traffic; flows far from the
        assigned centroid are candidates for anomalies.

    BLOCK 2 — THRESHOLD CALIBRATION :
        Finds the Euclidean distance threshold that maximizes F1 on
        the labeled training data (Wednesday/Friday), without
        touching the test set (Tuesday/Thursday).

    BLOCK 3 — BATCH EVALUATION :
        Evaluates model_v1 on the test set with the calibrated
        threshold. Reports binary metrics, distance distribution per
        class, analysis of UNSEEN classes and system metrics.

    BLOCK 4 — OPTIMIZATION :
        Manual hyperparameter search (KMeans is not compatible with
        MLlib's CrossValidator). Selects by Silhouette, retrains on
        the full benign set, recalibrates the threshold and generates
        a comparative report against the base model.

INPUTS (S3 input paths):
    - s3://proyecto-anomalias-emr/data/cic-ids2017/train/processed/
    - s3://proyecto-anomalias-emr/data/cic-ids2017/test/processed/
    - s3://proyecto-anomalias-emr/models/random_forest_pipeline/preprocessors/indexer/

OUTPUTS (S3 output paths):
    - .../models/kmeans_pipeline/model_v1/
    - .../models/kmeans_pipeline/model_optimized/
    - .../models/kmeans_pipeline/threshold/threshold_v1.json
    - .../models/kmeans_pipeline/threshold/threshold_opt.json
    - .../results/kmeans_evaluation/train_metrics_kmeans.json
    - .../results/kmeans_evaluation/test_metrics_kmeans.json
    - .../results/kmeans_evaluation/predictions_kmeans/       (Parquet)
    - .../results/kmeans_evaluation/optimization_report_kmeans.json

"""


import time                                          # time measurement
import json                                          # metrics serialization
import logging                                       # structured logging
from urllib.parse import urlparse                    # parse s3:// paths

import boto3                                         # JSON I/O with S3
import numpy as np                                   # threshold grid

from pyspark.sql import SparkSession                 # Spark entry point
from pyspark.sql.functions import (                  # SQL functions
    udf, col, when,
    sum as Fsum, mean as Fmean, stddev as Fstddev, count as Fcount,
)
from pyspark.sql.types import DoubleType            # type of the distance UDF
from pyspark.ml import Pipeline, PipelineModel       # MLlib pipeline
from pyspark.ml.clustering import KMeans             # clustering algorithm
from pyspark.ml.feature import StringIndexerModel, StandardScalerModel
from pyspark.ml.linalg import Vectors, VectorUDT     # vector for AUC
from pyspark.ml.evaluation import (
    ClusteringEvaluator,
    BinaryClassificationEvaluator,
)



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("PipelineKMeans")



# GLOBAL CONFIGURATION 

BUCKET           = "s3://project-anomalies-emr"                         # root bucket
TRAIN_RAW        = f"{BUCKET}/data/cic-ids2017/train/"                   # training
TEST_RAW         = f"{BUCKET}/data/cic-ids2017/test/"                    # test
RF_MODELS_PATH   = f"{BUCKET}/models/random_forest_pipeline/"           # shared preprocessors (RF)
KM_MODELS_PATH   = f"{BUCKET}/models/kmeans_pipeline/"                  # KMeans models
RESULTS_PATH     = f"{BUCKET}/results/kmeans_evaluation/"               # results

# Derived paths.
TRAIN_PROCESSED  = f"{TRAIN_RAW}processed/"                             # training Parquet
TEST_PROCESSED   = f"{TEST_RAW}processed/"                              # test Parquet
SCALER_PATH      = f"{RF_MODELS_PATH}preprocessors/scaler/"            # StandardScaler fitted in gestion_datos.py
INDEXER_PATH     = f"{RF_MODELS_PATH}preprocessors/indexer/"           # fitted StringIndexer
MODEL_V1_OUT     = f"{KM_MODELS_PATH}model_v1/"                        # base model
MODEL_OPT_OUT    = f"{KM_MODELS_PATH}model_optimized/"                 # optimized model
THRESHOLD_V1     = f"{KM_MODELS_PATH}threshold/threshold_v1.json"      # calibrated base threshold
THRESHOLD_OPT    = f"{KM_MODELS_PATH}threshold/threshold_opt.json"     # optimized threshold
TRAIN_METRICS    = f"{RESULTS_PATH}train_metrics_kmeans.json"          # training metrics
TEST_METRICS_OUT = f"{RESULTS_PATH}test_metrics_kmeans.json"           # test metrics
PREDICTIONS_OUT  = f"{RESULTS_PATH}predictions_kmeans/"                # Parquet predictions
OPT_REPORT_OUT   = f"{RESULTS_PATH}optimization_report_kmeans.json"    # comparative report


K_CLUSTERS       = 10                                                   # number of base clusters
INIT_MODE        = "k-means||"                                          # stable parallel initialization
MAX_ITER         = 50                                                   # maximum iterations
TOL              = 1e-4                                                 # convergence tolerance
SEED             = 42                                                   # reproducible seed


SAMPLE_FRACTION    = 0.30                                               # 30% sample for the search


EXECUTOR_INSTANCES = 2                                                  # number of executors (base cluster)
EXECUTOR_CORES     = 2                                                  # cores per executor
SHUFFLE_PARTITIONS = 16                                                 # shuffle partitions



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


def build_spark():
    """SparkSession with ALL cluster parameters via .config()."""
    spark = (
        SparkSession.builder
        .appName("ProjectAnomalies-KMeans")                                      # exact required name
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
        .config("spark.kryo.registrationRequired", "false")                       # avoids warnings with MLlib classes
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



def build_distance_udf(spark, kmeans_model):
    """Builds and returns a UDF of Euclidean distance to the centroid
    using the centers of the given model (broadcasted)."""
    centers_list = [c.tolist() for c in kmeans_model.clusterCenters()]
    centers_broadcast = spark.sparkContext.broadcast(centers_list)

    @udf(DoubleType())
    def _distance_udf(features, cluster_id):
        import numpy as np                                              # import inside the UDF
        centroid   = np.array(centers_broadcast.value[cluster_id])
        vector     = np.array(features.toArray())
        diff       = vector - centroid
        return float(np.sqrt(np.dot(diff, diff)))

    return _distance_udf


@udf(VectorUDT())
def distance_to_vector_udf(distance):
    """Converts the scalar distance into a vector [-d, d] for the
    BinaryClassificationEvaluator (AUC-ROC)."""
    return Vectors.dense([-distance, distance])


def confusion_at_threshold(df, threshold):
    """Computes TP, FP, FN, TN for a given threshold in a SINGLE pass.
    - prediction = 1 if distancia_centroide > threshold (anomalous)
    - label_binaria = 1 if the row is a real attack
    """
    row = df.agg(
        Fsum(when((col("distancia_centroide") > threshold) & (col("label_binaria") == 1), 1).otherwise(0)).alias("tp"),
        Fsum(when((col("distancia_centroide") > threshold) & (col("label_binaria") == 0), 1).otherwise(0)).alias("fp"),
        Fsum(when((col("distancia_centroide") <= threshold) & (col("label_binaria") == 1), 1).otherwise(0)).alias("fn"),
        Fsum(when((col("distancia_centroide") <= threshold) & (col("label_binaria") == 0), 1).otherwise(0)).alias("tn"),
    ).first()
    return int(row["tp"]), int(row["fp"]), int(row["fn"]), int(row["tn"])


def metrics_from_confusion(tp, fp, fn, tn):
    """Derives Precision, Recall, F1 and FPR from the confusion matrix."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return precision, recall, f1, fpr


def _calibrate_threshold_internal(spark, model, full_train_df):
    """Finds the threshold that maximizes F1 (tie -> lowest FPR) over the
    full training DataFrame. Function shared between blocks 2 and 4."""
    kmeans_model = model.stages[-1]
    distance_udf = build_distance_udf(spark, kmeans_model)

    df = model.transform(full_train_df)                               # cluster_id
    df = df.withColumn("distancia_centroide",
                       distance_udf(col("scaled_features"), col("cluster_id")))
    df = df.withColumn("label_binaria",
                       when(col("label_index") == 0.0, 0).otherwise(1))
    df = df.cache()
    df.count()                                                         # materialize

    # Percentiles over benign data to build the threshold grid.
    df_benign = df.filter(col("label_binaria") == 0)
    p90, p95, p97, p99, p999 = df_benign.approxQuantile(
        "distancia_centroide", [0.90, 0.95, 0.97, 0.99, 0.999], 0.001)

    candidates = sorted(set([p90, p95, p97, p99, p999] + list(np.linspace(p90, p999, 20))))
    logger.info("Benign percentiles -> P90=%.4f P95=%.4f P97=%.4f P99=%.4f P99.9=%.4f",
                p90, p95, p97, p99, p999)
    logger.info("Total threshold candidates to evaluate: %d", len(candidates))

    best = None
    for threshold in candidates:
        tp, fp, fn, tn = confusion_at_threshold(df, float(threshold))
        precision, recall, f1, fpr = metrics_from_confusion(tp, fp, fn, tn)
        cand = {"threshold": float(threshold), "precision": precision,
                "recall": recall, "f1": f1, "fpr": fpr}
        if (best is None or cand["f1"] > best["f1"]
                or (cand["f1"] == best["f1"] and cand["fpr"] < best["fpr"])):
            best = cand

    # AUC-ROC global (independent of the threshold).
    df = df.withColumn("raw_prediction_vec", distance_to_vector_udf(col("distancia_centroide")))
    auc = float(BinaryClassificationEvaluator(
        rawPredictionCol="raw_prediction_vec", labelCol="label_binaria",
        metricName="areaUnderROC").evaluate(df))

    df.unpersist()
    best["auc_roc"] = auc
    best["percentiles"] = {"p90": p90, "p95": p95, "p97": p97, "p99": p99, "p999": p999}
    return best


def _evaluate_on_test_internal(spark, model, df_test, threshold):
    """Computes binary metrics of the model on the test set with
    the given threshold. Function shared between blocks 3 and 4."""
    kmeans_model = model.stages[-1]
    distance_udf = build_distance_udf(spark, kmeans_model)

    df = model.transform(df_test)
    df = df.withColumn("distancia_centroide",
                       distance_udf(col("scaled_features"), col("cluster_id")))
    df = df.withColumn("prediccion_binaria",
                       (col("distancia_centroide") > threshold).cast("int"))
    df = df.withColumn("label_binaria",
                       when(col("label_index") == 0.0, 0).otherwise(1))
    df = df.cache()
    df.count()

    row = df.agg(
        Fsum(when((col("prediccion_binaria") == 1) & (col("label_binaria") == 1), 1).otherwise(0)).alias("tp"),
        Fsum(when((col("prediccion_binaria") == 1) & (col("label_binaria") == 0), 1).otherwise(0)).alias("fp"),
        Fsum(when((col("prediccion_binaria") == 0) & (col("label_binaria") == 1), 1).otherwise(0)).alias("fn"),
        Fsum(when((col("prediccion_binaria") == 0) & (col("label_binaria") == 0), 1).otherwise(0)).alias("tn"),
    ).first()
    tp, fp, fn, tn = int(row["tp"]), int(row["fp"]), int(row["fn"]), int(row["tn"])
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision, recall, f1, fpr = metrics_from_confusion(tp, fp, fn, tn)

    df = df.withColumn("raw_prediction_vec", distance_to_vector_udf(col("distancia_centroide")))
    auc = float(BinaryClassificationEvaluator(
        rawPredictionCol="raw_prediction_vec", labelCol="label_binaria",
        metricName="areaUnderROC").evaluate(df))

    df.unpersist()
    return {"auc_roc": auc, "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "fpr": fpr}


# BLOCK 1: BASE MODEL TRAINING 
def block1_train_kmeans(spark):
    """Trains KMeans ONLY on benign traffic (label_index == 0.0).
    Saves the base model and its intrinsic metrics to S3."""
    logger.info("=" * 60)
    logger.info("BLOCK 1 — KMeans TRAINING (model_v1, benign only)")
    logger.info("=" * 60)

  
    # (c) LOADING OF THE PROCESSED TRAINING DATAFRAME

    try:
        df_train = spark.read.parquet(TRAIN_PROCESSED)
        logger.info("Training Parquet loaded from %s", TRAIN_PROCESSED)
    except Exception as e:
        logger.error("ERROR reading the training Parquet (%s): %s", TRAIN_PROCESSED, e)
        raise


    # (d) CRITICAL FILTERING: KMeans learns ONLY on benign traffic

    df_train_benign = df_train.filter(col("label_index") == 0.0)       # only majority benign class
    num_benign = df_train_benign.count()                               # count of benign records
    logger.info("Benign records for KMeans training: %d", num_benign)

  
    # (e) VERIFY / GENERATE THE scaled_features COLUMN
   
    if "scaled_features" in df_train_benign.columns:                   # normal path: it already comes scaled
        logger.info("Column 'scaled_features' present: reusing the scaling from gestion_datos.py.")
    else:
        # Alternative path: load the RF StandardScaler and apply it.
        logger.info("Column 'scaled_features' missing: the RF StandardScaler will be loaded to generate it.")
        try:
            scaler_model = StandardScalerModel.load(SCALER_PATH)
            df_train_benign = scaler_model.transform(df_train_benign)
            logger.info("StandardScaler applied from %s", SCALER_PATH)
        except Exception as e:
            logger.error("ERROR loading/applying the StandardScaler (%s): %s", SCALER_PATH, e)
            raise

    # Persist the benign subset: reused in fit + Silhouette + UDF.
    df_train_benign = df_train_benign.cache()
    df_train_benign.count()                                            # materialize the cache

  
    
   
    kmeans = KMeans(
        k=K_CLUSTERS,                       # 10 initial clusters
        initMode=INIT_MODE,                 # "k-means||": parallel and stable initialization
        maxIter=MAX_ITER,                   # Lloyd iteration cap
        tol=TOL,                            # convergence tolerance
        seed=SEED,                          # reproducibility
        featuresCol="scaled_features",      # input vector already scaled
        predictionCol="cluster_id",         # assigned cluster column
        distanceMeasure="euclidean",        # distance metric
    )
    pipeline = Pipeline(stages=[kmeans])


    # (h) DISTRIBUTED TRAINING WITH TIME MEASUREMENT
    
    logger.info("Starting KMeans training (k=%d, maxIter=%d) ONLY on benign data...",
                K_CLUSTERS, MAX_ITER)
    t_start = time.time()
    model = pipeline.fit(df_train_benign)
    t_end = time.time()
    training_time = t_end - t_start
    logger.info("Training completed in %.2f s", training_time)

    kmeans_model = model.stages[-1]                                    # fitted KMeansModel


    # (i) MODEL SAVING

    try:
        model.write().overwrite().save(MODEL_V1_OUT)
        logger.info("KMeans model saved to %s", MODEL_V1_OUT)
    except Exception as e:
        logger.error("ERROR saving the model to %s: %s", MODEL_V1_OUT, e)
        raise


    # (j) INTRINSIC CLUSTERING METRICS

    wssse = float(kmeans_model.summary.trainingCost)                   # training cost 
    logger.info("WSSSE (trainingCost): %.4f", wssse)

    cluster_sizes = [int(s) for s in kmeans_model.summary.clusterSizes]
    logger.info("Cluster sizes: %s", cluster_sizes)

    centers = kmeans_model.clusterCenters()
    logger.info("Number of centroids: %d (dimension = %d)", len(centers), len(centers[0]))

    # Transform the benign set for Silhouette and the distance UDF.
    df_train_pred = model.transform(df_train_benign)                   # adds cluster_id

    silhouette_evaluator = ClusteringEvaluator(
        featuresCol="scaled_features", predictionCol="cluster_id",
        metricName="silhouette", distanceMeasure="squaredEuclidean")
    silhouette = float(silhouette_evaluator.evaluate(df_train_pred))
    logger.info("Silhouette score (squaredEuclidean): %.4f", silhouette)


    # (k) EUCLIDEAN DISTANCE UDF TO THE ASSIGNED CENTROID

    distance_udf = build_distance_udf(spark, kmeans_model)
    df_train_pred = df_train_pred.withColumn(
        "distancia_centroide",
        distance_udf(col("scaled_features"), col("cluster_id")),
    )
    dist_min, dist_max = df_train_pred.selectExpr(
        "min(distancia_centroide)", "max(distancia_centroide)"
    ).first()
    logger.info("Distance to centroid (benign) -> min=%.4f, max=%.4f", dist_min, dist_max)

    
    # (l) SAVING TRAINING METRICS

    metrics = {
        "training_time_seconds": round(training_time, 4),
        "k": K_CLUSTERS,
        "init_mode": INIT_MODE,
        "max_iter": MAX_ITER,
        "wssse": wssse,
        "silhouette_score": silhouette,
        "cluster_sizes": cluster_sizes,
        "num_training_records": int(num_benign),
        "num_partitions": SHUFFLE_PARTITIONS,
        "executor_instances": EXECUTOR_INSTANCES,
        "cluster_cores_total": EXECUTOR_INSTANCES * EXECUTOR_CORES,
    }
    try:
        write_json_s3(metrics, TRAIN_METRICS)
        logger.info("Training metrics saved to %s", TRAIN_METRICS)
    except Exception as e:
        logger.error("ERROR writing metrics to %s: %s", TRAIN_METRICS, e)
        raise

    df_train_benign.unpersist()
    logger.info("BLOCK 1 completed successfully.")



#  BLOCK 2: THRESHOLD CALIBRATION 


def block2_calibrate_kmeans(spark):
    """Calibrates the distance threshold that maximizes F1 on the
    labeled training data (Wednesday/Friday of attacks).
    The test data (Tuesday/Thursday) is NOT touched here."""
    logger.info("=" * 60)
    logger.info("BLOCK 2 — THRESHOLD CALIBRATION (model_v1, labeled train)")
    logger.info("=" * 60)

    
    # (c) LOADING THE MODEL AND THE FULL TRAINING DATAFRAME
    

    try:
        model = PipelineModel.load(MODEL_V1_OUT)
        logger.info("KMeans model loaded from %s", MODEL_V1_OUT)
    except Exception as e:
        logger.error("ERROR loading the model (%s): %s", MODEL_V1_OUT, e)
        raise

    try:
        df_train = spark.read.parquet(TRAIN_PROCESSED)          # full Parquet (benign + attack)
        logger.info("Training Parquet loaded from %s", TRAIN_PROCESSED)
    except Exception as e:
        logger.error("ERROR reading the training Parquet (%s): %s", TRAIN_PROCESSED, e)
        raise

    
    # (e)-(j) CALIBRATION VIA THE INTERNAL HELPER
    
    logger.info("Calibrating threshold with a grid of thresholds (>= 20 candidates)...")
    best = _calibrate_threshold_internal(spark, model, df_train)
    best_threshold = best["threshold"]

    logger.info("CALIBRATION SUMMARY -> threshold=%.4f | F1=%.4f | P=%.4f | R=%.4f | FPR=%.4f | AUC=%.4f",
                best_threshold, best["f1"], best["precision"], best["recall"],
                best["fpr"], best["auc_roc"])

    
    # (k) SAVING THE THRESHOLD AND METRICS
    
    p = best["percentiles"]
    output = {
        "threshold_optimal": best_threshold,
        "f1_at_threshold":   best["f1"],
        "precision":         best["precision"],
        "recall":            best["recall"],
        "fpr":               best["fpr"],
        "auc_roc":           best["auc_roc"],
        "percentile_90":  p["p90"],
        "percentile_95":  p["p95"],
        "percentile_97":  p["p97"],
        "percentile_99":  p["p99"],
        "percentile_999": p["p999"],
        "calibration_records": None,                          # calculated internally
        "calibration_days": ["Wednesday", "Friday"],
    }
    try:
        write_json_s3(output, THRESHOLD_V1)
        logger.info("Calibrated threshold saved to %s", THRESHOLD_V1)
    except Exception as e:
        logger.error("ERROR writing the threshold to %s: %s", THRESHOLD_V1, e)
        raise

    logger.info("BLOCK 2 completed successfully.")



#  BLOCK 3: BATCH EVALUATION OF THE BASE MODEL 


def block3_evaluate_kmeans(spark):
    """Evaluates model_v1 on the test set (Tuesday + Thursday)
    with the calibrated threshold. Reports binary metrics, distance
    distribution per class and analysis of UNSEEN classes."""
    logger.info("=" * 60)
    logger.info("BLOCK 3 — KMeans BATCH EVALUATION (model_v1 on test)")
    logger.info("=" * 60)

    
    # (b) LOADING THE MODEL, TEST DATAFRAME, THRESHOLD AND INDEXER
    
    try:
        model = PipelineModel.load(MODEL_V1_OUT)
        logger.info("KMeans model loaded from %s", MODEL_V1_OUT)
    except Exception as e:
        logger.error("ERROR loading the model (%s): %s", MODEL_V1_OUT, e)
        raise

    try:
        df_test = spark.read.parquet(TEST_PROCESSED)
        logger.info("Test Parquet loaded from %s", TEST_PROCESSED)
    except Exception as e:
        logger.error("ERROR reading the test Parquet (%s): %s", TEST_PROCESSED, e)
        raise

    try:
        threshold_json = read_json_s3(THRESHOLD_V1)
        threshold_optimal = float(threshold_json["threshold_optimal"])
        logger.info("Calibrated threshold read: %.4f", threshold_optimal)
    except Exception as e:
        logger.error("ERROR reading the threshold (%s): %s", THRESHOLD_V1, e)
        raise

    try:
        indexer_model = StringIndexerModel.load(INDEXER_PATH)
        labels = list(indexer_model.labels)                           # labels[i] = name of index i
        logger.info("StringIndexer loaded: %d known classes.", len(labels))
    except Exception as e:
        logger.error("ERROR loading the StringIndexer (%s): %s", INDEXER_PATH, e)
        raise

    kmeans_model = model.stages[-1]

    
    # (c) DISTANCE UDF (centers of the loaded model)
    
    distance_udf = build_distance_udf(spark, kmeans_model)

    
    # (d) DISTRIBUTED INFERENCE WITH TIME MEASUREMENT
    
    num_partitions = df_test.rdd.getNumPartitions()
    logger.info("Starting batch inference (partitions=%d)...", num_partitions)

    t_start = time.time()
    df_pred = model.transform(df_test)                                 # assigns cluster_id
    df_pred = df_pred.withColumn(                                      # adds distance to the centroid
        "distancia_centroide",
        distance_udf(col("scaled_features"), col("cluster_id")),
    )
    # (e) Binary classification with the calibrated threshold.
    df_pred = df_pred.withColumn(
        "prediccion_binaria",
        (col("distancia_centroide") > threshold_optimal).cast("int"),
    )
    df_pred = df_pred.withColumn(
        "label_binaria",
        when(col("label_index") == 0.0, 0).otherwise(1),
    )
    df_pred = df_pred.cache()
    num_records = df_pred.count()                                      # forces the inference
    t_end = time.time()

    inference_time = t_end - t_start
    throughput = num_records / inference_time if inference_time > 0 else 0.0
    latency_us = (inference_time / num_records * 1_000_000) if num_records > 0 else 0.0
    logger.info("Inference: %d records in %.2f s | throughput=%.1f rec/s | latency=%.2f µs/rec",
                num_records, inference_time, throughput, latency_us)

    
    # (f) BINARY METRICS
    
    cm_row = df_pred.agg(
        Fsum(when((col("prediccion_binaria") == 1) & (col("label_binaria") == 1), 1).otherwise(0)).alias("tp"),
        Fsum(when((col("prediccion_binaria") == 1) & (col("label_binaria") == 0), 1).otherwise(0)).alias("fp"),
        Fsum(when((col("prediccion_binaria") == 0) & (col("label_binaria") == 1), 1).otherwise(0)).alias("fn"),
        Fsum(when((col("prediccion_binaria") == 0) & (col("label_binaria") == 0), 1).otherwise(0)).alias("tn"),
    ).first()
    tp, fp, fn, tn = int(cm_row["tp"]), int(cm_row["fp"]), int(cm_row["fn"]), int(cm_row["tn"])

    total = tp + fp + fn + tn
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0               # critical FPR for IDS

    # AUC-ROC using the distance as a continuous score.
    df_pred = df_pred.withColumn(
        "raw_prediction_vec",
        distance_to_vector_udf(col("distancia_centroide")),
    )
    auc_roc = float(BinaryClassificationEvaluator(
        rawPredictionCol="raw_prediction_vec", labelCol="label_binaria",
        metricName="areaUnderROC").evaluate(df_pred))

    logger.info("BINARY METRICS -> AUC=%.4f | Acc=%.4f | P=%.4f | R(DR)=%.4f | F1=%.4f | FPR=%.4f",
                auc_roc, accuracy, precision, recall, f1, fpr)
    logger.info("Confusion matrix -> TP=%d FP=%d FN=%d TN=%d", tp, fp, fn, tn)

    
    # (g) DISTANCE DISTRIBUTION PER REAL CLASS
    
    df_by_class = (
        df_pred.groupBy("label_index")
        .agg(
            Fmean("distancia_centroide").alias("mean_dist"),
            Fstddev("distancia_centroide").alias("std_dist"),
            Fcount("*").alias("num_records"),
            Fmean(col("prediccion_binaria").cast("double")).alias("detection_rate"),
        )
        .orderBy("label_index")
    )

    class_rows = df_by_class.collect()
    per_class_table = []
    logger.info("%-28s | %-10s | %-10s | %-12s | %-10s",
                "class", "mean_dist", "std_dist", "num_rec", "det_rate")
    for f in class_rows:
        idx = int(f["label_index"])
        name = labels[idx] if 0 <= idx < len(labels) else f"idx_{idx}"
        record = {
            "class": name, "label_index": idx,
            "mean_dist": float(f["mean_dist"]) if f["mean_dist"] is not None else None,
            "std_dist": float(f["std_dist"]) if f["std_dist"] is not None else None,
            "num_records": int(f["num_records"]),
            "detection_rate": float(f["detection_rate"]) if f["detection_rate"] is not None else None,
        }
        per_class_table.append(record)
        logger.info("%-28s | %-10.4f | %-10.4f | %-12d | %-10.4f",
                    name,
                    record["mean_dist"] or 0.0,
                    record["std_dist"] or 0.0,
                    record["num_records"],
                    record["detection_rate"] or 0.0)

    
    # (h) ANALYSIS OF UNSEEN CLASSES (Thursday: Web attacks + Infiltration)
    
    unseen_indices = [i for i, name in enumerate(labels)
                      if ("web attack" in name.lower()) or ("infiltration" in name.lower())]
    unseen_names = [labels[i] for i in unseen_indices]
    logger.info("Classes unseen in training/calibration: %s", unseen_names)

    unseen_analysis = []
    if unseen_indices:
        df_unseen = df_pred.filter(col("label_index").isin([float(i) for i in unseen_indices]))
        unseen_rows = (
            df_unseen.groupBy("label_index")
            .agg(
                Fmean("distancia_centroide").alias("mean_dist"),
                Fcount("*").alias("num_records"),
                Fmean(col("prediccion_binaria").cast("double")).alias("detection_rate"),
            )
            .collect()
        )
        for f in unseen_rows:
            idx = int(f["label_index"])
            name = labels[idx]
            detection_rate = float(f["detection_rate"]) if f["detection_rate"] is not None else 0.0
            mean = float(f["mean_dist"]) if f["mean_dist"] is not None else 0.0
            unseen_analysis.append({
                "class": name, "detection_rate": detection_rate,
                "mean_dist": mean, "num_records": int(f["num_records"]),
            })
            status = "fully detects them" if detection_rate >= 0.5 else "partial/low detection"
            logger.info("[UNSEEN] %s -> det_rate=%.4f | mean_dist=%.4f | %s",
                        name, detection_rate, mean, status)
    else:
        logger.info("No 'Web Attack'/'Infiltration' classes found in the indexer.")

    
    # (i) DISTRIBUTED SYSTEM METRICS
    
    system_metrics = {
        "throughput_records_per_second": round(throughput, 2),
        "average_latency_us": round(latency_us, 4),
        "num_partitions": num_partitions,
        "inference_time_seconds": round(inference_time, 4),
    }
    logger.info("System metrics: %s", system_metrics)

    
    # (j) SAVING METRICS AND PREDICTIONS
    
    output = {
        "threshold_used": threshold_optimal,
        "binary": {
            "auc_roc": auc_roc, "accuracy": accuracy,
            "precision": precision, "recall_detection_rate": recall,
            "f1": f1, "fpr": fpr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "total_records": total,
        },
        "per_class_distribution": per_class_table,
        "unseen_classes_analysis": unseen_analysis,
        "system_metrics": system_metrics,
    }
    try:
        write_json_s3(output, TEST_METRICS_OUT)
        logger.info("Test metrics saved to %s", TEST_METRICS_OUT)
    except Exception as e:
        logger.error("ERROR writing metrics to %s: %s", TEST_METRICS_OUT, e)
        raise

    try:
        (df_pred
         .select("distancia_centroide", "cluster_id", "prediccion_binaria",
                 "label_binaria", "label_index")
         .write.mode("overwrite").parquet(PREDICTIONS_OUT))
        logger.info("Predictions saved to %s", PREDICTIONS_OUT)
    except Exception as e:
        logger.error("ERROR writing predictions to %s: %s", PREDICTIONS_OUT, e)
        raise

    df_pred.unpersist()
    logger.info("BLOCK 3 completed successfully.")



#  BLOCK 4: HYPERPARAMETER OPTIMIZATION 


def block4_optimize_kmeans(spark):
    """Manual grid search (8 combinations) over a 30% sample of the
    benign data. Selects by Silhouette (tie -> lowest WSSSE), retrains
    on the full benign set, recalibrates the threshold and generates
    the comparative report."""
    logger.info("=" * 60)
    logger.info("BLOCK 4 — KMeans OPTIMIZATION (manual grid search)")
    logger.info("=" * 60)

    # WARNING: with spark.executor.instances=2 (base cluster), the grid
    # search is expensive. A 30% sample of the benign data is used for the
    # search; the winning model is retrained on the full benign set.
    # KMeans is NOT compatible with MLlib's CrossValidator -> manual search.

    
    # (b) LOADING THE BENIGN DATA + 30% SAMPLE
    
    try:
        df_train = spark.read.parquet(TRAIN_PROCESSED)
        logger.info("Training Parquet loaded from %s", TRAIN_PROCESSED)
    except Exception as e:
        logger.error("ERROR reading the training Parquet (%s): %s", TRAIN_PROCESSED, e)
        raise

    df_benign = df_train.filter(col("label_index") == 0.0).cache()
    n_benign = df_benign.count()
    df_sample = df_benign.sample(withReplacement=False,
                                 fraction=SAMPLE_FRACTION, seed=SEED).cache()
    n_sample = df_sample.count()
    logger.info("Total benign=%d | sample (30%%)=%d", n_benign, n_sample)

    
    # (c) MANUAL HYPERPARAMETER GRID (8 combinations)
    
    param_grid = [
        {"k": 5,  "maxIter": 20, "tol": 1e-4},
        {"k": 10, "maxIter": 20, "tol": 1e-4},
        {"k": 15, "maxIter": 20, "tol": 1e-4},
        {"k": 20, "maxIter": 50, "tol": 1e-4},
        {"k": 30, "maxIter": 50, "tol": 1e-4},
        {"k": 10, "maxIter": 50, "tol": 1e-5},
        {"k": 15, "maxIter": 50, "tol": 1e-5},
        {"k": 20, "maxIter": 50, "tol": 1e-5},
    ]
    # Total: 8 combinations. Each training runs on 30% of the benign data.
    logger.info("Search grid: %d combinations on the 30%% sample.", len(param_grid))

    silhouette_evaluator = ClusteringEvaluator(
        featuresCol="scaled_features", predictionCol="cluster_id",
        metricName="silhouette", distanceMeasure="squaredEuclidean")

    
    # (d) SEARCH LOOP
    
    grid_results = []
    for i, params in enumerate(param_grid, start=1):
        logger.info("[%d/%d] Training k=%d maxIter=%d tol=%g on the sample...",
                    i, len(param_grid), params["k"], params["maxIter"], params["tol"])
        km = KMeans(
            k=params["k"], maxIter=params["maxIter"], tol=params["tol"],
            seed=SEED, initMode="k-means||",
            featuresCol="scaled_features", predictionCol="cluster_id",
            distanceMeasure="euclidean")
        pipe = Pipeline(stages=[km])

        t0 = time.time()
        model_tmp = pipe.fit(df_sample)
        t_train = time.time() - t0

        km_tmp = model_tmp.stages[-1]
        wssse = float(km_tmp.summary.trainingCost)

        df_eval = model_tmp.transform(df_sample)
        silhouette = float(silhouette_evaluator.evaluate(df_eval))

        # Mean/std deviation of distances over the sample.
        distance_udf = build_distance_udf(spark, km_tmp)
        df_eval = df_eval.withColumn(
            "distancia_centroide",
            distance_udf(col("scaled_features"), col("cluster_id")))
        stats = df_eval.agg(
            Fmean("distancia_centroide").alias("m"),
            Fstddev("distancia_centroide").alias("s")).first()
        mean_dist = float(stats["m"]) if stats["m"] is not None else 0.0
        dist_std  = float(stats["s"]) if stats["s"] is not None else 0.0

        result = {
            "k": params["k"], "maxIter": params["maxIter"], "tol": params["tol"],
            "silhouette": silhouette, "wssse": wssse,
            "mean_dist": mean_dist, "dist_std": dist_std,
            "training_time_s": round(t_train, 4),
        }
        grid_results.append(result)
        logger.info("   -> Silhouette=%.4f | WSSSE=%.2f | mean_dist=%.4f | T=%.2fs",
                    silhouette, wssse, mean_dist, t_train)

    
    # (e) SELECTION OF THE BEST MODEL
    
    # Maximize Silhouette; tie (difference < 0.01) -> lowest WSSSE.
    best = grid_results[0]
    for r in grid_results[1:]:
        diff = r["silhouette"] - best["silhouette"]
        if diff > 0.01:                                              # clear Silhouette improvement
            best = r
        elif abs(diff) <= 0.01 and r["wssse"] < best["wssse"]:       # tie -> lowest WSSSE
            best = r
    logger.info("BEST CONFIG: k=%d maxIter=%d tol=%g | Silhouette=%.4f | WSSSE=%.2f "
                "(criterion: max Silhouette, tie<0.01 -> min WSSSE)",
                best["k"], best["maxIter"], best["tol"], best["silhouette"], best["wssse"])

    
    # (f) RETRAINING ON THE FULL BENIGN SET
    
    logger.info("Retraining the best model on the FULL benign set (%d records)...", n_benign)
    km_final = KMeans(
        k=best["k"], maxIter=best["maxIter"], tol=best["tol"],
        seed=SEED, initMode="k-means||",
        featuresCol="scaled_features", predictionCol="cluster_id",
        distanceMeasure="euclidean")
    pipe_final = Pipeline(stages=[km_final])

    t0 = time.time()
    model_opt = pipe_final.fit(df_benign)
    final_training_time = time.time() - t0
    logger.info("Retraining completed in %.2f s", final_training_time)

    km_opt = model_opt.stages[-1]
    silhouette_opt = float(silhouette_evaluator.evaluate(model_opt.transform(df_benign)))
    wssse_opt = float(km_opt.summary.trainingCost)

    
    # (g) THRESHOLD RECALIBRATION (optimized model)
    
    logger.info("Recalibrating the threshold of the optimized model (full training DataFrame)...")
    opt_calib = _calibrate_threshold_internal(spark, model_opt, df_train)
    opt_threshold = opt_calib["threshold"]
    logger.info("Optimized threshold: %.4f (F1=%.4f, FPR=%.4f, AUC=%.4f)",
                opt_threshold, opt_calib["f1"], opt_calib["fpr"], opt_calib["auc_roc"])

    threshold_opt_json = {
        "threshold_optimal": opt_threshold,
        "f1_at_threshold": opt_calib["f1"],
        "precision": opt_calib["precision"],
        "recall": opt_calib["recall"],
        "fpr": opt_calib["fpr"],
        "auc_roc": opt_calib["auc_roc"],
        "percentile_90": opt_calib["percentiles"]["p90"],
        "percentile_95": opt_calib["percentiles"]["p95"],
        "percentile_97": opt_calib["percentiles"]["p97"],
        "percentile_99": opt_calib["percentiles"]["p99"],
        "percentile_999": opt_calib["percentiles"]["p999"],
        "calibration_days": ["Wednesday", "Friday"],
        "model": "model_optimized",
    }
    try:
        write_json_s3(threshold_opt_json, THRESHOLD_OPT)
        logger.info("Optimized threshold saved to %s", THRESHOLD_OPT)
    except Exception as e:
        logger.error("ERROR writing the optimized threshold (%s): %s", THRESHOLD_OPT, e)
        raise

    
    # (h) RE-EVALUATION ON THE TEST SET
    
    try:
        df_test = spark.read.parquet(TEST_PROCESSED)
        logger.info("Test Parquet loaded from %s", TEST_PROCESSED)
    except Exception as e:
        logger.error("ERROR reading the test Parquet (%s): %s", TEST_PROCESSED, e)
        raise

    logger.info("Evaluating the optimized model on the test set with threshold=%.4f...", opt_threshold)
    test_opt = _evaluate_on_test_internal(spark, model_opt, df_test, opt_threshold)
    logger.info("Optimized TEST -> AUC=%.4f | Acc=%.4f | P=%.4f | R=%.4f | F1=%.4f | FPR=%.4f",
                test_opt["auc_roc"], test_opt["accuracy"], test_opt["precision"],
                test_opt["recall"], test_opt["f1"], test_opt["fpr"])

    
    # (i) SAVING THE BEST MODEL + COMPARATIVE REPORT
    
    try:
        model_opt.write().overwrite().save(MODEL_OPT_OUT)
        logger.info("Optimized model saved to %s", MODEL_OPT_OUT)
    except Exception as e:
        logger.error("ERROR saving the optimized model (%s): %s", MODEL_OPT_OUT, e)
        raise

    # Load metrics of the base model for the comparison.
    train_base = read_json_s3(TRAIN_METRICS) or {}
    test_base  = read_json_s3(TEST_METRICS_OUT) or {}
    test_base_bin = test_base.get("binary", {})

    report = {
        "base_model": {
            "hyperparameters": {
                "k": train_base.get("k", 10),
                "maxIter": train_base.get("max_iter", 50),
                "tol": 1e-4,
                "initMode": train_base.get("init_mode", "k-means||"),
            },
            "silhouette": train_base.get("silhouette_score"),
            "wssse": train_base.get("wssse"),
            "auc_roc": test_base_bin.get("auc_roc"),
            "binary_f1": test_base_bin.get("f1"),
            "fpr": test_base_bin.get("fpr"),
            "binary_accuracy": test_base_bin.get("accuracy"),
            "calibrated_threshold": test_base.get("threshold_used"),
            "training_time_s": train_base.get("training_time_seconds"),
        },
        "optimized_model": {
            "hyperparameters": {
                "k": best["k"], "maxIter": best["maxIter"],
                "tol": best["tol"], "initMode": "k-means||",
            },
            "silhouette": silhouette_opt,
            "wssse": wssse_opt,
            "auc_roc": test_opt["auc_roc"],
            "binary_f1": test_opt["f1"],
            "fpr": test_opt["fpr"],
            "binary_accuracy": test_opt["accuracy"],
            "calibrated_threshold": opt_threshold,
            "training_time_s": round(final_training_time, 4),
        },
        "search_grid": grid_results,
        "selection_criterion": "max Silhouette; tie(<0.01) -> min WSSSE",
        "sample_fraction": SAMPLE_FRACTION,
    }
    try:
        write_json_s3(report, OPT_REPORT_OUT)
        logger.info("Optimization report saved to %s", OPT_REPORT_OUT)
    except Exception as e:
        logger.error("ERROR writing the report (%s): %s", OPT_REPORT_OUT, e)
        raise

    
    # (j) COMPARATIVE TABLE IN CONSOLE
    
    def _fmt(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "N/D"

    base = report["base_model"]
    opt  = report["optimized_model"]
    logger.info("| Configuration       | Silhouette | AUC-ROC | F1 bin | FPR   | Train time |")
    logger.info("|---------------------|------------|---------|--------|-------|------------|")
    logger.info("| Base model (k=%-2s)   |   %-6s   |  %-5s  | %-5s  | %-5s | %8s s |",
                base["hyperparameters"]["k"], _fmt(base["silhouette"]), _fmt(base["auc_roc"]),
                _fmt(base["binary_f1"]), _fmt(base["fpr"]), _fmt(base["training_time_s"]))
    logger.info("| Optimized model     |   %-6s   |  %-5s  | %-5s  | %-5s | %8s s |",
                _fmt(opt["silhouette"]), _fmt(opt["auc_roc"]), _fmt(opt["binary_f1"]),
                _fmt(opt["fpr"]), _fmt(opt["training_time_s"]))

    df_sample.unpersist()
    df_benign.unpersist()
    logger.info("BLOCK 4 completed successfully.")



#  MAIN: COMPLETE ORCHESTRATION OF THE KMeans PIPELINE 

def main():
    """Runs in sequence the four blocks of the KMeans pipeline with
    a single shared SparkSession."""
    spark = build_spark()
    logger.info("SparkSession 'ProjectAnomalies-KMeans' started.")
    logger.info("Starting the complete KMeans pipeline (4 blocks).")

    try:
        block1_train_kmeans(spark)          # Base model training (only benign)
        block2_calibrate_kmeans(spark)      # Distance threshold calibration
        block3_evaluate_kmeans(spark)       # Batch evaluation on test
        block4_optimize_kmeans(spark)       # Hyperparameter optimization
    except Exception as e:
        logger.error("ERROR in the KMeans pipeline: %s", e)
        spark.stop()
        raise

    spark.stop()
    logger.info("KMeans pipeline finished successfully (4/4 blocks).")


if __name__ == "__main__":
    main()