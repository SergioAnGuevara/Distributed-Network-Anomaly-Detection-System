#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_management_cic_ids2017.py

PURPOSE:
    Ingest, clean, encode, assemble, scale and weight the
    CIC-IDS2017 dataset to feed ALL the project pipelines
    (Random Forest and KMeans). It is the FIRST script of the chain: it produces
    the processed Parquets and the preprocessors (StringIndexer +
    StandardScaler) that the rest of the scripts reuse.


INPUTS (input S3 paths):
    - s3://project-anomalies-emr/data/cic-ids2017/train/   (Monday/Wednesday/Friday CSVs)
    - s3://project-anomalies-emr/data/cic-ids2017/test/    (Tuesday/Thursday CSVs)

OUTPUTS (output S3 paths):
    - .../train/processed/                            (training Parquet)
    - .../test/processed/                             (test Parquet)
    - .../models/random_forest_pipeline/preprocessors/indexer/  (StringIndexerModel)
    - .../models/random_forest_pipeline/preprocessors/scaler/   (StandardScalerModel)
    - .../models/random_forest_pipeline/preprocessors/features_meta.json
      JSON list with the VectorAssembler column names in deterministic
      alphabetical order. Required by data_management_cic_ids2018.py
      to align the IDS2018 feature schema with the IDS2017 one.

"""


import os                                               # environment variables and local paths
import re                                               # column name normalization
import json                                             # auxiliary serialization
import shutil                                           # temp directory cleanup
import logging                                          # structured logging (no print)
import subprocess                                       # kaggle installation and CLI download
import tempfile                                         # temp directory for download
from functools import reduce                            # accumulated unions/conditions
from urllib.parse import urlparse                       # s3:// path parsing

import boto3                                            # S3 CSV listing and Secrets Manager

from pyspark.sql import SparkSession                    # Spark entry point
from pyspark import StorageLevel                        # persistence levels (cache)
from pyspark.sql import functions as F                  # SQL functions (alias F)
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("DataManagement")



BUCKET        = "s3://project-anomalies-emr"                   # project root bucket
TRAIN_RAW     = f"{BUCKET}/data/cic-ids2017/train/"             # training CSVs
TEST_RAW      = f"{BUCKET}/data/cic-ids2017/test/"              # test CSVs
MODELS_PATH   = f"{BUCKET}/models/random_forest_pipeline/"     # models/preprocessors
RESULTS_PATH  = f"{BUCKET}/results/batch_evaluation/"          # results
SCRIPTS_PATH  = f"{BUCKET}/scripts/"                           # project scripts

# Derived paths.
TRAIN_PROCESSED   = f"{TRAIN_RAW}processed/"                   # training Parquet output
TEST_PROCESSED    = f"{TEST_RAW}processed/"                    # test Parquet output
INDEXER_OUT       = f"{MODELS_PATH}preprocessors/indexer/"     # StringIndexer output
SCALER_OUT        = f"{MODELS_PATH}preprocessors/scaler/"      # StandardScaler output
FEATURES_META_OUT = f"{MODELS_PATH}preprocessors/features_meta.json"  # VectorAssembler feature list

EXECUTOR_INSTANCES = 2                                          # number of executors (base cluster)
SHUFFLE_PARTITIONS = 16                                         # shuffle partitions


# Kaggle dataset slug (CSVs named by day, required by _detect_day):
# https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

KAGGLE_DATASET = "chethuhn/network-intrusion-dataset"

# Day-of-week -> split mapping (matches the CIC-IDS2017 CSV naming convention)
_TRAIN_DAYS = {"monday", "wednesday", "friday"}
_TEST_DAYS  = {"tuesday", "thursday"}



def _parse_s3(s3_path):
    """Decompose s3://bucket/key into (bucket, key)."""
    p = urlparse(s3_path)
    return p.netloc, p.path.lstrip("/")


def list_csv(s3_prefix):
    """Lists all s3:// paths of .csv files under a prefix."""
    bucket, prefix = _parse_s3(s3_prefix)                    # bucket and prefix
    client = boto3.client("s3")                               # S3 client
    paginator = client.get_paginator("list_objects_v2")       # object pagination
    paths = []                                                 # path accumulator
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):                 # page objects
            key = obj["Key"]                                 # object key
            if key.lower().endswith(".csv"):                 # only CSV
                paths.append(f"s3://{bucket}/{key}")          # full path
    return paths


def build_spark():
    """SparkSession with ALL the cluster parameters via .config()."""
    spark = (
        SparkSession.builder
        .appName("AnomalyProject-DataManagement")                                 # exact required name
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
        .config("spark.kryo.registrationRequired", "false")                        # avoids Kryo warnings with MLlib
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


# AUTOMATIC DOWNLOAD FROM KAGGLE

def _get_kaggle_credentials():
    """
    Sets up the Kaggle credentials. Accepts, in priority order:
      1. KAGGLE_API_TOKEN  (new 'KGAT_...' token via environment variable).
      2. KAGGLE_USERNAME + KAGGLE_KEY  (classic kaggle.json format).
      3. ~/.kaggle/access_token  or  ~/.kaggle/kaggle.json  (the client reads
         them automatically, nothing to configure).
      4. AWS Secrets Manager: secret 'kaggle/credentials' with JSON
         {"api_token": "..."}  or  {"username": "...", "key": "..."}.

    Returns a dict describing the found method. The caller is responsible
    for exporting to the environment whatever is needed. Raises RuntimeError
    if there are no credentials through any means.
    """
    # 1. New token (KGAT_...) via environment variable.
    if os.environ.get("KAGGLE_API_TOKEN"):
        logger.info("Kaggle credentials: KAGGLE_API_TOKEN (new token) in the environment.")
        return {"method": "api_token"}

    # 2. Classic username + key via environment variables.
    username = os.environ.get("KAGGLE_USERNAME")
    key      = os.environ.get("KAGGLE_KEY")
    if username and key:
        logger.info("Kaggle credentials: KAGGLE_USERNAME/KAGGLE_KEY in the environment.")
        return {"method": "user_key", "username": username, "key": key}

    # 3. Files that the kaggle client reads on its own.
    home = os.path.expanduser("~")
    if (os.path.exists(os.path.join(home, ".kaggle", "access_token"))
            or os.path.exists(os.path.join(home, ".kaggle", "kaggle.json"))):
        logger.info("Kaggle credentials: file in ~/.kaggle/ (read automatically by the client).")
        return {"method": "file"}

    # 4. Fallback: AWS Secrets Manager.
    logger.info(
        "No credentials in the environment nor in ~/.kaggle/. "
        "Trying Secrets Manager (kaggle/credentials)..."
    )
    try:
        sm = boto3.client("secretsmanager")
        secret = sm.get_secret_value(SecretId="kaggle/credentials")
        creds = json.loads(secret["SecretString"])
        logger.info("Kaggle credentials obtained from Secrets Manager.")
        if creds.get("api_token"):
            return {"method": "api_token", "token": creds["api_token"]}
        return {"method": "user_key", "username": creds["username"], "key": creds["key"]}
    except Exception as exc:
        raise RuntimeError(
            "No Kaggle credentials found.\n"
            "Available options:\n"
            "  1. export KAGGLE_API_TOKEN=KGAT_...          (new token)\n"
            "  2. export KAGGLE_USERNAME=...  and  KAGGLE_KEY=...\n"
            "  3. file ~/.kaggle/access_token  or  ~/.kaggle/kaggle.json\n"
            "  4. AWS Secrets Manager: secret 'kaggle/credentials'"
        ) from exc


def _detect_day(file_name):
    """
    Detects the day of the week from a CIC-IDS2017 CSV name.
    The files have names like:
      Monday-WorkingHours.pcap_ISCX.csv
      Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    Returns the day in lowercase or None if it is not recognized.
    """
    name = file_name.lower()
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday"):
        if day in name:
            return day
    return None


def _list_local_csvs(directory):
    """Recursively lists all .csv files inside a local directory."""
    result = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".csv"):
                result.append(os.path.join(root, file))
    return result


def _has_csv_in_s3(client, bucket, prefix):
    """Returns True if at least one .csv already exists under the given prefix."""
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=10)
    return any(
        obj["Key"].lower().endswith(".csv")
        for obj in resp.get("Contents", [])
    )


def download_and_upload_cic_ids2017():
    """
    Downloads the CIC-IDS2017 dataset from Kaggle and uploads the CSVs to S3.

    Split by day of the week:
      - Monday, Wednesday, Friday  ->  TRAIN_RAW  (training)
      - Tuesday, Thursday          ->  TEST_RAW   (test)

    Idempotent: if both prefixes already contain CSVs in S3, the function
    returns immediately without re-downloading or re-uploading anything.

    The 'kaggle' package is installed automatically on the master node
    if it is not available (requires pip/PyPI access from the EMR).
    """
    bucket, train_prefix = _parse_s3(TRAIN_RAW)
    _,      test_prefix  = _parse_s3(TEST_RAW)
    client_s3 = boto3.client("s3")


    train_ok = _has_csv_in_s3(client_s3, bucket, train_prefix)
    test_ok  = _has_csv_in_s3(client_s3, bucket, test_prefix)
    if train_ok and test_ok:
        logger.info(
            "CIC-IDS2017 CSVs already present in S3 (train and test). "
            "Skipping download from Kaggle."
        )
        return

    logger.info(
        "CSVs missing in S3 (train_ok=%s, test_ok=%s). "
        "Starting download from Kaggle [%s]...",
        train_ok, test_ok, KAGGLE_DATASET,
    )

    #  Credentials 
    creds = _get_kaggle_credentials()
    if creds["method"] == "user_key":
        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"]      = creds["key"]
    elif creds["method"] == "api_token" and creds.get("token"):
        # New token retrieved from Secrets Manager: export it to the environment.
        os.environ["KAGGLE_API_TOKEN"] = creds["token"]
    # For 'file' or 'api_token' already present in the environment: nothing to do,
    # the kaggle client reads it automatically.

    #  Install / upgrade the kaggle package 

    logger.info("Installing/upgrading the 'kaggle' package (>=1.7) via pip...")
    try:
        subprocess.check_call(
            ["pip", "install", "--quiet", "--upgrade", "kaggle>=1.7"],
            timeout=180,
        )
        logger.info("'kaggle' package ready.")
    except Exception as e:
        logger.warning("Could not upgrade 'kaggle' (%s). It will use the current version.", e)

  
    tmpdir = tempfile.mkdtemp(prefix="cicids2017_")
    logger.info("Temporary download directory: %s", tmpdir)

    try:
        logger.info("Running: kaggle datasets download -d %s ...", KAGGLE_DATASET)
        subprocess.check_call(
            [
                "kaggle", "datasets", "download",
                "-d", KAGGLE_DATASET,
                "-p", tmpdir,
                "--unzip",                              # unzips the ZIP automatically
            ],
            timeout=3600,                               # 1 hour max (dataset ~1 GB)
        )
        logger.info("Dataset downloaded and unzipped in %s.", tmpdir)

        # ── Upload of each CSV to the correct S3 prefix ─────────────────
        csvs = _list_local_csvs(tmpdir)
        if not csvs:
            raise FileNotFoundError(
                f"No .csv files found in {tmpdir} after the download. "
                f"Verify the dataset slug: {KAGGLE_DATASET}"
            )
        logger.info("CSVs found locally: %d", len(csvs))

        n_train = n_test = n_omitted = 0
        for local_path in csvs:
            file_name = os.path.basename(local_path)
            day = _detect_day(file_name)

            if day in _TRAIN_DAYS:
                dest_prefix = train_prefix
                n_train += 1
            elif day in _TEST_DAYS:
                dest_prefix = test_prefix
                n_test += 1
            else:
                logger.warning(
                    "Could not classify '%s' by day — skipping.",
                    file_name,
                )
                n_omitted += 1
                continue

            s3_key = dest_prefix + file_name
            logger.info(
                "Uploading %-60s -> s3://%s/%s",
                file_name, bucket, s3_key,
            )
            client_s3.upload_file(local_path, bucket, s3_key)

        logger.info(
            "S3 upload complete: %d train | %d test | %d omitted.",
            n_train, n_test, n_omitted,
        )

        # Minimal validation: at least one CSV per split.
        if n_train == 0:
            raise RuntimeError(
                "No training CSV was uploaded (Monday/Wednesday/Friday). "
                f"Review the file names in the dataset {KAGGLE_DATASET}."
            )
        if n_test == 0:
            raise RuntimeError(
                "No test CSV was uploaded (Tuesday/Thursday). "
                f"Review the file names in the dataset {KAGGLE_DATASET}."
            )

    finally:
        # Always free space on the master, regardless of the result.
        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.info("Temporary directory removed: %s", tmpdir)




def normalize_name(name):
    """Normalizes a column name: strip, spaces->'_', lowercase,
    and removal of non-alphanumeric characters (keeping '_')."""
    n = name.strip()                                # (1) remove outer spaces
    n = re.sub(r"\s+", "_", n)                        # (2) internal spaces -> underscore
    n = n.lower()                                     # (3) lowercase
    n = re.sub(r"[^0-9a-z_]", "", n)                  # (4) remove non-alphanumeric
    return n


def load_and_normalize(spark, s3_prefix):
    """Loads ALL the CSVs in the prefix in ONE single read (Spark processes them
    in parallel) and normalizes the column names only once. Much more
    efficient than reading file by file and combining them with unionByName."""
    paths = list_csv(s3_prefix)                    # explicit .csv list (excludes processed/)
    if not paths:                                     # validate that there are files
        raise FileNotFoundError(f"No CSVs found in {s3_prefix}")
    logger.info("CSVs found in %s: %d (single-pass read).", s3_prefix, len(paths))

    # Read WITHOUT schema inference from the complete file list.
    # Passing the list avoids reading the processed/ subfolder and avoids unionByName.
    df = (spark.read
          .option("header", True)
          .option("inferSchema", False)
          .csv(paths))

    # (d) Normalize names ONCE over the combined schema.
    for original in df.columns:
        new_name = normalize_name(original)
        if new_name != original:
            df = df.withColumnRenamed(original, new_name)

    # (f) Traceability: real source file of each row.
    df = df.withColumn("_source_file", F.input_file_name())
    return df


def detect_numeric_columns(df, excluded):
    """Detects columns castable to double (real numerics). It works on a
    BOUNDED sample of rows: a column is either numeric or not consistently,
    so scanning millions of rows is unnecessary — a sample is enough to
    classify the type and avoid a very expensive full scan."""
    columns = [c for c in df.columns if c not in excluded]     # candidates
    sample = df.select(*columns).limit(20000)                # enough sample to type
    exprs = []                                                 # per-column aggregations
    for c in columns:
        exprs.append(F.count(F.when(F.col(c).isNotNull(), True)).alias(f"nn__{c}"))            # not null
        exprs.append(F.count(F.when(F.col(c).cast("double").isNotNull(), True)).alias(f"cd__{c}"))  # castable
    row = sample.agg(*exprs).first()                         # a single action on the sample
    numeric = [c for c in columns
                 if row[f"nn__{c}"] > 0 and row[f"nn__{c}"] == row[f"cd__{c}"]]
    logger.info("Numeric columns detected: %d of %d candidates",
                len(numeric), len(columns))
    return numeric


def clean(df, numeric_columns, split_label):
    """Cleaning in ONE single pass: a single filter that discards rows with
    a null label, or with any null / infinite / NaN numeric column; then
    removes duplicates. Reduces from 4 counts (that recomputed everything) to 2."""
    # (f) Logical partition column.
    df = df.withColumn("_split", F.lit(split_label))
    n0 = df.count()                                            # initial count (over cache)

    # Combined filter: label not null AND each numeric is valid (no null/inf/NaN).
    cond = F.col("label").isNotNull()
    for c in numeric_columns:
        cond = (cond
                & F.col(c).isNotNull()
                & ~F.isnan(F.col(c))
                & (F.col(c) != float("inf"))
                & (F.col(c) != float("-inf")))
    df = df.filter(cond)                                       # single cleaning pass
    df = df.dropDuplicates()                                   # single heavy shuffle
    n1 = df.count()                                            # final count

    logger.info("[%s] Cleaning: %d -> %d records (dropped=%d: null/inf/NaN/duplicates)",
                split_label, n0, n1, n0 - n1)
    return df


def compute_class_weights(df_train):
    """Computes class_weight = (1/relative_frequency) normalized so that
    the sum of class weights equals the number of classes."""
    counts = df_train.groupBy("label_index").count().collect()   # per-class counts (small)
    total = sum(r["count"] for r in counts)                       # total records
    num_classes = len(counts)                                      # number of classes

    raw_weights = {int(r["label_index"]): total / r["count"] for r in counts}  # 1/freq = total/n
    weight_sum = sum(raw_weights.values())                        # sum of raw weights
    factor = num_classes / weight_sum                               # normalization factor
    norm_weights = {idx: w * factor for idx, w in raw_weights.items()}  # normalized weights

    logger.info("Normalized class weights (sum=%d classes): %s", num_classes,
                {k: round(v, 4) for k, v in norm_weights.items()})

    # Build a mapping DataFrame and join it to the training one.
    spark = df_train.sql_ctx.sparkSession                          # associated session
    rows = [(float(idx), float(w)) for idx, w in norm_weights.items()]
    df_weights = spark.createDataFrame(rows, ["label_index", "class_weight"])
    return df_train.join(df_weights, on="label_index", how="left")


def save_features_meta(feature_columns):
    """
    Serializes the VectorAssembler column list into a JSON file in S3
    under the path FEATURES_META_OUT.

    This file is required by data_management_cic_ids2018.py to align the
    IDS2018 columns exactly with those that the StandardScaler and the
    VectorAssembler were trained on in IDS2017. Without this file, the
    2018 script falls back to feature inference from the data, which can
    cause discrepancies.

    The JSON has the form:
        {"feature_cols": ["col_a", "col_b", ...]}
    where the columns are in deterministic alphabetical order, which is
    the same criterion used by VectorAssembler in main().
    """
    bucket, key = _parse_s3(FEATURES_META_OUT)                    # decomposes the S3 path
    client = boto3.client("s3")                                     # S3 client of the EMR role

    # Writes the JSON to a temporary local file and uploads it to S3.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump({"feature_cols": list(feature_columns)}, tmp,
                  ensure_ascii=False, indent=2)
        tmp_path = tmp.name                                          # keeps the path for later

    try:
        client.upload_file(tmp_path, bucket, key)                # uploads to S3
        logger.info(
            "features_meta.json saved in %s (%d columns).",
            FEATURES_META_OUT, len(feature_columns),
        )
    except Exception as e:
        logger.error(
            "ERROR saving features_meta.json in %s: %s",
            FEATURES_META_OUT, e,
        )
        raise
    finally:
        os.remove(tmp_path)                                         # cleans up the temp file


def main():
    """Orchestrates the complete ingestion and preparation of the dataset."""

    # STEP 0: AUTOMATIC DOWNLOAD FROM KAGGLE (if the CSVs are not in S3)
    try:
        download_and_upload_cic_ids2017()
    except Exception as e:
        logger.error("ERROR in Kaggle download/upload: %s", e)
        raise

    spark = build_spark()
    logger.info("SparkSession 'AnomalyProject-DataManagement' started.")

    # (c)(d) LOAD + NORMALIZATION OF TRAINING AND TEST
    try:
        df_train = load_and_normalize(spark, TRAIN_RAW)        # training CSVs
        df_test  = load_and_normalize(spark, TEST_RAW)         # test CSVs
        logger.info("Training and test CSVs loaded and normalized.")
    except Exception as e:
        logger.error("ERROR loading/normalizing the CSVs: %s", e)
        spark.stop()
        raise

    # DETECTION OF NUMERIC COLUMNS (over training) + CAST
    # Explicitly excludes 'label' and traceability columns.
    excluded = {"label", "_source_file"}
    numeric_columns = detect_numeric_columns(df_train, excluded)
    if "label" not in df_train.columns:                         # validates the label column
        logger.error("The column 'label' does not exist after normalization. Aborting.")
        spark.stop()
        raise KeyError("label")

    def cast(df):
        """Casts the numeric columns to double, discards non-numeric ones
        (except label/_source_file) and keeps the expected schema."""
        columns_to_keep = numeric_columns + ["label", "_source_file"]
        df = df.select(*[c for c in columns_to_keep if c in df.columns])
        df = df.select(*[
            F.col(c).cast("double").alias(c) if c in set(numeric_columns) else F.col(c)
            for c in df.columns
        ])
        return df

    df_train = cast(df_train).persist(StorageLevel.MEMORY_AND_DISK)   # cast + cache (reads S3 once)
    df_test  = cast(df_test).persist(StorageLevel.MEMORY_AND_DISK)     # cast + test cache

    # (e) CLEANING WITH LOGGING + CACHE
    # The cast DataFrames are cached so the 2 counts of clean() read from
    # memory and do NOT re-read the CSVs from S3. The cleaned result is cached
    # again because from here on it is reused by the StringIndexer, the
    # VectorAssembler, the StandardScaler, the class weights and the writing.
    # Without these caches every action recomputed everything (the reason it
    # used to take 1 hour).
    df_train_clean = clean(df_train, numeric_columns, "train").persist(StorageLevel.MEMORY_AND_DISK)
    df_test_clean  = clean(df_test,  numeric_columns, "test").persist(StorageLevel.MEMORY_AND_DISK)
    df_train_clean.count(); df_test_clean.count()   # materializes the clean cache
    df_train.unpersist(); df_test.unpersist()          # frees the cast cache (no longer used)
    df_train, df_test = df_train_clean, df_test_clean

    # (g) STRINGINDEXER (fit ONLY on training)
    # frequencyDesc -> the most frequent class (benign) gets index 0.0.
    # handleInvalid='keep' lets test labels NOT seen in training
    # (Web attacks, Infiltration) not break the transform.
    indexer = StringIndexer(
        inputCol="label", outputCol="label_index",
        stringOrderType="frequencyDesc", handleInvalid="keep")
    indexer_model = indexer.fit(df_train)                       # FIT only in training
    df_train = indexer_model.transform(df_train)                # apply to training
    df_test  = indexer_model.transform(df_test)                 # apply to test
    logger.info("StringIndexer fitted. Known classes: %s", indexer_model.labels)

    # (h) VECTORASSEMBLER (only numeric columns, deterministic alphabetical order)
    feature_columns = sorted(numeric_columns)              # stable alphabetical order
    assembler = VectorAssembler(
        inputCols=feature_columns, outputCol="features", handleInvalid="skip")
    df_train = assembler.transform(df_train)                    # assemble training
    df_test  = assembler.transform(df_test)                     # assemble test
    logger.info("VectorAssembler: %d features assembled.", len(feature_columns))

    # (i) STANDARDSCALER (fit ONLY on training)
    scaler = StandardScaler(
        inputCol="features", outputCol="scaled_features",
        withMean=True, withStd=True)
    scaler_model = scaler.fit(df_train)                         # FIT only in training (avoids leakage)
    df_train = scaler_model.transform(df_train)                 # scale training
    df_test  = scaler_model.transform(df_test)                  # scale test
    logger.info("StandardScaler fitted (withMean=True, withStd=True).")

    # (j) CLASS WEIGHTS (only training)
    df_train = compute_class_weights(df_train)

    # Selection of the final columns to persist.
    cols_train = ["label", "label_index", "features", "scaled_features",
                  "class_weight", "_source_file", "_split"]
    cols_test  = ["label", "label_index", "features", "scaled_features",
                  "_source_file", "_split"]
    df_train_out = df_train.select(*cols_train)                 # final training schema
    df_test_out  = df_test.select(*cols_test)                   # final test schema

    # (k) SAVE TO S3
    try:
        df_train_out.write.mode("overwrite").parquet(TRAIN_PROCESSED)   # training Parquet
        logger.info("Processed training saved in %s", TRAIN_PROCESSED)
    except Exception as e:
        logger.error("ERROR saving training Parquet (%s): %s", TRAIN_PROCESSED, e)
        spark.stop()
        raise

    try:
        df_test_out.write.mode("overwrite").parquet(TEST_PROCESSED)     # test Parquet
        logger.info("Processed test saved in %s", TEST_PROCESSED)
    except Exception as e:
        logger.error("ERROR saving test Parquet (%s): %s", TEST_PROCESSED, e)
        spark.stop()
        raise

    try:
        indexer_model.write().overwrite().save(INDEXER_OUT)             # fitted StringIndexer
        scaler_model.write().overwrite().save(SCALER_OUT)               # fitted StandardScaler
        logger.info("Preprocessors saved in %s and %s", INDEXER_OUT, SCALER_OUT)
    except Exception as e:
        logger.error("ERROR saving preprocessors: %s", e)
        spark.stop()
        raise

   
    try:
        save_features_meta(feature_columns)
    except Exception as e:
        # The pipeline does not stop: the 2018 script has a fallback.
        # It is logged as a WARNING because the absence of the file is not
        # critical for the training and evaluation pipelines.
        logger.warning(
            "Could not save features_meta.json (%s). "
            "data_management_cic_ids2018.py will operate in fallback mode.",
            e,
        )

    # (m) CLOSURE
    spark.stop()
    logger.info("Ingestion and preparation finished successfully.")


if __name__ == "__main__":
    main()
