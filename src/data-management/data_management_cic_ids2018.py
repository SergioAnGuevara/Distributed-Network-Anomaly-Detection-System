
import re
import json
import logging
from functools import reduce
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("DataManagement2018")



# Public bucket of the CIC-IDS2018 dataset (read-only, anonymous access).
BUCKET_2018  = "s3://cse-cic-ids2018"
PREFIX_2018 = "Processed Traffic Data for ML Algorithms"

# Files for the streaming demo .
DEMO_FILES = [
    f"{BUCKET_2018}/{PREFIX_2018}/02-20-2018.csv",
    f"{BUCKET_2018}/{PREFIX_2018}/02-21-2018.csv",
]



PROJECT_BUCKET  = "s3://project-anomalies-emr"
PREPROC_PATH     = f"{PROJECT_BUCKET}/models/random_forest_pipeline/preprocessors"
FEATURES_META_IN = f"{PREPROC_PATH}/features_meta.json"

# Aligned Parquet output (input for tcp_sender.py).
STREAMING_OUT = f"{PROJECT_BUCKET}/data/cic-ids2018/streaming/processed/"

# Label column (normalized name, same as in data_management_cic_ids2017.py).
LABEL_COL = "label"

# Columns that NEVER enter the feature vector.
EXCLUDED_FEATURES = {
    "label", "_source_file", "_split",
    "label_index", "class_weight", "features", "scaled_features",
    "timestamp",   # extra column from IDS2018
    "flow_id",     # metadata from IDS2018
    "src_ip", "src_port", "dst_ip", "dst_port",  # network metadata
}


EXECUTOR_INSTANCES = 2
SHUFFLE_PARTITIONS = 16


HOMOLOGATION = {
    # Flow rates
    "flow_byts_s":      "flow_bytes_s",
    "flow_pkts_s":      "flow_packets_s",
    "fwd_pkts_s":       "fwd_packets_s",
    "bwd_pkts_s":       "bwd_packets_s",
    # Packet length — forward
    "fwd_pkt_len_max":  "fwd_packet_length_max",
    "fwd_pkt_len_min":  "fwd_packet_length_min",
    "fwd_pkt_len_mean": "fwd_packet_length_mean",
    "fwd_pkt_len_std":  "fwd_packet_length_std",
    # Packet length — backward
    "bwd_pkt_len_max":  "bwd_packet_length_max",
    "bwd_pkt_len_min":  "bwd_packet_length_min",
    "bwd_pkt_len_mean": "bwd_packet_length_mean",
    "bwd_pkt_len_std":  "bwd_packet_length_std",
    # Packet length — overall
    "pkt_len_max":      "max_packet_length",
    "pkt_len_min":      "min_packet_length",
    "pkt_len_mean":     "packet_length_mean",
    "pkt_len_std":      "packet_length_std",
    "pkt_len_var":      "packet_length_variance",
    "pkt_size_avg":     "average_packet_size",
    # Segment
    "fwd_seg_size_avg": "avg_fwd_segment_size",
    "bwd_seg_size_avg": "avg_bwd_segment_size",
    "fwd_seg_size_min": "min_seg_size_forward",
    # Bulk
    "fwd_byts_b_avg":   "fwd_avg_bytes_bulk",
    "fwd_pkts_b_avg":   "fwd_avg_packets_bulk",
    "fwd_blk_rate_avg": "fwd_avg_bulk_rate",
    "bwd_byts_b_avg":   "bwd_avg_bytes_bulk",
    "bwd_pkts_b_avg":   "bwd_avg_packets_bulk",
    "bwd_blk_rate_avg": "bwd_avg_bulk_rate",
    # Flags
    "fin_flag_cnt":     "fin_flag_count",
    "syn_flag_cnt":     "syn_flag_count",
    "rst_flag_cnt":     "rst_flag_count",
    "psh_flag_cnt":     "psh_flag_count",
    "ack_flag_cnt":     "ack_flag_count",
    "urg_flag_cnt":     "urg_flag_count",
    "ece_flag_cnt":     "ece_flag_count",
    # Packet and byte volume
    "tot_fwd_pkts":     "total_fwd_packets",
    "tot_bwd_pkts":     "total_backward_packets",
    "totlen_fwd_pkts":  "total_length_of_fwd_packets",   # no duplicate
    "totlen_bwd_pkts":  "total_length_of_bwd_packets",   # no duplicate
    # Subflow
    "subflow_fwd_pkts": "subflow_fwd_packets",
    "subflow_bwd_pkts": "subflow_bwd_packets",
    "subflow_fwd_byts": "subflow_fwd_bytes",
    "subflow_bwd_byts": "subflow_bwd_bytes",
    # TCP init
    "init_bwd_win_byts": "init_win_bytes_backward",
    "init_fwd_win_byts": "init_win_bytes_forward",
    # Active / segment
    "fwd_act_data_pkts": "act_data_pkt_fwd",
}




def _parse_s3(s3_path):
    """Decompose s3://bucket/key into (bucket, key)."""
    p = urlparse(s3_path)
    return p.netloc, p.path.lstrip("/")


def read_features_meta():
    """
    Loads the list of feature columns of the 2017 pipeline from S3.
    Returns the list if it exists, or None if it was not found.
    """
    bucket, key = _parse_s3(FEATURES_META_IN)
    client = boto3.client("s3")
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        data = json.loads(resp["Body"].read().decode("utf-8"))
        feature_cols = data.get("feature_cols", None)
        if feature_cols:
            logger.info(
                "features_meta.json loaded: %d feature columns.",
                len(feature_cols),
            )
        return feature_cols
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            logger.warning(
                "features_meta.json NOT found in %s. "
                "Columns will be inferred from IDS2018 data. "
                "See NOTA_METADATA in this script's docstring.",
                FEATURES_META_IN,
            )
            return None
        raise




def build_spark():
    """
    SparkSession configured for EMR.

   
    """
    spark = (
        SparkSession.builder
        .appName("AnomalyProject-DataManagement2018")
        # Resources — same as data_management_cic_ids2017.py
        .config("spark.executor.cores",           "2")
        .config("spark.executor.memory",          "6g")
        .config("spark.executor.memoryOverhead",  "1024")
        .config("spark.driver.cores",             "2")
        .config("spark.driver.memory",            "6g")
        .config("spark.executor.instances",       str(EXECUTOR_INSTANCES))
        .config("spark.default.parallelism",      str(SHUFFLE_PARTITIONS))
        .config("spark.sql.shuffle.partitions",   str(SHUFFLE_PARTITIONS))
        .config("spark.serializer",
                "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "256m")
        .config("spark.kryo.registrationRequired", "false")
        .config("spark.memory.fraction",          "0.75")
        .config("spark.memory.storageFraction",   "0.40")
        .config("spark.task.maxFailures",         "4")
        .config("spark.speculation",              "true")
        .config("spark.speculation.multiplier",   "1.5")
        .config("spark.hadoop.fs.s3a.fast.upload","true")
   
        .config(
            "spark.hadoop.fs.s3a.bucket.cse-cic-ids2018"
            ".aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider",
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark




def normalize_name(name):
    """
    Normalizes a CICFlowMeter column name applying the
    same rules as data_management_cic_ids2017.py:
        1. strip() of outer whitespace
        2. internal whitespace → underscore
        3. conversion to lowercase
        4. removal of non-alphanumeric characters (keeps '_')
    """
    n = name.strip()
    n = re.sub(r"\s+", "_", n)
    n = n.lower()
    n = re.sub(r"[^0-9a-z_]", "", n)
    return n




def load_and_normalize(spark, paths):
    """
    Loads the CIC-IDS2018 CSVs from the public S3 bucket and
    normalizes the column names immediately.
    """
    if not paths:
        raise ValueError("Empty CSV paths list. Check DEMO_FILES.")

    dataframes = []
    for path in paths:
        logger.info("Loading: %s", path)
        try:
            df = (
                spark.read
                .option("header",       True)
                .option("inferSchema",  False)    # all as string
                .option("mode",         "PERMISSIVE")
                .option("encoding",     "UTF-8")
                .csv(path)
            )
        except Exception as e:
            logger.warning(
                "UTF-8 failed on %s (%s). Retrying with latin-1.", path, e)
            df = (
                spark.read
                .option("header",      True)
                .option("inferSchema", False)
                .option("mode",        "PERMISSIVE")
                .option("encoding",    "ISO-8859-1")
                .csv(path)
            )

        # Normalize names immediately after loading.
        for original in df.columns:
            new_name = normalize_name(original)
            if original != new_name:
                df = df.withColumnRenamed(original, new_name)

        # Row-level traceability.
        df = df.withColumn("_source_file", F.input_file_name())

        logger.info(
            "  %s: %d columns loaded.",
            path.split("/")[-1], len(df.columns),
        )
        dataframes.append(df)

    df_union = reduce(
        lambda a, b: a.unionByName(b, allowMissingColumns=True),
        dataframes,
    )
    logger.info(
        "Combined 2018 dataset: %d columns after unionByName.",
        len(df_union.columns),
    )
    return df_union


#COLUMN HOMOLOGATION 

def homologate_columns(df):
    """
    Renames IDS2018 columns (normalized) to their IDS2017 equivalents.
    Only renames if the source column exists AND the
    destination column does NOT exist (avoids collisions).
    """
    renamed = []
    for name_2018, name_2017 in HOMOLOGATION.items():
        if name_2018 in df.columns and name_2017 not in df.columns:
            df = df.withColumnRenamed(name_2018, name_2017)
            renamed.append(f"{name_2018} → {name_2017}")

    logger.info(
        "Homologation: %d columns renamed.", len(renamed),
    )
    if renamed:
        logger.debug("Details: %s", renamed)
    return df


#NUMERIC CAST ===========

def cast_numeric_columns(df, numeric_columns):
    """
    Casts the feature columns to DoubleType keeping only
    the needed columns to reduce the subsequent shuffle.
    """
    cols_to_keep = (
        list(numeric_columns)
        + [c for c in ["label", "_source_file"] if c in df.columns]
    )
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    df = df.select(*cols_to_keep)

    set_num = set(numeric_columns)
    df = df.select(*[
        F.col(c).cast(DoubleType()).alias(c)
        if c in set_num else F.col(c)
        for c in df.columns
    ])
    return df


#CLEANING 

def clean(df, numeric_columns):
    """
    Applies the same cleaning as data_management_cic_ids2017.py:
        Step 1: remove nulls in key columns.
        Step 2: replace ±Inf with NaN and remove those rows.
        Step 3: remove exact duplicates.
    """
    df = df.withColumn("_split", F.lit("streaming_demo"))
    key_columns = numeric_columns + [LABEL_COL]

    
    df.cache()

    n0 = df.count()
    logger.info("[IDS2018] Initial records: %d", n0)

    # Step 1: nulls.
    df = df.dropna(how="any", subset=key_columns)
    n1 = df.count()
    logger.info("[IDS2018] After removing nulls: %d (removed=%d)", n1, n0 - n1)

    # Step 2: infinities → NaN → filter.
    set_num = set(numeric_columns)
    df = df.select(*[
        F.when(
            (F.col(c) == float("inf")) | (F.col(c) == float("-inf")),
            float("nan"),
        ).otherwise(F.col(c)).alias(c)
        if c in set_num else F.col(c)
        for c in df.columns
    ])
    cond_nan = reduce(
        lambda a, b: a | b,
        [F.isnan(F.col(c)) for c in numeric_columns],
    )
    df = df.filter(~cond_nan)
    n2 = df.count()
    logger.info(
        "[IDS2018] After removing infinities/NaN: %d (removed=%d)", n2, n1 - n2,
    )
    if (n1 - n2) > (n1 * 0.30):
        logger.warning(
            "[IDS2018] Removed >30%% of records due to infinities/NaN. "
            "Known behavior in CIC-IDS2018. Continue."
        )

    # Step 3: duplicates.
    df = df.dropDuplicates()
    n3 = df.count()
    logger.info(
        "[IDS2018] After removing duplicates: %d (removed=%d)", n3, n2 - n3,
    )

    df.unpersist()
    return df




def align_with_2017_schema(df, feature_cols_2017):
    """
    Ensures that the 2018 DataFrame has exactly the columns
    that the 2017 PipelineModel expects.

    - Present columns: used directly.
    - Missing columns: imputed with 0.0 (no activity).
    - Extra 2018 columns: ignored.
    """
    present_cols          = set(df.columns)
    added_columns  = []

    for col in feature_cols_2017:
        if col not in present_cols:
            df = df.withColumn(col, F.lit(0.0).cast(DoubleType()))
            added_columns.append(col)

    if added_columns:
        logger.warning(
            "%d columns expected by the 2017 pipeline missing in IDS2018 "
            "→ imputed with 0.0: %s",
            len(added_columns), added_columns,
        )
    else:
        logger.info(
            "Perfect alignment: all 2017 columns present in IDS2018."
        )

    extra_cols = [
        c for c in present_cols
        if c not in set(feature_cols_2017) and c not in EXCLUDED_FEATURES
    ]
    if extra_cols:
        logger.info(
            "%d IDS2018 columns ignored (not in 2017 schema): %s",
            len(extra_cols), extra_cols[:10],
        )

    return df



def detect_numeric_columns(df, excluded):
    """
    Detects columns 100% castable to DoubleType.
    Only used as a fallback if features_meta.json does not exist.
    Identical to the function in data_management_cic_ids2017.py.
    """
    columns = [c for c in df.columns if c not in excluded]
    exprs = []
    for c in columns:
        exprs.append(
            F.count(F.when(F.col(c).isNotNull(), True)).alias(f"nn__{c}"))
        exprs.append(
            F.count(F.when(F.col(c).cast("double").isNotNull(),
                           True)).alias(f"cd__{c}"))
    row     = df.agg(*exprs).first()
    numeric = [
        c for c in columns
        if row[f"nn__{c}"] > 0 and row[f"nn__{c}"] == row[f"cd__{c}"]
    ]
    logger.info(
        "Numeric columns detected (fallback): %d of %d candidates.",
        len(numeric), len(columns),
    )
    return numeric



def main():
    """
    Orchestrates the preparation of the CIC-IDS2018 dataset for the demo.

    Pipeline:
        1. Load features_meta.json from S3 (list of 2017 columns).
        2. Load and normalize IDS2018 CSVs from the public bucket.
        3. Homologate IDS2018 → IDS2017 columns.
        4. Determine the feature columns to use.
        5. Align with the 2017 schema (impute missing with 0.0).
        6. Cast to DoubleType + keep only the needed columns.
        7. Cleaning (nulls → infinities → duplicates).
        8. Save Parquet to the project bucket.
        9. Diagnostic summary.
    """
    spark = build_spark()
    logger.info("SparkSession 'AnomalyProject-DataManagement2018' started.")

    
    feature_cols_2017 = read_features_meta()


    try:
        df = load_and_normalize(spark, DEMO_FILES)
    except Exception as e:
        logger.error("ERROR loading IDS2018 CSV: %s", e)
        spark.stop()
        raise

    if LABEL_COL not in df.columns:
        logger.error(
            "Column '%s' not found after normalizing. "
            "Available columns: %s", LABEL_COL, df.columns,
        )
        spark.stop()
        raise KeyError(LABEL_COL)


    df = homologate_columns(df)

  
    if feature_cols_2017 is not None:
        logger.info(
            "Using %d feature columns from the 2017 pipeline (features_meta.json).",
            len(feature_cols_2017),
        )
        numeric_columns = feature_cols_2017
    else:
        logger.warning(
            "FALLBACK: inferring numeric columns from IDS2018. "
            "Save features_meta.json in data_management_cic_ids2017.py (see NOTA_METADATA)."
        )
        numeric_columns = sorted(
            detect_numeric_columns(df, EXCLUDED_FEATURES)
        )


    df = align_with_2017_schema(df, numeric_columns)


    columns_to_cast = [c for c in numeric_columns if c in df.columns]
    df = cast_numeric_columns(df, columns_to_cast)

    df = clean(df, columns_to_cast)


    output_cols = (
        columns_to_cast
        + [c for c in ["label", "_source_file", "_split"] if c in df.columns]
    )
    output_cols = [c for c in output_cols if c in df.columns]
    df_out = df.select(*output_cols)

    try:
        df_out.write.mode("overwrite").parquet(STREAMING_OUT)
        logger.info("Aligned Parquet saved to: %s", STREAMING_OUT)
    except Exception as e:
        logger.error("ERROR saving Parquet to %s: %s", STREAMING_OUT, e)
        spark.stop()
        raise


    final_count = df_out.count()
    distribution = {
        r[LABEL_COL]: r["count"]
        for r in df_out.groupBy(LABEL_COL).count().collect()
    }
    logger.info(
        "FINAL SUMMARY | Records: %d | Features: %d | Distribution: %s",
        final_count, len(columns_to_cast), distribution,
    )
    logger.info(
        "Parquet ready for streaming. tcp_sender.py must read from: %s",
        STREAMING_OUT,
    )

    spark.stop()
    logger.info("data_management_cic_ids2018.py completed successfully.")


if __name__ == "__main__":
    main()
