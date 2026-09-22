# Distributed Near Real-Time Network Anomaly Detection System

A distributed, near real-time network intrusion detection system built entirely on AWS — from training and validation to live streaming inference.

> Big Data course project — Universidad del Rosario, Bogotá, Colombia

---

## Overview

This project takes a problem that is routinely solved with standard, locally-trained machine learning models network intrusion detection  and re-architects it as a **fully distributed system**. Every stage of the workflow, from data ingestion and preprocessing through model training, validation, and live inference, runs on a distributed Apache Spark cluster on AWS EMR. The goal was not just to train an accurate classifier, but to build and validate the infrastructure required to train, calibrate, evaluate, and serve that classifier at scale.

The system combines two complementary approaches:
- A **supervised Random Forest classifier**, trained to recognize known attack patterns.
- An **unsupervised KMeans clustering model**, trained exclusively on benign traffic, designed to flag anomalies as deviations from normal behavior  including, in principle, attack types never seen during training.

---

## Architecture

```
 ┌──────────────────┐     ┌───────────────────┐     ┌────────────────────┐     ┌───────────────────┐
 │   Data Ingestion  │ --> │   Preprocessing    │ --> │   Model Training    │ --> │  Streaming          │
 │  (S3 raw CSVs)    │     │ (indexing/scaling) │     │  (RF + KMeans on    │     │  Inference Demo     │
 │                  │     │  on Spark           │     │   Spark MLlib)      │     │  (TCP → EMR)        │
 └──────────────────┘     └───────────────────┘     └────────────────────┘     └───────────────────┘
```

**Infrastructure:**
- **Compute:** AWS EMR cluster  core nodes, Spark executors configured at 2 cores / 6 GB memory each, 2 executor instances, 16 shuffle partitions.
- **Storage:** AWS S3 for raw data, processed Parquet datasets, trained model artifacts, and evaluation results.
- **Processing engine:** Apache Spark (PySpark, Spark MLlib) with Kryo serialization for performance.
- **Serialization/model persistence:** Spark ML Pipelines, persisted directly to S3.

Full pipeline code is organized into four stages, mirroring the project scripts:
1. `data_management_cic_ids2017/2018.py` — data ingestion, cleaning, indexing, and scaling.
2. `pipeline_random_forest.py` — Random Forest training, evaluation, and hyperparameter tuning.
3. `pipeline_kmeans.py` — KMeans training (benign-only), threshold calibration, batch evaluation, and hyperparameter optimization.
4. `streaming/` — live TCP-streaming inference demo against the deployed pipeline. 

---

## Dataset

- **Training:** [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — Monday traffic (benign only) used to train KMeans; Wednesday/Friday (benign + attack) used to calibrate the anomaly threshold.
- **Testing:** Tuesday/Thursday traffic (held out, never used in training or calibration) — includes both known and previously unseen attack categories (e.g. Web Attack, Infiltration).
- **Streaming demo:** [CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) traffic, replayed over TCP to simulate a live production feed.

This train/calibrate/test split was deliberately chosen to prevent data leakage — the calibration step (threshold selection) never touches the final test set.

---

## Methodology

**Random Forest (supervised)**
Trained directly on labeled traffic to classify flows into benign or specific attack categories.

**KMeans (unsupervised)**
Trained **exclusively on benign traffic** the model never sees an attack during training. At inference time, a flow's distance to its assigned cluster centroid is used as an anomaly score: flows far from any "normal" centroid are flagged as anomalous. The anomaly threshold was calibrated by grid-searching over percentile-based distance candidates (P90–P99.9) on the labeled calibration set, selecting the value that maximizes F1.

**Hyperparameter optimization**
Since KMeans is not compatible with Spark MLlib's `CrossValidator`, a manual grid search (8 configurations, varying `k`, `maxIter`, `tol`) was run over a 30% sample of benign traffic, selecting by Silhouette score, then re-trained on the full benign set.

---

## Results

Metrics below are from **batch evaluation on the held-out test set**, run on the distributed EMR cluster.

| Model | Metric | Value | Note |
|---|---|---|---|
| Random Forest | Accuracy | 0.974 | Inflated by class imbalance — 97% of traffic is benign |
| Random Forest | Precision (macro) | 0.493 | Averaged across classes; struggles on rare attack types |
| Random Forest | Recall (macro) | 0.493 | Misses attack types underrepresented in training |
| Random Forest | F1 (macro / weighted) | 0.493 / 0.973 | Weighted score is dominated by the benign class |
| Random Forest | False Positive Rate | 1.3% | Low false-alarm rate on benign traffic |
| KMeans | Silhouette Score | 0.428 | Moderate cluster separation; improved from k=10 to k=30 |
| KMeans | AUC-ROC | 0.653 | Only modestly above random (0.5) |
| KMeans | False Positive Rate | 4.1% | Higher false-alarm rate than Random Forest |
| KMeans | Recall | ~1% | Very low — see interpretation below |

### Interpretation

The headline accuracy numbers look strong but are misleading on their own: CIC-IDS2017 is heavily imbalanced toward benign traffic, so a model can score well on accuracy and weighted F1 while performing poorly on the minority (attack) classes that actually matter for intrusion detection. The macro-averaged metrics, which weight every class equally regardless of frequency  tell the more honest story: Random Forest's macro F1 of 0.49 reflects real difficulty generalizing to rarer or unseen attack types.

KMeans, evaluated as a standalone anomaly detector, currently underperforms: an AUC-ROC of 0.653 is only modestly better than chance, and its recall (~1%) indicates it is not effectively separating attacks from benign traffic using distance-to-centroid alone. This is a genuine limitation of the current approach, not a hidden one, see *Limitations & Future Work* below for how this could be improved.

---

## System Performance

Measured during **distributed batch training and validation** on the EMR cluster (not the live streaming demo — see below):

| Metric | Random Forest | KMeans |
|---|---|---|
| Throughput | ~71,700 records/sec | ~55,700 records/sec |
| Training time | 150s (optimized from 214s) | 62s |
| Avg. per-record processing cost | ~13.9 µs | ~18.0 µs |

The per-record figures above are derived from total batch wall-clock time divided by record count under full cluster parallelization — they characterize distributed batch throughput, not an independently measured single-record latency.

**Not yet measured:** horizontal scalability (cluster size 2 → 5 nodes) is a planned future benchmark, not a completed one.

---

## Live Streaming Demo

To validate that the trained pipeline actually functions as a deployed, real-time system — not just a batch-trained model — a live demo streams CIC-IDS2018 traffic over TCP into the deployed Spark pipeline for real-time inference.

This demo was designed to be observed live rather than benchmarked, so no throughput/latency numbers are reported for it here. The full setup, scripts, and a step-by-step guide to running the demo is in the final section.

---

## Limitations & Future Work

- **Class imbalance** significantly affects macro-averaged classification metrics; techniques such as class weighting, SMOTE, or focal loss could improve minority-class recall.
- **KMeans recall is very low** as a standalone detector; a hybrid scoring approach (combining distance-to-centroid with other features, or using a different anomaly-scoring method such as isolation forests) is a promising next step.
- **Horizontal scalability** (cluster size vs. throughput) has not yet been benchmarked.
- **Streaming demo metrics** (live throughput/latency under load) were not captured and would be a valuable addition for a production-readiness assessment.

---

## Tech Stack

- **Language:** Python
- **Distributed processing:** Apache Spark, Spark MLlib
- **Cloud infrastructure:** AWS EMR, AWS S3
- **Libraries:** boto3, NumPy
- **Dataset:** CIC-IDS2017, CIC-IDS2018

---

## How to Run


1. Provision an AWS EMR cluster with Spark installed.
2. Upload raw CIC-IDS2017/2018 data to the configured S3 bucket paths.
3. Run the pipeline scripts in order:

   ```bash
   spark-submit gestion_datos.py
   spark-submit pipeline_random_forest.py
   spark-submit pipeline_kmeans.py
   ```

4. To run the live streaming demo, 

This is the live demo. It requires **three terminals open simultaneously**.

 Prepare the scripts on each machine
On the EC2 (Terminal 1):
```bash
ssh -i <key.pem> ec2-user@<EC2-PUBLIC-IP>
aws s3 cp s3://project-anomalies-emr/scripts/tcp_sender.py .
```

On the EMR master (Terminal 2):
```bash
ssh -i <key.pem> hadoop@<MASTER-DNS>
aws s3 cp s3://project-anomalies-emr/scripts/streaming_demo.py .
aws s3 cp s3://project-anomalies-emr/scripts/monitor.py .
```

**Step 1 — Launch the TCP producer (Terminal 1 — EC2)**

```bash
# Benign traffic (start of the demo)
python3 tcp_sender.py --rate 500

# DDoS-HOIC spike simulation (critical phase)
python3 tcp_sender.py --rate 3500

# Max speed (pure throughput test)
python3 tcp_sender.py --rate 0

# Continuous loop (for longer demos)
python3 tcp_sender.py --rate 500 --loop
```

The terminal will show:

```
[14:30:00] TCP producer initialized
[14:30:00] Source  : s3://proyecto-anomalias-emr/data/cic-ids2018/streaming/processed/
[14:30:00] Host    : 0.0.0.0:9999
[14:30:00] Rate    : 500 rows/s
[14:30:00] Parquet found: 3 file(s)
[14:30:01] Listening on 0.0.0.0:9999 — waiting for Spark...
```

> **The producer blocks, waiting for Spark to connect.** It won't proceed until the consumer connects. Launch step 2 within 60 seconds.

**Step 2 — Launch the Spark consumer (Terminal 2 — EMR master)**

```bash
spark-submit --master yarn \
  --conf spark.executor.cores=2 \
  --conf spark.executor.memory=6g \
  --conf spark.executor.instances=2 \
  streaming_demo_v2.py \
  --producer-host <EC2-PRIVATE-IP> \
  --producer-port 9999 \
  --trigger 10
```

Available parameters:

| Parameter | Default | Description |
|---|---|---|
| `--producer-host` | `localhost` | Private IP of the producer EC2 |
| `--producer-port` | `9999` | TCP port of the producer |
| `--trigger` | `10` | Micro-batch duration, in seconds |

The terminal will show, per micro-batch:

```
══════════════════════════════════════════════════════════════════════
  BATCH #  12   14:32:10   rows:  5,000   proc:  3,420 ms  [cache: 3 jobs]
──────────────────────────────────────────────────────────────────────

  RANDOM FOREST — Supervised classification
    benign                  ████████████████░░░░░░░░   4,100  ( 82.0%)
    ddos_loic_http          ████░░░░░░░░░░░░░░░░░░░░     700  ( 14.0%)
    ddos_hoic               ██░░░░░░░░░░░░░░░░░░░░░░     200  (  4.0%)

    RF anomaly rate  :  18.0% ↑

  KMEANS — Unsupervised detection  (threshold: 15.0)
    Normal   ████████████░░░░░░░░░░░░   3,800  ( 76.0%)
    Anomaly  ████████░░░░░░░░░░░░░░░░   1,200  ( 24.0%)

    Mean dist. → centroid   :   18.43
    Max dist. in batch      :   87.21
══════════════════════════════════════════════════════════════════════
```

**Step 3 — Launch the monitor (Terminal 3 — local)**

```bash
# Refresh every 5 seconds (default)
python3 monitor.py

# Refresh every 3 seconds
python3 monitor.py --interval 3

# Keep a 50-batch history
python3 monitor.py --interval 5 --history 50
```

**Step 4 — Open the Spark UI (local terminal)**

```bash
# SSH tunnel to access the Spark UI
ssh -i <key.pem> \
    -L 4040:localhost:4040 \
    -N \
    hadoop@<MASTER-DNS>

# Open in your browser:
# http://localhost:4040

```
---

## Authors
Sergio Andres Guevara Ramirez 

Samuel David Rojas Cardenas

