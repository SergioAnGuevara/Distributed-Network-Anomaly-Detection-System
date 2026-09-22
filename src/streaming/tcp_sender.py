#!/usr/bin/env python3


import socket
import time
import argparse
import json
import sys
import boto3
from io import BytesIO

#  ANSI colors 
GRN = "\033[92m"; YLW = "\033[93m"; RED = "\033[91m"
CYN = "\033[96m"; BLD = "\033[1m";  RST = "\033[0m"

#  Arguments 
parser = argparse.ArgumentParser(
    description="TCP producer — reads the homologated IDS2018 Parquet and emits over a socket."
)
parser.add_argument("--host",   default="0.0.0.0",
                    help="Listening IP of the TCP server (default: 0.0.0.0)")
parser.add_argument("--port",   default=9999, type=int,
                    help="TCP port (default: 9999)")
parser.add_argument("--rate",   default=500,  type=int,
                    help="Rows/s to emit. 0 = maximum speed (default: 500)")
parser.add_argument("--bucket", default="project-anomalies-emr",
                    help="Project S3 bucket")
parser.add_argument("--prefix", default="data/cic-ids2018/streaming/processed/",
                    help="S3 prefix of the homologated Parquet")
parser.add_argument("--region", default="us-east-1")
parser.add_argument("--loop",   action="store_true",
                    help="Repeat the dataset indefinitely")
args = parser.parse_args()

DELAY = 1.0 / args.rate if args.rate > 0 else 0


#  Helpers 
def log(msg, color=CYN):
    ts = time.strftime("%H:%M:%S")
    print(f"{color}[{ts}] {msg}{RST}", flush=True)

def log_progress(sent, start_time):
    elapsed = time.time() - start_time
    rate    = sent / elapsed if elapsed > 0 else 0
    log(f"{sent:>10,} rows | {rate:>7.0f} rows/s | "
        f"{elapsed:>6.1f}s", GRN)


#  Reading Parquet from S3 ─
def list_parquet(bucket, prefix, region):
    """Lists all the .parquet files under the S3 prefix."""
    s3   = boto3.client("s3", region_name=region)
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [
        obj["Key"] for obj in resp.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]
    if not keys:
        raise FileNotFoundError(
            f"No .parquet files found in "
            f"s3://{bucket}/{prefix}\n"
            f"Did you run data_management_cic_ids2018.py first?"
        )
    log(f"Parquet found: {len(keys)} file(s) in s3://{bucket}/{prefix}")
    return keys


def read_parquet_s3(bucket, key, region):
    """
    Downloads a Parquet file from S3 and returns an iterator of
    rows serialized as CSV (string), ready to be sent over TCP.

    """
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
    except ImportError:
        log("ERROR: pyarrow is not installed.", RED)
        log("Install with: pip3 install pyarrow --break-system-packages", YLW)
        sys.exit(1)

    s3  = boto3.client("s3", region_name=region)
    log(f"Downloading {key.split('/')[-1]} ...")
    obj  = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()

    table   = pq.read_table(BytesIO(data))
    columns = table.column_names

    log(f"  {len(columns)} columns | {table.num_rows:,} rows")

    # Order columns: features (alphabetical) + label + metadata
    meta_cols    = {"label", "_source_file", "_split"}
    feature_cols = sorted(c for c in columns if c not in meta_cols)
    ordered_cols = feature_cols + [c for c in ["label", "_source_file", "_split"]
                                   if c in columns]

    table = table.select(ordered_cols)

    # Generate CSV row by row
    for batch in table.to_batches(max_chunksize=1000):
        batch_dict = batch.to_pydict()
        n_rows     = batch.num_rows
        for i in range(n_rows):
            row = [str(batch_dict[c][i]) for c in ordered_cols]
            yield ",".join(row) + "\n"



def main():
    log(f"TCP producer initialized", BLD)
    log(f"Source  : s3://{args.bucket}/{args.prefix}")
    log(f"Host    : {args.host}:{args.port}")
    log(f"Rate    : {'MAXIMUM' if args.rate == 0 else f'{args.rate} rows/s'}")
    log(f"Loop    : {'yes' if args.loop else 'no'}")
    print()

    # Verify that the Parquet exists before opening the socket
    try:
        parquet_keys = list_parquet(args.bucket, args.prefix, args.region)
    except FileNotFoundError as e:
        log(str(e), RED)
        sys.exit(1)

    # Open the TCP socket as a server
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((args.host, args.port))
    except OSError as e:
        log(f"Cannot open port {args.port}: {e}", RED)
        log("Is there another process on that port? Try: lsof -i :9999", YLW)
        sys.exit(1)

    srv.listen(1)
    log(f"Listening on {args.host}:{args.port} — waiting for Spark...", YLW)

    conn, addr = srv.accept()
    log(f"Spark consumer connected from {addr}", GRN)
    print()

    sent = 0
    start_time = time.time()

    try:
        while True:
            for key in parquet_keys:
                for row in read_parquet_s3(args.bucket, key, args.region):
                    conn.sendall(row.encode("utf-8"))
                    sent += 1
                    if DELAY > 0:
                        time.sleep(DELAY)
                    if sent % 10_000 == 0:
                        log_progress(sent, start_time)

            if not args.loop:
                break
            log("Dataset complete. Restarting...", YLW)

    except BrokenPipeError:
        log("Spark consumer disconnected.", YLW)
    except KeyboardInterrupt:
        log("Manual interrupt (Ctrl+C).", YLW)
    finally:
        elapsed = time.time() - start_time
        rate    = sent / elapsed if elapsed > 0 else 0
        print()
        log(f"{'─'*48}")
        log(f"FINAL SUMMARY", BLD)
        log(f"  Rows sent     : {sent:,}")
        log(f"  Total time    : {elapsed:.1f}s")
        log(f"  Average rate  : {rate:.0f} rows/s")
        log(f"{'─'*48}")
        conn.close()
        srv.close()


if __name__ == "__main__":
    main()