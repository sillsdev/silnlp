import argparse
import csv
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import boto3


def try_n_times(func: Callable, n=10):
    for i in range(n):
        try:
            return func()
        except Exception as e:
            if i < n - 1:
                print(f"Failed {i+1} of {n} times.  Retrying.")
                time.sleep(2**i)
            else:
                raise e


def get_bucket_objects(bucket) -> dict:
    bucket_objects = {}
    for obj in bucket.objects.all():
        bucket_objects[obj.key] = obj.last_modified
    return bucket_objects


def sync_buckets(include_checkpoints: bool, dry_run: bool, output_path: str = None) -> None:
    minio_resource = boto3.resource(
        service_name="s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
        # Verify is false if endpoint_url is an IP address. Aqua/Cheetah connecting to MinIO need this disabled for now.
        verify=False if re.match(r"https://\d+\.\d+\.\d+\.\d+", os.getenv("MINIO_ENDPOINT_URL")) else True,
    )
    minio_bucket = minio_resource.Bucket("nlp-research")

    b2_resource = boto3.resource(
        service_name="s3",
        endpoint_url=os.getenv("B2_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("B2_KEY_ID"),
        aws_secret_access_key=os.getenv("B2_APPLICATION_KEY"),
        verify=True,
    )
    b2_bucket = b2_resource.Bucket("silnlp")

    # Get all objects in the MinIO bucket
    print("Getting objects from MinIO")
    minio_objects: dict[str, datetime] = try_n_times(lambda: get_bucket_objects(minio_bucket))

    # Get all objects in the B2 bucket
    print("Getting objects from B2")
    b2_objects: dict[str, datetime] = try_n_times(lambda: get_bucket_objects(b2_bucket))

    if not include_checkpoints:
        print("Excluding model checkpoints from the sync")
        keys_to_remove = set()
        for key in minio_objects.keys():
            # Check if key matches regex
            if re.match(
                r"^MT/experiments/.+/run/(checkpoint.*(pytorch_model\.bin|\.safetensors)$|ckpt.+\.(data-00000-of-00001|index)$)",
                key,
            ):
                keys_to_remove.add(key)

        for key in keys_to_remove:
            b2_objects.pop(key, None)
            minio_objects.pop(key, None)

    output_csv_file_name = f"sync_output_{time.strftime('%Y%m%d-%H%M%S')}" + ("_dryrun" if dry_run else "") + ".csv"
    if output_path is not None:
        output_path = Path(output_path)
        # Create the output path if it doesn't exist
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        if output_path.is_dir():
            output_path = Path(output_path / output_csv_file_name)
    else:
        output_path = Path(output_csv_file_name)

    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Filename", "Action"])
        # Get the objects that are in the MinIO bucket but not in the B2 bucket, or have been modified
        objects_to_sync = []
        for key, value in minio_objects.items():
            if key not in b2_objects.keys():
                objects_to_sync.append(key)
            elif value > b2_objects[key]:
                objects_to_sync.append(key)

        objects_to_delete = []
        for key in b2_objects.keys():
            if key not in minio_objects.keys():
                objects_to_delete.append(key)
                if not dry_run:
                    csv_writer.writerow([key, "Deleted from B2"])
                else:
                    csv_writer.writerow([key, "Would be deleted from B2"])
        if not dry_run:
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i : i + 1000]
                delete_params = {"Objects": [{"Key": key} for key in batch]}
                try_n_times(lambda: b2_bucket.delete_objects(Delete=delete_params))

        # Sync the objects to the B2 bucket
        length = len(objects_to_sync)
        if not dry_run:
            print(f"Total objects to sync: {len(objects_to_sync)}")
        else:
            print(f"Total objects that would be synced: {len(objects_to_sync)}")
        x = 0
        for key in objects_to_sync:
            x += 1
            if not dry_run:
                print(f"Syncing, {x}/{length}: {key}")
                with tempfile.TemporaryDirectory() as temp_dir:
                    obj_path = Path(temp_dir) / key
                    obj_path.parent.mkdir(parents=True, exist_ok=True)
                    try_n_times(lambda: minio_bucket.download_file(key, str(obj_path)))
                    try_n_times(lambda: b2_bucket.upload_file(str(obj_path), key))
                csv_writer.writerow([key, "Synced to B2"])
            else:
                print(f"Would be syncing, {x}/{length}: {key}")
                csv_writer.writerow([key, "Would be synced to B2"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync MinIO and B2 buckets")
    parser.add_argument(
        "--include-checkpoints", default=False, action="store_true", help="Include model checkpoints in the sync"
    )
    parser.add_argument(
        "--dry-run",
        default=False,
        action="store_true",
        help="Don't sync any files, just report what would be synced",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        type=str,
        help="The path to the output folder for the CSV file. If not specified, the CSV file will be created in the current working directory.",
    )
    args = parser.parse_args()

    sync_buckets(args.include_checkpoints, args.dry_run, args.output_folder)


if __name__ == "__main__":
    main()
