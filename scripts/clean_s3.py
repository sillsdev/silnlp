import argparse
import csv
import datetime
import os
import re
import time
from typing import Tuple

import boto3

MONTH_IN_SECONDS = 2628288


def stats(
    num_deleted_research: int,
    num_deleted_production: int,
    space_freed_research: int,
    space_freed_production: int,
    dry_run: bool,
) -> None:

    total_deleted = num_deleted_research + num_deleted_production
    total_space_freed = space_freed_research + space_freed_production

    print("Research:")
    print("Number of files " + ("that would be " if dry_run else "") + f"deleted: {num_deleted_research}")
    print("Storage space " + ("that would be " if dry_run else "") + f"freed (GB): {space_freed_research / 2 ** 30}")
    print("Production:")
    print("Number of files " + ("that would be " if dry_run else "") + f"deleted: {num_deleted_production}")
    print("Storage space " + ("that would be " if dry_run else "") + f"freed (GB): {space_freed_production / 2 ** 30}")
    print("Total:")
    print("Number of files " + ("that would be " if dry_run else "") + f"deleted: {total_deleted}")
    print("Storage space " + ("that would be " if dry_run else "") + f"freed (GB): {total_space_freed / 2 ** 30}")


def clean_research(max_months: int, dry_run: bool) -> Tuple[int, int]:
    print("Cleaning research")
    regex_to_delete = re.compile(
        r"^MT/experiments/.+/run/(checkpoint.*(pytorch_model\.bin|\.safetensors)$|ckpt.+\.(data-00000-of-00001|index)$)"
    )
    # create a csv filename to store the deleted files that includes the current datetime
    output_csv = f"deleted_research_files_{time.strftime('%Y%m%d-%H%M%S')}" + ("_dryrun" if dry_run else "") + ".csv"
    return _delete_data(
        max_months, dry_run, regex_to_delete, output_csv, bucket_service="minio", checkpoint_protection=True
    )


def clean_production(max_months: int, dry_run: bool) -> Tuple[int, int]:
    print("Cleaning production")
    regex_to_delete = re.compile(r"^(production|dev|int-qa|ext-qa)/builds/.+")
    output_csv = f"deleted_production_files_{time.strftime('%Y%m%d-%H%M%S')}" + ("_dryrun" if dry_run else "") + ".csv"
    return _delete_data(max_months, dry_run, regex_to_delete, output_csv, bucket_service="aws")


def _delete_data(
    max_months: int,
    dry_run: bool,
    regex_to_delete: str,
    output_csv: str,
    bucket_service: str,
    checkpoint_protection: bool = False,
) -> Tuple[int, int]:
    max_age = max_months * MONTH_IN_SECONDS
    if bucket_service == "minio":
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
        )
        bucket_name = "nlp-research"
    else:
        s3 = boto3.client("s3")
        bucket_name = "silnlp"
    paginator = s3.get_paginator("list_objects_v2")
    total_deleted = 0
    storage_space_freed = 0
    keep_until_dates = {}
    # First pass, identify keep until files
    # which must follow the format keep_until_YYYY-MM-DD.lock and be located in the same folder
    # as the experiment's config.yml file
    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page["Contents"]:
            s3_filename = obj["Key"]
            parts = s3_filename.split("/")

            if parts[-1].startswith("keep_until_"):
                try:
                    date_str = parts[-1].split("_")[-1].replace(".lock", "")
                    keep_until_timestamp = datetime.datetime.strptime(date_str, "%Y-%m-%d").timestamp()

                    folder_path = "/".join(parts[:-1])
                    keep_until_dates[folder_path] = keep_until_timestamp
                except ValueError:
                    print(f"Invalid keep_until format in {s3_filename}. Should follow keep_until_YYYY-MM-DD.lock")

    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        if dry_run:
            csv_writer.writerow(["Filename", "LastModified", "Eligible for Deletion", "Extra Info"])
        else:
            csv_writer.writerow(["Filename", "LastModified", "Deleted", "Extra Info"])
        for page in paginator.paginate(Bucket=bucket_name):
            for obj in page["Contents"]:
                s3_filename = obj["Key"]
                if regex_to_delete.search(s3_filename) is None:
                    continue

                last_modified = obj["LastModified"].timestamp()
                now = time.time()

                delete = False
                if checkpoint_protection:
                    parts = s3_filename.split("/")
                    if len(parts) >= 4:
                        experiment_folder = "/".join(parts[:-3])
                        if experiment_folder in keep_until_dates:
                            protect_until = keep_until_dates[experiment_folder]
                            if now < protect_until:
                                print(
                                    f"Skipping {s3_filename} (Experiment '{experiment_folder}' protected until "
                                    f"{datetime.datetime.fromtimestamp(protect_until, tz=datetime.timezone.utc)})"
                                )
                                csv_writer.writerow(
                                    [
                                        s3_filename,
                                        last_modified,
                                        delete,
                                        f"Protected until "
                                        f"{datetime.datetime.fromtimestamp(protect_until, tz=datetime.timezone.utc)})",
                                    ]
                                )
                                continue
                    else:
                        raise RuntimeError(
                            f"Invalide checkpoint path: {s3_filename}. "
                            f"Either disable checkpoint protection "
                            f"or double check that only checkpoints are included in the regex_to_delete."
                        )

                if now - last_modified > max_age:
                    delete = True
                    print(s3_filename)
                    print(f"{(now - last_modified) / MONTH_IN_SECONDS} months old")
                    if not dry_run:
                        s3.delete_object(Bucket=bucket_name, Key=s3_filename)
                        print("Deleted")
                    total_deleted += 1
                    storage_space_freed += obj["Size"]
                csv_writer.writerow([s3_filename, last_modified, delete])
    return total_deleted, storage_space_freed


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove old files from S3 bucket")
    parser.add_argument(
        "--max-months-research", type=int, default=1, help="Maximum age of research checkpoints to keep in months"
    )
    parser.add_argument(
        "--max-months-production", type=int, default=2, help="Maximum age of production files to keep in months"
    )
    parser.add_argument(
        "--dry-run",
        default=False,
        action="store_true",
        help="Don't delete any files, just report what would be deleted and how much space would be saved",
    )
    args = parser.parse_args()

    num_deleted_research, space_freed_research = clean_research(args.max_months_research, args.dry_run)
    num_deleted_production, space_freed_production = clean_production(args.max_months_production, args.dry_run)

    stats(num_deleted_research, num_deleted_production, space_freed_research, space_freed_production, args.dry_run)


if __name__ == "__main__":
    main()
