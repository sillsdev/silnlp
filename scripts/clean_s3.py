import argparse
import re
import time
from typing import Tuple

import boto3

MONTH_IN_SECONDS = 2628288


def clean_s3(max_months: int, dry_run: bool) -> None:
    print("Cleaning research")
    research_total_deleted, research_total_space_freed = clean_research(max_months, dry_run)
    print("Cleaning production")
    production_total_deleted, production_total_space_freed = clean_production(max_months, dry_run)

    total_deleted = research_total_deleted + production_total_deleted
    storage_space_freed = research_total_space_freed + production_total_space_freed

    print("Research:")
    print("Number of files " + ("that would be " if dry_run else "") + f"deleted: {research_total_deleted}")
    print(
        "Storage space " + ("that would be " if dry_run else "") + f"freed (GB): {research_total_space_freed / 2 ** 30}"
    )
    print("Production:")
    print("Number of files " + ("that would be " if dry_run else "") + f"deleted: {production_total_deleted}")
    print(
        "Storage space "
        + ("that would be " if dry_run else "")
        + f"freed (GB): {production_total_space_freed / 2 ** 30}"
    )
    print("Total:")
    print("Number of files " + ("that would be " if dry_run else "") + f"deleted: {total_deleted}")
    print("Storage space " + ("that would be " if dry_run else "") + f"freed (GB): {storage_space_freed / 2 ** 30}")


def clean_research(max_months: int, dry_run: bool) -> Tuple[int, int]:
    regex_to_delete = re.compile(
        r"^MT/experiments/.+/run/(checkpoint.*(pytorch_model\.bin|\.safetensors)$|ckpt.+\.(data-00000-of-00001|index)$)"
    )
    return _delete_data(max_months, dry_run, regex_to_delete)


def clean_production(max_months: int, dry_run: bool) -> Tuple[int, int]:
    regex_to_delete = re.compile(
        r"^(production|dev|int-qa|ext-qa)/builds/.+"
        r"(pretranslate.src.json|pretranslate.tgt.json|train.src.txt|train.tgt.txt)"
    )
    return _delete_data(max_months, dry_run, regex_to_delete)


def _delete_data(max_months: int, dry_run: bool, regex_to_delete: str) -> Tuple[int, int]:
    max_age = max_months * MONTH_IN_SECONDS

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    total_deleted = 0
    storage_space_freed = 0
    for page in paginator.paginate(Bucket="silnlp"):
        for obj in page["Contents"]:
            if regex_to_delete.search(obj["Key"]) is None:
                continue
            last_modified = obj["LastModified"].timestamp()
            now = time.time()
            if now - last_modified <= max_age:
                continue
            print(obj["Key"])
            print(f"{(now - last_modified) / MONTH_IN_SECONDS} months old")
            if not dry_run:
                s3.delete_object(Bucket="silnlp", Key=obj["Key"])
                print("Deleted")
            total_deleted += 1
            storage_space_freed += obj["Size"]
    return total_deleted, storage_space_freed


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove old model checkpoints from S3 bucket")
    parser.add_argument("--max-months", type=int, default=1, help="Maximum age of checkpoints to keep in months")
    parser.add_argument(
        "--dry-run",
        default=False,
        action="store_true",
        help="Don't delete any files, just report what would be deleted and how much space would be saved",
    )
    args = parser.parse_args()

    clean_s3(args.max_months, args.dry_run)


if __name__ == "__main__":
    main()
