import re
import time

import boto3

MONTH_IN_SECONDS = 2628288


def main() -> None:

    max_age = 3 * MONTH_IN_SECONDS
    regex_to_delete = re.compile(
        r"^MT/experiments/.+/run/(checkpoint.*(pytorch_model\.bin|\.safetensors)$|ckpt.+\.(data-00000-of-00001|index)$)"
    )

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    total_deleted = 0
    memory_freed = 0
    for page in paginator.paginate(Bucket="aqua-ml-data"):
        for obj in page["Contents"]:
            if regex_to_delete.search(obj["Key"]) is None:
                continue
            last_modified = obj["LastModified"].timestamp()
            now = time.time()
            if now - last_modified <= max_age:
                continue
            print(obj["Key"])
            print(f"{(now - last_modified) / MONTH_IN_SECONDS} months old")
            s3.delete_object(Bucket="aqua-ml-data", Key=obj["Key"])
            total_deleted += 1
            memory_freed += obj["Size"]
    print(f"Number of files deleted: {total_deleted}")
    print(f"Memory freed (GB): {memory_freed / 2 ** 30}")


if __name__ == "__main__":
    main()
