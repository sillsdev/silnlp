#!/bin/sh
CLEANED_URL=$(echo "$MINIO_ENDPOINT_URL" | sed -E "s|^https://||" | sed -E "s|:[^:]*$||")
if [ -n "$MINIO_ENDPOINT_IP" ] && [ -n "$CLEANED_URL" ]; then
    echo "$MINIO_ENDPOINT_IP $CLEANED_URL" | tee -a /etc/hosts > /dev/null
    echo "Updated /etc/hosts: $MINIO_ENDPOINT_IP $CLEANED_URL"
else
    echo "Skipping /etc/hosts update (Environment variables MINIO_ENDPOINT_IP and/or MINIO_ENDPOINT_URL are not set)"
fi