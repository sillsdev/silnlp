#!/bin/bash
env_file="$1"
while IFS= read -r line; do
    var_name=$(echo "$line" | cut -d'=' -f1)
    sed -i "/^export $var_name=/d" ~/.bashrc
    export "$line"
    echo "export $line" >> ~/.bashrc
done < "$env_file"

CLEANED_URL=$(echo "$MINIO_ENDPOINT_URL" | sed -E "s|^https://||" | sed -E "s|:[^:]*$||")
if [ -n "$MINIO_ENDPOINT_IP" ] && [ -n "$CLEANED_URL" ]; then
    # Remove existing entries
    sudo sed -i "/^$MINIO_ENDPOINT_IP /d" /etc/hosts
    sudo sed -i "/$CLEANED_URL/d" /etc/hosts
    # Add new entry
    echo "$MINIO_ENDPOINT_IP $CLEANED_URL" | sudo tee -a /etc/hosts > /dev/null
    echo "Updated /etc/hosts: $MINIO_ENDPOINT_IP $CLEANED_URL"
else
    echo "Skipping /etc/hosts update (Environment variables MINIO_ENDPOINT_IP and/or MINIO_ENDPOINT_URL are not set)"
fi