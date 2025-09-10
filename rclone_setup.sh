#!/bin/bash
apt-get install --no-install-recommends -y fuse3 rclone
mkdir -p /root/.config/rclone
cp scripts/rclone/rclone.conf /root/.config/rclone/
BUCKET_TYPE=$1
if [ "$BUCKET_TYPE" = "minio" ]; then
    export SIL_NLP_DATA_PATH="/root/M"
    mkdir -p /root/M
    sed -i -e "s#access_key_id = x*#access_key_id = $MINIO_ACCESS_KEY#" /root/.config/rclone/rclone.conf
    sed -i -e "s#secret_access_key = x*#secret_access_key = $MINIO_SECRET_KEY#" /root/.config/rclone/rclone.conf

    echo "Mounting MinIO bucket..."
    rclone mount --daemon --log-file=rclone_log.txt --log-level=DEBUG  --vfs-cache-mode full --use-server-modtime miniosilnlp:nlp-research ~/M
    echo "Done"
elif [ "$BUCKET_TYPE" = "backblaze" ]; then
    export SIL_NLP_DATA_PATH="/root/B"
    mkdir -p /root/B
    sed -i -e "s#account = x*#account = $B2_KEY_ID#" /root/.config/rclone/rclone.conf
    sed -i -e "s#key = x*#key = $B2_APPLICATION_KEY#" /root/.config/rclone/rclone.conf

    echo "Mounting Backblaze bucket..."
    rclone mount --daemon --log-file=rclone_log.txt --log-level=DEBUG  --vfs-cache-mode full --use-server-modtime b2silnlp:silnlp ~/B
    echo "Done"
fi