#!/bin/bash

BUCKET_TYPE=$1

# Pick the mountpoint / remote for the requested bucket type.
case "$BUCKET_TYPE" in
    minio)
        MOUNTPOINT="/root/M"
        REMOTE="miniosilnlp:nlp-research"
        ;;
    backblaze)
        MOUNTPOINT="/root/B"
        REMOTE="b2silnlp:silnlp"
        ;;
    *)
        echo "rclone_setup.sh: unknown bucket type '$BUCKET_TYPE' (expected 'minio' or 'backblaze')" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

# If the mountpoint is already a live, responsive rclone mount, there is nothing
# to do. This is the common case when opening a new terminal, and re-mounting
# would fail with "mountpoint is not empty, refusing to mount".
if mountpoint -q "$MOUNTPOINT" && ls "$MOUNTPOINT" >/dev/null 2>&1; then
    export SIL_NLP_DATA_PATH="$MOUNTPOINT"
    echo "rclone: $MOUNTPOINT already mounted, skipping."
    return 0 2>/dev/null || exit 0
fi

# Install rclone only if it isn't already available.
if ! command -v rclone >/dev/null 2>&1; then
    curl https://rclone.org/install.sh | bash
fi

mkdir -p /root/.config/rclone
cp scripts/rclone/rclone.conf /root/.config/rclone/

echo "Cleaning up any stale rclone mount at $MOUNTPOINT..."
pkill -f "rclone mount $REMOTE" || true
fusermount -uz "$MOUNTPOINT" 2>/dev/null || true
sleep 2

if [ "$BUCKET_TYPE" = "minio" ]; then
    export SIL_NLP_DATA_PATH="$MOUNTPOINT"
    mkdir -p "$MOUNTPOINT"
    sed -i -e "s#access_key_id = x*#access_key_id = $MINIO_ACCESS_KEY#" /root/.config/rclone/rclone.conf
    sed -i -e "s#secret_access_key = x*#secret_access_key = $MINIO_SECRET_KEY#" /root/.config/rclone/rclone.conf

    echo "Mounting MinIO bucket..."
    rclone mount --daemon --log-file=rclone_log.txt --log-file-max-size 10M --log-file-max-age 1d --log-level=DEBUG  --vfs-cache-mode full --use-server-modtime "$REMOTE" "$MOUNTPOINT"
    echo "Done"
elif [ "$BUCKET_TYPE" = "backblaze" ]; then
    export SIL_NLP_DATA_PATH="$MOUNTPOINT"
    mkdir -p "$MOUNTPOINT"
    sed -i -e "s#account = x*#account = $B2_KEY_ID#" /root/.config/rclone/rclone.conf
    sed -i -e "s#key = x*#key = $B2_APPLICATION_KEY#" /root/.config/rclone/rclone.conf

    echo "Mounting Backblaze bucket..."
    rclone mount --daemon --log-file=rclone_log.txt --log-file-max-size 10M --log-file-max-age 1d --log-level=DEBUG  --vfs-cache-mode full --use-server-modtime "$REMOTE" "$MOUNTPOINT"
    echo "Done"
fi
