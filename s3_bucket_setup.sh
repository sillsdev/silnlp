cp scripts/rclone/rclone.conf ~/.config/rclone

sed -i -e "s#access_key_id = x*#access_key_id = $MINIO_ACCESS_KEY#" ~/.config/rclone/rclone.conf
sed -i -e "s#secret_access_key = x*#secret_access_key = $MINIO_SECRET_KEY#" ~/.config/rclone/rclone.conf
sed -i -e "s#endpoint = x*#endpoint = $MINIO_ENDPOINT_URL#" ~/.config/rclone/rclone.conf

echo "Mounting MinIO bucket..."
rclone mount --vfs-cache-mode full --use-server-modtime miniosilnlp:nlp-research ~/M
echo "Done"
