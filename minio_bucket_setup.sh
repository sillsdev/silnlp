cp /workspaces/silnlp/scripts/rclone/rclone.conf ~/.config/rclone

sed -i -e "s#access_key_id = x*#access_key_id = $MINIO_ACCESS_KEY#" ~/.config/rclone/rclone.conf
sed -i -e "s#secret_access_key = x*#secret_access_key = $MINIO_SECRET_KEY#" ~/.config/rclone/rclone.conf

echo "Mounting MinIO bucket..."
rclone mount --daemon --log-file=rclone_log.txt --log-level=DEBUG  --vfs-cache-mode full --use-server-modtime miniosilnlp:nlp-research ~/M
echo "Done"
