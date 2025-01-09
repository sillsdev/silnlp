cp /workspaces/silnlp/scripts/rclone/rclone.conf ~/.config/rclone

sed -i -e "s#access_key_id = x*#access_key_id = $AWS_ACCESS_KEY_ID#" ~/.config/rclone/rclone.conf
sed -i -e "s#secret_access_key = x*#secret_access_key = $AWS_SECRET_ACCESS_KEY#" ~/.config/rclone/rclone.conf

echo "Mounting S3 bucket..."
rclone mount --vfs-cache-mode full --use-server-modtime s3silnlp:silnlp /silnlp
echo "Done"
