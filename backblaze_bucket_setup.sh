cp /workspaces/silnlp/scripts/rclone/rclone.conf ~/.config/rclone

sed -i -e "s#account = x*#account = $B2_KEY_ID#" ~/.config/rclone/rclone.conf
sed -i -e "s#key = x*#key = $B2_APPLICATION_KEY#" ~/.config/rclone/rclone.conf

echo "Mounting Backblaze bucket..."
rclone mount --daemon --log-file=/root/rclone_log.txt --log-level=DEBUG --vfs-cache-mode full --use-server-modtime miniosilnlp:nlp-research ~/B
echo "Done"
