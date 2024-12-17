echo "Installing fuse3..."
apt update
apt install -y fuse3

echo "Downloading rclone..."
curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip
unzip rclone-current-linux-amd64.zip
cd rclone-*-linux-amd64

echo "Installing rclone..."
cp rclone /usr/bin/
chown root:root /usr/bin/rclone
chmod 755 /usr/bin/rclone

mkdir -p /usr/local/share/man/man1
cp rclone.1 /usr/local/share/man/man1/
command -v mandb && echo "Creating manpages..." || echo "Skipping manpages..."

cd ..
rm -r rclone-*-linux-amd64

echo "Configuring rclone..."
mkdir -p ~/.config/rclone

cp scripts/rclone/rclone.conf ~/.config/rclone

sed -i -e "s#access_key_id = x*#access_key_id = $AWS_ACCESS_KEY_ID#" ~/.config/rclone/rclone.conf
sed -i -e "s#secret_access_key = x*#secret_access_key = $AWS_SECRET_ACCESS_KEY#" ~/.config/rclone/rclone.conf

mkdir -p /silnlp

echo "Mounting S3 bucket..."
rclone mount --vfs-cache-mode full --use-server-modtime s3silnlp:silnlp /silnlp
echo "Done"