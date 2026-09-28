def get_docker_setup_bash_script() -> list[str]:
    return [
        "uv sync --locked --no-dev --no-install-project",
        "apt-get install --no-install-recommends -y fuse3 rclone",
        "mkdir -p /root/M",
        "mkdir -p /root/.config/rclone",
        "cp scripts/rclone/rclone.conf /root/.config/rclone/",
        'sed -i -e "s#access_key_id = x*#access_key_id = $MINIO_ACCESS_KEY#" ~/.config/rclone/rclone.conf',
        'sed -i -e "s#secret_access_key = x*#secret_access_key = $MINIO_SECRET_KEY#" ~/.config/rclone/rclone.conf',
        'sed -i -e "s#endpoint = .*#endpoint = $MINIO_ENDPOINT_URL#" ~/.config/rclone/rclone.conf',
        "rclone mount --daemon --no-check-certificate --log-file=/root/rclone_log.txt --log-level=DEBUG "
        "--vfs-cache-mode full --vfs-cache-max-size 15G --use-server-modtime miniosilnlp:nlp-research /root/M",
    ]

def get_docker_args() -> list[str]:
    return [
        "--env TOKENIZERS_PARALLELISM='false'",
        "--cap-add SYS_ADMIN",
        "--device /dev/fuse",
        "--security-opt apparmor=docker-apparmor",
        "--env CHECK_TRANSFERS=1",
        "--env SIL_NLP_DATA_PATH=/root/M",
    ]