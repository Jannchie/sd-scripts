#!/bin/bash

# 进入无限循环
while true
do
    # 执行 rsync 命令
    watch -n 600 rsync -av /mnt/nfs-mnj-hot-09/tmp/pan/model/ /mnt/nfs-mnj-archive-12/group/creative/pan/models/ckpt/trained

    # 休眠 600 秒 (即 10 分钟)
    sleep 600
done