if [ $# -eq 0 ]; then
    docker run -itd --gpus all \
      --name 4d-humans \
      -v `pwd`:/4D-Humans \
      -v $(readlink -f `pwd`/../PHALP):/PHALP \
      -v $(readlink -f `pwd`/../YOLOv6):/YOLOv6 \
      -v /home/$USER/.gsutil:/root/.gsutil \
      -v /home/$USER/.config:/root/.config \
      --restart on-failure \
      ralphhan/4d-humans \
      bash /4D-Humans/$0 1
    exit 0
fi

cd /4D-Humans
NUM_GPUS=$(nvidia-smi -L | wc -l)
for GPU_ID in $(seq 0 $((NUM_GPUS - 1)))
do
    CUDA_VISIBLE_DEVICES=$GPU_ID /opt/conda/envs/4D-humans/bin/python track_all.py &
done
wait
