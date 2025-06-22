CATEGORIES=("airplane" "car" "bag" "table")
MAX_ITERS=200000
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
if [ ${#CATEGORIES[@]} -gt $NUM_GPUS ]; then
  echo "错误：需要 ${#CATEGORIES[@]} 个GPU，但只有 $NUM_GPUS 个可用。"
  exit 1
fi
for i in ${!CATEGORIES[@]}; do
  CATEGORY=${CATEGORIES[$i]}
  GPU_ID=$i
  TAG="${CATEGORY}_${MAX_ITERS}"
  echo "在 GPU $GPU_ID 上启动对类别 '$CATEGORY' 的训练..."
  echo "日志标签 (tag): $TAG"
  echo "最大迭代次数: $MAX_ITERS"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_gen.py \
    --categories $CATEGORY \
    --max_iters $MAX_ITERS \
    --tag $TAG &

  echo "进程已启动，PID: $!"
  echo "--------------------------------------------------"
done

echo "所有训练进程已启动。等待它们全部完成..."
wait
echo "所有训练任务已完成。"