LOGS_ROOT_DIR="logs_exp"
OUTPUT_ROOT_DIR="extracted_point_clouds"
POINT_CLOUD_TAG="val/pointcloud_VERTEX"
PYTHON_EXECUTABLE="python"
if [ ! -d "$LOGS_ROOT_DIR" ]; then
    echo "错误: 日志根目录 '$LOGS_ROOT_DIR' 不存在。"
    exit 1
fi
mkdir -p "$OUTPUT_ROOT_DIR"
echo "自动化提取开始..."
echo "日志来源: $LOGS_ROOT_DIR"
echo "结果将保存至: $OUTPUT_ROOT_DIR"
echo "使用的点云标签: $POINT_CLOUD_TAG"
for experiment_dir in "$LOGS_ROOT_DIR"/GEN_*; do
    if [ -d "$experiment_dir" ]; then
        $PYTHON_EXECUTABLE utils/extract_pcs.py --input_dir "$experiment_dir" --output_root "$OUTPUT_ROOT_DIR" --tag "$POINT_CLOUD_TAG"
    fi
done

echo -e "\n所有目录处理完毕！"