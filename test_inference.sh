#!/bin/bash
# ActivityNet 推理快速测试脚本

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║              ActivityNet 推理快速测试                                         ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 检查必需文件
echo "检查必需文件..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ANNOTATION_FILE="data/activitynet_hf/activity_net.v1-3.min.json"
VIDEO_DIR="data/activitynet_hf/v1-3/train_val"
CONFIG_FILE="configs/anet_uniformer.yaml"
INFERENCE_SCRIPT="inference_activitynet.py"

if [ ! -f "$ANNOTATION_FILE" ]; then
    echo "❌ 标注文件不存在: $ANNOTATION_FILE"
    echo "   请先下载数据集"
    exit 1
fi
echo "✅ 标注文件: $ANNOTATION_FILE"

if [ ! -d "$VIDEO_DIR" ]; then
    echo "❌ 视频目录不存在: $VIDEO_DIR"
    echo "   请先下载示例视频"
    exit 1
fi
echo "✅ 视频目录: $VIDEO_DIR"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi
echo "✅ 配置文件: $CONFIG_FILE"

if [ ! -f "$INFERENCE_SCRIPT" ]; then
    echo "❌ 推理脚本不存在: $INFERENCE_SCRIPT"
    exit 1
fi
echo "✅ 推理脚本: $INFERENCE_SCRIPT"

# 检查示例视频数量
VIDEO_COUNT=$(ls -1 $VIDEO_DIR/*.mp4 2>/dev/null | wc -l)
echo "✅ 示例视频数量: $VIDEO_COUNT"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查 checkpoint
echo "查找可用的 checkpoint..."
CHECKPOINT_DIRS=(
    "ckpt/anet_uniformer"
    "ckpt/badminton_video_uniformer*"
    "ckpt"
)

CHECKPOINT=""
for dir in "${CHECKPOINT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        LATEST_CKPT=$(ls -t $dir/*.pth.tar 2>/dev/null | head -1)
        if [ -n "$LATEST_CKPT" ]; then
            CHECKPOINT="$LATEST_CKPT"
            echo "✅ 找到 checkpoint: $CHECKPOINT"
            break
        fi
    fi
done

if [ -z "$CHECKPOINT" ]; then
    echo "⚠️  未找到 checkpoint 文件"
    echo ""
    echo "请提供 checkpoint 路径，或按 Ctrl+C 退出："
    read -p "Checkpoint 路径: " CHECKPOINT
    
    if [ ! -f "$CHECKPOINT" ]; then
        echo "❌ Checkpoint 文件不存在: $CHECKPOINT"
        exit 1
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 创建输出目录
OUTPUT_DIR="inference_results/test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "测试配置:"
echo "  配置文件: $CONFIG_FILE"
echo "  Checkpoint: $CHECKPOINT"
echo "  标注文件: $ANNOTATION_FILE"
echo "  视频目录: $VIDEO_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  测试视频数: 最多 $VIDEO_COUNT 个"
echo ""

# 询问是否继续
read -p "是否开始测试? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "测试已取消"
    exit 0
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                          开始推理测试                                         ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 运行推理
python3 $INFERENCE_SCRIPT \
    --config "$CONFIG_FILE" \
    --checkpoint "$CHECKPOINT" \
    --annotation "$ANNOTATION_FILE" \
    --video-dir "$VIDEO_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --max-videos $VIDEO_COUNT \
    --no-eval \
    --device cuda 2>&1 | tee "$OUTPUT_DIR/inference.log"

RESULT=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $RESULT -eq 0 ]; then
    echo "✅ 推理测试完成!"
    echo ""
    echo "结果文件:"
    echo "  - 预测结果: $OUTPUT_DIR/predictions.json"
    echo "  - Pickle 格式: $OUTPUT_DIR/predictions.pkl"
    echo "  - 日志文件: $OUTPUT_DIR/inference.log"
    echo ""
    
    # 显示预测数量
    if [ -f "$OUTPUT_DIR/predictions.json" ]; then
        PRED_COUNT=$(python3 -c "
import json
with open('$OUTPUT_DIR/predictions.json') as f:
    data = json.load(f)
total = sum(len(preds) for preds in data.get('results', {}).values())
print(f'总预测数: {total}')
for vid, preds in list(data.get('results', {}).items())[:3]:
    print(f'  {vid}: {len(preds)} 个预测')
" 2>/dev/null)
        echo "预测统计:"
        echo "$PRED_COUNT"
    fi
    
    echo ""
    echo "查看结果:"
    echo "  cat $OUTPUT_DIR/predictions.json | jq . | head -50"
    echo ""
    echo "查看详细使用说明:"
    echo "  cat INFERENCE_GUIDE.md"
else
    echo "❌ 推理测试失败"
    echo ""
    echo "查看日志文件获取详细错误信息:"
    echo "  cat $OUTPUT_DIR/inference.log"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                          测试完成                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"

