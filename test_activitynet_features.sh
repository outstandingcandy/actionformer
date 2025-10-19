#!/bin/bash
# 测试 ActivityNet 特征提取功能

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║              测试 ActivityNet 特征提取                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 配置
ANNOTATION_FILE="data/activitynet_hf/activity_net.v1-3.min.json"
VIDEO_DIR="data/activitynet_hf/v1-3/train_val"
OUTPUT_DIR="data/activitynet_demo_features"
FEATURE_TYPE="uniformer"

# 检查文件
echo "检查必需文件..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -f "$ANNOTATION_FILE" ]; then
    echo "❌ 标注文件不存在: $ANNOTATION_FILE"
    echo "   请先运行: python3 download_activitynet_selective.py --annotations"
    exit 1
fi
echo "✅ 标注文件: $ANNOTATION_FILE"

if [ ! -d "$VIDEO_DIR" ]; then
    echo "❌ 视频目录不存在: $VIDEO_DIR"
    echo "   请先运行: python3 download_activitynet_selective.py --samples"
    exit 1
fi
echo "✅ 视频目录: $VIDEO_DIR"

# 统计视频数量
VIDEO_COUNT=$(ls -1 $VIDEO_DIR/*.mp4 2>/dev/null | wc -l)
echo "✅ 示例视频数量: $VIDEO_COUNT"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 显示测试配置
echo "测试配置:"
echo "  标注文件: $ANNOTATION_FILE"
echo "  视频目录: $VIDEO_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  特征类型: $FEATURE_TYPE"
echo "  数据集格式: activitynet"
echo "  视频数量: $VIDEO_COUNT"
echo ""

read -p "是否开始提取特征? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "测试已取消"
    exit 0
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                        开始提取特征                                           ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 运行特征提取
python3 extract_features.py \
    --annotations_file "$ANNOTATION_FILE" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --feature_type "$FEATURE_TYPE" \
    --dataset_format activitynet \
    --file_prefix v_ \
    --target_fps 25.0 2>&1 | tee "${OUTPUT_DIR}_extraction.log"

RESULT=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $RESULT -eq 0 ]; then
    echo "✅ 特征提取完成!"
    echo ""
    
    # 统计结果
    if [ -d "$OUTPUT_DIR/features" ]; then
        FEAT_COUNT=$(ls -1 $OUTPUT_DIR/features/*.npy 2>/dev/null | wc -l)
        echo "提取的特征:"
        echo "  特征文件数: $FEAT_COUNT"
        echo "  特征目录: $OUTPUT_DIR/features/"
        
        # 显示第一个特征文件的信息
        FIRST_FEAT=$(ls $OUTPUT_DIR/features/*.npy 2>/dev/null | head -1)
        if [ -n "$FIRST_FEAT" ]; then
            echo ""
            echo "示例特征文件:"
            FEAT_INFO=$(python3 -c "
import numpy as np
feat = np.load('$FIRST_FEAT')
print(f'  文件: $(basename $FIRST_FEAT)')
print(f'  形状: {feat.shape}')
print(f'  维度: {feat.shape[1]}')
print(f'  时间步: {feat.shape[0]}')
print(f'  大小: {feat.nbytes/1024/1024:.2f} MB')
" 2>/dev/null)
            echo "$FEAT_INFO"
        fi
        
        echo ""
        echo "查看所有特征:"
        echo "  ls -lh $OUTPUT_DIR/features/"
        
        # 检查失败列表
        if [ -f "$OUTPUT_DIR/failed_videos.txt" ]; then
            FAILED_COUNT=$(wc -l < "$OUTPUT_DIR/failed_videos.txt")
            echo ""
            echo "⚠️  失败视频: $FAILED_COUNT 个"
            echo "   查看: cat $OUTPUT_DIR/failed_videos.txt"
        fi
    fi
    
    echo ""
    echo "下一步:"
    echo "  1. 检查特征文件: ls $OUTPUT_DIR/features/"
    echo "  2. 配置训练: nano configs/anet_uniformer.yaml"
    echo "  3. 开始训练: python3 train.py configs/anet_uniformer.yaml"
    echo "  4. 或者推理: python3 inference_activitynet.py --feature-dir $OUTPUT_DIR/features ..."
else
    echo "❌ 特征提取失败"
    echo ""
    echo "查看日志:"
    echo "  cat ${OUTPUT_DIR}_extraction.log"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                          测试完成                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"

