#!/bin/bash
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 二分类训练监控面板"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 训练信息:"
if [ -f binary_train.pid ]; then
    PID=$(cat binary_train.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "  ✅ 训练进程运行中 (PID: $PID)"
    else
        echo "  ❌ 训练进程已停止"
    fi
else
    echo "  ⚠️  未找到PID文件"
fi
echo ""
echo "📊 最新训练进度:"
grep -E "Epoch [0-9]+ (started|finished)" log.binary_serve | tail -n 5
echo ""
echo "📈 最近评估结果:"
grep -E "mAP@0.5:|整体mAP:" log.binary_serve | tail -n 6
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "实时日志 (按Ctrl+C停止):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -f log.binary_serve
