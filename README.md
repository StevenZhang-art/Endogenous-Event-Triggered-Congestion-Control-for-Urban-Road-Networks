# 实验复现指南

## 运行环境
- **SUMO 版本**：1.27.1（请确保系统路径已正确配置，或按需修改脚本中的 SUMO 环境变量）
- **Python 依赖**：见 `requirements.txt`（如 traci、numpy、matplotlib 等）

## 启动实验
1. **主实验**：运行  
   `python experiment/traffic_simulation.py`  
   该脚本执行完整的主对比实验

2. **Kp 参数敏感性分析**：运行  
   `python experiment/Kp.py`  
   该脚本执行反馈增益 `K_p` 的消融实验。

## 输出结果
所有图表均以 **PDF 格式** 保存，按目录组织如下：

| 目录 | 内容 |
|------|------|
| `graphs_1/` | Kp 不同阈值下的柱状图（消融实验对比） |
| `graphs_2/` | 主实验中不同阈值下的柱状图（控制性能对比） |
| `graphs_3/` | Kp 敏感性分析及不同阈值下各指标的变化趋势 |

> 注意：若需更改 SUMO 安装路径，请在traffic_simulation.py脚本第2357附近修改 `SUMO_EXE` 变量。