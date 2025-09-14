# 隧道火灾多模态预测模型
本项目实现了一个基于深度学习的隧道火灾多模态预测系统，能够结合火焰图像和温度传感器数据来预测热释放率（HRR）。该系统具有动态权重调整、跨模态映射和传感器异常检测等先进功能。
多模态融合
图像模态：基于VGG架构的火焰图像特征提取
传感器模态：温度上升数据的深度处理
动态权重调整：根据传感器可靠性自动调整模态权重
### 环境要求
```bash
Python >= 3.8
PyTorch >= 1.9.0
PyTorch Lightning >= 1.5.0
torchvision >= 0.10.0
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
```
### 安装依赖

```bash
pip install torch torchvision pytorch-lightning
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 数据准备

1. 创建数据目录结构：
```
data_multi/
├── Fire.pkl          # 包含时间、温度上升、HRR等数据
└── Fire/             # 火焰图像文件夹
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

2. 数据格式要求：
   - `Fire.pkl`：包含列 `['Time', 'Tem_rise', 'HRR']`
   - 图像文件：以时间戳命名的JPG格式火焰图像

### 训练模型

```bash
python multiinput_regression.py
```
训练过程将自动：
- 加载和预处理数据
- 训练多模态模型
- 保存最佳模型检查点
- 生成训练日志和可视化结果

### 模型测试
```bash
python multiinput_test_onescenario.py
```
## 项目结构

```
├── multiinput_regression.py      # 主训练脚本
├── multiinput_test_onescenario.py # 模型测试脚本
├── data_multi/                        # 数据目录
│   ├── Fire.pkl                      # 表格数据
│   └── Fire/                         # 图像数据
├── Fire/                             # 训练日志目录
│   └── dynamic_multimodal_Fire/      # TensorBoard日志
└── README.md                         # 项目文档
```

##  核心功能

### 1.异常检测机制
```python
def detect_abnormality(self, tabular_data, window_size=5, zero_threshold=1e-6):
    # 1. 突发异常值检测
    abnorm_mask = torch.abs(tabular_data) > 500.0
    
    # 2. 持续零值检测（传感器失效）
    if len(tabular_data) >= window_size:
        zero_windows = torch.stack([
            (torch.abs(tabular_data[i:i + window_size]) < zero_threshold)
            for i in range(len(tabular_data) - window_size + 1)
        ])
        full_zero_mask = torch.all(zero_windows, dim=1)
        # ... 扩展为逐点标记
```

### 2.动态权重调整
```python
# 当检测到传感器异常时
if abnorm_flags.any():
    weights = torch.stack([
        weights[:, 0] + 0.4,  # 提高图像权重
        weights[:, 1] * 0.1   # 降低温度权重
    ], dim=1)
    weights = F.softmax(weights, dim=1)
```

### 3. 缺失数据处理
```python
def handle_missing_temp(self, img):
    # 仅使用图像特征进行预测
    img_feat = self.image_fc(self.conv_blocks(img))
    temp_pred = self.last_temp_pred  # 使用缓存的温度预测
    tab_feat = self.tabular_fc(temp_pred)
    
    # 调整权重偏向图像模态
    weights = torch.tensor([[0.9, 0.1]], device=self.device)
    # ... 加权融合
```


如有问题或建议，请通过以下方式联系：

- 发送邮件至 chaoguo@shnu.edu.cn
