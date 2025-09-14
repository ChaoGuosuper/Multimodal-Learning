# 隧道火灾多模态预测模型

本项目实现了一个基于深度学习的隧道火灾多模态预测系统，能够结合火焰图像和温度传感器数据来预测热释放率（HRR）。该系统具有动态权重调整、跨模态映射和传感器异常检测等先进功能。


## ✨ 主要特性

### 🔥 多模态融合
- **图像模态**：基于VGG架构的火焰图像特征提取
- **传感器模态**：温度上升数据的深度处理
- **动态权重调整**：根据传感器可靠性自动调整模态权重

### 🧠 智能特性
- **跨模态映射**：学习火焰面积与温度的映射关系
- **异常检测**：自动检测传感器故障和异常数据
- **缺失数据处理**：当传感器失效时，仅使用图像数据进行预测

### 📊 模型架构
- **FlameAreaExtractor**：提取火焰面积特征
- **AreaToTempRegressor**：学习面积-温度映射关系
- **WeightGenerator**：动态生成模态可靠性权重
- **MultiModalModel**：主预测模型

## 🚀 快速开始

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
python multiinput_regression-DSV1.py
```

训练过程将自动：
- 加载和预处理数据
- 训练多模态模型
- 保存最佳模型检查点
- 生成训练日志和可视化结果

### 模型测试

```bash
python multiinput_test_onescenario-DSV1.py
```

## 📁 项目结构

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

## 🏗️ 模型架构详解

### 1. 图像特征提取
```python
# VGG风格的卷积块
conv_blocks = nn.Sequential(
    vgg_block(3, 64),      # 输入RGB图像
    vgg_block(64, 128),    # 特征提取
    vgg_block(128, 256),   # 深层特征
    vgg_block(256, 512),   # 高级特征
    vgg_block(512, 512),   # 最终特征
    nn.AdaptiveAvgPool2d((7, 7))
)
```

### 2. 跨模态映射
```python
# 火焰面积提取器
area_extractor = FlameAreaExtractor()
# 面积-温度回归器
temp_regressor = AreaToTempRegressor()
```

### 3. 动态权重生成
```python
# 权重生成器
weight_generator = WeightGenerator(
    img_feat_dim=256,
    tab_feat_dim=64,
    window_size=10
)
```

## 📈 训练配置

### 超参数设置
- **学习率**：1e-4
- **批次大小**：32
- **最大训练轮数**：300
- **早停耐心**：20轮
- **梯度裁剪**：0.5
- **梯度累积**：4个批次

### 损失函数
```python
total_loss = main_loss + 0.3 * temp_loss + 0.1 * weight_penalty
```
- **主损失**：HRR预测的L1损失
- **温度损失**：跨模态映射的MSE损失
- **权重正则化**：防止模态权重过于极端

### 学习率调度
使用`ReduceLROnPlateau`调度器：
- 监控验证损失
- 耐心值：10轮
- 衰减因子：0.5

## 🔧 核心功能

### 1. 异常检测机制
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

### 2. 动态权重调整
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

## 📊 评估指标

模型训练完成后会自动生成以下评估指标：

- **MAE**：平均绝对误差
- **RMSE**：均方根误差
- **R² Score**：决定系数
- **动态权重分布**：各模态的权重变化趋势

## 📈 可视化结果

训练和测试过程会生成以下可视化图表：

1. **真实值 vs 预测值散点图**：显示预测精度
2. **时间序列对比**：展示预测趋势
3. **动态权重调整**：显示模态权重变化
4. **温度预测对比**：跨模态映射效果

## 🛠️ 自定义配置

### 修改模型参数
```python
model = MultiModalModel(
    lr=1e-4,           # 学习率
    batch_size=32      # 批次大小
)
```

### 调整训练参数
```python
trainer = pl.Trainer(
    max_epochs=300,           # 最大训练轮数
    gradient_clip_val=0.5,    # 梯度裁剪
    accumulate_grad_batches=4 # 梯度累积
)
```

### 自定义数据路径
```python
data_path = "./your_data_path/"
```

## 🔍 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小批次大小
   - 使用梯度累积
   - 启用混合精度训练

2. **数据加载错误**
   - 检查数据路径是否正确
   - 确认图像文件格式为JPG
   - 验证pkl文件包含必要列

3. **模型收敛缓慢**
   - 调整学习率
   - 增加训练轮数
   - 检查数据质量

### 性能优化建议

1. **数据预处理**
   - 使用多进程数据加载
   - 预计算图像特征
   - 数据缓存机制

2. **模型优化**
   - 使用混合精度训练
   - 模型量化
   - 知识蒸馏

## 📚 参考文献

1. VGG网络架构
2. 多模态学习理论
3. 动态权重调整方法
4. 传感器异常检测技术

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目：

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 发送邮件至 chaoguo@shnu.edu.cn
