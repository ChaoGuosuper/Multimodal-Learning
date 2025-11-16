# 隧道火灾多模态预测模型
本项目实现了一个基于深度学习的隧道火灾多模态预测系统，能够结合火焰图像和温度传感器数据来预测热释放率（HRR）。该系统具有动态权重调整、跨模态映射和传感器异常检测等先进功能。
多模态融合
多模态模型获取网址：链接: https://pan.baidu.com/s/1Q05fBduWIri3WG3BQ5mWRw 提取码: 9shx

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
fds fire test datasets/
├── Fuel.pkl          # 包含时间、温度上升、HRR等数据
└── Fuel/             # 火焰图像文件夹
    ├── FDSone1.jpg
    ├── FDSone6.jpg
    └── ...
```

2. 数据格式要求：
   - `Fire.pkl`：包含列 `['Time', 'Tem_rise', 'HRR']`
   - 图像文件：以时间戳命名的JPG格式火焰图像


### 模型测试
```bash
python multiinput_test_onescenario-full-scale fuel fire.py
```


##  核心功能

### 1.异常检测机制
```python
    def detect_abnormality(self, tabular_data, zero_window_size=5, change_window_size=35, zero_threshold=1e-6, change_threshold=1e-3):
        """
        改进的传感器异常检测机制，支持三种异常：
        1. 突发异常值（原逻辑）：单点数据超过阈值
        2. 持续零值故障：连续多个采样点数据接近零（传感器失效典型特征）
        3. 长时间温度数据没有变化：连续多个采样点的数据变化小于阈值（传感器不响应）

        :param zero_window_size: 持续零值故障检测的滑动窗口大小
        :param change_window_size: 长时间没有变化检测的滑动窗口大小
        :param zero_threshold: 用于判断零值的阈值
        :param change_threshold: 用于判断数据变化的阈值
        """
        # 确保 tabular_data 是张量并展平
        if not isinstance(tabular_data, torch.Tensor):
            tabular_data = torch.tensor(tabular_data)
        tabular_data = tabular_data.flatten()
        data_len = tabular_data.shape[0]
        
        # 初始化异常掩码
        abnorm_mask = torch.zeros_like(tabular_data, dtype=torch.bool)

        # 2. 持续零值故障检测（传感器完全失效特征）
        if data_len >= zero_window_size:
            # 构造滑动窗口：检测连续 zero_window_size 个零值
            zero_windows = tabular_data.unfold(0, zero_window_size, 1)
            zero_mask = torch.all(torch.abs(zero_windows) < zero_threshold, dim=1)

            # 扩展为逐点标记（若某点属于全零窗口则标记异常）
            zero_abnorm = torch.zeros_like(tabular_data, dtype=torch.bool)
            for i, is_abnormal in enumerate(zero_mask):
                if bool(is_abnormal):
                    zero_abnorm[i:i + zero_window_size] = True

            # 合并零值异常
            abnorm_mask = abnorm_mask | zero_abnorm

        # 3. 长时间温度数据没有变化检测
        if data_len >= change_window_size:
            # 构造滑动窗口：检测连续 change_window_size 个数据点的变化范围
            change_windows = tabular_data.unfold(0, change_window_size, 1)
            change_range = torch.max(change_windows, dim=1).values - torch.min(change_windows, dim=1).values
            no_change_mask = change_range < change_threshold

            # 扩展为逐点标记（若某点属于长时间无变化窗口则标记异常）
            no_change_abnorm = torch.zeros_like(tabular_data, dtype=torch.bool)
            for i, is_abnormal in enumerate(no_change_mask):
                if bool(is_abnormal):
                    no_change_abnorm[i:i + change_window_size] = True

            # 合并长时间无变化异常
            abnorm_mask = abnorm_mask | no_change_abnorm

        return abnorm_mask.squeeze()

    def forward(self, img_feat, tab_feat, tabular_data):
        # 异常检测
        abnorm_flags = self.detect_abnormality(tabular_data)
        
        # 确保 abnorm_flags 是张量并且可以正确使用 .any()
        if not isinstance(abnorm_flags, torch.Tensor):
            abnorm_flags = torch.tensor(abnorm_flags)
        if abnorm_flags.dim() == 0:  # 标量
            has_abnorm = abnorm_flags.item() if abnorm_flags.dtype == torch.bool else bool(abnorm_flags.item())
        else:
            has_abnorm = abnorm_flags.any().item()

        # 特征拼接
        combined = torch.cat([img_feat, tab_feat], dim=1)

        # 更新状态缓冲区
        self.state_buffer = torch.roll(self.state_buffer, shifts=-1, dims=0)
        self.state_buffer[-1] = combined[0].detach()

        # 生成权重
        window_feats = self.state_buffer.mean(dim=0).unsqueeze(0)
        weights = self.fc(window_feats)

        # 异常处理
        if has_abnorm:
            weights = torch.stack([
                weights[:, 0] + 0.4,
                weights[:, 1] * 0.1
            ], dim=1)
            weights = F.softmax(weights, dim=1)

        return weights
```

### 2.动态权重调整
```python
        # 异常处理
        if has_abnorm:
            weights = torch.stack([
                weights[:, 0] + 0.4,
                weights[:, 1] * 0.1
            ], dim=1)
            weights = F.softmax(weights, dim=1)

        return weights
```




如有问题或建议，请通过以下方式联系：

- 发送邮件至 chaoguo@shnu.edu.cn
