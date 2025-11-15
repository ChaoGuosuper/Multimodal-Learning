import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from sklearn.metrics import r2_score, mean_squared_error
import time
# 配置参数
DATA_PATH = "./fds fire test datasets/"
CHECKPOINT_PATH = "./checkpoints/FDS Fire.ckpt"
BATCH_SIZE = 32
SEED = 42  # 保持与训练时相同的随机种子
torch.manual_seed(SEED)


class ImageDataset(Dataset):
    """多模态数据集"""

    def __init__(self, pickle_file, image_dir):
        self.image_dir = image_dir
        self.tabular = pd.read_pickle(pickle_file)

    def __len__(self):
        return len(self.tabular)

    def __getitem__(self, idx):
        tabular = self.tabular.iloc[idx, 0:]
        y = tabular["HRR"]

        # 加载并预处理图像
        image = Image.open(f"{self.image_dir}/{tabular['Time']}.jpg")
        image = transforms.functional.to_tensor(image)  # 自动转为[0,1]范围

        # 处理表格数据
        tabular = tabular[["Tem_rise"]].values.astype(np.float32)
        tabular = torch.FloatTensor(tabular).squeeze()  # 确保形状为[1]

        return image, tabular, y


def vgg_block(input_size, output_size):
    return nn.Sequential(
        nn.Conv2d(input_size, output_size, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(output_size, output_size, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


# ============================ 火焰面积提取器 ============================
class FlameAreaExtractor(nn.Module):
    """火焰面积提取器"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.conv(x)
        return self.fc(features.view(x.size(0), -1))


class AreaToTempRegressor(nn.Module):
    """温度回归器"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.fc(x)


# ============================ 动态权重生成器 ============================
class WeightGenerator(nn.Module):
    """模态可靠性权重生成器"""

    def __init__(self, img_feat_dim, tab_feat_dim, window_size=10):
        super().__init__()
        self.window_size = window_size
        self.fc = nn.Sequential(
            nn.Linear(img_feat_dim + tab_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 输出两个模态的可靠性权重
            nn.Softmax(dim=1)  # 权重归一化
        )
        # 状态缓存器
        self.register_buffer('state_buffer', torch.zeros(window_size, img_feat_dim + tab_feat_dim))

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


class MultiModalModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-3, batch_size: int = 32):
        super().__init__()
        self.save_hyperparameters()

        # 图像处理部分
        self.conv_blocks = nn.Sequential(
            vgg_block(3, 64),
            vgg_block(64, 128),
            vgg_block(128, 256),
            vgg_block(256, 512),
            vgg_block(512, 512),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # 多模态融合部分
        self.image_fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 256)
        )

        self.tabular_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )

        # 联合预测层 - 与检查点兼容的结构
        self.final_fc = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 注意: 只有2层，与检查点匹配
        )

        # 火焰面积提取器和温度回归器
        self.area_extractor = FlameAreaExtractor()
        self.area_temp_regressor = AreaToTempRegressor()

        # 动态权重生成器
        self.weight_generator = WeightGenerator(
            img_feat_dim=256,
            tab_feat_dim=64,
            window_size=10
        )

        # 温度预测缓存
        self.register_buffer('last_temp_pred', torch.tensor([0.0]).view(1, 1))

        # 初始化权重参数
        self.dynamic_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))

    def forward(self, img, tab):
        # 提取基本特征
        img_base = self.conv_blocks(img)
        img_base = img_base.view(img.size(0), -1)
        img_feat = self.image_fc(img_base)
        tab_feat = self.tabular_fc(tab.unsqueeze(1))

        # 跨模态映射
        area_feat = self.area_extractor(img)
        temp_pred = self.area_temp_regressor(area_feat)
        self.last_temp_pred = temp_pred.detach()

        # 动态权重调整
        weights = self.weight_generator(img_feat, tab_feat, tab)
        img_weight, tab_weight = weights[:, 0], weights[:, 1]

        # 加权融合特征
        weighted_img = img_feat * img_weight.unsqueeze(1)
        weighted_tab = tab_feat * tab_weight.unsqueeze(1)
        combined = torch.cat([weighted_img, weighted_tab], dim=1)

        # 最终预测
        pred = self.final_fc(combined)

        return pred, area_feat, temp_pred, weights

    def handle_missing_temp(self, img):
        """处理温度传感器缺失场景"""
        img_base = self.conv_blocks(img)
        img_base = img_base.view(img.size(0), -1)
        img_feat = self.image_fc(img_base)

        # 使用当前图像预测温度值
        area_feat = self.area_extractor(img)
        temp_pred = self.area_temp_regressor(area_feat)
        tab_feat = self.tabular_fc(temp_pred)

        # 仅使用图像权重
        weights = torch.tensor([[0.9, 0.1]], device=self.device)
        weighted_img = img_feat * weights[:, 0].unsqueeze(1)
        weighted_tab = tab_feat * weights[:, 1].unsqueeze(1)
        combined = torch.cat([weighted_img, weighted_tab], dim=1)

        return self.final_fc(combined).squeeze()

    def training_step(self, batch, batch_idx):
        image, tabular, y = batch
        y_pred, area_feat, temp_pred, weights = self(image, tabular)
        y_pred = y_pred.squeeze()

        # 主任务损失
        main_loss = nn.L1Loss()(y_pred, y)
        # 跨模态映射损失
        temp_loss = nn.MSELoss()(temp_pred.squeeze(), tabular)
        # 权重正则化
        weight_penalty = torch.abs(weights - 0.5).mean()
        # 总损失
        total_loss = main_loss + 0.3 * temp_loss + 0.1 * weight_penalty

        self.log("train_loss", total_loss)
        self.log("train_temp_loss", temp_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        image, tabular, y = batch
        y_pred, _, _, _ = self(image, tabular)
        y_pred = y_pred.squeeze()
        loss = nn.L1Loss()(y_pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'reduce_on_plateau': True
            }
        }


def load_compatible_model(checkpoint_path):
    """加载兼容检查点的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型实例
    model = MultiModalModel()

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']

    # 检查点键列表
    checkpoint_keys = list(state_dict.keys())
    model_keys = list(model.state_dict().keys())

    print("\033[1;33mCheckpoint keys vs Model keys:\033[0m")
    print(f"Missing in model: {set(checkpoint_keys) - set(model_keys)}")
    print(f"Missing in checkpoint: {set(model_keys) - set(checkpoint_keys)}")

    # 修复形状问题
    if 'last_temp_pred' in state_dict:
        if state_dict['last_temp_pred'].shape != model.last_temp_pred.shape:
            # 简化形状匹配问题
            state_dict['last_temp_pred'] = state_dict['last_temp_pred'][-1].view(1, 1)

    # 加载状态字典 - 忽略不匹配的键
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model


def main():
    # 1. 数据准备
    full_dataset = ImageDataset(
        pickle_file=f"{DATA_PATH}/Fuel.pkl",
        image_dir=f"{DATA_PATH}/Fuel"
    )

    test_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\033[1;34mUsing device: {device}\033[0m")

    model = load_compatible_model(CHECKPOINT_PATH)


    # 3. 执行预测
    all_preds = []
    all_targets = []
    inference_times = []  # 存储每个batch的推理时间

    with torch.no_grad():
        for images, tabular, y in test_loader:
            images = images.to(device)
            tabular = tabular.to(device)

            # 记录推理开始时间
            start_time = time.time()
            
            # 模型前向传播
            preds, _, _, _ = model(images, tabular)
            
            # 记录推理结束时间
            end_time = time.time()
            
            # 计算batch推理时间（毫秒）
            batch_time_ms = (end_time - start_time) * 1000
            batch_size = images.size(0)
            # 计算单帧推理时间（毫秒）
            single_frame_time_ms = batch_time_ms / batch_size
            inference_times.append(single_frame_time_ms)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y.numpy().flatten())

    # 4. 结果分析
    df = pd.DataFrame({
        'True HRR': all_targets,
        'Predicted HRR': all_preds
    })

    # 计算关键指标
    mae = np.mean(np.abs(df['True HRR'] - df['Predicted HRR']))
    rmse = np.sqrt(mean_squared_error(df['True HRR'], df['Predicted HRR']))
    r2 = r2_score(df['True HRR'], df['Predicted HRR'])
    
    # 计算误差统计
    error = df['Predicted HRR'] - df['True HRR']
    ci_lower = np.percentile(error, 2.5)
    ci_upper = np.percentile(error, 97.5)
    n_outliers = len(error[(error < ci_lower) | (error > ci_upper)])
    outlier_percentage = (n_outliers / len(error)) * 100
    
    # 计算单帧推理时间统计
    avg_inference_time = np.mean(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)
    std_inference_time = np.std(inference_times)

    # 5. 可视化
    plt.figure(figsize=(14, 6), dpi=120)
    plt.rcParams['font.family'] = 'Times New Roman'

    # 子图1：实际 vs 预测散点图
    plt.subplot(1, 2, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    sns.scatterplot(x='True HRR', y='Predicted HRR', data=df, alpha=0.7, s=60)

    # 绘制参考线
    min_val = min(df.min())
    max_val = max(df.max())
    x_range = np.linspace(min_val, max_val, 100)
    
    # 1:1参考线
    plt.plot(x_range, x_range, 'r-', linewidth=2, label='1:1 Line')
    
    # ±15%误差参考线
    plt.plot(x_range, x_range * 1.20, 'b--', linewidth=1.0, alpha=0.7, label='+20% Error')
    plt.plot(x_range, x_range * 0.80, 'b--', linewidth=1.0, alpha=0.7, label='-20% Error')

    # 设置标题和标签
    plt.title(f'HRR Prediction Results (R²={r2:.3f})', fontsize=16)
    plt.xlabel('True HRR (kW)', fontsize=14)
    plt.ylabel('Predicted HRR (kW)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.8)

    # 子图2：误差分布（完整直方图 + 95%置信区间 + 异常值标记）
    plt.subplot(1, 2, 2)
    error = df['Predicted HRR'] - df['True HRR']
    
    # 计算95%置信区间
    ci_lower = np.percentile(error, 2.5)  # 2.5th percentile
    ci_upper = np.percentile(error, 97.5)  # 97.5th percentile
    
    # 识别异常值（超出95%置信区间的点）
    outliers = error[(error < ci_lower) | (error > ci_upper)]
    inliers = error[(error >= ci_lower) & (error <= ci_upper)]
    
    # 绘制完整直方图
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制正常值直方图
    if len(inliers) > 0:
        sns.histplot(inliers, bins=30, kde=True, color='skyblue', alpha=0.7, label='Inliers (95% CI)')
    
    # 绘制异常值直方图
    if len(outliers) > 0:
        sns.histplot(outliers, bins=30, kde=False, color='red', alpha=0.8, label=f'Outliers (n={len(outliers)})')
    
    # 标注95%置信区间边界
    plt.axvline(x=ci_lower, color='orange', linestyle='--', linewidth=2, 
                label=f'95% CI Lower: {ci_lower:.1f} kW')
    plt.axvline(x=ci_upper, color='orange', linestyle='--', linewidth=2, 
                label=f'95% CI Upper: {ci_upper:.1f} kW')
    
    # 填充95%置信区间区域
    ylim = plt.ylim()
    x_fill = np.linspace(ci_lower, ci_upper, 100)
    plt.fill_between(x_fill, ylim[0], ylim[1], 
                     color='orange', alpha=0.2, label='95% Confidence Interval')
    plt.ylim(ylim)
    
    # 标注关键统计量
    # plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5, label='Zero Error')
    mean_error = error.mean()
    plt.axvline(x=mean_error, color='green', linestyle='-', linewidth=2, 
                label=f'Mean Error: {mean_error:.1f} kW')
    # median_error = error.median()
    # plt.axvline(x=median_error, color='purple', linestyle=':', linewidth=2, 
    #             label=f'Median Error: {median_error:.1f} kW')

    # 设置标题和标签
    plt.title('Prediction Error Distribution with 95% CI', fontsize=16)
    plt.xlabel('Error (kW)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9, loc='best')

    # 使 X 轴关于 0 对称并增加正向留白
    max_abs_error = max(abs(error.min()), abs(error.max()))
    symmetric_limit = max_abs_error * 1.1 if max_abs_error > 0 else 1.0
    plt.xlim(-symmetric_limit, symmetric_limit)

    plt.tight_layout(pad=3.0)
    plt.savefig('hrr_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 保存结果
    # 7. 输出统计信息
    stats_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'R²', 'Max Error', 'Min Error', 
                   '95% CI Lower', '95% CI Upper', 'Number of Outliers', 'Outlier Percentage',
                   'Avg Inference Time', 'Min Inference Time', 'Max Inference Time', 'Std Inference Time'],
        'Value': [mae, rmse, r2, error.max(), error.min(),
                  ci_lower, ci_upper, n_outliers, outlier_percentage,
                  avg_inference_time, min_inference_time, max_inference_time, std_inference_time],
        'Units': ['kW', 'kW', '', 'kW', 'kW', 'kW', 'kW', 'count', '%', 'ms', 'ms', 'ms', 'ms']
    })
    
    # 保存预测结果和统计信息到Excel（不同sheet）
    with pd.ExcelWriter('prediction_results.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Predictions', index=False)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)

    # 打印结果
    print("\n\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1;36m" + "PREDICTION PERFORMANCE SUMMARY".center(60) + "\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    print(f"\033[1;32m{'MAE:':<15}\033[0m {mae:.1f} kW")
    print(f"\033[1;32m{'RMSE:':<15}\033[0m {rmse:.1f} kW")
    print(f"\033[1;32m{'R² Score:':<15}\033[0m {r2:.3f}")
    print(f"\033[1;32m{'Max Error:':<15}\033[0m {error.max():.1f} kW")
    print(f"\033[1;32m{'Min Error:':<15}\033[0m {error.min():.1f} kW")
    print("\033[1m" + "-" * 60 + "\033[0m")
    print("\033[1;35m" + "ERROR DISTRIBUTION (95% CI)".center(60) + "\033[0m")
    print("\033[1m" + "-" * 60 + "\033[0m")
    print(f"\033[1;32m{'95% CI Lower:':<15}\033[0m {ci_lower:.1f} kW")
    print(f"\033[1;32m{'95% CI Upper:':<15}\033[0m {ci_upper:.1f} kW")
    print(f"\033[1;32m{'Outliers:':<15}\033[0m {n_outliers} ({outlier_percentage:.1f}%)")
    print("\033[1m" + "-" * 60 + "\033[0m")
    print("\033[1;35m" + "INFERENCE TIME STATISTICS".center(60) + "\033[0m")
    print("\033[1m" + "-" * 60 + "\033[0m")
    print(f"\033[1;32m{'Avg Time:':<15}\033[0m {avg_inference_time:.2f} ms")
    print(f"\033[1;32m{'Min Time:':<15}\033[0m {min_inference_time:.2f} ms")
    print(f"\033[1;32m{'Max Time:':<15}\033[0m {max_inference_time:.2f} ms")
    print(f"\033[1;32m{'Std Time:':<15}\033[0m {std_inference_time:.2f} ms")
    print("\033[1m" + "=" * 60 + "\033[0m")
    print(f"\033[1;36m{'Results saved to:'}\033[0m")
    print(f"  - hrr_prediction_results.png")
    print(f"  - prediction_results.xlsx")
    print("\033[1m" + "=" * 60 + "\033[0m\n")


if __name__ == "__main__":
    main()