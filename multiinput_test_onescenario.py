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
# 配置参数
DATA_PATH = "./data_multi/"
CHECKPOINT_PATH = "./Fire/dynamic_multimodal_Fire/version_0/best-model-epoch=125-val_loss=32.18.ckpt"
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

    def detect_abnormality(self, tabular_data, window_size=5, zero_threshold=1e-6):
        """传感器异常检测"""
        # 1. 阈值检测
        abnorm_mask = torch.abs(tabular_data) > 500.0

        # 2. 持续零值检测
        if len(tabular_data) >= window_size:
            zero_windows = torch.stack([
                (torch.abs(tabular_data[i:i + window_size]) < zero_threshold)
                for i in range(len(tabular_data) - window_size + 1)
            ])

            full_zero_mask = torch.all(zero_windows, dim=1)
            zero_abnorm = torch.zeros_like(tabular_data, dtype=torch.bool)
            for i, is_abnormal in enumerate(full_zero_mask):
                if is_abnormal:
                    zero_abnorm[i:i + window_size] = True

            abnorm_mask = abnorm_mask | zero_abnorm

        return abnorm_mask.squeeze()

    def forward(self, img_feat, tab_feat, tabular_data):
        # 异常检测
        abnorm_flags = self.detect_abnormality(tabular_data)

        # 特征拼接
        combined = torch.cat([img_feat, tab_feat], dim=1)

        # 更新状态缓冲区
        self.state_buffer = torch.roll(self.state_buffer, shifts=-1, dims=0)
        self.state_buffer[-1] = combined[0].detach()

        # 生成权重
        window_feats = self.state_buffer.mean(dim=0).unsqueeze(0)
        weights = self.fc(window_feats)

        # 异常处理
        if abnorm_flags.any():
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
        pickle_file=f"{DATA_PATH}/BH.pkl",
        image_dir=f"{DATA_PATH}/BH"
    )

    test_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\033[1;34mUsing device: {device}\033[0m")

    model = load_compatible_model(CHECKPOINT_PATH)

    # 3. 执行预测
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, tabular, y in test_loader:
            images = images.to(device)
            tabular = tabular.to(device)

            # 模型前向传播
            preds, _, _, _ = model(images, tabular)
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
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='1:1 Line')

    # 设置标题和标签
    plt.title(f'HRR Prediction Results (R²={r2:.3f})', fontsize=16)
    plt.xlabel('True HRR (kW)', fontsize=14)
    plt.ylabel('Predicted HRR (kW)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.8)

    # 子图2：相对误差分布
    plt.subplot(1, 2, 2)
    relative_error = (df['Predicted HRR'] - df['True HRR']) / (df['True HRR'] + 1e-6) * 100
    plt.grid(True, linestyle='--', alpha=0.6)
    sns.histplot(relative_error, bins=20, kde=True, color='skyblue')

    # 标注关键统计量
    plt.axvline(x=0, color='r', linestyle='--')
    mean_error = relative_error.mean()
    plt.axvline(x=mean_error, color='g', linestyle='-', label=f'Mean Error: {mean_error:.1f}%')

    # 设置标题和标签
    plt.title('Prediction Error Distribution', fontsize=16)
    plt.xlabel('Relative Error (%)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.8)

    plt.tight_layout(pad=3.0)
    plt.savefig('hrr_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 保存结果
    df.to_excel('prediction_results.xlsx', index=False)

    # 7. 输出统计信息
    stats_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'R²', 'Max Error', 'Min Error'],
        'Value': [mae, rmse, r2, relative_error.max(), relative_error.min()],
        'Units': ['kW', 'kW', '', '%', '%']
    })

    # 打印结果
    print("\n\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1;36m" + "PREDICTION PERFORMANCE SUMMARY".center(60) + "\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    print(f"\033[1;32m{'MAE:':<15}\033[0m {mae:.1f} kW")
    print(f"\033[1;32m{'RMSE:':<15}\033[0m {rmse:.1f} kW")
    print(f"\033[1;32m{'R² Score:':<15}\033[0m {r2:.3f}")
    print(f"\033[1;32m{'Max Error:':<15}\033[0m {relative_error.max():.1f}%")
    print(f"\033[1;32m{'Min Error:':<15}\033[0m {relative_error.min():.1f}%")
    print("\033[1m" + "=" * 60 + "\033[0m")
    print(f"\033[1;36m{'Results saved to:'}\033[0m")
    print(f"  - hrr_prediction_results.png")
    print(f"  - prediction_results.xlsx")
    print("\033[1m" + "=" * 60 + "\033[0m\n")


if __name__ == "__main__":
    main()