import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import time  # 新增：用于计算推理时间
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import pytorch_lightning as pl

# 配置参数
DATA_PATH = "./full-scale fire test datasets"
CHECKPOINT_PATH = "./checkpoints/Full scale EV Fire.ckpt"  # 替换为实际检查点路径
BATCH_SIZE = 32
SEED = 42  # 保持与训练时相同的随机种子


class ImageDataset(Dataset):
    """多模态数据集"""

    def __init__(self, pickle_file, image_dir):
        self.image_dir = image_dir
        self.pickle_file = pickle_file
        self.tabular = pd.read_pickle(pickle_file)

    def __len__(self):
        return len(self.tabular)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tabular = self.tabular.iloc[idx, 0:]
        y = tabular["MLR"]

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


class MultiModalModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-3, batch_size: int = 32):
        super().__init__()
        self.save_hyperparameters()

        # 新增结果存储容器
        self.test_preds = []
        self.test_targets = []

        # 图像处理部分
        self.conv_blocks = nn.Sequential(
            vgg_block(3, 64),
            vgg_block(64, 128),
            vgg_block(128, 256),
            vgg_block(256, 512),
            vgg_block(512, 512),
            nn.AdaptiveAvgPool2d((7, 7))  # 确保不同尺寸输入统一为7x7
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

        # 联合预测层
        self.final_fc = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, img, tab):
        # 图像特征提取
        img_feat = self.conv_blocks(img)
        img_feat = img_feat.view(img.size(0), -1)
        img_feat = self.image_fc(img_feat)

        # 表格特征提取
        tab_feat = self.tabular_fc(tab.unsqueeze(1))  # 添加特征维度

        # 特征融合
        combined = torch.cat([img_feat, tab_feat], dim=1)
        return self.final_fc(combined)


def main():
    # 1. 数据准备
    full_dataset = ImageDataset(
        pickle_file=f"{DATA_PATH}/LC.pkl",
        image_dir=f"{DATA_PATH}/LC/"
    )

    test_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\033[1;34mUsing device: {device}\033[0m")

    model = MultiModalModel.load_from_checkpoint(CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()  # 设置评估模式

    # 3. 执行预测（新增单帧推理时间计算）
    all_preds = []
    all_targets = []
    inference_times = []  # 存储单帧推理时间（毫秒）

    with torch.no_grad():
        for images, tabular, y in test_loader:
            images = images.to(device)
            tabular = tabular.to(device)

            # 记录推理时间
            start_time = time.time()
            preds = model(images, tabular).cpu().numpy().flatten()
            end_time = time.time()

            # 计算单帧推理时间（毫秒）
            batch_time_ms = (end_time - start_time) * 1000
            batch_size = images.size(0)
            single_frame_time = batch_time_ms / batch_size
            inference_times.extend([single_frame_time] * batch_size)

            all_preds.extend(preds)
            all_targets.extend(y.numpy().flatten())

    # 4. 结果分析
    df = pd.DataFrame({
        'True MLR': all_targets,
        'Predicted MLR': all_preds,
        'Inference Time (ms)': inference_times  # 新增：单帧推理时间
    })

    # 计算关键指标
    mae = np.mean(np.abs(df['True MLR'] - df['Predicted MLR']))
    rmse = np.sqrt(mean_squared_error(df['True MLR'], df['Predicted MLR']))
    r2 = r2_score(df['True MLR'], df['Predicted MLR'])
    
    # 计算误差统计（新增）
    errors = df['Predicted MLR'] - df['True MLR']
    max_error = errors.max()
    min_error = errors.min()
    ci_lower = np.percentile(errors, 2.5)  # 95%置信区间下限
    ci_upper = np.percentile(errors, 97.5)  # 95%置信区间上限
    n_outliers = len(errors[(errors < ci_lower) | (errors > ci_upper)])
    outlier_percentage = (n_outliers / len(errors)) * 100 if len(errors) > 0 else 0

    # 计算推理时间统计（新增）
    avg_inference = np.mean(inference_times)
    min_inference = np.min(inference_times)
    max_inference = np.max(inference_times)
    std_inference = np.std(inference_times)

    # 5. 可视化优化
    plt.figure(figsize=(14, 6), dpi=120)
    plt.rcParams['font.family'] = 'Times New Roman'

    # 子图1：实际 vs 预测散点图（优化版）
    plt.subplot(1, 2, 1)
    plt.grid(True, linestyle='--', alpha=0.6)  # 新增网格线
    sns.scatterplot(
        x='True MLR', 
        y='Predicted MLR', 
        data=df, 
        alpha=0.7, 
        s=60,  # 点大小
        edgecolor='w', 
        linewidth=0.5
    )

    # 绘制参考线
    min_val = min(df['True MLR'].min(), df['Predicted MLR'].min())
    max_val = max(df['True MLR'].max(), df['Predicted MLR'].max())
    x_range = np.linspace(min_val, max_val, 100)
    
    # 1:1参考线
    plt.plot(x_range, x_range, 'r-', linewidth=2, label='1:1 Line')
    
    # ±30%误差线（原代码为0.7和1.3，即±30%）
    plt.plot(x_range, 0.7 * x_range, 'g--', linewidth=1.0, alpha=0.7, label='±30% Error')
    plt.plot(x_range, 1.3 * x_range, 'g--', linewidth=1.0, alpha=0.7)
    plt.fill_between(x_range, 0.7 * x_range, 1.3 * x_range, color='gray', alpha=0.1)

    # 设置标题和标签
    plt.title(f'Prediction Accuracy (R²={r2:.3f})', fontsize=16)
    plt.xlabel('True MLR (kg/s)', fontsize=14)
    plt.ylabel('Predicted MLR (kg/s)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.8)

    # 子图2：误差分布（优化版）
    plt.subplot(1, 2, 2)
    plt.grid(True, linestyle='--', alpha=0.6)  # 新增网格线

    # 区分异常值和正常值
    inliers = errors[(errors >= ci_lower) & (errors <= ci_upper)]
    outliers = errors[(errors < ci_lower) | (errors > ci_upper)]

    # 绘制直方图
    if len(inliers) > 0:
        sns.histplot(
            inliers, 
            bins=30, 
            kde=True, 
            color='skyblue', 
            alpha=0.7, 
            label='Inliers (95% CI)'
        )
    if len(outliers) > 0:
        sns.histplot(
            outliers, 
            bins=30, 
            kde=False, 
            color='red', 
            alpha=0.8, 
            label=f'Outliers (n={len(outliers)})'
        )

    # 标注95%置信区间
    plt.axvline(x=ci_lower, color='orange', linestyle='--', linewidth=2, 
                label=f'95% CI Lower: {ci_lower:.3f}')
    plt.axvline(x=ci_upper, color='orange', linestyle='--', linewidth=2, 
                label=f'95% CI Upper: {ci_upper:.3f}')

    # 填充置信区间区域
    ylim = plt.ylim()
    x_fill = np.linspace(ci_lower, ci_upper, 100)
    plt.fill_between(x_fill, ylim[0], ylim[1], color='orange', alpha=0.2, 
                     label='95% Confidence Interval')
    plt.ylim(ylim)

    # 标注均值误差
    mean_error = errors.mean()
    plt.axvline(x=mean_error, color='green', linestyle='-', linewidth=2, 
                label=f'Mean Error: {mean_error:.3f}')

    # 设置标题和标签
    plt.title('Prediction Error Distribution', fontsize=16)
    plt.xlabel('Error (kg/s)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9, loc='best')

    # 对称化X轴范围
    max_abs_error = max(abs(errors.min()), abs(errors.max())) if len(errors) > 0 else 1.0
    symmetric_limit = max_abs_error * 1.1
    plt.xlim(-symmetric_limit, symmetric_limit)

    plt.tight_layout(pad=3.0)
    plt.savefig('MLR_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 保存结果（优化：分sheet保存预测结果和统计信息）
    stats_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'R²', 'Max Error', 'Min Error', 
                   '95% CI Lower', '95% CI Upper', 'Number of Outliers', 'Outlier Percentage',
                   'Avg Inference Time', 'Min Inference Time', 'Max Inference Time', 'Std Inference Time'],
        'Value': [mae, rmse, r2, max_error, min_error,
                  ci_lower, ci_upper, n_outliers, outlier_percentage,
                  avg_inference, min_inference, max_inference, std_inference],
        'Units': ['kg/s', 'kg/s', '', 'kg/s', 'kg/s', 'kg/s', 'kg/s', 'count', '%', 
                  'ms', 'ms', 'ms', 'ms']
    })

    with pd.ExcelWriter('MLRprediction_results.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Predictions', index=False)  # 预测结果
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)  # 统计信息

    # 7. 打印结果
    print("\033[1m" + "=" * 50 + "\033[0m")
    print(f"\033[1;32mMAE:\033[0m {mae:.3f} kg/s")
    print(f"\033[1;32mRMSE:\033[0m {rmse:.3f} kg/s")
    print(f"\033[1;32mR² Score:\033[0m {r2:.3f}")
    print(f"\033[1;32mAverage Inference Time:\033[0m {avg_inference:.3f} ms")
    print(f"\033[1;36mResults saved to:\033[0m")
    print(f"  - MLR_comparison525.png")
    print(f"  - MLRprediction_results.xlsx")
    print("\033[1m" + "=" * 50 + "\033[0m")


if __name__ == "__main__":
    main()