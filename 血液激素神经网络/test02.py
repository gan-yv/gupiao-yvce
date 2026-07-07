import os
import torch
import torch.nn.functional as F
# 设置 CUDA 同步模式，方便调试 GPU 错误
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 打印 PyTorch 版本
print(torch.__version__)

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)

# 导入必要的库
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import random

# 设置随机种子以确保结果的可复现性
torch.manual_seed(42)  # 设置 PyTorch 的随机种子
np.random.seed(42)     # 设置 NumPy 的随机种子
random.seed(42)        # 设置 Python 内置随机模块的种子
print("Torch version:", torch.__version__)
print("Numpy version:", np.__version__)
print("Pandas version:", pd.__version__)



# 检测并设置设备（CPU 或 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 打印设备信息
print("Using device:", device)

# 如果使用的是 GPU，打印 GPU 的详细信息
if device.type == "cuda":
    print("GPU Model:", torch.cuda.get_device_name(0))  # 打印 GPU 模型
    print("CUDA Version:", torch.version.cuda)          # 打印 CUDA 版本
    


# -*- coding: utf-8 -*-
"""
股票多变量时间序列数据预处理脚本
功能：读取原始股票数据 → 计算技术指标 → 数据标准化 → 构造时序窗口 → 保存训练数据
适用模型：LSTM / Transformer / 时间序列预测模型
"""

# ========================== 1. 导入依赖库 ==========================
import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import RobustScaler  # 抗异常值标准化器，适合金融数据
import pickle

# ========================== 2. 全局参数配置 ==========================
# 原始数据路径
csv_file = r"E:\大学\学习\血液激素神经网络\archive\all_stocks_5yr.csv"
# 预处理后数据保存路径
output_dir = r"E:\大学\学习\血液激素神经网络\archive\processed"
os.makedirs(output_dir, exist_ok=True)  # 自动创建文件夹，不存在则创建

# 选择要预测的股票代码列表
tickers = ['AAPL','MSFT','GOOGL','AMZN','MYL']
T = 30                # 时间窗口大小：用前30天数据预测未来1天
train_ratio = 0.8     # 训练集占比 80%，验证集占比 20%
CLIP_BOUND = 0.1      # 收益率上下限裁剪，防止极端异常值

# ========================== 3. 技术指标计算函数 ==========================
def compute_RSI(series, period=14):
    """
    计算相对强弱指数 RSI
    :param series: 股票收盘价序列
    :param period: RSI计算周期，默认14
    :return: RSI指标序列
    """
    delta = series.diff()                          # 计算价格日变化量
    up = delta.clip(lower=0)                       # 上涨幅度（只取正值）
    down = -delta.clip(upper=0)                    # 下跌幅度（只取正值）
    roll_up = up.rolling(period).mean()            # 上涨平滑均值
    roll_down = down.rolling(period).mean()        # 下跌平滑均值
    rs = roll_up / roll_down                       # 相对强弱值
    rsi = 100 - (100 / (1 + rs))                   # 转换为 0~100 范围
    return rsi

def add_technical_indicators(df, ticker):
    """
    为单只股票添加所有技术指标
    :param df: 单只股票原始数据
    :param ticker: 股票代码
    :return: 包含技术指标的DataFrame
    """
    close = df[f"{ticker}_Close"]
    vol = df[f"{ticker}_Volume"]

    # 1. 日收益率 + 对数收益率（裁剪异常值）
    df[f"{ticker}_Return"] = close.pct_change().clip(-CLIP_BOUND, CLIP_BOUND)
    df[f"{ticker}_LogReturn"] = (np.log(close) - np.log(close.shift(1))).clip(-CLIP_BOUND, CLIP_BOUND)

    # 2. 移动平均指标：简单均线 + 指数均线
    for n in [10]:
        df[f"{ticker}_SMA_{n}"] = close.rolling(n).mean()
        df[f"{ticker}_EMA_{n}"] = close.ewm(span=n, adjust=False).mean()

    # 3. 波动率（10日标准差）
    df[f"{ticker}_Volatility_10"] = close.rolling(10).std()

    # 4. RSI 指标
    df[f"{ticker}_RSI_14"] = compute_RSI(close, 14)

    # 5. MACD 指标
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df[f"{ticker}_MACD"] = exp1 - exp2

    return df

# ========================== 4. 读取并清洗原始数据 ==========================
df = pd.read_csv(csv_file)
df['date'] = pd.to_datetime(df['date'])  # 转换日期格式
df = df.sort_values('date')              # 按时间升序排序
df.set_index('date', inplace=True)       # 将日期设为索引

# ========================== 5. 多股票数据合并与特征工程 ==========================
processed_df = None

for t in tickers:
    # 筛选单只股票数据
    df_t = df[df['Name'] == t].copy()
    if df_t.empty:
        continue

    df_t = df_t.sort_index()

    # 重命名列，区分不同股票
    df_t.rename(columns={
        'open': f'{t}_Open',
        'high': f'{t}_High',
        'low': f'{t}_Low',
        'close': f'{t}_Close',
        'volume': f'{t}_Volume'
    }, inplace=True)

    # 保留核心列
    df_t = df_t[[f'{t}_Open', f'{t}_High', f'{t}_Low', f'{t}_Close', f'{t}_Volume']]

    # 添加技术指标
    df_t = add_technical_indicators(df_t, t)

    # 横向合并多只股票数据
    if processed_df is None:
        processed_df = df_t
    else:
        processed_df = processed_df.join(df_t, how='outer')

# ========================== 6. 缺失值处理 ==========================
processed_df = processed_df.sort_index()   # 按时间排序
processed_df = processed_df.ffill()        # 前向填充缺失值
processed_df = processed_df.dropna()        # 删除仍存在的缺失值

# ========================== 7. 筛选最终建模特征 ==========================
selected_features = []
for t in tickers:
    selected_features += [
        f"{t}_Close",           # 收盘价
        f"{t}_Return",          # 收益率（预测目标）
        f"{t}_LogReturn",       # 对数收益率
        f"{t}_SMA_10",          # 10日均线
        f"{t}_EMA_10",          # 10日指数均线
        f"{t}_RSI_14",          # RSI
        f"{t}_MACD",            # MACD
        f"{t}_Volatility_10"    # 波动率
    ]

# 只保留存在的列，防止报错
selected_features = [col for col in selected_features if col in processed_df.columns]
processed_df = processed_df[selected_features]

# ========================== 8. 数据标准化（防止数据泄露） ==========================
scalers = {}
train_size = int(train_ratio * len(processed_df))

# 严格按时间划分训练集与验证集
train_df = processed_df.iloc[:train_size].copy()
val_df = processed_df.iloc[train_size:].copy()

# 对每一列特征独立标准化，仅使用训练集数据拟合
for col in processed_df.columns:
    scaler = RobustScaler()
    train_df[col] = scaler.fit_transform(train_df[[col]])   # 训练集：拟合+转换
    val_df[col] = scaler.transform(val_df[[col]])           # 验证集：只转换，不学习
    scalers[col] = scaler                                   # 保存scaler用于后续逆变换

# 合并标准化后的数据并按时间排序
processed_df = pd.concat([train_df, val_df]).sort_index()

# ========================== 9. 保存预处理工具与数据 ==========================
# 保存标准化器
with open(os.path.join(output_dir, "scalers.pkl"), "wb") as f:
    pickle.dump(scalers, f)

# 保存特征列名
feature_cols = processed_df.columns.tolist()
with open(os.path.join(output_dir, "feature_cols.pkl"), "wb") as f:
    pickle.dump(feature_cols, f)

# 保存预测目标列（收益率）
y_col_names = [f"{t}_Return" for t in tickers if f"{t}_Return" in feature_cols]
with open(os.path.join(output_dir, "y_cols.pkl"), "wb") as f:
    pickle.dump(y_col_names, f)

# 保存完整处理后数据表
processed_df.to_csv(os.path.join(output_dir, "all_stocks_processed.csv"), index=False)

# ========================== 10. 构造时序输入窗口 ==========================
features = processed_df.values
y_cols = [feature_cols.index(col) for col in y_col_names]

X_list, y_list = [], []
# 滑动窗口构造样本：用 T 天历史数据 → 预测第 T+1 天的收益率
for i in range(len(features) - T):
    X_list.append(features[i:i+T])          # 输入：前 T 天所有特征
    y_list.append(features[i+T, y_cols])    # 标签：第 T+1 天的股票收益率

# 转为 numpy 数组
X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

# ========================== 11. 最终训练/验证集划分 ==========================
train_size = int(train_ratio * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:], y[train_size:]

# 转为 PyTorch 张量
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val)

# 打印数据维度信息
print("✅ X_train shape:", X_train.shape)   # (样本数, 时间步, 特征数)
print("✅ y_train shape:", y_train.shape)   # (样本数, 股票数)
print("✅ X_val shape:", X_val.shape)
print("✅ y_val shape:", y_val.shape)

# ========================== 12. 保存最终训练数据 ==========================
np.save(os.path.join(output_dir, "X_train.npy"), X_train.numpy())
np.save(os.path.join(output_dir, "y_train.npy"), y_train.numpy())
np.save(os.path.join(output_dir, "X_val.npy"), X_val.numpy())
np.save(os.path.join(output_dir, "y_val.npy"), y_val.numpy())

print("💾 数据预处理全部完成！")


import torch
import numpy as np
import pickle
from torch.utils.data import Dataset

# ==============================
# 股票时间序列数据集类 (标准规范版)
# 功能：加载预处理的npy数据 + 支持标签反归一化 + 适配LSTM训练
# ==============================
class StockDataset(Dataset):
    def __init__(self, X_path, y_path, scalers_path=None,
                 feature_cols=None, y_col_names=None, inverse=False):
        """
        初始化股票数据集加载类
        :param X_path: 输入特征数据路径 (.npy)
        :param y_path: 标签数据路径 (.npy)
        :param scalers_path: 标准化器路径 (.pkl)
        :param feature_cols: 特征列名列表
        :param y_col_names: 标签列名列表（收益率列名）
        :param inverse: 是否开启反归一化
        """
        # 加载数据并转为 float32 张量
        self.X = torch.from_numpy(np.load(X_path)).float()
        self.y = torch.from_numpy(np.load(y_path)).float()
        
        # 反归一化开关
        self.inverse = inverse

        # 列名信息（用于反归一化）
        self.feature_cols = feature_cols
        self.y_col_names = y_col_names

        # 加载标准化器
        self.scalers = None
        if scalers_path is not None:
            with open(scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)

    def __len__(self):
        """返回数据集总长度"""
        return len(self.X)

    def __getitem__(self, idx):
        """根据索引获取单条样本"""
        return self.X[idx], self.y[idx]

    def inverse_transform_y(self, y_batch):
        """
        对模型输出的预测值 batch 进行反归一化（验证/绘图时使用）
        :param y_batch: 模型输出的标准化后张量
        :return: 反归一化后的真实尺度张量
        """
        # 如果未开启反归一化 或 无scaler，直接返回原值
        if not self.inverse or self.scalers is None:
            return y_batch

        # 将张量转为numpy进行反归一化计算
        y_np = y_batch.detach().cpu().numpy()
        y_inv = np.zeros_like(y_np)

        # 逐列（逐只股票）反归一化
        for i, col_name in enumerate(self.y_col_names):
            scaler = self.scalers.get(col_name, None)
            
            # 若没有对应scaler，直接使用原值
            if scaler is None:
                y_inv[:, i] = y_np[:, i]
            else:
                # 反归一化并恢复形状
                y_inv[:, i] = scaler.inverse_transform(
                    y_np[:, i].reshape(-1, 1)
                ).flatten()

        # 转回与输入相同设备（CPU/GPU）和类型的张量
        return torch.tensor(y_inv, dtype=y_batch.dtype, device=y_batch.device)
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    血液流动注意力层 (Blood Flow Attention Layer)
    论文创新点：基于时间序列的因果注意力 + 生理启发式衰减机制
    适用于：股票预测、时序预测、生理信号分析（心电/血液激素）
    
    核心设计理念：
    1. 因果性约束：信息只能从过去流向未来，禁止使用未来信息
    2. 距离衰减机制：时间距离越远，信息传递强度越弱
    3. 可学习流动强度：模拟血液流速/血压，自动学习信息传播强度
    
    数学公式：
    Attention(i,j) = Sim(Q_i,K_j) * flow_strength + log( exp(-|i-j|/tau) )
    其中：
    - Sim(Q_i,K_j)：标准多头注意力相似度
    - tau：可学习衰减系数（控制有效依赖长度）
    - flow_strength：可学习流动强度系数
    """
    def __init__(self, in_dim, out_dim, n_heads=8, dropout=0.1):
        super().__init__()

        # ========================= 基础参数配置 =========================
        # 确保输出维度能被多头数整除
        assert out_dim % n_heads == 0, "out_dim 必须能被 n_heads 整除"
        self.n_heads = n_heads                  # 注意力头数（并行建模多尺度依赖）
        self.head_dim = out_dim // n_heads      # 单头特征维度

        # ========================= QKV 线性投影 =========================
        # Query：当前时刻查询向量
        self.q_proj = nn.Linear(in_dim, out_dim)
        # Key：历史时刻键向量
        self.k_proj = nn.Linear(in_dim, out_dim)
        # Value：待聚合的信息向量
        self.v_proj = nn.Linear(in_dim, out_dim)

        # 多头合并后的输出投影
        self.out_proj = nn.Linear(out_dim, out_dim)

        # ========================= 正则化与Dropout =========================
        self.norm = nn.LayerNorm(out_dim)       # 层归一化（稳定训练）
        self.dropout = nn.Dropout(dropout)      # Dropout 防止过拟合

        # ========================= 残差连接投影 =========================
        # 输入输出维度不同时进行线性映射，保持残差维度一致
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        # ========================= 血液流动注意力核心可学习参数 =========================
        # 衰减系数 tau：控制信息随时间衰减的速度（模拟血液传播距离）
        self.decay_tau = nn.Parameter(torch.tensor(10.0))
        # 流动强度：模拟血压/信号强度，缩放注意力权重
        self.flow_strength = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        前向传播
        输入 x: [batch_size, seq_len, in_dim] → 批量、时间步、输入维度
        输出 out: [batch_size, seq_len, out_dim]
        """
        B, T, _ = x.shape  # B=批次大小，T=序列长度

        # ========================= Step 1: QKV 线性投影 + 多头拆分 =========================
        # 投影 → 重塑维度 → 交换头和时间维度
        # 输出 shape: [B, n_heads, T, head_dim]
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # ========================= Step 2: 计算标准 scaled dot-product attention =========================
        # 注意力相似度矩阵：Q * K^T / sqrt(d_k)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # shape: [B, n_heads, T, T]

        # ========================= Step 3: 血液流动衰减机制（核心创新） =========================
        # 生成时间步索引矩阵
        idx = torch.arange(T, device=x.device)
        # 计算时间绝对距离矩阵 |i - j|
        dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)).float()
        
        # 限制衰减系数范围，保证训练稳定
        tau = torch.clamp(self.decay_tau, 1.0, 50.0)
        # 指数衰减：距离越远，权重越小
        decay = torch.exp(-dist / (tau + 1e-6))
        # 将衰减加入注意力（log 空间相加，等价于原始空间相乘）
        attn = attn * self.flow_strength + torch.log(decay + 1e-9)
        
        # 数值稳定性：限制注意力范围，防止梯度爆炸
        attn = torch.clamp(attn, -20, 20)

        # ========================= Step 4: 因果掩码（单向血液流动，禁止看未来） =========================
        # 生成上三角掩码矩阵（mask 掉未来时刻）
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        # 掩码位置填充 -inf，softmax 后权重为 0
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # ========================= Step 5: 归一化得到注意力权重 =========================
        alpha = torch.softmax(attn, dim=-1)
        alpha = self.dropout(alpha)

        # ========================= Step 6: 信息加权聚合 =========================
        out = torch.matmul(alpha, V)

        # ========================= Step 7: 多头合并 + 输出投影 =========================
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.out_proj(out)
        out = self.dropout(out)

        # ========================= Step 8: 残差连接 + 层归一化 =========================
        out = self.norm(out + self.residual_proj(x))

        return out
    
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class HormoneModule(nn.Module):
    """
    激素动态调控模块 (Hormone Regulation Module)
    论文创新点：基于GRU的时序门控机制，模拟生物激素分泌与动态调控
    
    核心生物学思想：
    1. 激素分泌具有时间累积效应（使用GRU模拟）
    2. 动态生成门控权重，随时间序列自适应变化
    3. 满足生理约束：调控权重和为1（alpha + beta = 1）
    4. 用于全局信息融合的动态加权控制

    功能：
    - 输入时间序列特征
    - 输出动态门控系数 alpha, beta
    - 支持训练过程可视化（激素动态曲线）
    """

    def __init__(self, input_dim, hidden_dim, output_dim, visualize=False):
        """
        初始化激素调控模块
        :param input_dim: 输入特征维度
        :param hidden_dim: GRU 隐藏层维度
        :param output_dim: 输出门控维度（与待融合特征维度一致）
        :param visualize: 是否开启时序动态可视化（调试/展示用）
        """
        super().__init__()

        # 可视化开关（仅在 batch_size=1 时生效）
        self.visualize = visualize

        # ===================== 核心激素生成网络 =====================
        # GRU：模拟激素的时序分泌与累积过程
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # ===================== 门控控制器 =====================
        # 从GRU状态生成动态调控权重
        self.controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 特征映射
            nn.ReLU(),                           # 非线性激活
            nn.Linear(hidden_dim, 2 * output_dim)# 输出 2*D 维门控（alpha + beta）
        )

        # Dropout 正则化，提升泛化能力
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, prev_state=None):
        """
        前向传播：生成激素动态门控
        :param x: 输入序列 [batch_size, seq_len, input_dim]
        :param prev_state: 上一时刻GRU状态（可选）
        :return:
            alpha: 第一路门控权重 [B, T, D]
            beta: 第二路门控权重 [B, T, D]
            h_last: GRU最终隐藏状态
        """
        # 获取批次与时间维度
        B, T, _ = x.shape

        # ===================== Step 1: GRU 时序编码 =====================
        # 对输入序列进行编码，输出每个时间步的激素状态
        h_seq, h_last = self.gru(x, prev_state)
        # h_seq shape: [batch_size, seq_len, hidden_dim]

        # ===================== Step 2: 生成动态门控 =====================
        # 全连接层生成原始门控值
        gate = self.controller(h_seq)  # [B, T, 2 * output_dim]

        # 重塑为 [B, T, 2, D]，分别对应 alpha 和 beta
        gate = gate.view(B, T, 2, -1)

        # 在维度2上做Softmax，严格满足生理约束：alpha + beta = 1
        gate = torch.softmax(gate, dim=2)

        # 拆分得到两路动态权重
        alpha = gate[:, :, 0, :]  # [B, T, output_dim]
        beta  = gate[:, :, 1, :]  # [B, T, output_dim]

        # ===================== Step 3: 正则化 =====================
        alpha = self.dropout(alpha)
        beta = self.dropout(beta)

        # ===================== Step 4: 激素动态可视化 =====================
        # 仅在单样本情况下绘制时间维度的激素变化曲线
        if self.visualize and B == 1:
            plt.figure(figsize=(6, 3))
            plt.plot(alpha[0].mean(dim=-1).detach().cpu(), label='alpha_mean')
            plt.plot(beta[0].mean(dim=-1).detach().cpu(), label='beta_mean')
            plt.title("Hormone Dynamics Over Time")
            plt.xlabel("Time Step")
            plt.ylabel("Gate Value")
            plt.legend()
            plt.show()

        return alpha, beta, h_last
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleTemporalEncoder(nn.Module):
    """
    多尺度时间编码器 (Multi-Scale Temporal Encoder)
    论文创新模块：多尺度时序特征提取 + 自适应融合

    核心设计思想：
    1. 多尺度建模：分别捕捉短期波动、中期趋势、长期规律
       - scale=1 → 精细短期特征
       - scale=2 → 中期趋势特征
       - scale=4 → 长期全局趋势
    2. 独立编码：每个尺度使用独立Transformer，避免信息干扰
    3. 对齐融合：上采样回原始时间轴，使用可学习权重自适应融合
    4. 残差连接：保证梯度稳定，深层训练更可靠

    适用场景：股票预测、生理信号、长时间序列建模
    """

    def __init__(self, input_dim, n_heads=4, num_layers=2,
                 dropout=0.1, scales=[1,2,4], max_seq_len=512):
        """
        初始化多尺度时间编码器
        :param input_dim: 输入特征维度
        :param n_heads: Transformer 注意力头数
        :param num_layers: 每个尺度的 Transformer 层数
        :param dropout: 正则化 dropout
        :param scales: 时间尺度列表 [1,2,4]
        :param max_seq_len: 最大序列长度（用于位置编码）
        """
        super().__init__()

        self.scales = scales                  # 时间尺度配置
        self.input_dim = input_dim            # 输入特征维度

        # ===================== 多尺度 Transformer 编码器 =====================
        # 每个时间尺度独立使用一个 Transformer 编码器
        self.encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=input_dim,
                    nhead=n_heads,
                    dim_feedforward=input_dim * 4,  # 前馈层维度 4倍
                    dropout=dropout,
                    batch_first=True               # 输入形状 [B, T, D]
                ),
                num_layers=num_layers               # 编码器层数
            ) for _ in scales
        ])

        # ===================== 多尺度位置编码 =====================
        # 为不同尺度生成独立位置编码，符合时序特性
        pos_enc_list = [
            self._generate_pos_encoding(max_seq_len, input_dim, s)
            for s in scales
        ]
        # 注册为模型缓冲区，不参与梯度更新
        self.register_buffer("pos_enc", torch.stack(pos_enc_list, dim=0))

        # ===================== 可学习尺度融合权重 =====================
        # 自动学习不同尺度的重要性
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))

        # 标准化与正则化
        self.layernorm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def _generate_pos_encoding(self, seq_len, dim, scale):
        """
        生成多尺度正弦余弦位置编码
        :param seq_len: 序列长度
        :param dim: 特征维度
        :param scale: 时间尺度因子
        :return: 位置编码 [1, seq_len, dim]
        """
        pos = torch.arange(seq_len).unsqueeze(1).float()
        i = torch.arange(dim).unsqueeze(0).float()

        # 尺度缩放位置频率
        angle_rates = scale / torch.pow(10000, (2 * (i // 2)) / dim)
        angle = pos * angle_rates

        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(angle[:, 0::2])  # 偶数位 sin
        pe[:, 1::2] = torch.cos(angle[:, 1::2])  # 奇数位 cos

        return pe.unsqueeze(0)  # [1, T, D]

    def forward(self, x):
        """
        前向传播
        :param x: 输入序列 [batch_size, seq_len, input_dim]
        :return: 多尺度融合输出 [B, T, D]
        """
        B, T, D = x.shape

        multi_outputs = []

        # ===================== 逐尺度编码 =====================
        for i, scale in enumerate(self.scales):
            # Step 1: 时间下采样（降低时间分辨率）
            if scale > 1:
                x_scaled = x[:, ::scale, :]  # 间隔采样
            else:
                x_scaled = x

            # Step 2: 加入对应尺度的位置编码
            x_scaled = x_scaled + self.pos_enc[i, :, :x_scaled.shape[1], :]

            # Step 3: Transformer 特征提取
            out = self.encoders[i](x_scaled)

            # Step 4: 上采样回原始时间长度（保证对齐）
            if scale > 1:
                out = F.interpolate(
                    out.transpose(1, 2),  # [B, D, T]
                    size=T,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)       # 恢复 [B, T, D]

            multi_outputs.append(out)

        # ===================== 可学习多尺度融合 =====================
        # 堆叠多尺度输出
        multi_outputs = torch.stack(multi_outputs, dim=0)  # [S, B, T, D]

        # 对尺度维度做 softmax，保证权重和为1
        weights = torch.softmax(self.scale_weights, dim=0)

        # 加权融合
        out = (multi_outputs * weights[:, None, None, None]).sum(dim=0)

        # ===================== 残差连接 + 归一化 =====================
        out = self.dropout(out)
        out = self.layernorm(out + x)  # 残差连接，稳定训练

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F

# === 最终脑启发式模型 ===
class BrainInspiredNetV2(nn.Module):
    """
    脑启发式时间序列预测模型 (Brain-Inspired Temporal Predictive Network V2)
    论文核心创新模型：融合双路径信息流动 + 激素动态调控 + 多尺度时序编码

    核心生物启发逻辑：
    1. 双路径信息处理
       - Direct Path (直接通路)：模拟大脑局部感知，快速捕捉短期直接特征
       - Indirect Path (间接通路)：模拟注意力/关联机制，捕捉全局依赖与交互
    2. 激素动态调控：模拟内分泌系统，自适应调节双路径信息融合比例
    3. 多尺度时序编码：模拟大脑多尺度时间感知，捕捉长短期趋势
    4. 输出约束：使用 Tanh 激活，适配金融收益率预测（-1~1 归一化范围）

    适用任务：
    - 股票收益率预测（多股票联合预测）
    - 生理时间序列预测（激素、心电）
    - 多变量时序预测任务
    """

    def __init__(self, input_dim, hidden_dim=64, hormone_dim=32, 
                 max_seq_len=72, pooling="last", output_dim=5):
        """
        模型初始化
        :param input_dim: 输入特征维度 D
        :param hidden_dim: 模型全局隐藏维度 H
        :param hormone_dim: 激素模块内部隐藏维度
        :param max_seq_len: 支持的最大序列长度
        :param pooling: 时序池化方式，可选 [mean, max, last, attention]
        :param output_dim: 输出维度（对应预测股票数量）
        """
        super().__init__()

        # -------------------------
        # ① 直接路径 MLP：局部特征提取通路
        # 模拟大脑浅层感知，快速编码局部时序模式
        # -------------------------
        self.direct_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # [B, T, D] -> [B, T, H]
            nn.ReLU(),                          
            nn.LayerNorm(hidden_dim),          
            nn.Dropout(0.2),                   
            nn.Linear(hidden_dim, hidden_dim)  # 输出 [B, T, H]
        )

        # -------------------------
        # ② 间接路径：血液流动注意力层
        # 模拟全局关联与信息传播，捕捉时序依赖
        # -------------------------
        self.indirect_path = GraphAttentionLayer(input_dim, hidden_dim, n_heads=8)

        # -------------------------
        # ③ 激素动态调控模块
        # 生物启发：动态生成门控，自适应融合双路径信息
        # -------------------------
        self.hormone_mod = HormoneModule(hidden_dim, hormone_dim, hidden_dim)

        # -------------------------
        # ④ 多尺度时间编码器
        # 捕捉短期波动、中期趋势、长期全局模式
        # -------------------------
        self.temporal_encoder = MultiScaleTemporalEncoder(
            input_dim=hidden_dim,      
            n_heads=4,
            num_layers=2,
            dropout=0.1,
            max_seq_len=max_seq_len,
            scales=[1,2,4]
        )

        # -------------------------
        # 特征融合与输出层
        # -------------------------
        self.fusion_dim = nn.Linear(hidden_dim, hidden_dim)  

        # 最终预测头：输出预测值 + Tanh 约束范围（适配金融数据）
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            # nn.Tanh()   
        )

        # -------------------------
        # 双路径维度对齐层
        # 保证 direct / indirect 特征维度一致
        # -------------------------
        self.direct_proj = nn.Identity()         
        self.indirect_proj = nn.Linear(hidden_dim, hidden_dim)

        # -------------------------
        # 时序池化策略
        # 将 [B, T, H] 压缩为 [B, H] 用于最终预测
        # -------------------------
        self.pooling = pooling
        if pooling == "attention":
            # 可学习注意力池化：自动关注重要时间步
            self.att_pool = nn.Sequential(
                nn.Linear(hidden_dim, 128),  
                nn.Tanh(),
                nn.Linear(128, 1)           
            )

    def forward(self, x, adj_matrix=None, hormone_prev=None, mask=None):
        """
        模型前向传播
        :param x: 输入时序特征 [batch_size, seq_len, input_dim]
        :param adj_matrix: 邻接矩阵（本模型未使用，保留接口）
        :param hormone_prev: 上一时刻激素隐藏状态
        :param mask: 时序掩码（padding 部分）
        :return: 
            y: 模型预测输出 [B, output_dim]
            hormone_state: 当前激素状态
        """
        B, T, D = x.shape

        # -------------------------
        # ① 双路径并行编码
        # -------------------------
        direct_out = self.direct_mlp(x)                     # 直接路径：局部特征 [B, T, H]
        indirect_out = self.indirect_path(x)                # 间接路径：全局依赖 [B, T, H]

        # -------------------------
        # ② 双路径维度对齐
        # -------------------------
        direct_out = self.direct_proj(direct_out)          
        indirect_out = self.indirect_proj(indirect_out)    

        # -------------------------
        # ③ 激素动态门控融合
        # 核心创新：alpha + beta = 1，动态加权双路径
        # -------------------------
        alpha, beta, hormone_state = self.hormone_mod(direct_out + indirect_out, hormone_prev)
        fused = alpha * direct_out + beta * indirect_out   

        # -------------------------
        # ④ 多尺度时序特征增强
        # -------------------------
        fused = self.fusion_dim(fused)                     
        encoded = self.temporal_encoder(fused)             

        # -------------------------
        # ⑤ 时序池化：压缩时间维度
        # -------------------------
        if self.pooling == "mean":
            pooled = encoded.mean(dim=1)                    
        elif self.pooling == "max":
            pooled, _ = encoded.max(dim=1)                  
        elif self.pooling == "last":
            pooled = encoded[:, -1, :]                     
        elif self.pooling == "attention":
            att_weights = self.att_pool(encoded).squeeze(-1)  
            if mask is not None:
                att_weights = att_weights.masked_fill(mask.bool(), float('-inf'))
            att_weights = torch.softmax(att_weights, dim=1).unsqueeze(-1) 
            pooled = (encoded * att_weights).sum(dim=1)     
        else:
            raise ValueError(f"未知的池化类型: {self.pooling}")

        # -------------------------
        # ⑥ 最终预测输出
        # -------------------------
        y = self.out_layer(pooled)  

        return y, hormone_state  
import matplotlib.pyplot as plt
import math
import torch

def visualize_alpha_beta(alpha, beta, batch_idx=0, title="Alpha/Beta Weights", save_path=None, max_subplots=6):
    """
    可视化激素调控模块输出的动态门控权重 Alpha & Beta
    
    功能说明：
    ----------
    对激素模块生成的双路径融合权重（alpha, beta）进行时序可视化
    每一行代表一个隐藏维度的权重随时间步的变化曲线
    支持自动分页，解决高维隐藏层无法一图展示的问题

    生物学意义：
    ----------
    Alpha：直接路径（局部特征）权重
    Beta：间接路径（全局注意力）权重
    两者满足约束：Alpha + Beta = 1，体现动态调控机制

    参数：
    ----------
    alpha: torch.Tensor  [B, T, H]  直接路径门控权重
    beta : torch.Tensor  [B, T, H]  间接路径门控权重
    batch_idx: int       选择要可视化的批次样本
    title: str           图表总标题
    save_path: str       图片保存路径（不含后缀），为 None 则只显示不保存
    max_subplots: int    单页最大子图数量，防止图表过高
    """
    # 从指定批次提取数据，并从计算图中分离，转移到 CPU 转为 numpy
    alpha_np = alpha[batch_idx].detach().cpu().numpy()  # [T, H]
    beta_np  = beta[batch_idx].detach().cpu().numpy()   # [T, H]

    # 获取维度信息
    B, T, H = alpha.shape
    num_pages = math.ceil(H / max_subplots)  # 计算总页数

    # 逐页绘制
    for page in range(num_pages):
        start_idx = page * max_subplots
        end_idx = min((page + 1) * max_subplots, H)
        num_plots = end_idx - start_idx

        plt.figure(figsize=(12, 3 * num_plots))
        
        # 逐个隐藏维度绘制
        for i in range(start_idx, end_idx):
            plt.subplot(num_plots, 1, i - start_idx + 1)
            plt.plot(alpha_np[:, i], label=f"Alpha {i+1}", color='tab:blue')
            plt.plot(beta_np[:, i], label=f"Beta {i+1}", color='tab:orange')
            plt.xlabel("Time Step")
            plt.ylabel("Gate Weight")
            plt.title(f"Hormone Gate Hidden Dim {i+1}")
            plt.legend()
            plt.grid(True)

        # 总标题 + 布局优化
        plt.suptitle(f"{title} (Page {page+1}/{num_pages})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 保存图片（如果需要）
        if save_path:
            plt.savefig(f"{save_path}_page{page+1}.png", dpi=200, bbox_inches='tight')
        plt.show()


import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

def evaluate_model(model, data_loader, criterion, device, use_mask=False):
    """
    时间序列预测模型通用评估函数
    功能：计算模型在验证集/测试集上的损失与各项回归指标

    核心指标：
    ----------
    - Loss: 模型预测损失（MSE / MAE等）
    - R2: 决定系数，衡量预测拟合程度（越接近1越好）
    - MAE: 平均绝对误差，稳健衡量误差大小
    - RMSE: 均方根误差，对大误差敏感

    支持特性：
    ----------
    1. 支持多变量/多股票同时评估
    2. 支持缺失值掩码（NaN屏蔽）
    3. 兼容模型返回 (pred, state) 类型的输出
    4. 自动适配单输出/多输出维度

    参数：
    ----------
    model: 训练好的脑启发时序预测模型
    data_loader: 验证/测试集数据加载器
    criterion: 损失函数（如MSELoss）
    device: 运行设备 cuda / cpu
    use_mask: 是否开启缺失值屏蔽，默认False

    返回：
    ----------
    avg_loss: 平均损失
    r2: R2 得分（标量或列表）
    mae: 平均绝对误差
    rmse: 均方根误差
    """
    # 切换模型为评估模式，关闭Dropout/BatchNorm
    model.eval()
    loss_sum = 0.0
    y_true_list, y_pred_list = [], []

    # 禁用自动求导，加速推理
    with torch.no_grad():
        for x_batch, y_batch in tqdm(data_loader, desc="Evaluating", leave=False):
            # 数据迁移到指定设备
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # 前向传播（兼容多输出模型，如返回(pred, hormone_state)）
            outputs = model(x_batch)
            y_pred = outputs[0] if isinstance(outputs, tuple) else outputs

            # 维度对齐：去除多余的单维度
            if y_pred.dim() == 2 and y_pred.size(1) == 1:
                y_pred = y_pred.squeeze(1)
            if y_batch.dim() == 2 and y_batch.size(1) == 1:
                y_batch = y_batch.squeeze(1)

            # 缺失值掩码处理：忽略标签中的 NaN 位置
            if use_mask:
                mask = ~torch.isnan(y_batch)
                y_pred_masked = y_pred[mask]
                y_batch_masked = y_batch[mask]
            else:
                y_pred_masked = y_pred
                y_batch_masked = y_batch

            # 计算批次损失
            try:
                batch_loss = criterion(y_pred_masked, y_batch_masked).item()
            except Exception as e:
                print(f"[Warning] 计算损失失败: {e}")
                batch_loss = 0.0
            loss_sum += batch_loss

            # 保存真实值与预测值（用于全局指标计算）
            y_true_list.append(y_batch_masked.cpu().numpy())
            y_pred_list.append(y_pred_masked.cpu().numpy())

    # ===================== 计算平均损失 =====================
    avg_loss = loss_sum / len(data_loader)

    # 拼接所有批次数据
    y_true_all = np.concatenate(y_true_list, axis=0)
    y_pred_all = np.concatenate(y_pred_list, axis=0)

    # 确保至少为2维 [N, D]
    if y_true_all.ndim == 1:
        y_true_all = y_true_all[:, None]
    if y_pred_all.ndim == 1:
        y_pred_all = y_pred_all[:, None]

    # ===================== 逐维度计算评估指标 =====================
    metrics = {"R2": [], "MAE": [], "RMSE": []}
    num_targets = y_true_all.shape[-1]

    for i in range(num_targets):
        y_true_i = y_true_all[:, i]
        y_pred_i = y_pred_all[:, i]
        try:
            metrics["R2"].append(r2_score(y_true_i, y_pred_i))
            metrics["MAE"].append(mean_absolute_error(y_true_i, y_pred_i))
            metrics["RMSE"].append(np.sqrt(mean_squared_error(y_true_i, y_pred_i)))
        except Exception as e:
            print(f"[Warning] 维度 {i} 指标计算失败: {e}")
            metrics["R2"].append(np.nan)
            metrics["MAE"].append(np.nan)
            metrics["RMSE"].append(np.nan)

    # 单输出时返回标量，多输出返回列表
    if num_targets == 1:
        for key in metrics:
            metrics[key] = metrics[key][0]

    return avg_loss, metrics["R2"], metrics["MAE"], metrics["RMSE"]

import os
import time
import json
import numpy as np
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

# 假设你已有这两个函数
# from utils import setup_logger, evaluate_model

logger = logging.getLogger(__name__)

class Trainer:
    """
    脑启发时序预测模型 - 统一训练与验证框架
    功能：封装完整训练流程、早停、学习率调度、模型保存、日志记录

    核心机制：
    1. 训练/验证双循环
    2. 早停机制（防止过拟合）
    3. 自适应学习率衰减
    4. 缺失值掩码支持
    5. 最优模型自动保存
    6. 结构化日志记录（便于论文绘图）

    适用模型：BrainInspiredNetV2 及所有时序预测模型
    """

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, 
                 max_epochs=100, patience=10, lr_scheduler_patience=3, 
                 lr_scheduler_factor=0.5, save_path=None, log_path=None, 
                 custom_callbacks=None, use_mask=False, save_state_dict=False):
        """
        初始化训练器
        :param model: 待训练模型（BrainInspiredNetV2）
        :param train_loader: 训练集数据加载器
        :param val_loader: 验证集数据加载器
        :param optimizer: 优化器（Adam / AdamW）
        :param criterion: 损失函数（MSE / MAE）
        :param device: 运行设备 cuda / cpu
        :param max_epochs: 最大训练轮数
        :param patience: 早停轮数
        :param lr_scheduler_patience: 学习率衰减等待轮数
        :param lr_scheduler_factor: 学习率衰减系数
        :param save_path: 模型保存路径
        :param log_path: 训练日志路径
        :param custom_callbacks: 自定义回调函数
        :param use_mask: 是否启用缺失值掩码
        :param save_state_dict: 是否只保存模型参数
        """
        # 基础组件
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # 训练策略
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor

        # 保存与日志
        self.save_path = save_path
        self.log_path = log_path
        self.custom_callbacks = custom_callbacks or []
        self.use_mask = use_mask
        self.save_state_dict = save_state_dict

        # 初始化日志
        if log_path is not None:
            logging.basicConfig(
                filename=log_path,
                level=logging.INFO,
                format='%(message)s'
            )

        # 设备检查与迁移
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，自动切换至 CPU")
            self.device = "cpu"
        self.device = torch.device(self.device)
        self.model.to(self.device)

    def train(self):
        """
        启动完整训练流程
        :return: 训练历史记录（损失、指标、最优模型信息）
        """
        # ===================== 初始化训练状态 =====================
        best_loss = float("inf")             # 最佳验证损失
        best_model_state = None              # 最优模型参数
        best_epoch = 0                        # 最优轮数
        train_losses = []                     # 训练损失历史
        val_losses = []                       # 验证损失历史
        val_metrics = {"R2": [], "MAE": [], "RMSE": []}  # 评估指标历史

        # 学习率调度器（自适应衰减）
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=self.lr_scheduler_patience,
            factor=self.lr_scheduler_factor
        )

        early_stop_counter = 0  # 早停计数器
        total_start_time = time.time()

        # ===================== 开始训练循环 =====================
        for epoch in range(1, self.max_epochs + 1):
            epoch_start_time = time.time()
            self.model.train()  # 切换训练模式
            train_loss_sum = 0.0

            # ===================== 批次训练 =====================
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch:03d} Training", leave=True):
                x_batch, y_batch = batch
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()  # 梯度清零

                # 前向传播（兼容多输出模型：pred + hormone_state）
                outputs = self.model(x_batch)
                pred = outputs[0] if isinstance(outputs, tuple) else outputs

                # 自动对齐输出维度（处理维度不匹配）
                if pred.size(1) != y_batch.size(1):
                    if pred.size(1) > y_batch.size(1):
                        pred = pred[:, :y_batch.size(1)]
                    else:
                        pad = torch.zeros(pred.size(0), y_batch.size(1)-pred.size(1), device=self.device)
                        pred = torch.cat([pred, pad], dim=1)

                # 缺失值掩码
                if self.use_mask:
                    mask = ~torch.isnan(y_batch)
                    pred_masked = pred[mask]
                    y_batch_masked = y_batch[mask]
                else:
                    pred_masked = pred
                    y_batch_masked = y_batch

                # 损失计算 + 反向传播
                loss = self.criterion(pred_masked, y_batch_masked)
                loss.backward()
                self.optimizer.step()

                train_loss_sum += loss.item()

            # ===================== 训练集平均损失 =====================
            avg_train_loss = train_loss_sum / len(self.train_loader)
            train_losses.append(avg_train_loss)

            # ===================== 验证集评估 =====================
            avg_val_loss, r2, mae, rmse = evaluate_model(
                self.model, self.val_loader, self.criterion, self.device, self.use_mask
            )
            val_losses.append(avg_val_loss)
            val_metrics["R2"].append(r2)
            val_metrics["MAE"].append(mae)
            val_metrics["RMSE"].append(rmse)

            # 更新学习率
            scheduler.step(avg_val_loss)

            # ===================== 日志记录 =====================
            epoch_duration = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            r2_mean = np.mean(r2) if isinstance(r2, (list, np.ndarray)) else r2
            mae_mean = np.mean(mae) if isinstance(mae, (list, np.ndarray)) else mae
            rmse_mean = np.mean(rmse) if isinstance(rmse, (list, np.ndarray)) else rmse

            log_data = {
                "epoch": epoch,
                "train_loss": round(float(avg_train_loss), 4),
                "val_loss": round(float(avg_val_loss), 4),
                "R2": round(float(r2_mean), 4),
                "MAE": round(float(mae_mean), 4),
                "RMSE": round(float(rmse_mean), 4),
                "lr": float(current_lr),
                "time_sec": round(epoch_duration, 2)
            }
            logger.info(json.dumps(log_data))

            # ===================== 早停 & 最优模型保存 =====================
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_state = self.model.state_dict()
                best_epoch = epoch
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.patience:
                    logger.info(f"Early stop at epoch {epoch}")
                    break

            # 自定义回调
            for cb in self.custom_callbacks:
                try:
                    cb(self.model, epoch, avg_train_loss, avg_val_loss, self.optimizer, scheduler, best_loss, best_epoch)
                except Exception as e:
                    logger.warning(f"Callback error: {e}")

        # ===================== 训练结束 =====================
        total_duration = time.time() - total_start_time
        logger.info(f"Training done. Time: {total_duration/60:.1f} min")

        # 加载最优模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

            # 保存模型
            if self.save_path:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_path = f"{os.path.splitext(self.save_path)[0]}_{ts}.pth"
                if self.save_state_dict:
                    torch.save(best_model_state, final_path)
                else:
                    torch.save(self.model, final_path)
                logger.info(f"Best model saved: {final_path}")

        # 输出最佳指标
        best_metrics = {
            "best_epoch": best_epoch,
            "best_loss": round(float(best_loss), 4),
            "best_R2": round(float(np.mean(val_metrics["R2"][best_epoch-1])), 4),
            "best_MAE": round(float(np.mean(val_metrics["MAE"][best_epoch-1])), 4),
            "best_RMSE": round(float(np.mean(val_metrics["RMSE"][best_epoch-1])), 4)
        }
        logger.info(f"Best: {best_metrics}")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_metrics": val_metrics,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "total_time": total_duration
        }
import os
import torch
import logging
from datetime import datetime

# 模块级日志器
logger = logging.getLogger(__name__)


def save_intermediate_model(model, epoch, train_loss, val_loss, optimizer=None, scheduler=None, best_loss=None, best_epoch=None):
    """
    训练中间模型保存回调函数
    功能：在训练过程中自动保存 Checkpoint，包含模型、优化器、调度器状态
    用于：断点续训、训练过程分析、实验复现

    保存内容：
    - 模型参数 (model_state_dict)
    - 当前轮数 epoch
    - 训练损失 / 验证损失
    - 优化器状态（可选）
    - 学习率调度器状态（可选）

    参数：
    ----------
    model: 训练中的模型
    epoch: 当前训练轮数
    train_loss: 本轮训练损失
    val_loss: 本轮验证损失
    optimizer: 优化器（可选）
    scheduler: 学习率调度器（可选）
    best_loss: 历史最佳损失（可选）
    best_epoch: 最佳轮数（可选）

    返回：
    ----------
    save_path: 保存的文件路径
    """
    # 创建模型保存目录
    dir_path = "models"
    os.makedirs(dir_path, exist_ok=True)

    # 生成时间戳，避免文件名重复
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(dir_path, f"intermediate_epoch_{epoch}_{timestamp}.pth")

    # 构建 checkpoint 字典
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    # 保存优化器状态（支持断点续训）
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        current_lr = optimizer.param_groups[0].get("lr", "unknown")
        logger.info(f"[Epoch {epoch}] 当前学习率: {current_lr:.6f}")

    # 保存调度器状态
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        scheduler_patience = getattr(scheduler, "patience", "N/A")
        logger.info(f"[Epoch {epoch}] 调度器耐心值: {scheduler_patience}")

    # 保存 checkpoint
    torch.save(checkpoint, save_path)
    logger.info(f"✅ 中间模型已保存: {save_path}")

    return save_path

"""
脑启发时序预测模型 - 批量训练与消融实验脚本
================================================
本脚本用于批量训练 BrainInspiredNetV2 及其多个消融版本模型。
主要功能包括：
1. 多模型/消融版本自动循环训练
2. 日志自动记录（控制台 + 文件双输出）
3. 集成早停、自适应学习率调度
4. 自动保存训练曲线、模型权重、配置文件
5. 自动生成模型对比图表，便于论文写作
"""

import os
import json
import time
import torch
import logging
import traceback
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

# ---------------------- 中文字体配置 ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']    # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# ==================== 股票数据集类 ====================
class StockDataset(Dataset):
    """
    股票时间序列数据集类
    功能：封装训练/验证数据，适配 PyTorch DataLoader
    输入：X (时序特征), y (预测目标)
    输出：单条样本 (X[idx], y[idx])
    """
    def __init__(self, X, y):
        # self.X = torch.tensor(X, dtype=torch.float32)
        # self.y = torch.tensor(y, dtype=torch.float32)
        self.X = X.clone().detach().float()
        self.y = y.clone().detach().float()
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==================== 数据加载器创建函数 ====================
def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=8, shuffle=True):
    """
    创建训练与验证数据加载器
    :param X_train: 训练特征 [N, T, D]
    :param y_train: 训练标签 [N, C]
    :param X_val: 验证特征
    :param y_val: 验证标签
    :param batch_size: 批次大小
    :param shuffle: 是否打乱
    :return: 训练/验证 DataLoader
    """
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# ==================== 模型名称与类的映射（消融实验） ====================
model_dict = {
    "Full": BrainInspiredNetV2,                    # 完整脑启发模型（所有模块）
    # "NoHormone": BrainInspiredNetV2_NoHormone,     # 无激素模块
    # "NoIndirect": BrainInspiredNetV2_NoIndirect,   # 无间接注意力路径
    # "NoTemporal": BrainInspiredNetV2_NoTemporal,   # 无多尺度时间编码器
    # "SimpleFC": BrainInspiredNetV2_SimpleFC        # 仅全连接基线模型
}

# ==================== 训练超参数配置 ====================
train_config = {
    "max_epochs": 50,              # 最大训练轮数
    "patience": 20,                 # 早停轮数
    "lr_scheduler_patience": 8,    # 学习率衰减等待轮数
    "lr_scheduler_factor": 0.5,    # 学习率衰减系数
    "batch_size": 16,               # 批次大小
    "learning_rate": 2e-4          # 初始学习率
}

# ==================== 日志配置函数 ====================
def setup_logger(log_path=None, level=logging.INFO):
    """
    日志系统配置：控制台 + 文件双输出
    自动创建日志目录，每条日志带时间戳与级别
    """
    logger_name = f"Logger_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# ==================== 训练曲线可视化 ====================
def plot_training_curve(training_results, model_key, save_dir="figures"):
    """
    绘制单模型训练/验证损失曲线
    自动保存为高清 PNG，用于论文展示
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(training_results['train_losses'], label='Train Loss')
    plt.plot(training_results['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'模型：{model_key} 训练损失曲线')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, f"training_curve_{model_key}.png"), dpi=150)
    plt.close()

# ==================== 设备与路径初始化 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_base_path = "logs"
figure_path = "figures"
os.makedirs(log_base_path, exist_ok=True)
os.makedirs(figure_path, exist_ok=True)

# ==================== 加载数据（用户需自行定义 X_train, y_train, X_val, y_val） ====================
# 示例：
# X_train: [样本数, 时间步, 特征维度]
# y_train: [样本数, 预测维度]
# ==================== 请在这里加载你的数据 ====================

# ==================== 创建数据加载器 ====================
train_loader, val_loader = create_data_loaders(
    X_train, y_train, X_val, y_val, batch_size=train_config["batch_size"]
)

# ==================== 存储所有模型训练结果 ====================
all_training_results = {}

# ==================== 遍历所有模型进行训练 ====================
for model_key in model_dict:
    try:
        # 日志
        log_path = os.path.join(log_base_path, f"training_{model_key}.log")
        logger = setup_logger(log_path=log_path)
        logger.info(f"===== 开始训练模型：{model_key} =====")

        # 构建模型
        model_class = model_dict[model_key]
        # model = model_class(input_dim=X_train.shape[2]).to(device)
        # model = model_class(
        #         input_dim=X_train.shape[2], 
        #         hidden_dim=64, 
        #         hormone_dim=32, 
        #         max_seq_len=30, 
        #         pooling="last", 
        #         output_dim=5  
        #     ).to(device)
        model = model_class(
                input_dim=X_train.shape[2], 
                hidden_dim=16, 
                hormone_dim=16, 
                max_seq_len=30, 
                pooling="last", 
                output_dim=5  
            ).to(device)
        # 优化器 & 损失
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"])
        criterion = nn.MSELoss()

        # 训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_epochs=train_config["max_epochs"],
            patience=train_config["patience"],
            lr_scheduler_patience=train_config["lr_scheduler_patience"],
            lr_scheduler_factor=train_config["lr_scheduler_factor"],
            save_path=f"best_model_{model_key}.pth",
            log_path=log_path,
            custom_callbacks=[save_intermediate_model],
            use_mask=False,
            save_state_dict=True
        )

        # 开始训练
        start_time = time.time()
        training_results = trainer.train()
        duration = time.time() - start_time

        # 训练完成信息
        logger.info("✅ 训练完成！")
        logger.info(f"⏱️ 总耗时：{duration:.2f}s")
        logger.info(f"🏆 最佳损失：{training_results['best_loss']:.6f}（Epoch {training_results['best_epoch']}）")

        # 绘制曲线
        plot_training_curve(training_results, model_key, save_dir=figure_path)

        # 保存配置
        config = {
            "model": model_key,
            "input_dim": X_train.shape[2],
            "device": str(device),
            "time_cost": duration,
            "best_epoch": training_results["best_epoch"],
            "best_loss": float(training_results["best_loss"]),
            **train_config
        }
        with open(f"training_config_{model_key}.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)

        # 记录结果
        all_training_results[model_key] = {
            "train_losses": training_results["train_losses"],
            "val_losses": training_results["val_losses"],
            "best_epoch": training_results["best_epoch"],
            "best_loss": float(training_results["best_loss"])
        }

        # 清理显存
        if device.type == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"❌ 模型 {model_key} 训练失败：\n{traceback.format_exc()}")

# ==================== 保存所有结果 ====================
with open("all_training_results.json", "w", encoding='utf-8') as f:
    json.dump(all_training_results, f, indent=4)
print("✅ 所有训练结果已保存至 all_training_results.json")

# ==================== 绘制模型对比图 ====================
plt.figure(figsize=(12, 6))
for model_key, result in all_training_results.items():
    plt.plot(result["val_losses"], label=f"{model_key} | Best: {result['best_loss']:.4f}")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("各模型验证损失对比（消融实验）")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(figure_path, "validation_loss_comparison.png"), dpi=150)
plt.show()