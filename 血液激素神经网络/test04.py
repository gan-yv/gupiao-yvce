# -*- coding: utf-8 -*-
"""
股票多变量时间序列数据预处理脚本
核心功能：
  1. 读取多只股票原始历史数据
  2. 自动化生成技术指标特征（RSI/MACD/均线/波动率等）
  3. 构建时间序列滑动窗口（用前N天预测后1天）
  4. 数据标准化 + 训练/验证集划分
  5. 保存预处理结果供深度学习模型训练使用
可调参数：MACD周期 / RSI周期 / 时间窗口 / 训练集比例 等
"""

# ========================== 1. 导入依赖库 ==========================
# 数据处理核心库：pandas处理表格数据，numpy处理数值计算
import pandas as pd
import numpy as np
# 文件操作库：用于创建文件夹、拼接文件路径
import os
# 数据标准化：RobustScaler抗异常值，适合金融数据
from sklearn.preprocessing import RobustScaler
# 模型保存：用于持久化保存标准化器
import pickle

# ========================== 2. 全局路径 & 【可调节超参数】 ==========================
# 原始股票数据CSV文件路径（请根据自己电脑路径修改）
csv_file = r"E:\大学\学习\血液激素神经网络\archive\all_stocks_5yr.csv"
# 预处理后数据的保存目录
output_dir = r"E:\大学\学习\血液激素神经网络\archive\processed"
# 自动创建输出目录：不存在则新建，存在则不报错
os.makedirs(output_dir, exist_ok=True)

# 需要批量处理的股票代码列表（可自由增删）
tickers = ['AAPL','MSFT','GOOGL','AMZN','MYL']

# ==================== 【核心可调超参数】 ====================
TIME_STEP = 30          # 时间窗口大小：用前30天的历史数据预测下1天
TRAIN_RATIO = 0.8       # 训练集占总数据比例：80%训练，20%验证
CLIP_BOUND = 0.1        # 收益率极端值裁剪阈值：±10%以外的收益强制截断

# RSI 相对强弱指标计算周期（常用值：7/9/14/21）
RSI_PERIOD = 14

# MACD 指标三大周期参数（经典默认：12,26,9）
MACD_FAST = 12          # MACD快线周期
MACD_SLOW = 26          # MACD慢线周期
MACD_SIGNAL = 9         # MACD信号线周期

# 简单移动平均线SMA计算周期列表
SMA_PERIODS = [10, 20, 50]
# 指数移动平均线EMA计算周期列表
EMA_PERIODS = [10, 20]

# 价格波动率计算周期（基于收益率滚动标准差）
VOLATILITY_PERIODS = [10, 20]
# ======================================================================

# ========================== 3. RSI 相对强弱指标计算函数 ==========================
def compute_RSI(series, period=RSI_PERIOD):
    """
    专业RSI（相对强弱指数）计算函数
    RSI作用：判断股票超买/超卖状态，取值0~100
    :param series: 股票收盘价时间序列
    :param period: RSI计算周期，默认使用顶部全局参数
    :return: 计算完成的RSI指标序列
    """
    # 第一步：计算每日价格的涨跌差值（后一天 - 前一天）
    delta = series.diff()
    
    # 第二步：分离上涨幅度和下跌幅度
    up = delta.clip(lower=0)        # 只保留上涨，下跌记为0
    down = -delta.clip(upper=0)     # 只保留下跌并取正值，上涨记为0
    
    # 第三步：用指数加权移动平均计算平均上涨/平均下跌
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    
    # 第四步：计算RS相对强弱值 + 防止分母为0
    rs = roll_up / (roll_down + 1e-10)
    
    # 第五步：标准公式计算RSI（最终落在0-100之间）
    return 100 - (100 / (1 + rs))

# ========================== 4. 技术指标特征工程函数 ==========================
def add_technical_indicators(df_t, ticker):
    """
    为单只股票生成全套机器学习特征
    包含：收益特征、成交量特征、均线、波动率、动量指标、MACD全套指标
    :param df_t: 单只股票的原始数据表格
    :param ticker: 股票代码
    :return: 新增所有技术指标后的完整数据表
    """
    # 提取当前股票的收盘价列
    close = df_t[f"{ticker}_Close"]

    # 1. 日收益率计算 + 极端值裁剪（避免异常值干扰模型）
    df_t[f"{ticker}_Return"] = close.pct_change().clip(-CLIP_BOUND, CLIP_BOUND)
    ret = df_t[f"{ticker}_Return"]

    # 2. 成交量特征工程（对数变换+差分，消除量纲影响，提升稳定性）
    df_t[f"{ticker}_Volume_Log"] = np.log1p(df_t[f"{ticker}_Volume"])  # log(1+x)避免0值问题
    df_t[f"{ticker}_Volume_Change"] = df_t[f"{ticker}_Volume_Log"].diff()  # 成交量日变化率
    df_t[f"{ticker}_Volume_Change"].fillna(0, inplace=True)  # 空值填充为0

    # 3. 移动平均线指标（趋势判断核心特征）
    for p in SMA_PERIODS:
        df_t[f"{ticker}_SMA_{p}"] = close.rolling(p).mean()  # 简单移动平均
    for p in EMA_PERIODS:
        df_t[f"{ticker}_EMA_{p}"] = close.ewm(span=p, adjust=False).mean()  # 指数移动平均

    # 4. 价格波动率特征（风险指标：收益率滚动标准差）
    for p in VOLATILITY_PERIODS:
        df_t[f"{ticker}_Volatility_{p}"] = ret.rolling(p).std()

    # 5. RSI动量指标（判断超买超卖）
    df_t[f"{ticker}_RSI_{RSI_PERIOD}"] = compute_RSI(close, RSI_PERIOD)

    # 6. MACD全套指标（趋势+动能核心指标）
    exp_fast = close.ewm(span=MACD_FAST, adjust=False).mean()    # 快线
    exp_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()    # 慢线
    macd_line = exp_fast - exp_slow                              # MACD主线
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()  # 信号线
    histogram = macd_line - signal_line                          # MACD柱

    # 将MACD三个指标存入数据表
    df_t[f"{ticker}_MACD"] = macd_line
    df_t[f"{ticker}_MACD_Signal"] = signal_line
    df_t[f"{ticker}_MACD_Hist"] = histogram

    return df_t

# ========================== 5. 读取原始数据并基础预处理 ==========================
# 读取股票CSV原始数据
df = pd.read_csv(csv_file)
# 将日期列转换为标准时间格式
df['date'] = pd.to_datetime(df['date'])
# 将日期设为索引，并按时间升序排序（时间序列必须有序）
df = df.set_index('date').sort_index()

# ========================== 6. 多股票数据合并与缺失值填充 ==========================
# 初始化空数据表，用于存储多股票合并后的数据
price_df = None

# 循环遍历每一只目标股票
for t in tickers:
    # 筛选出当前股票的所有数据
    df_t = df[df['Name'] == t].copy()
    # 如果该股票无数据，跳过
    if df_t.empty: continue

    # 重命名列名：给每列加上股票代码前缀，避免多股票列名冲突
    df_t.rename(columns={
        'open':f'{t}_Open','high':f'{t}_High','low':f'{t}_Low',
        'close':f'{t}_Close','volume':f'{t}_Volume'
    }, inplace=True)
    
    # 只保留价格+成交量核心列
    df_t = df_t[[f'{t}_Open',f'{t}_High',f'{t}_Low',f'{t}_Close',f'{t}_Volume']]

    # 第一只股票直接赋值，后续股票按日期横向拼接
    price_df = df_t if price_df is None else price_df.join(df_t, how='outer')

# 按日期排序 + 前向填充缺失值（交易日缺失用前一天数据填充）
price_df = price_df.sort_index().ffill()

# ========================== 7. 批量生成所有股票技术指标特征 ==========================
# 复制原始价格数据，避免修改原表
processed_df = price_df.copy()

# 循环为每只股票添加技术指标
for t in tickers:
    if f"{t}_Close" in processed_df.columns:
        processed_df = add_technical_indicators(processed_df, t)

# 截断缺失行：滚动计算指标会产生开头空值，直接删除
processed_df = processed_df.dropna()
# 按最长均线周期截断，保证所有指标都有有效数据
processed_df = processed_df.iloc[max(SMA_PERIODS):]

# ========================== 8. 构建特征矩阵X与标签y ==========================
# 存储所有模型输入特征的列名
feature_cols = []
# 存储所有股票收盘价列名（用于计算标签）
close_cols = []

# 自动匹配所有生成的特征列
for t in tickers:
    if f"{t}_Return" in processed_df.columns:
        feature_cols += [
            f"{t}_Return", f"{t}_Volume_Log", f"{t}_Volume_Change",
            *[f"{t}_SMA_{p}" for p in SMA_PERIODS],
            *[f"{t}_EMA_{p}" for p in EMA_PERIODS],
            *[f"{t}_Volatility_{p}" for p in VOLATILITY_PERIODS],
            f"{t}_RSI_{RSI_PERIOD}",
            f"{t}_MACD", f"{t}_MACD_Signal", f"{t}_MACD_Hist"
        ]
    close_cols.append(f"{t}_Close")

# 特征矩阵X：所有技术指标（模型输入）
X = processed_df[feature_cols].copy()
# 收盘价矩阵：用于计算预测目标
close = processed_df[close_cols]

# 标签y：下一日对数收益率（模型要预测的目标）
# 计算方式：log(明日收盘价) - log(今日收盘价)，并裁剪极端值
y = (np.log(close.shift(-1)) - np.log(close)).clip(-CLIP_BOUND, CLIP_BOUND)

# 只保留标签不为空的行（最后一行无明日数据，删除）
valid_idx = y.dropna().index
X = X.loc[valid_idx]
y = y.loc[valid_idx]

# ========================== 9. 时间序列数据集划分 ==========================
# 总样本数量
n_total = len(X)
# 计算训练集分割索引（时间序列不能随机打乱，必须按时间顺序划分）
split_idx = int(n_total * TRAIN_RATIO)

# 前80%为训练集
X_train_df = X.iloc[:split_idx]
y_train_df = y.iloc[:split_idx]
# 后20%为验证集
X_val_df = X.iloc[split_idx:]
y_val_df = y.iloc[split_idx:]

# ========================== 10. 时间序列滑动窗口构建 ==========================
def build_windows(X_df, y_df, step):
    """
    构建LSTM/时序模型所需的3D窗口数据
    格式：[样本数, 时间步长, 特征数]
    :param X_df: 特征数据
    :param y_df: 标签数据
    :param step: 时间窗口长度
    :return: 窗口化后的特征X和标签y
    """
    X_win, y_win = [], []
    # 滑动窗口遍历：每次取step天数据，预测第step+1天的标签
    for i in range(len(X_df) - step):
        X_win.append(X_df.iloc[i:i+step].values)  # 取前step天特征
        y_win.append(y_df.iloc[i + step].values)  # 取第step+1天标签
    # 转换为numpy浮点数组，适配深度学习框架
    return np.array(X_win, np.float32), np.array(y_win, np.float32)

# 对训练集和验证集分别构建时间窗口
X_train, y_train = build_windows(X_train_df, y_train_df, TIME_STEP)
X_val, y_val     = build_windows(X_val_df, y_val_df, TIME_STEP)

# ========================== 11. 3D时序数据标准化 ==========================
# 将3D数据展平为2D，才能用sklearn标准化器拟合
X_train_2d = X_train.reshape(-1, X_train.shape[-1])
# 使用抗异常值的RobustScaler（金融数据必备）
scaler = RobustScaler()
# 仅用训练集数据拟合标准化器（严禁使用验证集数据）
scaler.fit(X_train_2d)

def scale_3d(X, scaler):
    """
    3D数据标准化工具函数
    先展平→标准化→恢复3D形状
    """
    shape = X.shape
    X_2d = X.reshape(-1, shape[-1])
    X_2d = scaler.transform(X_2d)
    return X_2d.reshape(shape)

# 标准化训练集和验证集
X_train = scale_3d(X_train, scaler)
X_val   = scale_3d(X_val, scaler)

# ========================== 12. 保存预处理结果 ==========================
# 保存numpy数组数据
np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "y_train.npy"), y_train)
np.save(os.path.join(output_dir, "X_val.npy"), X_val)
np.save(os.path.join(output_dir, "y_val.npy"), y_val)

# 保存标准化器（后续模型预测必须使用同一个scaler）
with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ========================== 任务完成输出 ==========================
print("=" * 70)
print(f"RSI周期: {RSI_PERIOD} | MACD周期: {MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}")
print(f"训练集 X: {X_train.shape} | y: {y_train.shape}")
print(f"验证集 X: {X_val.shape}   | y: {y_val.shape}")
print("=" * 70)

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    r"""
    血液流动注意力机制层（多尺度指数衰减注意力，生理启发式时序建模）

    本层基于标准多头自注意力机制改进，融合**生理血液流动启发的指数衰减先验**
    与**多尺度时序依赖建模**，专为连续时序/生理信号、图时序数据设计，
    同时保留因果约束、自适应距离矩阵、SDPA 推理加速等工程优化。

    核心数学公式总览：
    1. 标准自注意力：Attention(Q,K,V) = softmax(QK^T / √d_k)V
    2. 血液流动衰减掩码：decay(t) = exp(-|i-j| / τ)，τ为多尺度衰减系数
    3. 多尺度融合：log_decay = LogSumExp(-dist/τ_s, -dist/τ_m, -dist/τ_l)
    4. 最终注意力得分：attn = (QK^T/√d_k + log_decay) · flow_strength
    5. GLU门控：out = out_proj(x) ⊗ σ(Linear(out_proj(x)))
    6. 残差归一化：out = LayerNorm(out + Linear(x))

    Args:
        in_dim (int): 输入特征维度
        out_dim (int): 输出特征维度（必须能被 n_heads 整除）
        n_heads (int): 注意力头数量，默认=8
        dropout (float): 注意力层 dropout 概率，默认=0.1
        max_T (int): 支持的最大序列长度，用于缓存距离矩阵，默认=2048

    Inputs:
        x (torch.Tensor): 输入序列张量，形状 [batch_size, seq_len, in_dim]

    Outputs:
        torch.Tensor: 注意力编码输出张量，形状 [batch_size, seq_len, out_dim]
    """
    def __init__(self, in_dim, out_dim, n_heads=8, dropout=0.1, max_T=2048):
        super().__init__()
        # 合法性校验：输出维度必须可被注意力头数整除
        assert out_dim % n_heads == 0, \
            f"输出维度 {out_dim} 无法被头数 {n_heads} 整除，请调整参数"
        
        # 基础结构参数
        self.n_heads = n_heads                # 注意力头数
        self.head_dim = out_dim // n_heads    # 单头注意力维度
        self.out_dim = out_dim                # 最终输出维度
        self.dropout = dropout                # Dropout概率
        self.max_T = max_T                    # 最大缓存序列长度

        # ============================ QKV 线性投影层 ============================
        # 输入特征 → 查询Q / 键K / 值V 投影映射
        # 公式：Q = xW_q, K = xW_k, V = xW_v
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)

        # 注意力输出投影：多头拼接 → 目标输出维度
        self.out_proj = nn.Linear(out_dim, out_dim)

        # ============================ 残差与归一化模块 ============================
        # 层归一化：稳定训练，加速收敛
        self.norm = nn.LayerNorm(out_dim)
        # 残差路径投影：输入输出维度不一致时做维度匹配
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        # ====================== 多尺度衰减系数 τ（生理启发式初始化） ======================
        # 设计依据：生理信号存在短/中/长三种时序依赖模式
        # τ_short  短期依赖：强指数衰减，仅关注极近邻时刻
        # τ_mid    中期依赖：中等衰减，关注局部时序窗口
        # τ_long   长期依赖：弱指数衰减，建模长距离时序依赖
        tau_short = 2.0
        tau_mid   = 8.0
        tau_long  = 30.0
        tau_init = torch.tensor([tau_short, tau_mid, tau_long])

        # 可学习参数：每个注意力头共享3个尺度并独立微调
        self.decay_tau = nn.Parameter(tau_init.repeat(n_heads, 1))
        # 血液流动强度：全局缩放注意力得分，可学习
        self.flow_strength = nn.Parameter(torch.tensor(1.0))

        # ============================ GLU 门控模块 ============================
        # GLU门控：增强特征选择能力，抑制噪声
        # 输出维度 ×2：用于拆分 特征值 + 门控值
        self.gate = nn.Linear(out_dim, out_dim * 2)

        # ============================ 自适应距离矩阵缓存 ============================
        # 缓存序列位置距离矩阵，避免重复计算，提升推理速度
        self.register_buffer('dist_matrix', torch.zeros(1, 1))
        self.cached_T = 1  # 记录当前缓存的序列长度

    def get_distance_matrix(self, T, device):
        """
        构建/获取位置绝对距离矩阵：dist[i,j] = |i - j|
        用于指数衰减注意力的距离计算

        Args:
            T (int): 当前序列长度
            device (torch.device): 张量运行设备

        Returns:
            torch.Tensor: 距离矩阵 [T, T]
        """
        # 缓存复用：若当前长度 ≤ 缓存长度，直接切片返回
        if T <= self.cached_T:
            return self.dist_matrix[:T, :T]
        
        # 重建缓存：限制最大长度避免OOM
        new_T = min(T, self.max_T)
        idx = torch.arange(new_T, device=device)
        # 计算位置绝对距离矩阵：广播机制生成 [T, T]
        self.dist_matrix = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)).float()
        self.cached_T = new_T
        
        return self.dist_matrix[:T, :T]

    def forward(self, x):
        """
        前向传播主函数：完整血液流动注意力计算流程
        """
        # 输入张量形状解析
        B, T, _ = x.shape  # B=批次大小，T=序列长度
        
        # ====================== 步骤1：获取位置距离矩阵 ======================
        dist = self.get_distance_matrix(T, x.device)  # [T, T]

        # ====================== 步骤2：QKV 多头投影与维度拆分 ======================
        # 投影 + 维度变换：[B,T,out_dim] → [B,n_heads,T,head_dim]
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # ====================== 步骤3：标准缩放点积注意力基础得分 ======================
        # 公式：attn_base = Q·K^T / √d_k
        # 形状：[B,n_heads,T,T]
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # ====================== 步骤4：多尺度指数衰减掩码计算 ======================
        # 约束τ到合理范围，保证数值稳定性
        tau = torch.clamp(self.decay_tau, min=1.0, max=100.0)
        tau = tau.view(1, self.n_heads, 3, 1, 1)  # [1,heads,3,1,1] 广播适配
        
        # 约束流动强度
        flow = torch.clamp(self.flow_strength, 0.01, 10.0)

        # 多尺度对数衰减（无exp/log冗余计算，数值稳定）
        # 公式：log_decay_scale = -|i-j| / τ
        log_decay_multi = -dist.unsqueeze(0).unsqueeze(0).unsqueeze(0) / (tau + 1e-6)
        
        # 多尺度融合：LogSumExp 实现平滑加权融合
        # 公式：log_decay = LSE(-dist/τ_s, -dist/τ_m, -dist/τ_l)
        log_decay = torch.logsumexp(log_decay_multi, dim=2)

        # ====================== 步骤5：注意力得分融合与数值约束 ======================
        # 最终注意力得分 = 基础注意力 + 生理衰减掩码 → 全局强度缩放
        attn = (attn + log_decay) * flow
        # 数值截断防止softmax溢出
        attn = torch.clamp(attn, -20, 20)

        # ====================== 步骤6：双约束掩码（因果+长距离截断） ======================
        # 1. 因果掩码：下三角掩码，保证t时刻只能访问1~t时刻信息
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        # 2. 长距离掩码：超过3倍平均τ的依赖直接屏蔽，符合生理先验
        long_mask = (dist.unsqueeze(0).unsqueeze(0) > 3 * tau.mean(dim=2))
        
        # 总掩码：因果约束 + 长距离截断
        total_mask = causal_mask.unsqueeze(0).unsqueeze(0) | long_mask

        # ====================== 步骤7：SDPA 加速注意力计算 ======================
        # PyTorch原生缩放点积注意力，支持GPU加速、内存优化
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=total_mask,
            dropout_p=self.dropout if self.training else 0.0
        )

        # ====================== 步骤8：多头拼接与输出投影 ======================
        # 维度变换：[B,n_heads,T,head_dim] → [B,T,out_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, self.out_dim)
        # 输出投影
        out = self.out_proj(out)

        # ====================== 步骤9：GLU 门控融合 ======================
        # 公式：GLU(x) = x1 ⊗ σ(x2)
        out, gate = self.gate(out).chunk(2, dim=-1)
        out = out * torch.sigmoid(gate)

        # ====================== 步骤10：残差连接 + 层归一化 ======================
        # 公式：Output = LayerNorm( GLU_out + Linear(x) )
        out = self.norm(out + self.residual_proj(x))

        return out
    
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time
import numpy as np

class HormoneModule(nn.Module):
    r"""
    激素动态调控模块 (Hormone Regulation Module)
    基于生物启发的时序门控机制，用于动态特征加权与信息融合
    支持多门控扩展、可训练基线、温度系数稳定、GPU高效、可视化置信区间

    数学建模总览：
    1. 时序特征编码：GRU 提取序列隐藏状态
    2. 门控生成：全连接网络生成原始门控值
    3. 温度归一化：带温度系数的 Softmax 保证门控稀疏且可微分
    4. 动态融合：门控加权外部特征实现自适应信息聚合

   
    1. 支持多层GRU，增强长序列建模能力
    2. 动态特征融合内置接口
    3. 95%置信区间可视化
    4. 修复温度系数更新bug
    5. 更鲁棒的设备/维度对齐
    6. 支持外部特征加权融合
    """
    def __init__(
        self, 
        input_dim: int,                 # 输入特征维度
        hidden_dim: int,                # GRU 隐藏层维度
        output_dim: int,                # 输出门控维度
        n_gates: int = 2,               # 激素门控数量
        n_gru_layers: int = 1,          # GRU 层数，多层可提升长序列建模能力
        visualize: bool = False,        # 是否启用门控可视化
        save_path: str = "./hormone_figs"  # 可视化图片保存路径
    ):
        """
        初始化激素动态调控模块
        
        Args:
            input_dim: 输入时间序列特征维度
            hidden_dim: GRU 隐藏状态维度
            output_dim: 输出门控的特征维度
            n_gates: 生成的自适应门控个数
            n_gru_layers: 堆叠的 GRU 层数
            visualize: 推理阶段是否绘制门控变化曲线
            save_path: 可视化结果保存目录
        """
        super().__init__()

        # 维度与结构超参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_gates = n_gates
        self.n_gru_layers = n_gru_layers
        
        # 可视化配置
        self.visualize = visualize
        self.save_path = save_path

        # 自动创建可视化保存目录（仅开启可视化时生效）
        if visualize and not os.path.exists(save_path):
            os.makedirs(save_path)

        # ===================== 核心模块1：多层GRU时序编码器 =====================
        # 数学公式：h_t = GRU(x_t, h_{t-1})
        # 输入：[batch, seq_len, input_dim]
        # 输出：序列隐藏态 h_seq: [B, T, H]，末态 h_last: [layers, B, H]
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=0.1 if n_gru_layers > 1 else 0.0  # 仅多层GRU使用dropout防过拟合
        )
        
        # 可学习的初始隐藏状态，作为激素系统的"基线状态"
        # 形状：[num_layers, 1, hidden_dim]
        self.h0 = nn.Parameter(torch.randn(n_gru_layers, 1, hidden_dim))

        # ===================== 核心模块2：门控生成网络 =====================
        # 输入：GRU 隐藏状态 H
        # 输出：未归一化的原始门控值 (n_gates × output_dim)
        self.controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_gates * output_dim)
        )

        # 可学习温度系数：控制 Softmax 稀疏性，τ ∈ [0.1, 10]
        self.temp = nn.Parameter(torch.ones(1) * 1.0)
        
        # 正则化层
        self.dropout = nn.Dropout(0.1)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        初始化 GRU 隐藏状态，自动对齐批次与设备
        
        数学操作：h0 → 扩展为 [layers, batch_size, hidden_dim]
        
        Args:
            batch_size: 批次大小
            device: 运行设备 (CPU/GPU)
        
        Returns:
            初始化后的隐藏状态
        """
        return self.h0.repeat(1, batch_size, 1).to(device)

    def forward(
        self, 
        x: torch.Tensor, 
        prev_state: torch.Tensor = None, 
        external_features: list = None
    ) -> tuple:
        """
        前向传播：时序编码 → 门控生成 → 归一化 → 特征融合
        
        Args:
            x: 输入时间序列，形状 [batch_size, seq_len, input_dim]
            prev_state: 上一时刻 GRU 隐藏状态，用于增量推理
            external_features: 待融合的外部特征列表，长度 = n_gates
        
        Returns:
            gate_list: 归一化后的门控列表，每个元素 [B, T, output_dim]
            h_last: GRU 最后时刻隐藏状态
            fused_feature: 门控加权融合后的特征（若输入 external_features）
        
        数学公式：
            1. 隐藏状态计算：
               H = GRU(X, h_prev) ∈ R^{B×T×H}
            
            2. 原始门控生成：
               G_raw = Linear(H) ∈ R^{B×T×G×D}  
               (G=门控数，D=输出维度)
            
            3. 带温度系数的归一化：
               G = Softmax(G_raw / τ)，在门控维度上归一化
            
            4. 动态特征融合（可选）：
               F_fused = Σ (g_i · F_i)，i=1~G
        """
        # 获取批次、序列长度与运行设备
        B, T, _ = x.shape
        device = x.device

        # 初始化 GRU 隐藏状态
        if prev_state is None:
            prev_state = self.init_hidden(B, device)

        # ===================== 1. GRU 时序特征编码 =====================
        h_seq, h_last = self.gru(x, prev_state)  # h_seq: [B, T, H]

        # ===================== 2. 生成多通道原始门控 =====================
        # 映射 → 重塑为 [B, T, n_gates, output_dim]
        gate_raw = self.controller(h_seq).view(B, T, self.n_gates, self.output_dim)
        
        # ===================== 3. 温度约束 + Softmax 门控归一化 =====================
        # 限制温度系数范围，防止梯度消失/爆炸
        temp_clamped = torch.clamp(self.temp, 0.1, 10.0)
        # 在 dim=2（门控维度）做归一化，保证所有门控权重和为1
        gates = torch.softmax(gate_raw / temp_clamped, dim=2)

        # 将多门控拆分为独立列表，方便外部调用
        gate_list = [gates[:, :, i, :] for i in range(self.n_gates)]

        # 训练阶段启用 Dropout 正则化
        if self.training:
            gate_list = [self.dropout(g) for g in gate_list]

        # 推理阶段可视化门控动态变化曲线
        if self.visualize and not self.training:
            self._visualize(gate_list)

        # ===================== 4. 门控加权动态特征融合 =====================
        fused_feature = None
        if external_features is not None:
            # 校验：外部特征数量必须等于门控数量
            assert len(external_features) == self.n_gates, \
                f"外部特征数量 {len(external_features)} 必须等于门控数 {self.n_gates}"
            # 加权求和：自适应融合多源特征
            fused_feature = sum(gate * feat for gate, feat in zip(gate_list, external_features))

        # 返回：门控列表、GRU末态、融合特征
        return (*gate_list, h_last, fused_feature)

    def _visualize(self, gate_list: list) -> None:
        """
        增强版可视化：绘制门控激活值均值 + 95% 置信区间
        
        统计公式：
        均值：μ = mean_{B,D}(g_t)
        标准误：SE = std_{B,D}(g_t) / √B
        95% 置信区间：[μ - 1.96×SE, μ + 1.96×SE]
        
        Args:
            gate_list: 门控列表，每个门控 [B, T, D]
        """
        with torch.no_grad():
            plt.figure(figsize=(10, 5))
            
            for idx, gate in enumerate(gate_list):
                # 在 批次维度(0) 和 特征维度(-1) 上做统计平均
                g_mean = gate.mean(dim=(0, -1)).cpu().numpy()  # [T]
                g_std  = gate.std(dim=(0, -1)).cpu().numpy()   # [T]
                
                # 计算 95% 置信区间
                sample_size = gate.size(0)
                ci = 1.96 * g_std / np.sqrt(sample_size)  # 置信区间半径
                
                # 绘制均值曲线与填充区间
                plt.plot(g_mean, label=f"Hormone Gate {idx+1}", linewidth=2.5)
                plt.fill_between(range(len(g_mean)), g_mean-ci, g_mean+ci, alpha=0.25)

            # 图表样式
            plt.title("Hormone Gate Dynamics (Mean ± 95% CI)", fontsize=14)
            plt.xlabel("Time Step", fontsize=12)
            plt.ylabel("Gate Activation Value", fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(alpha=0.3, linestyle='--')
            plt.tight_layout()

            # 带时间戳保存，避免覆盖
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_path, f"hormone_gate_{timestamp}.png")
            plt.savefig(save_path, dpi=250, bbox_inches="tight")
            plt.close()
            print(f"✅ 门控可视化已保存：{save_path}")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================
# 1. 随机深度（DropPath）：训练时随机丢弃残差分支，防止过拟合
# 作用：增强模型泛化能力，常用于 Transformer / 深度网络残差结构
# ==============================================
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        # drop_prob：随机丢弃概率，0 表示不丢弃
        self.drop_prob = drop_prob

    def forward(self, x):
        # 推理阶段 / 丢弃概率为 0 → 直接返回原特征
        if self.drop_prob == 0. or not self.training:
            return x
        
        # 保留特征的概率
        keep_prob = 1 - self.drop_prob
        
        # 生成与输入同设备、同数据类型的随机掩码
        # shape 设计：只在 batch 维度随机，序列/通道维度保持一致
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        
        # 生成 [0,1) 均匀分布随机数 + keep_prob
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        
        # 向下取整 → 得到 0/1 二值掩码
        random_tensor.floor_()
        
        # 缩放保证训练/推理时特征期望一致 + 应用掩码
        return x.div(keep_prob) * random_tensor

# ==============================================
# 2. RoPE + 相对位置偏置：旋转位置编码 + 可学习相对偏置
# 优点：超长序列稳定性更强，同时保留绝对位置信息
# 适用：长时序信号（金融、心电、传感器、语音）
# ==============================================
class RoPEWithRelativeBias(nn.Module):
    """RoPE + 相对位置偏置，超长序列更稳定"""
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.dim = dim  # 特征维度
        # 可学习相对位置偏置：[1, max_len, dim]
        self.rel_bias = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

    def forward(self, x):
        # 输入维度：[B, T, D] → 批次、序列长度、特征维度
        B, T, D = x.shape
        
        # 生成位置序列：[T, 1]
        position = torch.arange(T, device=x.device).unsqueeze(1)
        
        # 位置编码分母项：标准 Transformer 正弦余弦位置编码公式
        div_term = torch.exp(torch.arange(0, D, 2, device=x.device) * (-math.log(10000.0) / D))
        
        # 计算 sin / cos 位置编码 → 扩展到全维度
        sin = torch.sin(position * div_term).repeat(1, 2)[:, :D]
        cos = torch.cos(position * div_term).repeat(1, 2)[:, :D]
        
        # ==================== RoPE 核心旋转公式 ====================
        # 对特征偶数/奇数维度分别旋转，保留相对位置信息
        x_rot = x * cos + torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape(x.shape) * sin
        
        # 旋转编码 + 可学习相对位置偏置
        return x_rot + self.rel_bias[:, :T, :]

# ==============================================
# 3. 残差膨胀卷积上采样：适合高频时序信号（金融/生理）
# 结构：上采样 + 深度可分离膨胀卷积 + 残差连接
# 优点：大感受野、轻量化、保留高频细节
# ==============================================
class ResidualDilatedConvUp(nn.Module):
    """膨胀卷积 + 大核 + 残差上采样，适合金融/生理高频信号"""
    def __init__(self, dim, scale, kernel_size=5, dilation=2):
        super().__init__()
        self.scale = scale  # 上采样倍率
        # ==================== 深度可分离卷积 ====================
        # DW 卷积：分组数=通道数，逐通道卷积，轻量化
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, dilation=dilation,
                                 padding=kernel_size//2 * dilation, groups=dim)
        # PW 卷积：1x1 卷积，融合通道信息
        self.pw_conv = nn.Conv1d(dim, dim, kernel_size=1)
        
        self.norm = nn.LayerNorm(dim)  # 层归一化
        self.act = nn.GELU()            # 激活函数

    def forward(self, x):
        # 输入：[B, T, D]
        residual = x  # 残差分支
        
        # 转换维度适配 Conv1d：[B, D, T]
        x = x.transpose(1, 2)
        
        # 线性插值上采样
        x = F.interpolate(x, scale_factor=self.scale, mode='linear', align_corners=False)
        
        # 深度可分离卷积
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        
        # 恢复维度：[B, T, D]
        x = x.transpose(1, 2)
        
        # 归一化 + 激活
        x = self.act(self.norm(x))
        
        # 残差连接
        return x + residual

# ==============================================
# 4. 尺度融合 Transformer：跨尺度特征交互
# 设计思路：把“尺度”当作序列长度，用 Transformer 做跨尺度注意力
# ==============================================
class ScaleFusionTransformer(nn.Module):
    """用微型 Transformer 做尺度间融合，增强交互"""
    def __init__(self, dim, n_heads=2, layers=1):
        super().__init__()
        self.proj = nn.Linear(dim, dim)  # 输入投影（可选）
        # 单层 Transformer 编码器：轻量跨尺度交互
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim*2,
                                       batch_first=True, activation='gelu'),
            num_layers=layers
        )

    def forward(self, scale_feats):
        # 输入：[B, S, T, D] → 批次、尺度数、序列长度、特征维度
        B, S, T, D = scale_feats.shape
        
        # 维度变换：把尺度维度当作序列
        # [B, T, S, D] → 合并批次与时间：[B*T, S, D]
        feats = scale_feats.permute(0, 2, 1, 3).reshape(B*T, S, D)
        
        # 跨尺度自注意力融合
        fused = self.transformer(feats)
        
        # 恢复原始维度：[B, S, T, D]
        return fused.reshape(B, T, S, D).permute(0, 2, 1, 3)

# ==============================================
# 5. 最终多尺度时序编码器（主模型）
# 核心流程：多尺度编码 → 上采样对齐 → 跨尺度融合 → 残差输出
# 适用场景：高频时序预测/异常检测/分类（金融、心电、传感器）
# ==============================================
class FinalMultiScaleTemporalEncoder(nn.Module):
    def __init__(self, input_dim, n_heads=4, num_layers=2, dropout=0.1,
                 scales=[1,2,4], max_seq_len=1024, drop_path=0.1,
                 share_encoder=True):
        super().__init__()
        self.scales = scales          # 多尺度列表，如 [1,2,4]
        self.n_scales = len(scales)   # 尺度数量
        self.share_encoder = share_encoder  # 是否共享 Transformer 编码器

        # ==================== 位置编码 ====================
        # RoPE + 相对位置偏置
        self.rope_rel = RoPEWithRelativeBias(input_dim, max_seq_len)
        # 可学习尺度嵌入：区分不同尺度特征
        self.scale_emb = nn.Parameter(torch.randn(len(scales), input_dim) * 0.02)

        # ==================== 多尺度 Transformer 编码器 ====================
        if share_encoder:
            # 共享权重编码器：节省参数量
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads,
                                           dim_feedforward=input_dim*4,
                                           dropout=dropout, batch_first=True, activation='gelu'),
                num_layers=num_layers
            )
        else:
            # 独立编码器：每个尺度单独建模，精度更高
            self.encoders = nn.ModuleList([
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads,
                                               dim_feedforward=input_dim*4,
                                               dropout=dropout, batch_first=True, activation='gelu'),
                    num_layers=num_layers
                ) for _ in scales
            ])

        # ==================== 多尺度上采样 ====================
        # 膨胀卷积残差上采样：将不同尺度特征对齐到原始长度
        self.upsamplers = nn.ModuleList([
            ResidualDilatedConvUp(input_dim, s, kernel_size=5, dilation=2) if s>1 else nn.Identity()
            for s in scales
        ])

        # ==================== 跨尺度融合模块 ====================
        self.scale_fusion = ScaleFusionTransformer(input_dim, n_heads=2)

        # ==================== 正则化与残差 ====================
        self.drop_path = DropPath(drop_path)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # 输入：[B, T, D] → 批次、时序长度、特征维度
        B, T, D = x.shape
        outs = []

        # ==================== 第一步：多尺度特征编码 ====================
        for i, s in enumerate(self.scales):
            # 下采样：尺度 s=2 → 每隔 1 个点取 1 个；s=4 → 每隔 3 个点取 1 个
            xs = x[:, ::s, :] if s > 1 else x
            
            # 位置编码：RoPE + 相对偏置
            xs = self.rope_rel(xs)
            
            # 尺度嵌入：给当前尺度添加专属特征标识
            xs = xs + self.scale_emb[i:i+1, :]

            # Transformer 编码
            if self.share_encoder:
                feat = self.encoder(xs)
            else:
                feat = self.encoders[i](xs)

            # 上采样：将短序列恢复为原始长度 T
            if s > 1:
                feat = self.upsamplers[i](feat)
            outs.append(feat)

        # 堆叠多尺度特征：[B, S, T, D]
        outs = torch.stack(outs, dim=1)

        # ==================== 第二步：跨尺度 Transformer 融合 ====================
        outs = self.scale_fusion(outs)

        # ==================== 第三步：特征融合 ====================
        # 1. 所有尺度直接求和融合
        fused = outs.sum(dim=1)
        # 2. 残差增强：额外叠加尺度 1（原始尺度）强化高频细节
        fused = fused + outs[:, 0]

        # ==================== 第四步：最终残差与归一化 ====================
        out = x + self.drop_path(fused)    # 残差连接 + 随机深度
        out = self.norm(out)               # 层归一化

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

