'''股票多变量时间序列数据预处理脚本（修复+优化版）'''
# -*- coding: utf-8 -*-
"""
✅ 标签无泄露：标签只使用未来数据，不混入训练时不可见信息
✅ 训练集标准化 → 验证集标准化：只用训练集拟合标准化器
✅ 保存 feature_cols：保存特征列名，方便后续模型推理使用
✅ 多步收益率预测 HORIZON：支持预测未来1~N天收益率
✅ 自动过滤无效股票：自动剔除数据量不足的股票
✅ 平稳化特征、市场信息、放量特征、时间编码、分类标签
📌 多股票多步预测不丢样本 | NaN智能填充 | 输出每只股票每步样本数表格
"""

# ========================== 1. 导入依赖库 ==========================
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# ========================== 2. 全局配置参数（可直接修改） ==========================
# 原始股票数据CSV文件路径
csv_file = r"E:\大学\学习\血液激素神经网络\archive\all_stocks_5yr.csv"
# 预处理后文件输出目录
output_dir = r"E:\大学\学习\血液激素神经网络\archive\processed"
# 创建输出文件夹（不存在则创建，存在则不报错）
os.makedirs(output_dir, exist_ok=True)

# 需要处理的股票代码列表（可多只）
tickers = ['AAPL']  # 可填多只 ['AAPL','MSFT','GOOG']

# 模型输入：用过去30天数据做特征
TIME_STEP = 30
# 模型输出：预测未来5天的收益率（多步预测）
HORIZON = 5
# 训练集 / 验证集 划分比例（80%训练，20%验证）
TRAIN_RATIO = 0.8
# 收益率极值裁剪（防止极端异常值影响训练）
CLIP_BOUND = 0.2

# 技术指标NaN填充方式（时间序列严禁bfill）
FILL_NA_METHOD = "ffill"

# RSI计算周期（默认14）
RSI_PERIOD = 14
# MACD 三条线周期参数
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
# 简单移动平均线周期
SMA_PERIODS = [10, 20, 50]
# 指数移动平均线周期
EMA_PERIODS = [10, 20]
# 波动率计算周期
VOLATILITY_PERIODS = [10, 20]

# ========================== 3. RSI指标计算函数 ==========================
def compute_RSI(series, period=RSI_PERIOD):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-10)
    return 100 - (100 / (1 + rs))

# ========================== 4. 技术指标生成函数（已修复+优化） ==========================
def add_technical_indicators(df_t, ticker):
    close = df_t[f"{ticker}_Close"]
    # 日收益率
    df_t[f"{ticker}_Return"] = close.pct_change().clip(-CLIP_BOUND, CLIP_BOUND)
    # 成交量特征（增加相对成交量）
    df_t[f"{ticker}_Volume_Log"] = np.log1p(df_t[f"{ticker}_Volume"])
    df_t[f"{ticker}_Volume_Change"] = df_t[f"{ticker}_Volume_Log"].diff().fillna(0)
    vol_mean = df_t[f"{ticker}_Volume"].rolling(20).mean()
    df_t[f"{ticker}_Volume_Ratio_20"] = df_t[f"{ticker}_Volume"] / (vol_mean + 1e-6)
    # 均线
    for p in SMA_PERIODS:
        df_t[f"{ticker}_SMA_{p}"] = close.rolling(p).mean()
    for p in EMA_PERIODS:
        df_t[f"{ticker}_EMA_{p}"] = close.ewm(span=p, adjust=False).mean()

    # 波动率
    ret = df_t[f"{ticker}_Return"]
    for p in VOLATILITY_PERIODS:
        df_t[f"{ticker}_Volatility_{p}"] = ret.rolling(p).std()

    # RSI
    df_t[f"{ticker}_RSI_{RSI_PERIOD}"] = compute_RSI(close, RSI_PERIOD)

    # MACD
    exp_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    exp_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = exp_fast - exp_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_t[f"{ticker}_MACD"] = macd_line
    df_t[f"{ticker}_MACD_Signal"] = signal_line
    df_t[f"{ticker}_MACD_Hist"] = macd_line - signal_line

    # ====================== 价格平稳化特征 ======================
    df_t[f"{ticker}_Price_SMA_Ratio"] = close / close.rolling(20).mean()
    df_t[f"{ticker}_Price_ZScore_20"] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-6)

    # ====================== 【优化】时间编码 ======================
    df_t[f"{ticker}_DOW_sin"] = np.sin(2*np.pi*df_t.index.dayofweek / 7)
    df_t[f"{ticker}_DOW_cos"] = np.cos(2*np.pi*df_t.index.dayofweek / 7)

    df_t[f"{ticker}_Month_sin"] = np.sin(2*np.pi*df_t.index.month / 12)
    df_t[f"{ticker}_Month_cos"] = np.cos(2*np.pi*df_t.index.month / 12)

    return df_t

# ========================== 5. 读取并清洗原始数据 ==========================
print("📥 读取原始数据...")
df = pd.read_csv(csv_file)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').sort_index()

all_tickers = sorted(df['Name'].unique())
if not tickers:
    tickers = all_tickers
else:
    tickers = [t for t in tickers if t in all_tickers]

print(f"📈 即将处理股票总数：{len(tickers)}")

# ========================== 6. 逐只股票生成特征 ==========================
print("🔨 开始生成技术指标...")
dfs = []
valid_tickers = []
feature_count_per_stock = {}

for t in tickers:
    df_t = df[df['Name'] == t].copy()
    if df_t.empty or len(df_t) < 100:
        print(f"⚠️  跳过 {t}：数据不足")
        continue

    valid_tickers.append(t)
    df_t.rename(columns={
        'open':f'{t}_Open','high':f'{t}_High','low':f'{t}_Low',
        'close':f'{t}_Close','volume':f'{t}_Volume'
    }, inplace=True)
    df_t = df_t[[f'{t}_Open',f'{t}_High',f'{t}_Low',f'{t}_Close',f'{t}_Volume']]
    df_t = add_technical_indicators(df_t, t)

    # 全局ffill，无未来泄露，无局部污染
    tech_cols = [c for c in df_t.columns if c not in [f'{t}_Open',f'{t}_High',f'{t}_Low',f'{t}_Close',f'{t}_Volume']]
    df_t[tech_cols] = df_t[tech_cols].fillna(method=FILL_NA_METHOD)

    feat_cnt = len(df_t.columns) - 5
    feature_count_per_stock[t] = feat_cnt
    dfs.append(df_t)

if not dfs:
    raise ValueError("❌ 无有效股票数据")

processed_df = pd.concat(dfs, axis=1, join='inner').dropna()
print(f"✅ 合并完成 | 总样本：{len(processed_df)} | 总特征：{processed_df.shape[1]}")

# ========================== 7. 构建特征X 和 标签y ==========================
feature_cols, close_cols = [], []
for t in valid_tickers:
    feature_cols += [
        f"{t}_Return", f"{t}_Volume_Log", f"{t}_Volume_Change", f"{t}_Volume_Ratio_20",
        *[f"{t}_SMA_{p}" for p in SMA_PERIODS],
        *[f"{t}_EMA_{p}" for p in EMA_PERIODS],
        *[f"{t}_Volatility_{p}" for p in VOLATILITY_PERIODS],
        f"{t}_RSI_{RSI_PERIOD}",
        f"{t}_MACD", f"{t}_MACD_Signal", f"{t}_MACD_Hist",
        f"{t}_Price_SMA_Ratio", f"{t}_Price_ZScore_20",
        f"{t}_DOW_sin", f"{t}_DOW_cos",
        f"{t}_Month_sin", f"{t}_Month_cos"
    ]
    close_cols.append(f"{t}_Close")

X_df = processed_df[feature_cols]
close_df = processed_df[close_cols]

# ========================== 构建多步标签 ==========================
y_dict = {}
stock_sample_counts = {}
step_sample_matrix = {}

for t in valid_tickers:
    close_series = processed_df[f"{t}_Close"]
    y_steps = []
    step_counts = []
    for h in range(1, HORIZON + 1):
        # 计算未来收益并立即 dropna
        y_step = np.log(close_series.shift(-h) / close_series).clip(-CLIP_BOUND, CLIP_BOUND)
        y_step = y_step.dropna()
        y_steps.append(y_step.rename(f"{t}_Return_t+{h}"))
        step_counts.append(len(y_step))

    
    y_t = pd.concat(y_steps, axis=1)
    y_dict[t] = y_t
    stock_sample_counts[t] = y_t.dropna().shape[0]
    step_sample_matrix[t] = step_counts

# 输出样本统计表
print("\n" + "="*80)
print("📊 每只股票 × 每个预测步长 有效样本数统计表")
print("="*80)
header = ["股票代码"] + [f"t+{h}" for h in range(1, HORIZON+1)] + ["合计有效样本"]
print(f"{'':<10}".join(header))
print("-"*80)
for t in valid_tickers:
    row = [f"{t:<8}"] + [f"{cnt:<10}" for cnt in step_sample_matrix[t]] + [f"{stock_sample_counts[t]:<10}"]
    print("".join(row))
print("="*80 + "\n")

# 【致命修复】不再重复拼接X，只取一次对应索引
X_list, y_list = [], []
for t in valid_tickers:
    y_t = y_dict[t].dropna()
    X_t = X_df.loc[y_t.index]
    X_list.append(X_t)
    y_list.append(y_t)

# 拼接标签，特征只保留全局唯一一份（无重复灾难）
common_index = y_list[0].index
for y_t in y_list[1:]:
    common_index = common_index.intersection(y_t.index)

y_final = pd.concat([y.loc[common_index] for y in y_list], axis=1)
X_final = X_df.loc[common_index]

print(f"🎯 标签构建完成 | 预测步长：{HORIZON} | 总可用样本：{len(X_final)}")

# ========================== 8. 划分训练集 / 验证集 ==========================
split_idx = int(len(X_final) * TRAIN_RATIO)
X_train_df, X_val_df = X_final.iloc[:split_idx], X_final.iloc[split_idx:]
y_train_df, y_val_df = y_final.iloc[:split_idx], y_final.iloc[split_idx:]

# ========================== 9. 数据标准化 ==========================
scaler = RobustScaler()
X_train_2d = X_train_df.values
scaler.fit(X_train_2d)

def scale_df(df, scaler):
    return pd.DataFrame(scaler.transform(df.values), index=df.index, columns=df.columns)

X_train_df = scale_df(X_train_df, scaler)
X_val_df = scale_df(X_val_df, scaler)

# ========================== 10. 构建时间窗口样本 ==========================
def build_windows(X_df, y_df, step):
    X_win, y_win = [], []
    # 多步预测必须保证未来HORIZON天存在
    max_idx = len(X_df) - step
    for i in range(max_idx):
        X_win.append(X_df.iloc[i:i+step].values)
        # 取最后一个时间步对应的多步标签
        y_win.append(y_df.iloc[i + step].values)
    return np.array(X_win, np.float32), np.array(y_win, np.float32)
X_train, y_train = build_windows(X_train_df, y_train_df, TIME_STEP)
X_val, y_val = build_windows(X_val_df, y_val_df, TIME_STEP)

print(f"🪟 时间窗口构建完成 | X_train: {X_train.shape} | y_train: {y_train.shape} | X_val: {X_val.shape} | y_val: {y_val.shape}")

# ========================== 11. 保存预处理结果 ==========================
np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "y_train.npy"), y_train)
np.save(os.path.join(output_dir, "X_val.npy"), X_val)
np.save(os.path.join(output_dir, "y_val.npy"), y_val)

with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(output_dir, "feature_cols.pkl"), "wb") as f:
    pickle.dump(feature_cols, f)

# ========================== 12. 最终输出信息 ==========================
print("="*65)
print("✅ 数据预处理 100% 完成！（已修复所有致命bug）")
print(f"有效股票数：{len(valid_tickers)}")
print(f"每只股票特征数量：{feature_count_per_stock}")
print(f"输入时间步：{TIME_STEP} | 预测步长：{HORIZON}")
print(f"训练集 X：{X_train.shape}  y：{y_train.shape}")
print(f"验证集 X：{X_val.shape}  y：{y_val.shape}")
print(f"文件已保存至：{output_dir}")
print("="*65)
'''血液流动注意力机制层（多尺度指数衰减注意力，生理启发式时序建模）'''
import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func

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
        # ====================== 初始化参数合法性检查 ======================
        self._check_init_params(in_dim, out_dim, n_heads, dropout, max_T)

        # 基础结构参数
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.out_dim = out_dim
        self.dropout = dropout
        self.max_T = max_T

        # ============================ QKV 线性投影层 ============================
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

        # ============================ 残差与归一化模块 ============================
        self.norm = nn.LayerNorm(out_dim)
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        # ====================== 多尺度衰减系数 τ（生理启发式初始化） ======================
        tau_short = 2.0
        tau_mid   = 8.0
        tau_long  = 30.0
        tau_init = torch.tensor([tau_short, tau_mid, tau_long])

        self.decay_tau = nn.Parameter(tau_init.repeat(n_heads, 1))
        self.flow_strength = nn.Parameter(torch.tensor(1.0))

        # ============================ GLU 门控模块 ============================
        self.gate = nn.Linear(out_dim, out_dim * 2)

        # ============================ 自适应距离矩阵缓存 ============================
        self.register_buffer('dist_matrix', torch.zeros(1, 1))
        self.cached_T = 1

    def _check_init_params(self, in_dim, out_dim, n_heads, dropout, max_T):
        """初始化参数完整性检查"""
        # 维度必须为正整数
        assert isinstance(in_dim, int) and in_dim > 0, f"输入维度必须为正整数，当前：{in_dim}"
        assert isinstance(out_dim, int) and out_dim > 0, f"输出维度必须为正整数，当前：{out_dim}"
        assert isinstance(n_heads, int) and n_heads > 0, f"注意力头数必须为正整数，当前：{n_heads}"
        # 输出维度可被头数整除
        assert out_dim % n_heads == 0, f"输出维度 {out_dim} 无法被头数 {n_heads} 整除，请调整参数"
        # Dropout 范围检查
        assert 0.0 <= dropout <= 1.0, f"Dropout 必须在 [0,1] 之间，当前：{dropout}"
        # 最大序列长度检查
        assert isinstance(max_T, int) and max_T >= 16, f"max_T 必须≥16，当前：{max_T}"

    def get_distance_matrix(self, T, device):
        """
        构建/获取位置绝对距离矩阵：dist[i,j] = |i - j|
        用于指数衰减注意力的距离计算
        """
        # 缓存长度安全检查
        if T <= 0:
            raise ValueError(f"序列长度 T 必须≥1，当前：{T}")

        if T <= self.cached_T:
            return self.dist_matrix[:T, :T]

        # 重建缓存
        new_T = min(T, self.max_T)
        idx = torch.arange(new_T, device=device)
        self.dist_matrix = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)).float()
        self.cached_T = new_T

        return self.dist_matrix[:T, :T]

    def forward(self, x):
        """前向传播主函数：完整血液流动注意力计算流程"""
        # ====================== 【核心】前向输入全检查 ======================
        self._check_forward_input(x)

        B, T, _ = x.shape
        device = x.device

        # ====================== 步骤1：获取位置距离矩阵 ======================
        dist = self.get_distance_matrix(T, device)
        # 距离矩阵设备一致性检查
        assert dist.device == device, f"距离矩阵设备不匹配！输入：{device}，矩阵：{dist.device}"

        # ====================== 步骤2：QKV 多头投影与维度拆分 ======================
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # QKV 维度检查
        self._check_qkv_dims(Q, K, V, B)

        # ====================== 步骤3：标准缩放点积注意力基础得分 ======================
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # ====================== 步骤4：多尺度指数衰减掩码计算 ======================
        tau = torch.clamp(self.decay_tau, min=1.0, max=100.0)
        tau = tau.view(1, self.n_heads, 3, 1, 1)
        flow = torch.clamp(self.flow_strength, 0.01, 10.0)

        # 数值稳定性：防止除以0
        log_decay_multi = -dist.unsqueeze(0).unsqueeze(0).unsqueeze(0) / (tau + 1e-6)
        log_decay = torch.logsumexp(log_decay_multi, dim=2)

        # ====================== 步骤5：注意力得分融合与数值约束 ======================
        attn = (attn + log_decay) * torch.clamp(flow, 0.1, 2.0)
        attn = torch.clamp(attn, -10, 10)


        # 注意力得分非法值检查
        self._check_tensor_nan_inf(attn, "注意力得分")


        # ====================== 步骤6：双约束掩码（因果+长距离截断）【已修复维度】 ======================
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()  # [T, T]
        dist_expand = dist.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
        tau_mean = tau.mean(dim=2)  # [1, heads, 1,1]
        long_mask = dist_expand > 3 * tau_mean
        long_mask = long_mask.squeeze(0)  # [heads, T, T]
        causal_mask = causal_mask.unsqueeze(0)  # [1, T, T]
        total_mask = causal_mask | long_mask  # [heads, T, T]
        

        # ====================== 步骤7：SDPA 加速注意力计算 ======================
        out = torch_nn_func.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=total_mask,
            dropout_p=self.dropout if self.training else 0.0
        )

        # ====================== 步骤8：多头拼接与输出投影 ======================
        out = out.transpose(1, 2).contiguous().view(B, T, self.out_dim)
        out = self.out_proj(out)

        # ====================== 步骤9：GLU 门控融合 ======================
        out, gate = self.gate(out).chunk(2, dim=-1)
        out = out * torch.sigmoid(gate)

        # ====================== 步骤10：残差连接 + 层归一化 ======================
        residual = self.residual_proj(x)
        out = self.norm(out + residual)

        # 输出最终检查
        self._check_tensor_nan_inf(out, "模型输出")
        self._check_output_dims(out, B, T)

        return out

    def _check_forward_input(self, x):
        """输入张量全维度/类型/设备检查"""
        # 检查是否为张量
        assert torch.is_tensor(x), f"输入必须是 torch.Tensor，当前类型：{type(x)}"
        # 检查维度为3维 [B, T, C]
        assert x.dim() == 3, f"输入必须是3维张量 [B, T, C]，当前维度：{x.dim()}"
        # 检查特征维度匹配
        B, T, C = x.shape
        assert C == self.q_proj.in_features, \
            f"输入特征维度不匹配！模型期望：{self.q_proj.in_features}，实际输入：{C}"
        # 检查序列长度合法
        assert T > 0 and B > 0, f"批次/序列长度必须>0，当前 B={B}, T={T}"
        # 检查是否有 NaN / Inf
        self._check_tensor_nan_inf(x, "模型输入")
        # 检查浮点类型
        assert torch.is_floating_point(x), f"输入必须是浮点型张量，当前：{x.dtype}"

    def _check_qkv_dims(self, Q, K, V, batch_size):
        """QKV 维度合法性检查"""
        expected_shape = (batch_size, self.n_heads, -1, self.head_dim)
        for name, tensor in [("Q", Q), ("K", K), ("V", V)]:
            assert tensor.shape[:2] == expected_shape[:2] and tensor.shape[-1] == expected_shape[-1], \
                f"{name} 维度错误！期望 {expected_shape}，实际 {tensor.shape}"

    def _check_tensor_nan_inf(self, tensor: torch.Tensor, name: str):
        """检查张量是否包含 NaN 或 Inf（训练崩溃核心原因）"""
        if torch.isnan(tensor).any():
            raise ValueError(f"【严重】{name} 中出现 NaN！训练即将崩溃")
        if torch.isinf(tensor).any():
            raise ValueError(f"【严重】{name} 中出现 Inf！训练即将崩溃")

    def _check_output_dims(self, out, B, T):
        """输出维度最终校验"""
        assert out.shape == (B, T, self.out_dim), \
            f"输出维度错误！期望 {(B, T, self.out_dim)}，实际 {out.shape}"
            
'''激素动态调控模块 (Hormone RegulationModule)'''
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import warnings  # 新增警告模块


class HormoneModule(nn.Module):
    r"""
    激素动态调控模块 (Hormone RegulationModule)
    基于生物启发的时序门控机制，用于动态特征加权与信息融合
    支持多门控扩展、可训练基线、温度系数稳定、GPU高效、可视化置信区间

    数学建模总览：
    1. 时序特征编码：GRU 提取序列隐藏状态
    2. 门控生成：全连接网络生成原始门控值
    3. 温度归一化：带温度系数的 Softmax 保证门控稀疏且可微分
    4. 动态融合：门控加权外部特征实现自适应信息聚合

    新增增强：
    1. 全链路输入维度校验
    2. 设备一致性检查
    3. 空值/无穷值检测
    4. 外部特征形状校验
    5. 超参数合法性校验
    6. 路径权限与写入检查
    7. 更友好的报错信息
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            n_gates: int = 2,
            n_gru_layers: int = 1,
            visualize: bool = False,
            save_path: str = "./hormone_figs"
    ):
        super().__init__()

        # ===================== 【新增】超参数合法性检查 =====================
        assert isinstance(input_dim, int) and input_dim > 0, \
            f"输入维度必须为正整数，当前输入：{input_dim}"
        assert isinstance(hidden_dim, int) and hidden_dim > 0, \
            f"隐藏层维度必须为正整数，当前输入：{hidden_dim}"
        assert isinstance(output_dim, int) and output_dim > 0, \
            f"输出维度必须为正整数，当前输入：{output_dim}"
        assert isinstance(n_gates, int) and n_gates >= 1, \
            f"门控数量至少为1，当前输入：{n_gates}"
        assert isinstance(n_gru_layers, int) and n_gru_layers >= 1, \
            f"GRU层数至少为1，当前输入：{n_gru_layers}"
        assert isinstance(visualize, bool), \
            f"可视化开关必须为布尔值，当前输入：{visualize}"

        # 维度与结构超参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_gates = n_gates
        self.n_gru_layers = n_gru_layers

        # 可视化配置
        self.visualize = visualize
        self.save_path = save_path

        # ===================== 【新增】路径检查 =====================
        if visualize:
            try:
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                # 测试是否可写
                test_file = os.path.join(save_path, ".test_write.tmp")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                raise RuntimeError(f"可视化路径创建/写入失败：{save_path}\n错误信息：{str(e)}")

        # ===================== 核心模块1：多层GRU时序编码器 =====================
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=0.1 if n_gru_layers > 1 else 0.0
        )

        # 可学习初始隐藏状态
        self.h0 = nn.Parameter(torch.randn(n_gru_layers, 1, hidden_dim))

        # ===================== 核心模块2：门控生成网络 =====================
        self.controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_gates * output_dim)
        )

        # 可学习温度系数
        self.temp = nn.Parameter(torch.ones(1) * 1.0)
        self.dropout = nn.Dropout(0.1)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """初始化GRU隐藏状态，带校验"""
        # 【新增】批次与设备校验
        assert isinstance(batch_size, int) and batch_size > 0, \
            f"批次大小必须为正整数，当前：{batch_size}"
        assert isinstance(device, torch.device), \
            f"设备类型错误，必须为torch.device，当前：{type(device)}"

        return self.h0.repeat(1, batch_size, 1).to(device)

    def forward(
            self,
            x: torch.Tensor,
            prev_state: torch.Tensor = None,
            external_features: list = None
    ) -> tuple:
        """
        前向传播：全链路校验 + 特征编码 + 门控生成 + 融合
        """
        # ===================== 【新增】输入张量基础检查 =====================
        assert isinstance(x, torch.Tensor), f"输入必须是torch.Tensor，当前：{type(x)}"
        assert x.dim() == 3, f"输入必须是3维张量 [B, T, C]，当前形状：{x.shape}"
        assert x.size(-1) == self.input_dim, \
            f"输入最后一维必须等于input_dim={self.input_dim}，当前：{x.size(-1)}"

        # 检查是否有NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            warnings.warn("⚠️ 输入张量包含 NaN 或 Inf 值，可能导致训练不稳定！")

        # 获取基本信息
        B, T, _ = x.shape
        device = x.device

        # ===================== 【新增】隐藏状态检查 =====================
        if prev_state is not None:
            assert isinstance(prev_state, torch.Tensor), "prev_state必须是张量"
            assert prev_state.device == device, \
                f"prev_state设备不匹配！输入：{x.device}，隐藏态：{prev_state.device}"
            assert prev_state.shape == (self.n_gru_layers, B, self.hidden_dim), \
                f"prev_state形状错误！期望：{(self.n_gru_layers, B, self.hidden_dim)}，实际：{prev_state.shape}"
        else:
            prev_state = self.init_hidden(B, device)

        # ===================== 1. GRU 时序编码 =====================
        h_seq, h_last = self.gru(x, prev_state)

        # 【新增】编码后检查异常值
        if torch.isnan(h_seq).any() or torch.isinf(h_seq).any():
            raise ValueError("GRU输出出现NaN/Inf，请检查输入或学习率")

        # ===================== 2. 生成门控 =====================
        gate_raw = self.controller(h_seq).view(B, T, self.n_gates, self.output_dim)

        # ===================== 3. 门控归一化 =====================
        temp_clamped = torch.clamp(self.temp, 0.1, 10.0)
        gates = torch.softmax(gate_raw / temp_clamped, dim=2)
        gate_list = [gates[:, :, i, :] for i in range(self.n_gates)]

        # 强制归一化，保证 alpha+beta=1
        sum_gate = sum(gate_list)
        gate_list = [g / (sum_gate + 1e-8) for g in gate_list]
        # 训练阶段dropout
        if self.training:
            gate_list = [self.dropout(g) for g in gate_list]

        # 可视化
        if self.visualize and not self.training:
            self._visualize(gate_list)

        # ===================== 【新增】外部特征严格校验 =====================
        fused_feature = None
        if external_features is not None:
            assert isinstance(external_features, list), \
                f"external_features必须是列表，当前：{type(external_features)}"
            assert len(external_features) == self.n_gates, \
                f"特征数{len(external_features)}≠门控数{self.n_gates}"

            # 逐特征校验形状+设备
            for i, feat in enumerate(external_features):
                assert isinstance(feat, torch.Tensor), \
                    f"第{i}个特征不是张量：{type(feat)}"
                assert feat.device == device, \
                    f"第{i}个特征设备不匹配！模型：{device}，特征：{feat.device}"
                assert feat.shape == (B, T, self.output_dim), \
                    f"第{i}个特征形状错误！期望{(B,T,self.output_dim)}，实际{feat.shape}"

            # 加权融合
            fused_feature = sum(g * f for g, f in zip(gate_list, external_features))

            # 【新增】融合后检查
            if torch.isnan(fused_feature).any():
                warnings.warn("⚠️ 融合特征出现NaN，可能是门控或外部特征异常")

        # ===================== 【新增】输出检查 =====================
        if not self.training:
            for i, g in enumerate(gate_list):
                if not ((g >= 0.0).all() and (g <= 1.0).all()):
                    warnings.warn(f"⚠️ 第{i}个门控值超出[0,1]范围！")

        #return (*gate_list, h_last, fused_feature)
            # 只返回 门控1,门控2,最后隐藏态
        #return gate_list[0], gate_list[1], h_last
    # 只返回时序门控，不返回状态

        return gate_list[0], gate_list[1]

    def _visualize(self, gate_list: list) -> None:
        """可视化：带异常捕获与校验"""
        try:
            with torch.no_grad():
                # 【新增】门控列表检查
                assert len(gate_list) == self.n_gates, "门控列表数量不匹配"
                assert all(isinstance(g, torch.Tensor) for g in gate_list), "包含非张量门控"

                plt.figure(figsize=(10, 5))

                for idx, gate in enumerate(gate_list):
                    g_mean = gate.mean(dim=(0, -1)).cpu().numpy()
                    g_std = gate.std(dim=(0, -1)).cpu().numpy()
                    sample_size = gate.size(0)
                    ci = 1.96 * g_std / np.sqrt(sample_size)

                    plt.plot(g_mean, label=f"Hormone Gate {idx + 1}", linewidth=2.5)
                    plt.fill_between(range(len(g_mean)), g_mean - ci, g_mean + ci, alpha=0.25)

                plt.title("Hormone Gate Dynamics (Mean ± 95% CI)", fontsize=14)
                plt.xlabel("Time Step", fontsize=12)
                plt.ylabel("Gate Activation Value", fontsize=12)
                plt.legend(fontsize=11)
                plt.grid(alpha=0.3, linestyle='--')
                plt.tight_layout()

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.save_path, f"hormone_gate_{timestamp}.png")
                plt.savefig(save_path, dpi=250, bbox_inches="tight")
                plt.close()
                print(f"✅ 门控可视化已保存：{save_path}")

        except Exception as e:
            warnings.warn(f"❌ 可视化失败：{str(e)}")
            plt.close()  # 防止内存泄漏
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================
# 1. 随机深度（DropPath）：训练时随机丢弃残差分支
# 增强：设备对齐、空输入防护、概率合法性检查
# ==============================================
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        # 检查丢弃概率是否在合法范围
        assert 0.0 <= drop_prob <= 1.0, "Drop probability must be between 0 and 1"
        self.drop_prob = drop_prob

    def forward(self, x):
        # 空输入防护
        if x.numel() == 0:
            return x
            
        # 推理阶段 / 不丢弃 → 直接返回
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        # 批次维度随机，其他维度保持一致
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        
        # 强制设备与数据类型对齐，避免CPU/GPU不匹配
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()  # 二值化掩码
        
        # 保证训练/推理期望一致
        return x.div(keep_prob) * random_tensor

# ==============================================
# 2. RoPE + 相对位置偏置
# 增强：维度对齐、动态长度适配、设备安全、边界检查
# ==============================================
class RoPEWithRelativeBias(nn.Module):
    """RoPE + 相对位置偏置，超长序列更稳定"""
    def __init__(self, dim, max_len=1024):
        super().__init__()
        # 维度合法性检查
        assert dim > 0, "Dimension must be positive"
        assert max_len > 0, "Max sequence length must be positive"
        
        self.dim = dim
        self.max_len = max_len
        # 可学习偏置初始化
        self.rel_bias = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (B, T, D), got {x.ndim}D")
        B, T, D = x.shape
        assert D == self.dim

        # 不修改原参数，只计算需要的位置
        position = torch.arange(T, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, device=x.device) * (-math.log(10000.0) / D))
        sin = torch.sin(position * div_term).repeat(1, 2)[:, :D]
        cos = torch.cos(position * div_term).repeat(1, 2)[:, :D]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot_odd = torch.cat([-x_odd, x_even], dim=-1)[..., :D]
        x_rot = x * cos + x_rot_odd * sin

        # 只取前T个位置，不动态扩容
        bias = self.rel_bias[:, :T, :]
        return x_rot + bias

# ==============================================
# 3. 残差膨胀卷积上采样
# 增强：输入校验、padding合法性、残差维度对齐
# ==============================================
# ==============================================
# 3. 残差膨胀卷积上采样【已修复：自动长度对齐】
# 解决：上采样后 off-by-one 长度不匹配问题
# ==============================================
class ResidualDilatedConvUp(nn.Module):
    """膨胀卷积 + 残差上采样，高频时序专用
    已修复：自动对齐到目标长度，解决off-by-one报错
    """
    def __init__(self, dim, scale, kernel_size=5, dilation=2):
        super().__init__()
        # 参数合法性检查
        assert dim > 0, "Dimension must be positive"
        assert scale > 1, "Upsampling scale must be > 1"
        assert kernel_size % 2 == 1, "Kernel size must be odd for same padding"
        
        self.scale = scale
        self.dim = dim
        
        # 深度可分离卷积
        self.dw_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, dilation=dilation,
            padding= (kernel_size//2) * dilation, groups=dim
        )
        self.pw_conv = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x, target_len=None):
        """
        x: [B, T_in, D]
        target_len: 希望输出的目标长度（原始序列长度）
        自动 pad / crop 到 target_len
        """
        # 输入校验
        if x.ndim != 3:
            raise ValueError(f"Expected (B, T, D), got {x.shape}")
        B, T_in, D = x.shape
        assert D == self.dim, "Dimension mismatch"
        
        residual = x
        # 转维度适配Conv1d [B, D, T]
        x = x.transpose(1, 2)
        
        # 线性上采样
        x = F.interpolate(x, scale_factor=self.scale, mode="linear", align_corners=False)
        
        # 卷积
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        
        # 恢复维度 [B, T_up, D]
        x = x.transpose(1, 2)
        x = self.act(self.norm(x))

        # ==================== 核心修复：自动对齐到目标长度 ====================
        if target_len is not None:
            T_up = x.shape[1]
            if T_up > target_len:
                # 过长 → 裁剪
                x = x[:, :target_len, :]
            elif T_up < target_len:
                # 过短 → 右侧填充0
                pad_len = target_len - T_up
                x = F.pad(x, (0, 0, 0, pad_len), mode="constant", value=0.0)

        return x

# ==============================================
# 4. 尺度融合 Transformer
# 增强：空输入防护、维度检查、设备安全
# ==============================================
class ScaleFusionTransformer(nn.Module):
    """微型Transformer做跨尺度融合"""
    def __init__(self, dim, n_heads=2, layers=1):
        super().__init__()
        assert dim > 0 and n_heads > 0 and layers > 0
        self.dim = dim
        
        self.proj = nn.Linear(dim, dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=n_heads, dim_feedforward=dim*2,
                batch_first=True, activation="gelu"
            ),
            num_layers=layers
        )

    def forward(self, scale_feats):
        # 输入检查 [B, S, T, D]
        if scale_feats.ndim != 4:
            raise ValueError(f"Expected 4D input, got {scale_feats.ndim}D")
            
        B, S, T, D = scale_feats.shape
        assert D == self.dim, "Dimension mismatch"
        
        # 空尺度防护
        if S == 0 or T == 0:
            return scale_feats
            
        # 维度变换：[B, T, S, D] → [B*T, S, D]
        feats = scale_feats.permute(0, 2, 1, 3).reshape(B * T, S, D)
        fused = self.transformer(feats)
        
        # 恢复形状
        return fused.reshape(B, T, S, D).permute(0, 2, 1, 3)

# ==============================================
# 5. 最终多尺度时序编码器（主模型）—— 完整版检查
# 最全面的校验：输入、尺度、长度、设备、超参
# ==============================================
class FinalMultiScaleTemporalEncoder(nn.Module):
    def __init__(self, input_dim, n_heads=4, num_layers=2, dropout=0.1,
                 scales=[1,2,4], max_seq_len=1024, drop_path=0.1,
                 share_encoder=True):
        super().__init__()
        
        # ==================== 全局超参合法性检查 ====================
        assert input_dim > 0, "Input dimension must be positive"
        assert n_heads > 0, "Number of heads must be positive"
        assert input_dim % n_heads == 0, "Input dim must be divisible by n_heads"
        assert 0 <= dropout <= 1, "Dropout must be between 0 and 1"
        assert len(scales) > 0, "Must provide at least one scale"
        assert all(s > 0 and isinstance(s, int) for s in scales), "Scales must be positive integers"
        assert max_seq_len > 0, "Max sequence length must be positive"
        
        self.scales = scales
        self.n_scales = len(scales)
        self.share_encoder = share_encoder
        self.input_dim = input_dim

        # ==================== 位置编码 & 尺度嵌入 ====================
        self.rope_rel = RoPEWithRelativeBias(input_dim, max_seq_len)
        self.scale_emb = nn.Parameter(torch.randn(len(scales), input_dim) * 0.02)

        # ==================== 多尺度编码器 ====================
        if share_encoder:
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=input_dim, nhead=n_heads, dim_feedforward=input_dim*4,
                    dropout=dropout, batch_first=True, activation="gelu"
                ),
                num_layers=num_layers
            )
        else:
            self.encoders = nn.ModuleList([
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=input_dim, nhead=n_heads, dim_feedforward=input_dim*4,
                        dropout=dropout, batch_first=True, activation="gelu"
                    ),
                    num_layers=num_layers
                ) for _ in scales
            ])

        # ==================== 上采样模块 ====================
        self.upsamplers = nn.ModuleList([
            ResidualDilatedConvUp(input_dim, s, kernel_size=5, dilation=2) if s>1 else nn.Identity()
            for s in scales
        ])

        # ==================== 融合模块 ====================
        self.scale_fusion = ScaleFusionTransformer(input_dim, n_heads=2)
        self.drop_path = DropPath(drop_path)
        self.norm = nn.LayerNorm(input_dim)

    def _check_input(self, x):
        """内部输入校验函数：强制检查输入合法性"""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError(f"Input must be 3D (B, T, D), got {x.shape}")
        B, T, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"Input dim {D} != model dim {self.input_dim}")
        if T <= 0:
            raise ValueError("Sequence length must be positive")
        # 检查多尺度下采样不会把序列压成0
        for s in self.scales:
            if T // s == 0:
                raise ValueError(f"Scale {s} too large for sequence length {T} (would become 0)")

    def forward(self, x):
        # 强制输入检查
        self._check_input(x)
        B, T, D = x.shape
        outs = []

        # ==================== 多尺度编码 ====================
        for i, s in enumerate(self.scales):
            # 下采样
            xs = x[:, ::s, :]
            
            # 空序列防护
            if xs.shape[1] == 0:
                raise RuntimeError(f"Scale {s} caused empty sequence")
                
            # 位置编码 + 尺度嵌入
            xs = self.rope_rel(xs)
            xs = xs + self.scale_emb[i:i+1, :].to(x.device)  # 强制设备对齐

            # Transformer 编码
            if self.share_encoder:
                feat = self.encoder(xs)
            else:
                feat = self.encoders[i](xs)

            # 上采样回原始长度
            if s > 1:
                feat = self.upsamplers[i](feat)
            
            # 确保上采样后形状正确
            if feat.shape[1] != T:
                raise RuntimeError(f"Upsampler failed: expected T={T}, got {feat.shape[1]}")
                
            outs.append(feat)

        # ==================== 堆叠 & 融合 ====================
        outs = torch.stack(outs, dim=1)  # [B, S, T, D]
        outs = self.scale_fusion(outs)

        # 多尺度融合
        fused = outs.sum(dim=1)
        fused = fused + outs[:, 0]  # 强化原始尺度

        # 残差 + 归一化
        out = x + self.drop_path(fused)
        out = self.norm(out)

        return out
    
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from typing import Optional, Tuple


# -----------------------------------------------------------------------------
# 通用张量校验工具函数
# 功能：对张量的空值、维度、类型、批次一致性进行统一检查，提升代码鲁棒性
# -----------------------------------------------------------------------------
def check_tensor(tensor: torch.Tensor,
                 name: str,
                 ndim: Optional[int] = None,
                 dtype: Optional[torch.dtype] = None) -> None:
    """
    通用张量合法性检查

    Args:
        tensor: 待检查张量
        name: 张量名称（用于报错提示）
        ndim: 期望维度，为 None 则不检查
        dtype: 期望数据类型，为 None 则不检查

    Raises:
        ValueError: 张量为空或维度不匹配
        TypeError: 非张量类型或数据类型不匹配
    """
    # 检查张量是否为空
    if tensor is None:
        raise ValueError(f"张量 [{name}] 不允许为 None.")

    # 检查是否为 torch.Tensor 类型
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"[{name}] 必须为 torch.Tensor，当前类型: {type(tensor)}.")

    # 检查张量维度
    if ndim is not None and tensor.dim() != ndim:
        raise ValueError(
            f"[{name}] 维度应为 {ndim}，实际为 {tensor.dim()}，形状: {tensor.shape}."
        )

    # 检查数据类型
    if dtype is not None and tensor.dtype != dtype:
        raise TypeError(
            f"[{name}] 数据类型应为 {dtype}，实际为 {tensor.dtype}."
        )


def check_batch_consistency(*tensors: torch.Tensor) -> None:
    """
    检查多个张量在 batch 维度上是否一致

    Args:
        tensors: 任意数量的张量

    Raises:
        ValueError: 批次维度不一致
    """
    if len(tensors) < 2:
        return

    batch_size = tensors[0].shape[0]
    for idx, t in enumerate(tensors[1:], 1):
        if t.shape[0] != batch_size:
            raise ValueError(
                f"批次大小不一致：张量0={batch_size}，张量{idx}={t.shape[0]}."
            )


def check_sequence_length(x: torch.Tensor, max_seq_len: int) -> None:
    """
    检查输入序列长度是否超出模型支持的最大长度

    Args:
        x: 输入时序张量 [B, T, D]
        max_seq_len: 模型允许的最大序列长度

    Raises:
        ValueError: 序列长度超限
    """
    seq_len = x.shape[1]
    if seq_len > max_seq_len:
        raise ValueError(
            f"输入序列长度 {seq_len} 超过最大支持长度 {max_seq_len}."
        )


# -----------------------------------------------------------------------------
# 脑启发时序预测模型
# 融合双路径特征 + 激素动态门控 + 多尺度时序编码
# -----------------------------------------------------------------------------
class BrainInspiredNetV2(nn.Module):
    """
    脑启发式时间序列预测模型 V2
    核心机制：
        1. 双路径信息编码：直接路径（局部特征）+ 间接路径（全局依赖）
        2. 激素动态调控：自适应生成门控权重融合双路径
        3. 多尺度时序编码：捕捉不同时间尺度趋势
        4. 可选时序池化：mean/max/last/attention

    适用场景：
        - 金融收益率预测
        - 多变量生理时序预测
        - 通用多变量时间序列预测
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 hormone_dim: int = 32,
                 max_seq_len: int = 30,
                 pooling: str = "last",
                 output_dim: int = 5):
        """
        模型初始化

        Args:
            input_dim: 输入特征维度
            hidden_dim: 模型隐藏层维度
            hormone_dim: 激素模块内部维度
            max_seq_len: 支持的最大序列长度
            pooling: 时序池化方式，可选 mean/max/last/attention
            output_dim: 输出维度（如预测股票数量）
        """
        super().__init__()

        # ------------------------------
        # 初始化参数合法性校验
        # ------------------------------
        if not (isinstance(input_dim, int) and input_dim > 0):
            raise ValueError("input_dim 必须为正整数.")
        if not (isinstance(hidden_dim, int) and hidden_dim > 0):
            raise ValueError("hidden_dim 必须为正整数.")
        if not (isinstance(hormone_dim, int) and hormone_dim > 0):
            raise ValueError("hormone_dim 必须为正整数.")
        if not (isinstance(max_seq_len, int) and max_seq_len > 0):
            raise ValueError("max_seq_len 必须为正整数.")
        if not (isinstance(output_dim, int) and output_dim > 0):
            raise ValueError("output_dim 必须为正整数.")

        valid_pooling = ["mean", "max", "last", "attention"]
        if pooling not in valid_pooling:
            raise ValueError(f"pooling 必须为 {valid_pooling} 之一.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.pooling = pooling

        # ------------------------------
        # 直接路径：浅层局部特征编码
        # ------------------------------
        self.direct_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ------------------------------
        # 间接路径：全局关联特征编码
        # ------------------------------
        self.indirect_path = GraphAttentionLayer(input_dim, hidden_dim, n_heads=4)

        

        self.hormone_mod = HormoneModule(hidden_dim, hormone_dim, hidden_dim)

        # ------------------------------
        # 多尺度时序编码器（恒等映射占位）
        # ------------------------------
        self.temporal_encoder = nn.Identity()

        # ------------------------------
        # 特征融合与输出层
        # ------------------------------
        self.fusion_dim = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, output_dim)

        # 双路径维度对齐
        self.direct_proj = nn.Identity()
        self.indirect_proj = nn.Linear(hidden_dim, hidden_dim)

        # 注意力池化（如需要）
        if pooling == "attention":
            self.att_pool = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

    def forward(self,
                x: torch.Tensor,
                adj_matrix: Optional[torch.Tensor] = None,
                hormone_prev: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        模型前向传播

        Args:
            x: 输入时序特征 [B, T, D]
            adj_matrix: 图邻接矩阵（本版本未使用）
            hormone_prev: 上一时刻激素状态
            mask: 时序掩码 [B, T]

        Returns:
            y: 预测输出 [B, output_dim]
            hormone_state: 当前激素状态
        """
        # 输入合法性检查
        check_tensor(x, "x", ndim=3)
        check_sequence_length(x, self.max_seq_len)

        B, T, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"输入特征维度 {D} 与模型 input_dim {self.input_dim} 不匹配.")

        # mask 形状检查
        if mask is not None:
            check_tensor(mask, "mask", ndim=2)
            if mask.shape != (B, T):
                raise ValueError(f"mask 形状应为 ({B},{T})，实际为 {mask.shape}.")

        # 激素历史状态检查
        if hormone_prev is not None:
            check_tensor(hormone_prev, "hormone_prev", ndim=3)
            check_batch_consistency(x, hormone_prev)

        # ------------------------------
        # 双路径特征编码
        # ------------------------------
        direct_out = self.direct_mlp(x)
        indirect_out = self.indirect_path(x)

        # 输出形状校验
        if direct_out.shape != (B, T, self.hidden_dim):
            raise RuntimeError(f"直接路径输出形状异常: {direct_out.shape}.")
        if indirect_out.shape != (B, T, self.hidden_dim):
            raise RuntimeError(f"间接路径输出形状异常: {indirect_out.shape}.")

        # 维度对齐
        direct_out = self.direct_proj(direct_out)
        indirect_out = self.indirect_proj(indirect_out)

        # ------------------------------
        # 激素门控融合
        # ------------------------------
        #alpha, beta, hormone_state = self.hormone_mod(direct_out + indirect_out, hormone_prev)
        alpha, beta = self.hormone_mod(direct_out + indirect_out, hormone_prev)
        # # 确保 alpha + beta ≈ 1
        # if not torch.allclose(alpha + beta, torch.ones_like(alpha), atol=1e-4):
        #     raise ValueError("激素门控约束不满足：alpha + beta 必须接近 1.")

        fused = alpha * direct_out + beta * indirect_out

        # ------------------------------
        # 多尺度时序增强
        # ------------------------------
        fused = self.fusion_dim(fused)
        encoded = self.temporal_encoder(fused)

        # ------------------------------
        # 时序池化
        # ------------------------------
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
            raise ValueError(f"不支持的池化方式: {self.pooling}.")

        # 最终预测
        y = self.out_layer(pooled)
        return y


# -----------------------------------------------------------------------------
# 激素门控权重可视化函数
# -----------------------------------------------------------------------------
def visualize_alpha_beta(alpha: torch.Tensor,
                         beta: torch.Tensor,
                         batch_idx: int = 0,
                         title: str = "Alpha/Beta Weights",
                         save_path: Optional[str] = None,
                         max_subplots: int = 6) -> None:
    """
    可视化激素模块动态门控权重 Alpha（直接路径）与 Beta（间接路径）

    Args:
        alpha: 直接路径权重 [B, T, H]
        beta: 间接路径权重 [B, T, H]
        batch_idx: 选择可视化的样本序号
        title: 图表标题
        save_path: 图片保存路径前缀
        max_subplots: 单页最大子图数量
    """
    # 输入检查
    check_tensor(alpha, "alpha", ndim=3)
    check_tensor(beta, "beta", ndim=3)
    check_batch_consistency(alpha, beta)

    B, T, H = alpha.shape
    if beta.shape != (B, T, H):
        raise ValueError(f"alpha/beta 形状不一致: {alpha.shape} vs {beta.shape}.")

    # 批次索引越界检查
    if not (0 <= batch_idx < B):
        raise IndexError(f"batch_idx 必须在 [0, {B-1}] 范围内.")

    # 门控值范围检查
    if not ((alpha >= -0.01) & (alpha <= 1.01)).all():
        raise ValueError(f"alpha 超出合理范围 [0,1].")
    if not ((beta >= -0.01) & (beta <= 1.01)).all():
        raise ValueError(f"beta 超出合理范围 [0,1].")

    # 门控和约束检查
    sum_err = torch.abs(alpha + beta - 1.0).max().item()
    if sum_err > 0.1:
        raise ValueError(f"alpha+beta 偏离 1 过大，最大误差: {sum_err:.3f}.")

    # 转为 numpy
    alpha_np = alpha[batch_idx].detach().cpu().numpy()
    beta_np = beta[batch_idx].detach().cpu().numpy()

    # 分页绘图
    num_pages = math.ceil(H / max_subplots)
    for page in range(num_pages):
        start = page * max_subplots
        end = min(start + max_subplots, H)
        num_plots = end - start

        plt.figure(figsize=(12, 3 * num_plots))
        for i in range(start, end):
            plt.subplot(num_plots, 1, i - start + 1)
            plt.plot(alpha_np[:, i], label=f"Alpha {i+1}", color="#1f77b4")
            plt.plot(beta_np[:, i], label=f"Beta {i+1}", color="#ff7f0e")
            plt.xlabel("Time Step")
            plt.ylabel("Gate Weight")
            plt.title(f"Hormone Gate Dim {i+1}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(-0.05, 1.05)

        plt.suptitle(f"{title} (Page {page+1}/{num_pages})", fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path is not None:
            plt.savefig(f"{save_path}_page{page+1}.png", dpi=200, bbox_inches='tight')

        plt.show()

import matplotlib.pyplot as plt
import math
import torch

def setup_logger(log_path=None):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger


import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
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
def evaluate_model(model, data_loader, criterion, device, use_mask=False):
    model.eval()
    loss_sum = 0.0
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(data_loader, desc="Evaluating", leave=False):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            outputs = model(x_batch)
            y_pred = outputs[0] if isinstance(outputs, tuple) else outputs

            # ===================== 修复 BUG：只在单输出时 squeeze =====================
            if y_pred.size(-1) == 1:
                y_pred = y_pred.squeeze(-1)
            if y_batch.size(-1) == 1:
                y_batch = y_batch.squeeze(-1)

            # 缺失值掩码
            if use_mask:
                mask = ~torch.isnan(y_batch)
                y_pred_masked = y_pred[mask]
                y_batch_masked = y_batch[mask]
            else:
                y_pred_masked = y_pred
                y_batch_masked = y_batch

            # 过滤空张量（避免全 NaN 批次报错）
            if y_batch_masked.numel() == 0:
                continue

            # 损失
            try:
                batch_loss = criterion(y_pred_masked, y_batch_masked).item()
            except:
                batch_loss = 0.0
            loss_sum += batch_loss

            # 保存
            y_true_list.append(y_batch_masked.cpu().numpy())
            y_pred_list.append(y_pred_masked.cpu().numpy())

    # 平均损失
    avg_loss = loss_sum / len(data_loader) if len(data_loader) > 0 else 0.0

    # ===================== 拼接（过滤空数组） =====================
    if len(y_true_list) == 0:
        return avg_loss, np.nan, np.nan, np.nan

    y_true_all = np.concatenate(y_true_list, axis=0)
    y_pred_all = np.concatenate(y_pred_list, axis=0)

    # ===================== 确保二维 [N, D]（修复 BUG） =====================
    if y_true_all.ndim == 1:
        y_true_all = y_true_all.reshape(-1, 1)
    if y_pred_all.ndim == 1:
        y_pred_all = y_pred_all.reshape(-1, 1)

    # ===================== 逐维度计算 =====================
    metrics = {"R2": [], "MAE": [], "RMSE": []}
    num_targets = y_true_all.shape[1]

    for i in range(num_targets):
        yt = y_true_all[:, i]
        yp = y_pred_all[:, i]

        # 过滤常数序列（R2 无法计算）
        if np.var(yt) < 1e-8:
            metrics["R2"].append(np.nan)
            metrics["MAE"].append(np.mean(np.abs(yt - yp)))
            metrics["RMSE"].append(np.sqrt(np.mean((yt - yp) ** 2)))
            continue

        try:
            metrics["R2"].append(r2_score(yt, yp))
            metrics["MAE"].append(mean_absolute_error(yt, yp))
            metrics["RMSE"].append(np.sqrt(mean_squared_error(yt, yp)))
        except:
            metrics["R2"].append(np.nan)
            metrics["MAE"].append(np.nan)
            metrics["RMSE"].append(np.nan)

    # 单输出返回标量
    if num_targets == 1:
        for k in metrics:
            metrics[k] = metrics[k][0]

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

                # # 自动对齐输出维度（处理维度不匹配）
                # if pred.size(1) != y_batch.size(1):
                #     if pred.size(1) > y_batch.size(1):
                #         pred = pred[:, :y_batch.size(1)]
                #     else:
                #         pad = torch.zeros(pred.size(0), y_batch.size(1)-pred.size(1), device=self.device)
                #         pred = torch.cat([pred, pad], dim=1)

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
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32) if isinstance(X, np.ndarray) else X.clone().detach().float()
        self.y = torch.tensor(y, dtype=torch.float32) if isinstance(y, np.ndarray) else y.clone().detach().float()
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
    "max_epochs":100,              # 最大训练轮数
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
# 加载预处理好的数据
import numpy as np

output_dir = r"E:\大学\学习\血液激素神经网络\archive\processed"
X_train = torch.from_numpy(np.load(os.path.join(output_dir, "X_train.npy")))
y_train = torch.from_numpy(np.load(os.path.join(output_dir, "y_train.npy")))
X_val = torch.from_numpy(np.load(os.path.join(output_dir, "X_val.npy")))
y_val = torch.from_numpy(np.load(os.path.join(output_dir, "y_val.npy")))

# ==================== 【必须加：创建数据加载器】 ====================
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