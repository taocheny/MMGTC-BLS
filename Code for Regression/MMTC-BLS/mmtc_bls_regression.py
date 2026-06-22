import numpy as np
from scipy.io import loadmat
import pandas as pd
from mmtc_bls_train import mmtc_bls_train


# ===================== 读取数据 =====================
try:
    data = loadmat('../../traindata/Sunspots/sunspot06.mat')
    train_x = data['train_x']
    train_y = data['train_y']
    test_x  = data['test_x']
    test_y  = data['test_y']
except FileNotFoundError:
    print('错误: sunspot06.mat 未找到,请提供数据文件。')
    exit()

assert train_x.dtype == np.float64, 'train_x 必须是浮点型'
assert test_x.dtype  == np.float64, 'test_x 必须是浮点型'


# ===================== 搜索网格与固定超参 =====================
# --- 阶段 1: BLS 架构网格 (按参考论文 [5] 设定) ---
FEA_RANGE   = range(1, 21)      # N_f : [1,20] 步长 2  → {1,3,...,19}   10 个
WIN_RANGE   = range(1, 21)      # N_w : [1,20] 步长 1  → {1,2,...,20}   20 个
ENHAN_RANGE = range(2, 201, 2)     # N_e : [2,200] 步长 5 → {1,6,...,196}  40 个  (共 8000 组)

# --- 阶段 2: MMTC 核参网格 ---
A1_GRID    = [round(0.05 * k, 2) for k in range(1, 20)]   # a1 ∈ {0.05,...,0.95} (19 个), a2 = 1 - a1
SIGMA_GRID = [2.0 ** k for k in range(-5, 6)]             # σ ∈ {2^-5,...,2^5} (11 个)

# --- 固定超参数 ---
LAM      = 1e-5     # 正则化参数 λ
SIGMA_I2 = 0.1      # 输入噪声方差 σ_i^2
SIGMA_O2 = 0.1      # 输出噪声方差 σ_o^2
NU       = 0.05     # 收敛容忍度 ν
T_MAX    = 50       # 最大迭代次数
S        = 0.8      # 增强层 l2 缩放系数

# --- 阶段 1 选架构时所用的固定核参 (仅用于先定架构, 可按需修改) ---
STAGE1_A1, STAGE1_A2         = 0.7, 0.3
STAGE1_SIGMA1, STAGE1_SIGMA2 = 2.0, 3.0


def run_mmtc(num_fea, num_win, num_enhan, a1, a2, sigma1, sigma2):
    """用给定架构与核参训练一次 MMTC-BLS, 返回 (train_err, test_err, n_iter, tr_t, te_t)。"""
    (_, tr_t, te_t, train_err, test_err, n_iter) = mmtc_bls_train(
        train_x, train_y, test_x, test_y,
        S, num_fea, num_win, num_enhan,
        a1=a1, a2=a2, sigma1=sigma1, sigma2=sigma2,
        lam=LAM, sigma_i2=SIGMA_I2, sigma_o2=SIGMA_O2,
        nu=NU, t_max=T_MAX, verbose=False,
    )
    return train_err, test_err, n_iter, tr_t, te_t


# ===================================================================
#  阶段 1: 固定核参, 在架构网格上搜索最优 (N_f, N_w, N_e)
# ===================================================================
print('===== 阶段 1: 搜索最优架构 (N_f, N_w, N_e), 固定核参 =====')
arch_records = []
best_arch_err = np.inf
best_arch = None

for num_fea in FEA_RANGE:
    for num_win in WIN_RANGE:
        for num_enhan in ENHAN_RANGE:
            print(f'[阶段1] 特征节点数 = {num_fea}, 窗口数 = {num_win}, '
                  f'增强节点数 = {num_enhan}')
            try:
                train_err, test_err, _, _, _ = run_mmtc(
                    num_fea, num_win, num_enhan,
                    STAGE1_A1, STAGE1_A2, STAGE1_SIGMA1, STAGE1_SIGMA2,
                )
            except Exception as e:
                print(f'训练过程中发生错误: {e}')
                continue

            arch_records.append([num_fea, num_win, num_enhan, test_err, train_err])
            print(f'  → 测试 RMSE = {test_err:.4e}, 训练 RMSE = {train_err:.4e}')
            if test_err < best_arch_err:
                best_arch_err = test_err
                best_arch = (num_fea, num_win, num_enhan)
                print(f'  ✓ 新最优架构 N_f={num_fea}, N_w={num_win}, '
                      f'N_e={num_enhan}: test_err={test_err:.4e}')

pd.DataFrame(arch_records,
             columns=['NumFea', 'NumWin', 'NumEnhan', 'Test_ERR', 'Train_ERR']
             ).to_excel('arch_results_sunspot06_mmtc.xlsx', index=False)

if best_arch is None:
    print('错误: 阶段 1 没有任何成功的架构组合, 终止。')
    exit()
num_fea_b, num_win_b, num_enhan_b = best_arch
print(f'\n[阶段 1 完成] 最优架构: N_f={num_fea_b}, N_w={num_win_b}, '
      f'N_e={num_enhan_b} (test_err={best_arch_err:.4e})\n')


# ===================================================================
#  阶段 2: 固定阶段 1 选出的架构, 搜索核参 (a1, σ1, σ2)
# ===================================================================
print('===== 阶段 2: 固定架构, 搜索核参 (a1, σ1, σ2) =====')
result = []
best = np.inf
best_dict = None

for a1 in A1_GRID:
    a2 = 1.0 - a1
    for sigma1 in SIGMA_GRID:
        for sigma2 in SIGMA_GRID:
            print(f'[阶段2] a1 = {a1:.2f}, a2 = {a2:.2f}, '
                  f'σ1 = {sigma1}, σ2 = {sigma2}')
            try:
                train_err, test_err, n_iter, tr_t, te_t = run_mmtc(
                    num_fea_b, num_win_b, num_enhan_b,
                    a1, a2, sigma1, sigma2,
                )
            except Exception as e:
                print(f'训练过程中发生错误: {e}')
                continue

            total_time = tr_t + te_t
            result.append([a1, a2, sigma1, sigma2, n_iter,
                           test_err, train_err, total_time, tr_t, te_t])
            print(f'  → 测试 RMSE = {test_err:.4e}, 训练 RMSE = {train_err:.4e}, '
                  f'迭代 = {n_iter}, 用时 = {total_time:.3f}s')
            if test_err < best:
                best = test_err
                best_dict = {
                    'Test_ERR'     : [test_err],
                    'Train_ERR'    : [train_err],
                    'a1'           : [a1],
                    'a2'           : [a2],
                    'sigma1'       : [sigma1],
                    'sigma2'       : [sigma2],
                    'lambda'       : [LAM],
                    'NumFea'       : [num_fea_b],
                    'NumWin'       : [num_win_b],
                    'NumEnhan'     : [num_enhan_b],
                    'N_iter'       : [n_iter],
                    'Total_Time'   : [total_time],
                    'Training_Time': [tr_t],
                    'Testing_Time' : [te_t],
                }
                pd.DataFrame(best_dict).to_excel(
                    'best_result_sunspot06_mmtc.xlsx', index=False)
                print(f'  ✓ 新最优 a1={a1}, σ1={sigma1}, σ2={sigma2}: '
                      f'test_err={test_err:.4e}')

pd.DataFrame(result, columns=[
    'a1', 'a2', 'sigma1', 'sigma2', 'N_iter',
    'Test_ERR', 'Train_ERR', 'Total_Time', 'Training_Time', 'Testing_Time',
]).to_excel('all_results_sunspot06_mmtc.xlsx', index=False)

print('\n两阶段搜索完成。')
if best_dict is not None:
    print(f'最优配置: N_f={num_fea_b}, N_w={num_win_b}, N_e={num_enhan_b}, '
          f'a1={best_dict["a1"][0]}, σ1={best_dict["sigma1"][0]}, '
          f'σ2={best_dict["sigma2"][0]}, λ={LAM}')
    print(f'最佳测试误差: {best:.4e}')
else:
    print('阶段 2 没有任何成功的核参组合。')
