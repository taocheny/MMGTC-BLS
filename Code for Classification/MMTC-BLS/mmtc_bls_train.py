import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import time

from sparse_bls import sparse_bls


# =========================================================================
#  MMTC-BLS (分类版): Broad Learning System via Maximum Mixture Total
#  Correntropy。训练算法与回归版完全一致, 仅有两处改动:
#    (1) 训练目标 Y 改为类别的 one-hot 编码 (N×C), 输出权重 W 为 (L, C);
#    (2) 预测取输出打分的 argmax 作为类别, 评价指标改为分类准确率 ACC。
#
#  目标函数 (Eq.1):
#    J1 = argmax_W  Σ_{j=1}^{2} Σ_{i=1}^{N} a_j·exp( -||u_i W - y_i||^2 / (2 σ_j^2 ||W̄||^2) )
#                   - (λ/2)·||W||^2 ,      a1 + a2 = 1,  a_j > 0
#    其中 y_i 为第 i 个样本的 one-hot 标签向量。
#
#  增广权重向量: W̄ = [ sqrt(σ_o^2/σ_i^2) ; -W^T ]^T ⇒ ||W̄||^2 = σ_o^2/σ_i^2 + ||W||_F^2。
#  注: 分类时 W 为 (N_k×C) 矩阵, ||W||^2 取 Frobenius 范数平方 (= 全元素平方和);
#      ||u_iW - y_i||^2 为每个样本 C 维残差向量的 2-范数平方 (按 C 个类别维度求和)。
#  不动点迭代 (Eqs.4-6) 与回归版完全相同, 已天然支持多输出 (C 列) 目标。
# =========================================================================


def _prepare_labels(train_y, test_y, device):
    """
    将分类标签整理为 (one-hot 训练目标, 整型类别索引)。
    - 若已是 one-hot (列数 > 1): 直接用作训练目标, 类别索引 = argmax;
    - 若为单列整型标签: 依据训练+测试集出现的类别构造 one-hot。
    返回 device 上的张量: Y_train(N,C) float, train_lab(N,) long, 测试同理。
    """
    y_tr = np.asarray(train_y)
    y_te = np.asarray(test_y)
    if y_tr.ndim == 1:
        y_tr = y_tr.reshape(-1, 1)
    if y_te.ndim == 1:
        y_te = y_te.reshape(-1, 1)

    if y_tr.shape[1] > 1:                       # 已是 one-hot 编码
        Y_train = y_tr.astype('float32')
        Y_test = y_te.astype('float32')
        train_lab = np.argmax(y_tr, axis=1)
        test_lab = np.argmax(y_te, axis=1)
    else:                                       # 单列整型标签 → 构造 one-hot
        classes = np.unique(np.concatenate([y_tr.ravel(), y_te.ravel()]))
        idx = {c: i for i, c in enumerate(classes)}
        C = len(classes)
        train_lab = np.array([idx[v] for v in y_tr.ravel()], dtype='int64')
        test_lab = np.array([idx[v] for v in y_te.ravel()], dtype='int64')
        eye = np.eye(C, dtype='float32')
        Y_train = eye[train_lab]
        Y_test = eye[test_lab]

    return (torch.from_numpy(Y_train).float().to(device),
            torch.from_numpy(train_lab).long().to(device),
            torch.from_numpy(Y_test).float().to(device),
            torch.from_numpy(test_lab).long().to(device))


def mmtc_fixed_point(U, Y, a1, a2, sigma1, sigma2, lam,
                     sigma_ratio, nu, t_max, verbose=True):
    """
    不动点迭代求解 MMTC-BLS 的输出权重 W (论文 Eqs. 3-6)。与回归版完全相同,
    此处 Y 为 (N, C) 的 one-hot 目标, 求得 W 为 (L, C)。

    参数:
        U:           (N, L) 隐藏表示 [F, E] (特征层与增强层拼接, 即论文中的 U)
        Y:           (N, C)  one-hot 目标
        a1, a2:      混合系数 (a1 + a2 = 1)
        sigma1:      小核带宽 σ1
        sigma2:      大核带宽 σ2
        lam:         Tikhonov 正则化参数 λ
        sigma_ratio: σ_o^2/σ_i^2, 增广权重范数 ||W̄||^2 = sigma_ratio + ||W||^2
        nu:          收敛容忍度 ν
        t_max:       最大迭代次数

    返回:
        W:      (L, C) 收敛后的输出权重
        n_iter: 实际迭代次数
    """
    device = U.device
    L = U.shape[1]
    I = torch.eye(L, device=device)

    # ---- 用岭回归 (标准 BLS 解) 初始化 W(0) ----
    A0 = U.T @ U + lam * I
    b0 = U.T @ Y
    try:
        W = torch.linalg.solve(A0, b0)
    except RuntimeError:
        W = torch.linalg.pinv(A0) @ b0

    n_iter = 0
    for t in range(t_max):
        W_prev = W.clone()

        # ||W̄||^2 = σ_o^2/σ_i^2 + ||W||^2  (标量, 增广权重向量范数)
        W_bar_norm2 = (sigma_ratio + torch.sum(W ** 2)).clamp(min=1e-12)

        # 残差 e_i = u_i W - y_i 及其平方范数 ||e_i||^2 (按输出维度求和)
        E = U @ W - Y                                   # (N, C)
        e_norm2 = torch.sum(E ** 2, dim=1)              # (N,)

        # ---- Λ1 对角元 (正定加权), 论文 Eq.(3) ----
        exp1 = torch.exp(-e_norm2 / (2.0 * sigma1 ** 2 * W_bar_norm2))
        exp2 = torch.exp(-e_norm2 / (2.0 * sigma2 ** 2 * W_bar_norm2))
        lam1_diag = (a1 / sigma1 ** 2) * exp1 + (a2 / sigma2 ** 2) * exp2   # (N,)

        # ---- s1, 论文 Eq.(3): 复用 Λ1 对角元 → s1 = (1/||W̄||^2)·Σ_i ||e_i||^2·Λ1[i,i] ----
        s1 = torch.sum(e_norm2 * lam1_diag) / W_bar_norm2

        # ---- ξ1 = λ·||W̄||^2 - s1, 论文 Eq.(4) ----
        xi1 = lam * W_bar_norm2 - s1

        # ---- W = (U^T Λ1 U + ξ1 I)^{-1} U^T Λ1 Y, 论文 Eqs.(4)-(6) ----
        ULam = U * lam1_diag.unsqueeze(1)               # (N, L) = Λ1 U
        A = ULam.T @ U + xi1 * I                         # (L, L)
        b = ULam.T @ Y                                   # (L, C)
        try:
            W = torch.linalg.solve(A, b)
        except RuntimeError:
            W = torch.linalg.pinv(A) @ b

        n_iter = t + 1
        delta = torch.sum((W - W_prev) ** 2).item()      # ||W(t) - W(t-1)||^2
        if verbose:
            print(f'  iter {n_iter:3d}: ||ΔW||^2 = {delta:.4e}, ξ1 = {xi1.item():.4e}')
        if delta < nu:
            if verbose:
                print(f'[MMTC-BLS] 不动点迭代收敛于第 {n_iter} 次')
            break

    return W, n_iter


# =========================================================================
#                       MMTC-BLS 分类主训练函数
# =========================================================================

def mmtc_bls_train(train_x, train_y, test_x, test_y,
                   s, num_fea, num_win, num_enhan,
                   a1, a2, sigma1, sigma2,
                   lam, sigma_i2, sigma_o2, nu, t_max, verbose):
    """
    MMTC-BLS 分类训练与测试。隐藏层生成与回归版一致; 输出层以类别 one-hot 为
    目标, 用最大混合总相关熵准则求解; 预测取输出打分 argmax, 评价指标为准确率 ACC。

    参数 (均为必填; 取值与论文一致, 由调用方/驱动脚本显式传入):
        s            BLS 增强层 l2 缩放系数
        num_fea      每个窗口的特征节点数
        num_win      特征窗口数
        num_enhan    增强节点数
        a1, a2       混合系数 (a1 + a2 = 1, a_j > 0)
        sigma1       小核带宽 σ1 (梯度陡峭, 精确拟合小残差)
        sigma2       大核带宽 σ2 (快速抑制重尾/大异常值)
        lam          Tikhonov 正则化参数 λ
        sigma_i2     输入噪声方差 σ_i^2 (论文取 0.1)
        sigma_o2     输出噪声方差 σ_o^2 (论文取 0.1)
                     二者决定增广权重: ||W̄||^2 = σ_o^2/σ_i^2 + ||W||^2
        nu           收敛容忍度 ν (论文取 0.05)
        t_max        最大迭代次数 (论文取 50)

    返回:
        (test_pred_label, training_time, testing_time, train_acc, test_acc, n_iter)
        其中 test_pred_label 为测试集预测类别 (整型索引)。
    """
    assert abs(a1 + a2 - 1.0) < 1e-6, 'a1 + a2 必须等于 1'
    assert a1 > 0 and a2 > 0, 'a1, a2 必须为正'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 增广权重向量尺度 σ_o^2/σ_i^2 (本文输入/输出噪声方差均为 0.1 ⇒ 比值为 1)
    sigma_ratio = sigma_o2 / sigma_i2

    train_x_t = torch.from_numpy(train_x).float().to(device)
    test_x_t  = torch.from_numpy(test_x).float().to(device)

    # 标签整理: one-hot 训练目标 + 整型类别索引 (用于 ACC)
    Y_train, train_labels, Y_test, test_labels = _prepare_labels(train_y, test_y, device)

    # ==================================================================
    #  生成 BLS 特征层 F、增强层 E, 拼接为隐藏层 U = [F, E]
    # ==================================================================
    scaler_train = StandardScaler()
    train_x_scaled_np = scaler_train.fit_transform(train_x_t.T.cpu().numpy()).T
    train_x_scaled = torch.from_numpy(train_x_scaled_np).float().to(device)

    x1 = torch.hstack([train_x_scaled,
                       0.1 * torch.ones((train_x_scaled.shape[0], 1), device=device)])
    feature_nodes = torch.zeros((train_x_scaled.shape[0], num_win * num_fea),
                                device=device)
    we_list, ps_list = [], []

    for i in range(num_win):
        wr = 2 * torch.rand(train_x_scaled.shape[1] + 1, num_fea, device=device) - 1
        a_proj = x1 @ wr
        a_mapped = 2 * (a_proj - torch.min(a_proj)) / \
            (torch.max(a_proj) - torch.min(a_proj) + 1e-10) - 1
        ws = sparse_bls(a_mapped, x1, 1e-3, 50).T
        we_list.append(ws)

        f1 = x1 @ ws
        ps1 = {'max': torch.max(f1, dim=0)[0], 'min': torch.min(f1, dim=0)[0]}
        f1_mapped = (f1 - ps1['min']) / (ps1['max'] - ps1['min'] + 1e-10)
        ps_list.append(ps1)
        feature_nodes[:, num_fea * i:num_fea * (i + 1)] = f1_mapped

    # 增强层: 正交随机权重
    x2 = torch.hstack([feature_nodes,
                       0.1 * torch.ones((feature_nodes.shape[0], 1), device=device)])
    m_dim = num_fea * num_win + 1
    rand_mat = torch.randn((m_dim, num_enhan), device=device)
    if m_dim >= num_enhan:
        q, _ = torch.linalg.qr(rand_mat, mode='reduced')
        wh = q[:, :num_enhan]
    else:
        q_t, _ = torch.linalg.qr(rand_mat.T, mode='reduced')
        wh = q_t.T

    raw_enh = x2 @ wh
    l2_scale = s / (torch.max(raw_enh) + 1e-10)
    enhancement_nodes = torch.tanh(raw_enh * l2_scale)

    # 隐藏表示 U = [F, E]  (特征层与增强层拼接, 即论文中的 U)
    U = torch.hstack([feature_nodes, enhancement_nodes])           # (N, L)
    N = U.shape[0]

    if verbose:
        print(f'[MMTC-BLS] 隐藏层生成完毕: N={N}, U.shape={tuple(U.shape)}, '
              f'类别数 C={Y_train.shape[1]}')

    # ==================================================================
    #  不动点迭代求解输出权重 W (目标为 one-hot 的 Y_train)
    # ==================================================================
    start_time = time.time()
    W, n_iter = mmtc_fixed_point(U, Y_train, a1, a2, sigma1, sigma2,
                                 lam, sigma_ratio, nu, t_max, verbose)
    training_time = time.time() - start_time

    # 训练准确率 ACC = argmax 预测与真实类别的一致率
    train_pred_label = torch.argmax(U @ W, dim=1)
    train_acc = (train_pred_label == train_labels).float().mean()
    if verbose:
        print(f'[MMTC-BLS] 训练完成, 用时 {training_time:.4f} s, 共迭代 {n_iter} 次')
        print(f'[MMTC-BLS] 训练 ACC = {train_acc.item():.4f}')

    # ==================================================================
    #  测试阶段: 用训练阶段得到的映射权重生成测试集隐藏层
    # ==================================================================
    start_time_test = time.time()

    scaler_test = StandardScaler()
    test_x_scaled_np = scaler_test.fit_transform(test_x_t.T.cpu().numpy()).T
    test_x_scaled = torch.from_numpy(test_x_scaled_np).float().to(device)

    xx1 = torch.hstack([test_x_scaled,
                        0.1 * torch.ones((test_x_scaled.shape[0], 1), device=device)])
    feature_nodes_test = torch.zeros((test_x_scaled.shape[0], num_fea * num_win),
                                     device=device)
    for i in range(num_win):
        ws = we_list[i]
        ps1 = ps_list[i]
        f2 = xx1 @ ws
        f2_mapped = (f2 - ps1['min']) / (ps1['max'] - ps1['min'] + 1e-10)
        feature_nodes_test[:, num_fea * i:num_fea * (i + 1)] = f2_mapped

    xx2 = torch.hstack([feature_nodes_test,
                        0.1 * torch.ones((feature_nodes_test.shape[0], 1),
                                         device=device)])
    enhancement_nodes_test = torch.tanh(xx2 @ wh * l2_scale)

    U_test = torch.hstack([feature_nodes_test, enhancement_nodes_test])

    test_pred_label = torch.argmax(U_test @ W, dim=1)
    test_acc = (test_pred_label == test_labels).float().mean()
    testing_time = time.time() - start_time_test

    if verbose:
        print(f'[MMTC-BLS] 测试完成, 用时 {testing_time:.4f} s')
        print(f'[MMTC-BLS] 测试 ACC = {test_acc.item():.4f}')

    return (test_pred_label.cpu().numpy(),
            training_time, testing_time,
            train_acc.item(),
            test_acc.item(),
            n_iter)
