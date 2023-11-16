import torch
# 这行代码导入了 torch 库，它是一个流行的机器学习框架，提供了构建和训练神经网络的工具。
import numpy as np
# 这行代码导入了 numpy 库，它是 Python 中的一个基础科学计算库。它提供了对大型多维数组和矩阵的支持，并且提供了一系列用于操作这些数组的数学函数。
import cvxopt
from cvxopt import matrix
# 这些行代码导入了 cvxopt 库，它是一个用于凸优化的 Python 包。从 cvxopt 导入的 matrix 模块特别用于处理矩阵和线性优化问题。
import os
# 这行代码导入了 os 模块，它提供了与操作系统进行交互的方法。通常用于文件和目录操作，如检查文件是否存在、创建目录等。
import copy
# 这行代码导入了 copy 模块，它提供了创建对象副本的函数。可以用于创建对象的独立副本，避免在处理可变对象时产生意外的副作用。
import math
# 这行代码导入了 math 模块，它提供了各种数学函数和常量。可以进行三角函数、对数等数学计算。
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 这行代码的作用是将环境变量KMP_DUPLICATE_LIB_OK设置为"TRUE"。
# 这个环境变量的含义是告诉Intel MKL（Math Kernel Library）在加载时是否允许重复的动态链接库（shared library）。
# 该环境变量的设置可以解决在使用Intel MKL时可能出现的一些问题，特别是在使用Jupyter Notebook或某些Python编辑器时，可能会报告动态链接库冲突的错误。

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
# 这行代码定义了一个名为 cvxopt_solve_qp 的函数，用于使用 cvxopt 库解决二次规划问题。函数接受多个参数：P、q、G、h、A、b，这些参数表示二次规划问题的组成部分。
    P = 0.5 * (P + P.T)  # make sure P is symmetric
# 这行代码确保矩阵 P 是对称的，通过将其与转置相加并除以2来实现。对称性是二次规划问题的要求之一。
    P = P.astype(np.double)
    q = q.astype(np.double)
# 这两行代码使用 astype 将矩阵 P 和 q 转换为 np.double 类型。这确保输入矩阵的数据类型与 cvxopt 库所需的类型一致。
    args = [matrix(P), matrix(q)]
# 这行代码创建了一个名为 args 的列表，其中包含将矩阵 P 和 q 转换为 cvxopt 库提供的 matrix 类型的对象。这些矩阵表示二次规划问题的二次项和线性项。
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
        # 这些代码检查是否提供了不等式约束条件 G 和 h。如果提供了这些条件，那么矩阵 G 和 h 将被追加到 args 列表中。
        # 此外，如果提供了等式约束条件 A 和 b，它们也会被追加到 args 列表中。矩阵 G、h、A 和 b 表示二次规划问题的不等式约束和等式约束。
    sol = cvxopt.solvers.qp(*args)
# 这行代码调用 cvxopt.solvers 模块中的 qp 函数，将 args 列表的元素作为参数传递给该函数。该函数解决了由输入矩阵定义的二次规划问题。
    return np.array(sol['x']).reshape((P.shape[1],))
# 这行代码从 sol 字典中获取解 x。然后，将解转换为 NumPy 数组，并调整其形状，使其具有与矩阵 P 相同的列数。调整形状后的解作为 cvxopt_solve_qp 函数的输出返回。

def setup_qp_and_solve(vec):
    # use cvxopt to solve QP
    P = np.dot(vec, vec.T)
    # 这行代码计算向量vec的外积，得到矩阵P。矩阵P用于定义二次规划问题的二次项。

    n = P.shape[0]
    q = np.zeros(n)
# 这两行代码获取矩阵P的维度，并创建一个长度为n的零向量q。向量q用于定义二次规划问题的线性项。

    G = - np.eye(n)
    h = np.zeros(n)
# 这两行代码创建一个维度为n×n的单位矩阵的负数G和一个长度为n的零向量h。矩阵G和向量h用于定义二次规划问题的不等式约束。

    A = np.ones((1, n))
    b = np.ones(1)
# 这两行代码创建一个维度为1×n的全1矩阵A和一个值为1的标量b。矩阵A和向量b用于定义二次规划问题的等式约束。

    cvxopt.solvers.options['show_progress'] = False
# 这行代码将cvxopt求解器的显示进度选项设置为False，以禁止显示求解进度。

    sol = cvxopt_solve_qp(P, q, G, h, A, b)
    return sol
# 这行代码调用之前定义的cvxopt_solve_qp函数，传递矩阵P、向量q、矩阵G、向量h、矩阵A和向量b作为参数，求解二次规划问题。最后，将求解得到的结果作为函数的输出返回。

def get_MGDA_d(grads, device):
    """ calculate the gradient direction for FedMGDA """

    vec = grads
    sol = setup_qp_and_solve(vec.cpu().detach().numpy())  # using CVX to solve the QP problem
# 这些代码将输入的梯度 grads 赋值给向量 vec。然后，调用 setup_qp_and_solve 函数使用 CVX 解决器求解二次规划问题，得到解 sol。

    sol = torch.from_numpy(sol).to(device)
    # print('sol: ', sol)
    d = torch.matmul(sol, grads)
# 解 sol 被转换为 PyTorch 张量，并移动到指定的设备上。然后，使用 torch.matmul 函数将解 sol 与梯度 grads 做矩阵乘法，得到梯度方向 d。

    # check descent direction
    descent_flag = 1
    c = - (grads @ d)
    if not torch.all(c <= 1e-6):
        descent_flag = 0
# 这些代码用于检查梯度方向是否为下降方向。首先，计算标量 c，其值为梯度 grads 与方向 d 的内积取负。
# 然后，通过比较 c 是否小于等于 1e-6，判断是否为下降方向。如果存在任何 c 不满足条件，则将下降标志 descent_flag 设置为 0，表示不是下降方向。
    return d, descent_flag
# 函数返回梯度方向 d 和下降标志 descent_flag

def quadprog(P, q, G, h, A, b):
# 代码定义了一个名为 quadprog 的函数，用于使用 cvxopt 库解决二次规划问题。函数接受多个参数：P、q、G、h、A、b，这些参数表示二次规划问题的组成部分。
    P = cvxopt.matrix(P.tolist())
    q = cvxopt.matrix(q.tolist(), tc='d')
    G = cvxopt.matrix(G.tolist())
    h = cvxopt.matrix(h.tolist())
    A = cvxopt.matrix(A.tolist())
    b = cvxopt.matrix(b.tolist(), tc='d')
# 代码将输入的矩阵和向量转换为 cvxopt 库所需的 matrix 类型。通过调用 cvxopt.matrix 函数并传递相应的参数，将输入转换为 cvxopt 的矩阵类型。
# tc='d' 参数用于指定转换后的矩阵的数据类型为双精度浮点型。
    cvxopt.solvers.options['show_progress'] = False
# cvxopt.solvers.options['show_progress'] = False 用于将求解器的显示进度选项设置为 False，以禁止显示求解进度。
    sol = cvxopt.solvers.qp(P, q.T, G.T, h.T, A.T, b)
# cvxopt.solvers.qp 函数调用 cvxopt 求解器，传递转换后的矩阵和向量作为参数，求解二次规划问题。
    return np.array(sol['x'])
# 将求解得到的结果从 sol 字典中提取出来，并将其转换为 NumPy 数组，作为函数的输出返回。

def setup_qp_and_solve_for_mgdaplus(vec, epsilon, lambda0):
    # 代码定义了一个名为 setup_qp_and_solve_for_mgdaplus 的函数，用于在 MGDA+ 算法中使用 cvxopt 库解决带边界约束的二次规划问题。
    # 函数接受三个参数：vec、epsilon 和 lambda0，这些参数表示二次规划问题的输入。
    # use cvxopt to solve QP
    P = np.dot(vec, vec.T)
# P 是通过 vec 的外积计算得到的矩阵
    n = P.shape[0]
# n 是矩阵 P 的维度
    q = np.array([[0] for i in range(n)])
    # 代码创建了一个形状为 (n, 1) 的 NumPy 数组 q，其中元素为零。n 是矩阵 P 的维度，表示数组 q 的长度。
    # 最终创建的数组 q 是一个垂直排列的列向量，其中每个元素都为零。它的形状为 (n, 1)，表示 n 行、1 列的数组。
    # equality constraint λ∈Δ
    A = np.ones(n).T
    b = np.array([1])
    # 这里创建了一个名为 A 的数组，它是一个形状为 (1, n) 的行向量，其中的每个元素都是 1。
    # np.ones(n) 创建了一个长度为 n 的数组，其中的每个元素都是 1。
    # .T 是转置操作，将行向量转换为列向量。
    # b 是一个长度为 1 的数组，只包含一个元素 1。

    # boundary
    lb = np.array([max(0, lambda0[i] - epsilon) for i in range(n)])
    ub = np.array([min(1, lambda0[i] + epsilon) for i in range(n)])
    # 这里创建了两个数组 lb 和 ub，分别表示下界和上界。
    # lb 是一个长度为 n 的数组，其每个元素都是通过计算 max(0, lambda0[i] - epsilon) 得到的。
    # ub 是一个长度为 n 的数组，其每个元素都是通过计算 min(1, lambda0[i] + epsilon) 得到的。

    G = np.zeros((2 * n, n))
    for i in range(n):
        G[i][i] = -1
        G[n + i][i] = 1
    # 这里创建了一个形状为 (2n, n) 的二维数组 G，所有元素初始化为零。
    # 使用循环遍历 n，并将对角线上的元素设置为 -1，将对角线下方的元素设置为 1。

    h = np.zeros((2 * n, 1))
    for i in range(n):
        h[i] = -lb[i]
        h[n + i] = ub[i]
    # 这里创建了一个形状为 (2n, 1) 的列向量 h，所有元素初始化为零。
    # 使用循环遍历 n，并将 h 的前 n 个元素设置为 -lb[i]，将后 n 个元素设置为 ub[i]。

    sol = quadprog(P, q, G, h, A, b).reshape(-1)
# 这里调用了 quadprog 函数，传递矩阵和向量作为参数，解决带边界约束的二次规划问题。
    # P 是二次规划问题的二次项系数矩阵。
    # q 是二次规划问题的线性项向量。
    # G 是二次规划问题的不等式约束矩阵。
    # h 是二次规划问题的不等式约束向量。
    # A 是二次规划问题的等式约束矩阵。
    # b 是二次规划问题的等式约束向量。

    return sol
# # 最后，将求解得到的结果进行形状重塑，并将其作为函数的输出返回。

def get_d_mgdaplus_d(grads, epsilon, lambda0, device):
    # 代码定义了一个名为 get_d_mgdaplus_d 的函数，用于计算 FedMGDA+ 算法中的梯度方向和下降标志。
    """ calculate the gradient direction for FedMGDA+ """

    vec = grads
    #  vec 就是传递给函数的 grads 参数，表示梯度向量。
    sol = setup_qp_and_solve_for_mgdaplus(vec.cpu().detach().numpy(), epsilon, lambda0)
    # setup_qp_and_solve_for_mgdaplus 函数用于设置带边界约束的二次规划问题，并使用 quadprog 函数解决该问题，返回求解的结果 sol。
    # print('sol: ', sol)

    sol = torch.from_numpy(sol).to(device)
    # 代码将求解结果 sol 转换为 PyTorch 张量，并将其移动到指定的设备上。
    d = torch.matmul(sol, grads)
    # 计算梯度方向 d，通过将 sol 与 grads 做矩阵乘法得到。

    # check descent direction
    descent_flag = 1
    c = -(grads @ d)
    if not torch.all(c <= 1e-5):
        descent_flag = 0
    # 这部分代码用于检查梯度方向是否是下降方向。
    # 首先，计算一个值 c，它是梯度向量 grads 与梯度方向 d 的内积的相反数。
    # 然后，通过检查是否存在 c 中的任何元素大于 1e-5（即大于一个小的阈值），来判断梯度方向是否是下降方向。
    # 如果存在大于阈值的元素，则将下降标志 descent_flag 设置为 0，表示梯度方向不是下降方向。

    return d, descent_flag
# 函数返回梯度方向 d 和下降标志 descent_flag。

def get_FedFV_d(grads, value, alpha, device):
    # 代码定义了一个名为 get_FedFV_d 的函数，用于计算 FedFV 算法中的梯度方向和下降标志。

    grads = [grads[i, :] for i in range(grads.shape[0])]
# 将 grads 转换为一个 Python 列表，其中每个元素都是 grads 中的一行。

    # project grads
    # 用于对梯度进行排序和投影。
    order_grads = copy.deepcopy(grads)
    # 代码创建了 grads 的一个深拷贝，将其赋值给 order_grads。这样做是为了保留原始的梯度值，在后续的投影过程中进行修改。
    order = [_ for _ in range(len(order_grads))]
    # 代码创建了一个列表 order，其中包含了从 0 到 len(order_grads)-1 的整数。该列表用于表示客户端的顺序，后续的排序过程将根据客户端的损失值进行调整。

    # sort client gradients according to their losses in ascending orders
    # 用于根据客户端的损失值对梯度进行排序。
    tmp = sorted(list(zip(value, order)), key=lambda x: x[0])
    # zip(value, order)：代码将 value 和 order 两个列表进行打包，得到一个包含元组的列表。每个元组包含了相应位置上的 value 和 order 的元素。
    # sorted(..., key=lambda x: x[0])：这行代码对打包后的列表进行排序，根据元组中的第一个元素（即 value）进行排序。
    # key=lambda x: x[0] 表示使用元组的第一个元素作为排序的依据。tmp：将排序后的列表赋值给 tmp 变量。
    order = [x[1] for x in tmp]
    # order = [x[1] for x in tmp]：这行代码从排序后的列表 tmp 中提取出元组的第二个元素（即 order），并重新构建一个新的列表赋值给 order 变量。



    # keep the original direction for clients with the αm largest losses
    keep_original = []
    # keep_original = []：首先，将保留原始方向的客户端索引列表 keep_original 初始化为空列表。
    if alpha > 0:
        # if alpha > 0:：这行代码检查 alpha 是否大于 0，即是否需要保留原始方向的客户端。
        keep_original = order[math.ceil((len(order) - 1) * (1 - alpha)):]
        # keep_original = order[math.ceil((len(order) - 1) * (1 - alpha)):]：如果需要保留原始方向的客户端
        # 则根据指定的比例参数 alpha 计算应该保留原始方向的客户端数量。具体地，使用 (len(order) - 1) * (1 - alpha) 计算保留的客户端数量（向上取整）
        # 然后从排序后的客户端列表 order 中获取这些客户端的索引，并赋值给 keep_original 变量。
        # 这行代码确定要保留原始方向的客户端的索引列表。
        # alpha 是一个介于 0 和 1 之间的值，用于确定要保留原始方向的客户端的比例。

    # calculate g_locals[j].L2_norm_square() first to be more faster.
    g_locals_L2_norm_square_list = []
    # g_locals_L2_norm_square_list = []：首先，创建一个空列表 g_locals_L2_norm_square_list，用于存储每个客户端梯度的 L2 范数的平方。
    for g_local in grads:
        # for g_local in grads:：遍历梯度列表 grads 中的每个客户端梯度，并将当前梯度赋值给变量 g_local。
        g_locals_L2_norm_square_list.append(torch.norm(g_local)**2)
        # g_locals_L2_norm_square_list.append(torch.norm(g_local)**2)：计算当前梯度 g_local 的 L2 范数的平方，并将结果添加到 g_locals_L2_norm_square_list 列表中。
# 这部分代码计算每个客户端梯度的 L2 范数的平方，并将其存储在 g_locals_L2_norm_square_list 列表中。

    # mitigate internal conflicts by iteratively projecting gradients
    for i in range(len(order_grads)):
        # for i in range(len(order_grads)):：这个循环遍历 order_grads 列表的索引，表示当前要更新的梯度的索引。len(order_grads) 是列表的长度。
        if i in keep_original:
            # if i in keep_original:：如果当前梯度的索引 i 在 keep_original 列表中（即需要保留原始方向的客户端），则跳过当前循环，继续下一个迭代。
            continue
        for j in order:
            # for j in order:：这个嵌套循环遍历 order 列表，表示要与当前梯度进行投影操作的其他梯度的索引。
            if j == i:
                # 如果当前梯度的索引 j 与外层循环的索引 i 相同，说明是同一个梯度，跳过当前循环，继续下一个迭代。
                continue
            else:
                # calculate the dot of gi and gj
                dot = grads[j] @ order_grads[i]
                # dot = grads[j] @ order_grads[i]：计算当前梯度 grads[j] 与要更新的梯度 order_grads[i] 的点积。
                if dot < 0:
                    # if dot < 0:：如果点积 dot 小于 0，表示当前梯度order_grads[i] 与 grads[j] 方向相反，需要进行投影操作。
                    order_grads[i] = order_grads[i] - dot / g_locals_L2_norm_square_list[j] * grads[j]
                    # order_grads[i] = order_grads[i] - dot / g_locals_L2_norm_square_list[j] * grads[j]：
                    # 根据投影操作的公式，更新梯度 order_grads[i]，将其与 grads[j] 投影到一起。

# 这部分代码对梯度进行投影，以解决梯度之间的内部冲突。
    # 首先，遍历排序后的客户端列表 order_grads。
    # 如果当前客户端不在 keep_original 列表中，则遍历 order 列表，计算当前客户端梯度与其他客户端梯度的点积，并根据一定规则对当前客户端梯度进行调整。

    # aggregate projected grads
    weights = torch.Tensor([1 / len(order_grads)] * len(order_grads)).to(device)
    # 创建一个张量 weights，其元素都是 1 / len(order_grads)，表示每个客户端梯度的权重。len(order_grads) 是客户端梯度的数量。.to(device) 是将张量移动到指定的设备上
    gt = weights @ torch.stack([order_grads[i] for i in range(len(order_grads))])
    # torch.stack([order_grads[i] for i in range(len(order_grads))])：将 order_grads 列表中的梯度张量按顺序堆叠起来，形成一个新的张量。
    # 通过列表推导式，我们遍历 order_grads 列表的索引，并将每个索引对应的梯度张量取出。这样生成的张量将作为梯度张量的矩阵表示，每一行对应一个客户端的梯度。
    # gt = weights @ ...：使用矩阵乘法运算符 @，计算权重矩阵 weights 与梯度矩阵的乘积，得到加权平均梯度 gt。

# 这行代码计算加权平均后的梯度 gt，其中每个客户端梯度按照权重 weights 进行加权平均。


    # ||gt||=||1/m*Σgi||
    gnorm = torch.norm(weights @ torch.stack([grads[i] for i in range(len(grads))]))
    # 这行代码计算加权平均后的梯度的 L2 范数 gnorm，其中每个客户端梯度按照权重 weights 进行加权平均。

    # check descent direction
    grads = torch.stack(grads)
    c = -(grads @ gt)
    descent_flag = 1
    if not torch.all(c <= 1e-5):
        descent_flag = 0
    # 这部分代码用于检查梯度方向是否是下降方向。
    # 首先，计算一个值 c，它是梯度向量 grads 与梯度方向 gt 的内积的相反数。
    # 然后，通过检查是否存在 c 中的任何元素大于 1e-5（即大于一个小的阈值），来判断梯度方向是否是下降方向。
    # 如果存在大于阈值的元素，则将下降标志 descent_flag 设置为 0，表示梯度方向不是下降方向。

    return gt, descent_flag
# 函数返回梯度方方 gt 和下降标志 descent_flag

def get_FedMGDP_d(grads, value, add_grads, alpha, fair_guidance_vec, force_active, device):
    # 用于计算 FedMGDP（联邦多目标优化）的梯度方向。
    """ calculate the gradient direction for FedMGDP """

    fair_grad = None
    # fair_grad = None：初始化 fair_grad 为 None。在后续的计算中，fair_grad 将被用来存储一个特定的梯度值。

    value_norm = torch.norm(value)
    # value_norm = torch.norm(value)：计算 value 张量的 L2 范数，将结果赋值给 value_norm。
    norm_values = value/value_norm
    # norm_values = value/value_norm：将 value 张量的每个元素除以 value_norm，得到归一化的 value 张量。
    fair_guidance_vec /= torch.norm(fair_guidance_vec)
    # fair_guidance_vec /= torch.norm(fair_guidance_vec)：将 fair_guidance_vec 张量除以其自身的 L2 范数，将其归一化为单位向量。

    m = grads.shape[0]
    # m = grads.shape[0]：获取 grads 张量的第一个维度的大小，即客户端梯度的数量。
    weights = torch.Tensor([1 / m] * m).to(device)
    # weights = torch.Tensor([1 / m] * m).to(device)：创建一个张量 weights，其中的元素都是 1 / m，表示每个客户端梯度的权重。
    # m 是客户端梯度的数量。.to(device) 是将张量移动到指定的设备上（例如 GPU）。
    g_norm = torch.norm(weights @ grads)
    # g_norm = torch.norm(weights @ grads)：将权重矩阵 weights 与客户端梯度矩阵 grads 的乘积作为一个向量，然后计算该向量的 L2 范数，将结果赋值给 g_norm。


    # new check active constraints
    cos = float(norm_values @ fair_guidance_vec)
    # cos = float(norm_values @ fair_guidance_vec)：计算 norm_values 和 fair_guidance_vec 的点积，并将结果赋值给 cos。
    cos = min(1, cos)  # prevent float error
    cos = max(-1, cos)  # prevent float error
    # #  cos = min(1, cos) 和 cos = max(-1, cos)：将 cos 的值限制在区间 [-1, 1] 内，以防止浮点数误差。
    bias = np.arccos(cos) / np.pi * 180
    # bias = np.arccos(cos) / np.pi * 180：计算 cos 的反余弦值，并将结果转换为角度制。
    # print('bias:', bias)
    pref_active_flag = (bias > alpha) | force_active
    # pref_active_flag = (bias > alpha) | force_active：判断 bias 是否大于给定的阈值 alpha 或者 force_active 是否为真，将结果赋值给 pref_active_flag。
    if not pref_active_flag:
        vec = grads
        pref_active_flag = 0
    # if not pref_active_flag:：如果 pref_active_flag 为假，即 bias 小于等于 alpha 并且 force_active 为假，则将 vec 设置为 grads。
    else:
        pref_active_flag = 1
        h_vec = (norm_values - fair_guidance_vec / torch.norm(fair_guidance_vec)).reshape(1, -1)
        # else:：如果 pref_active_flag 为真，则进行以下操作：
        # pref_active_flag = 1：将 pref_active_flag 设置为 1。
        # h_vec = (norm_values - fair_guidance_vec / torch.norm(fair_guidance_vec)).reshape(1, -1)：
        # 计算一个向量 h_vec，其中的元素为 norm_values 减去 fair_guidance_vec 的归一化值。通过 reshape 将其转换为形状为 (1, -1) 的张量。
        h_vec /= torch.norm(h_vec)
        # h_vec /= torch.norm(h_vec)：将 h_vec 进行归一化，使其成为单位向量。
        fair_grad = h_vec @ grads
        # fair_grad = h_vec @ grads：计算 h_vec 和 grads 的矩阵乘积，将结果赋值给 fair_grad。
        vec = torch.cat((grads, fair_grad))
        # vec = torch.cat((grads, fair_grad))：将 grads 和 fair_grad 连接起来，形成一个新的张量 vec。

    if add_grads is not None:
        vec = torch.vstack([vec, add_grads])
    # if add_grads is not None:：如果 add_grads 不为 None，则进行以下操作：
    # vec = torch.vstack([vec, add_grads])：将 vec 和 add_grads 在垂直方向上堆叠起来，形成一个新的张量 vec。


    sol = setup_qp_and_solve(vec.cpu().detach().numpy())  # using CVX to solve the QP problem
    # sol = setup_qp_and_solve(vec.cpu().detach().numpy())：调用 setup_qp_and_solve 函数，将 vec 的 NumPy 数组表示作为参数传递给该函数。
    # 该函数使用 CVX（凸优化库）来解决 QP（二次规划）问题。
    sol = torch.from_numpy(sol).to(device)
    # sol = torch.from_numpy(sol).to(device)：将 CVX 求解得到的结果转换为 PyTorch 张量，并将其移动到指定的设备上。
    d = sol @ vec  # get common gradient
    # d = sol @ vec：计算 sol 和 vec 的矩阵乘积，得到共享梯度 d。

    # check constraints
    descent_flag = 1
    c = - (vec @ d)
    if not torch.all(c <= 1e-5):
        descent_flag = 0
    # if not torch.all(c <= 1e-5):：如果存在任何一个约束条件 c 不满足小于等于 1e-5，则执行以下操作：
    # descent_flag = 0：将 descent_flag 设置为 0，表示不满足下降条件。

    return d, vec, pref_active_flag, fair_grad, descent_flag
    # 返回结果 d（共享梯度）、vec（用于计算共享梯度的向量）、pref_active_flag（指示是否满足优先约束条件的标志）
    # fair_grad（用于公平性优化的梯度）、descent_flag（指示是否满足下降条件的标志）。