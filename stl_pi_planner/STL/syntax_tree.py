from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Union
from .predicate import LinearPredicate
import numpy as np
from collections import deque
from typing import Tuple

Interval = Tuple[float, float]

def _sliding_max_1d(values: np.ndarray, window: int) -> np.ndarray:
    """
    一维数组上的 O(N) 滑动窗口最大值。
    返回数组 res，res[i] = max(values[i-window+1 : i+1]),
    对于 i < window-1，res[i] 设为 -inf。
    """
    n = values.shape[0]
    res = np.full(n, -np.inf, dtype=float)
    dq: deque[int] = deque()

    for i, v in enumerate(values):
        # 维护一个递减队列
        while dq and values[dq[-1]] <= v:
            dq.pop()
        dq.append(i)

        # 移除窗口外的元素
        if dq[0] <= i - window:
            dq.popleft()

        # 从第 window-1 个元素开始产生有效输出
        if i >= window - 1:
            res[i] = values[dq[0]]

    return res


def sliding_max_intervals(
    vals: np.ndarray,
    I: Tuple[int, int],
) -> np.ndarray:
    """
    对区间序列 vals 做 SlidingMax，近似论文中的 SlidingMax(worklist[vϕ], I)。

    参数
    ----
    vals : np.ndarray
        shape = (N, 2)，vals[k, 0] = 下界 L_k，vals[k, 1] = 上界 U_k。
    I : (a, b)
        离散时间区间，表示对每个 t 计算 max_{j ∈ [t+a, t+b]} vals[j]。

    返回
    ----
    out : np.ndarray
        shape = (N, 2)，out[t] = 该 t 处窗口内区间的“最大值”：
          - 下界 = 窗口内所有下界的最大值
          - 上界 = 窗口内所有上界的最大值
        对于窗口完全越界的 t，out[t] 保持为 [-inf, -inf]。
    """
    assert vals.ndim == 2 and vals.shape[1] == 2, "vals 需要是 (N,2) 的数组"
    a, b = I
    assert 0 <= a <= b, "这里只考虑离散非负时间区间 0 <= a <= b"

    N = vals.shape[0]
    L = vals[:, 0]
    U = vals[:, 1]

    window = b - a + 1

    # 先对整个序列做“一般意义上”的滑动最大值，
    # 得到的是：tmpL[i] = max(L[i-window+1 : i+1])
    tmpL = _sliding_max_1d(L, window)
    tmpU = _sliding_max_1d(U, window)

    # 然后根据 t 与 i 的关系 (i = t + b) 把结果移位到 t 上：
    outL = np.full(N, -np.inf, dtype=float)
    outU = np.full(N, -np.inf, dtype=float)

    # 有效的 t 需要满足 t + b < N  =>  t <= N-1-b
    t_max = N - 1 - b
    for t in range(0, max(t_max + 1, 0)):
        i = t + b
        outL[t] = tmpL[i]
        outU[t] = tmpU[i]

    out = np.stack([outL, outU], axis=1)
    return out

def sliding_min_intervals(
    vals: np.ndarray,
    I: Tuple[int, int],
) -> np.ndarray:
    """
    类似 sliding_max_intervals，只是把窗口上界改成“最小值”。

    利用区间取负 + 最大值实现：
      min_I [L,U]  等价于  - max_I [-U, -L]
    """
    assert vals.ndim == 2 and vals.shape[1] == 2, "vals 需要是 (N,2) 的数组"
    # 先把 [L, U] 变成 [-U, -L]
    vals_neg = np.empty_like(vals)
    vals_neg[:, 0] = -vals[:, 1]
    vals_neg[:, 1] = -vals[:, 0]

    tmp = sliding_max_intervals(vals_neg, I)

    # 再把 [-U', -L'] 变回 [L', U']
    out = np.empty_like(tmp)
    out[:, 0] = -tmp[:, 1]
    out[:, 1] = -tmp[:, 0]
    return out




@dataclass
class STLNode:
    """
    一个 STL 语法树结点。
    kind:  'pred', 'not', 'and', 'or', 'G', 'F', 'U', 'R'
    interval: 仅对时序算子有效，例如 [0, a], [b, c]
    children: 子结点列表
    payload: 叶子结点时，存放谓词对象（比如 LinearPredicate / NonlinearPredicate）
    """
    kind: str
    interval: Optional[Interval] = None
    children: List["STLNode"] = field(default_factory=list)
    payload: Optional[Any] = None
    name: Optional[str] = None

    # 在线监控算法需要的属性：hor(v), worklist
    horizon: Optional[Interval] = None  # [t_min, t_max]
    worklist: List[Tuple[float, float]] = field(default_factory=list)

    # --------- 语法糖：逻辑运算符重载 ---------
    def __invert__(self) -> "STLNode":
        """~phi 代表 ¬phi"""
        return STLNode(kind="not", children=[self])

    def __and__(self, other: "STLNode") -> "STLNode":
        """phi1 & phi2 代表 phi1 ∧ phi2"""
        return STLNode(kind="and", children=[self, other])

    def __or__(self, other: "STLNode") -> "STLNode":
        """phi1 | phi2 代表 phi1 ∨ phi2"""
        return STLNode(kind="or", children=[self, other])

    # --------- 语法糖：时序运算符接口 ---------
    def always(self, interval: Interval) -> "STLNode":
        """G_I phi"""
        return STLNode(kind="G", interval=interval, children=[self])

    def eventually(self, interval: Interval) -> "STLNode":
        """F_I phi"""
        return STLNode(kind="F", interval=interval, children=[self])

    def until(self, other: "STLNode", interval: Interval) -> "STLNode":
        """phi U_I psi"""
        return STLNode(kind="U", interval=interval, children=[self, other])

    def release(self, other: "STLNode", interval: Interval) -> "STLNode":
        """phi R_I psi"""
        return STLNode(kind="R", interval=interval, children=[self, other])

    # --------- 一点辅助函数 ---------
    def is_predicate(self) -> bool:
        return self.kind == "pred"

    def pretty_label(self) -> str:
        """返回类似图里的结点标签（不含 hor(v)）"""
        if self.kind == "pred":
            return self.name or "μ"
        if self.kind == "not":
            return "¬"
        if self.kind == "and":
            return "∧"
        if self.kind == "or":
            return "∨"
        if self.kind == "G":
            return f"□[{self.interval[0]}, {self.interval[1]}]"
        if self.kind == "F":
            return f"◇[{self.interval[0]}, {self.interval[1]}]"
        if self.kind == "U":
            return f"U[{self.interval[0]}, {self.interval[1]}]"
        if self.kind == "R":
            return f"R[{self.interval[0]}, {self.interval[1]}]"
        return self.kind

    def __str__(self) -> str:
        h_str = ""
        if self.horizon is not None:
            h_str = f" hor={self.horizon}"
        return f"{self.pretty_label()}{h_str}"


class STLSyntaxTree:
    """
    STL 语法树包装类，用于：
      - 保存根结点 root
      - 计算/存储 hor(v)
      - 后续挂 RoSI 算法
    """

    def __init__(self, root: STLNode):
        self.root = root
        self._assign_horizons()

        # 全局时域上界（根据 hor(v) 计算）
        self._T_H = self._compute_global_horizon()

        # 预先收集后序遍历的节点列表，方便自底向上更新
        self._post_order_nodes: List[STLNode] = []
        self._build_post_order()

        # 存当前已看到的输出前缀 y[0:k]（形状 (p, K)）
        self._y: Optional[np.ndarray] = None
        self._signal_dim: Optional[int] = None
        self.K: Optional[int] = None      # 总规划长度（可在 reset 时给）
        self._root_interval: Interval = (-np.inf, np.inf)

                # 按后序遍历保存所有结点（叶子在前，根在最后），方便每次 update 自底向上更新
        self._nodes_postorder: List[STLNode] = []

        def dfs_post(n: STLNode):
            for ch in n.children:
                dfs_post(ch)
            self._nodes_postorder.append(n)

        dfs_post(self.root)

    # --------- 工厂方法：把 predicate 包成叶子结点 ---------
    @staticmethod
    def from_predicate(predicate_obj: Any, name: Optional[str] = None) -> STLNode:
        """
        把已有的 predicate（比如 LinearPredicate/NonlinearPredicate）
        包成一个 STLNode(kind='pred').
        """
        return STLNode(kind="pred", payload=predicate_obj, name=name)

    # --------- 计算 hor(v) ---------
    def _assign_horizons(self) -> None:
        """根据式 (4.1) 递归给所有结点赋 horizon"""

        def add_interval(i1: Interval, i2: Interval) -> Interval:
            return (i1[0] + i2[0], i1[1] + i2[1])

        def dfs(node: STLNode, parent: Optional[STLNode]) -> None:
            if parent is None:
                # 根结点：hor(root) = [0, 0]
                node.horizon = (0.0, 0.0)
            else:
                # 如果父结点是时序算子 (G/F/U/R)，则 hor(v) = I ⊕ hor(parent)
                if parent.kind in {"G", "F", "U", "R"} and parent.interval is not None:
                    node.horizon = add_interval(parent.interval, parent.horizon)
                else:
                    # 否则 hor(v) = hor(parent)
                    node.horizon = parent.horizon

            for child in node.children:
                dfs(child, node)

        dfs(self.root, parent=None)

    # --------- 计算全局时域上界 T_H ---------
    def _compute_global_horizon(self) -> int:
        max_h = 0.0

        def dfs(node: STLNode):
            nonlocal max_h
            if node.horizon is not None:
                max_h = max(max_h, node.horizon[1])
            for ch in node.children:
                dfs(ch)

        dfs(self.root)
        return int(np.ceil(max_h))

    
    def _build_post_order(self):
        nodes: List[STLNode] = []

        def dfs(node: STLNode):
            for ch in node.children:
                dfs(ch)
            nodes.append(node)

        dfs(self.root)
        self._post_order_nodes = nodes


    # --------- 打印树结构（调试用） ---------
    def print_tree(self) -> None:
        """
        简单的缩进打印，方便检查结构和 hor(v)，
        以后真要画成图可以再接 graphviz / tikz。
        """

        def dfs(node: STLNode, depth: int) -> None:
            indent = "  " * depth
            print(f"{indent}- {node}")
            for ch in node.children:
                dfs(ch, depth + 1)

        dfs(self.root, 0)
    
    # ---- RoSI 的区间运算工具 ----
    def _interval_neg(self, I: Interval) -> Interval:
        L, U = I
        return (-U, -L)
    
    def _interval_array_neg(self, arr: np.ndarray) -> np.ndarray:
        """
        对一组区间做取负运算：
        arr[i] = [L_i, U_i]  ->  out[i] = [-U_i, -L_i]
        """
        out = np.empty_like(arr)
        L = arr[:, 0]
        U = arr[:, 1]
        out[:, 0] = -U
        out[:, 1] = -L
        return out


    def _interval_min(self, I1: Interval, I2: Interval) -> Interval:
        return (min(I1[0], I2[0]), min(I1[1], I2[1]))

    def _interval_max(self, I1: Interval, I2: Interval) -> Interval:
        return (max(I1[0], I2[0]), max(I1[1], I2[1]))

    def _interval_inf(self, intervals: List[Interval]) -> Interval:
        Ls = [I[0] for I in intervals]
        Us = [I[1] for I in intervals]
        return (min(Ls), min(Us))

    def _interval_sup(self, intervals: List[Interval]) -> Interval:
        Ls = [I[0] for I in intervals]
        Us = [I[1] for I in intervals]
        return (max(Ls), max(Us))


    # def reset(self, K: int, signal_dim: Optional[int] = None):
    #     """
    #     初始化 RoSI 监控器，包含 worklist 初始化。
    #     """
    #     self.K = K
    #     self._signal_dim = signal_dim
    #     self._y = None
    #     self._root_interval = (-np.inf, np.inf)

    #     # ==== 初始化 worklist ====
    #     def dfs(node: STLNode):
    #         assert node.horizon is not None
    #         h_min, h_max = node.horizon     # float
    #         L = int(np.ceil(h_max - h_min)) + 1

    #         # 初始化 L 个区间，每个都是 [-inf, inf]
    #         node.worklist = [(-np.inf, np.inf) for _ in range(L)]

    #         # 递归子节点
    #         for ch in node.children:
    #             dfs(ch)

    #     dfs(self.root)
    
    # def reset(self, K: int, signal_dim: Optional[int] = None):
    #     """
    #     重置 monitor 状态。
    #     K: 离散时间步数（规划/仿真长度）
    #     signal_dim: 输出维度 p，若为 None，则在第一次 update 时从 y_k 推断
    #     """
    #     self.K = K
    #     self._signal_dim = signal_dim
    #     self._y = None
    #     self._root_interval = (-np.inf, np.inf)

    #     # 为每个节点分配一个 (K,2) 的 worklist，并初始化为 [-inf, +inf]
    #     def dfs(node: STLNode):
    #         wl = np.empty((self.K, 2), dtype=float)
    #         wl[:, 0] = -np.inf   # 下界
    #         wl[:, 1] = +np.inf   # 上界
    #         node.worklist = wl

    #         for ch in node.children:
    #             dfs(ch)

    #     dfs(self.root)

    def reset(self, K: int, signal_dim: Optional[int] = None):
        """
        重置 monitor 状态。
        K: 离散时间步数（规划/仿真长度）
        signal_dim: 输出维度 p，若为 None，则在第一次 update 时从 y_k 推断
        """
        self.K = K
        self._signal_dim = signal_dim
        self._y = None
        self._root_interval = (-np.inf, np.inf)

        # 为每个结点分配一个 K×2 的 worklist，所有时间点初始化为 [-inf, +inf]
        for node in self._nodes_postorder:
            node.worklist = np.full((K, 2), (-np.inf, np.inf), dtype=float)


    def update(self, y_k: np.ndarray, k: int) -> Interval:
        """
        用当前时刻的输出 y_k 更新整棵语法树的 RoSI，并返回
        根公式在 tau=0 处的 RoSI 区间 [L, U].

        增量式做法：
          1. 先更新所有谓词结点在时间 k 的区间 [r,r]；
          2. 然后按照后序遍历，自底向上更新逻辑结点在时间 k 的区间；
          3. 对于 F/G 这样的时序算子，则对整条 time axis 做一次 SlidingMax/Min，
             复杂度 O(K)，K 很小（比如 15），在 PI 里开销也很小。
        """
        y_k = np.asarray(y_k).reshape(-1)

        # 初始化信号缓存 self._y
        if self._y is None:
            if self.K is None:
                # 如果没显式 reset，就给一个“至少够用”的长度
                self.K = max(k + 1, self._T_H + 1)
            self._signal_dim = y_k.shape[0] if self._signal_dim is None else self._signal_dim
            self._y = np.full((self._signal_dim, self.K), np.nan, dtype=float)

        assert y_k.shape[0] == self._y.shape[0], "y_k 维度和 monitor 初始化不一致"
        assert 0 <= k < self.K, "k 超出 reset 时设定的时域"

        # 写入当前时间步的观测
        self._y[:, k] = y_k

        # ---- 第一步：更新所有谓词结点在 time=k 的区间 ----
        for node in self._nodes_postorder:
            if node.kind == "pred":
                # 对 partial signal y[:, :k+1] 计算当前时刻的鲁棒度
                rob = node.payload.robustness(self._y[:, :k+1], k)
                if np.isscalar(rob):
                    r = float(rob)
                else:
                    r = float(rob[0])
                node.worklist[k, 0] = r
                node.worklist[k, 1] = r

        # ---- 第二步：自底向上更新非谓词结点 ----
        for node in self._nodes_postorder:
            if node.kind == "pred":
                continue

            # 布尔运算：只影响当前时刻 k
            if node.kind == "not":
                child = node.children[0]
                Lc, Uc = child.worklist[k]
                node.worklist[k, 0] = -Uc
                node.worklist[k, 1] = -Lc
                continue

            if node.kind == "and":
                left, right = node.children
                L1, U1 = left.worklist[k]
                L2, U2 = right.worklist[k]
                node.worklist[k, 0] = min(L1, L2)
                node.worklist[k, 1] = min(U1, U2)
                continue

            if node.kind == "or":
                left, right = node.children
                L1, U1 = left.worklist[k]
                L2, U2 = right.worklist[k]
                node.worklist[k, 0] = max(L1, L2)
                node.worklist[k, 1] = max(U1, U2)
                continue

            # 时序算子 F/G：用 SlidingMax/Min 在整个 [0, K-1] 上重新算一遍
            if node.kind in {"F", "G"}:
                child = node.children[0]
                a, b = node.interval
                a_i = int(np.floor(a))
                b_i = int(np.ceil(b))

                if node.kind == "F":
                    node.worklist[:, :] = sliding_max_intervals(
                        child.worklist, (a_i, b_i)
                    )
                else:  # "G"
                    node.worklist[:, :] = sliding_min_intervals(
                        child.worklist, (a_i, b_i)
                    )
                continue

            # 目前没实现 U/R，需要时再扩展
            raise NotImplementedError(f"Operator {node.kind} not implemented in incremental RoSI")

        # 根结点在 tau = 0 处的 RoSI 就是 worklist[0]
        L0, U0 = self.root.worklist[0]
        self._root_interval = (L0, U0)
        return self._root_interval


    def current_interval(self) -> Interval:
        """
        返回最近一次 update 后，根公式在 tau=0 处的 RoSI 区间 [L, U]
        """
        return self._root_interval


if __name__ == "__main__":
    # 例如：y > 0, x > 0（这里随便写，真实项目里用你自己定义的 a, b）
    y_pred = LinearPredicate(a=[1.0, 0.0], b=[0.0], name="y > 0")
    x_pred = LinearPredicate(a=[0.0, 1.0], b=[0.0], name="x > 0")

    # 把它们包成 STLNode 叶子
    y_gt_0 = STLSyntaxTree.from_predicate(y_pred, name="y > 0")
    x_gt_0 = STLSyntaxTree.from_predicate(x_pred, name="x > 0")

    a, b, c = 1, 2, 3  # 举例

    # 构造公式：□[0,a]( ¬(y>0) ∨ ◇[b,c](x>0) )
    phi = ((~y_gt_0) | x_gt_0.eventually((b, c))).always((0, a))

    # 生成语法树并计算 hor(v)
    tree = STLSyntaxTree(phi)

    # 打印看看结构 & horizon
    tree.print_tree()