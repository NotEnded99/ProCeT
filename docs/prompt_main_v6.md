# Prompt

基于 `New_repair/main_v4.py` 写一个 `New_repair/main_v6.py`，在保持其他代码不变的前提下，落地以下三个改进：

---

## Idea 1: 选择性保护（Top-N V_safe）

`compute_jacobian_rs()` 只在最脆弱的 N 个 V_safe 区域上计算，N 通过 `compute_simplex_bound_batch` 计算每个单纯形的 margin（= min_L - cbf_margin），选 margin 最小的 N 个。

新增 CLI 参数：`--top-n-protect`（默认 500）。

---

## Idea 2: 优先级修复

违规区域按以下优先级分批修复，每批全部修复完再进入下一批：

| 优先级 | 区域类型 |
|--------|----------|
| 1（最高）| F_h_positive_in_unsafe |
| 2 | F_safe_cbf_violation |
| 3 | F_depth_limit_reached |
| 4（最低）| F_unsafe_cannot_split |

同优先级内按 violation severity 排序（更严重的先修）。

---

## Idea 3: 每轮修复上限 M

每轮最多修复 M 个违规区域，超出则按优先级截断。M 通过 CLI 参数 `--max-repair-per-iter`（默认 100）控制。

---

## 输出要求

- 所有新增逻辑集中在 main_v6.py 中
- 不修改任何 geometry_module*.py 和 optimizer_module*.py
- 直接在 selected 后的区域上调用 `compute_jacobian_rs()` 和 `compute_repair_loss_and_grad()`
- CLI 参数新增 `--top-n-protect` 和 `--max-repair-per-iter`
- 运行流程：加载区域 → 按优先级+severity排序 → 取前M个 → 选择Top-N V_safe → 计算J_RS → 计算损失 → QP更新 → 验证 → 保存（与v4流程一致）
