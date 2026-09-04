# PCSC

Pareto-Coverage Stress Coevolution（PCSC，Pareto 脆弱性覆盖压力协同进化）的最小可运行实现。

该仓库聚焦 PCSC 的核心机制：

- 基于历史收益块构造有界压力场景；
- 计算投资组合级归一化脆弱性；
- 以边际集合覆盖贡献选择容量受限的场景档案；
- 在 Pareto 投资组合种群与压力场景档案之间执行双向反馈；
- 使用带单资产权重上限和单边换手率上限的权重修复。

仓库不包含 A 股原始数据、处理后数据、论文实验结果或论文手稿。`examples/run_pcsc_synthetic.py` 使用合成收益，仅用于验证代码能够执行，不能视为论文实验复现。

## 环境

建议使用 Python 3.11 或 3.12。在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 快速运行

```bash
python -m examples.run_pcsc_synthetic
```

脚本将输出 Pareto 解数量、场景档案大小、档案更新次数、选中解的两个目标值，以及权重和约束检查结果。

## 运行测试

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## 使用自己的收益数据

核心入口为 `src.pcsc.evolution.run_pcsc`。其第一个参数应为形状为 `T x n` 的有限日收益矩阵，其中 `T` 是回看窗口长度，`n` 是资产数量。主要输入还包括：

- `upper`：单资产权重上限；
- `previous_weights`：再平衡前权重；
- `max_one_way_turnover`：单边换手率上限；
- `scenario_bounds`：压力块长度、市场冲击倍率和残差压缩参数范围；
- `archive_size` 与 `scenario_candidate_size`：场景档案容量和候选数；
- `population_size`、`generations` 与 `scenario_update_interval`：进化预算；
- `seed`：随机种子。

可参考 `examples/run_pcsc_synthetic.py` 中的完整调用。真实回测必须在每个再平衡点仅使用当时及以前的数据，并在外部回测层处理持有期收益和交易成本核算。

## 目录

```text
src/pcsc/        PCSC 进化、覆盖、档案与路径损失
src/scenarios/   压力场景表示、生成、变异与修复
src/portfolio/   投资组合目标和权重约束修复
src/baselines/   PCSC 复用的归一化理想点选择及 Mean--CVaR 基础实现
examples/        无数据依赖的最小运行示例
tests/           核心机制单元测试
```
