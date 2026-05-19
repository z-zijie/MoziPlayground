# Abs PRD

## 1. Background / 背景
Abs 是基础逐元素算子，用于将输入 Tensor 的每个元素映射为绝对值。该算子在数值预处理、激活后处理、特征变换以及图算子组合中非常常见，属于 NPU 上高频基础算子之一。

本 PRD 面向 NPU ARCH `dav-2201`，定义 `Abs` 算子的功能、边界行为、输入输出约束以及验收标准。

## 2. Goal / 目标
- 实现 `Abs` 算子：输入一个 `float32` 类型 Tensor `x`，输出与输入 `shape` 相同、`dtype` 相同的 Tensor `y`。
- 满足逐元素语义：`y[i] = |x[i]|`。
- 支持任意维度 Tensor。
- 正确处理以下边界场景：
  - 正数
  - 负数
  - `+0.0`
  - `-0.0`
  - `NaN`
  - `+Inf`
  - `-Inf`
  - 空 Tensor
  - 标量 Tensor

## 3. Non-Goals / 非目标
- 不扩展到非 `float32` dtype。
- 不定义广播语义；输入为单个 Tensor 的逐元素变换，不涉及多输入对齐。
- 不定义复数、整数、布尔等其他数值类型的行为。
- 不定义梯度、反向传播或训练框架侧的自动微分实现。
- 不定义融合优化策略或与其他算子的图级重写规则。

## 4. User Scenarios / 使用场景
- 上游图中需要将特征值、权重或中间激活转换为非负值。
- 数值后处理需要保留原 Tensor 结构，只改变元素符号。
- 模型推理图中需要在 NPU 上执行基础逐元素绝对值运算。

## 5. Functional Requirements / 功能需求
- 算子输入为单个 `float32` Tensor `x`。
- 输出为单个 `float32` Tensor `y`，且：
  - `y.shape == x.shape`
  - `y.dtype == x.dtype == float32`
- 逐元素计算绝对值：
  - 对有限正数，输出其自身。
  - 对有限负数，输出其相反数。
  - 对 `+0.0`，输出 `+0.0`。
  - 对 `-0.0`，输出 `+0.0`。
  - 对 `NaN`，输出 `NaN`，并保持 NaN 语义不被规约为其他值。
  - 对 `+Inf`，输出 `+Inf`。
  - 对 `-Inf`，输出 `+Inf`。
- 支持任意 rank 的 Tensor，包括：
  - 1D、2D、3D 及更高维 Tensor
  - 标量 Tensor
  - 空 Tensor
- 对空 Tensor 的输出应保持同样的空 `shape`，且不产生额外元素。
- 不修改输入 Tensor 的内容。

## 6. Input and Output Overview / 输入输出概述
- 输入：
  - `x`: `float32` Tensor
  - 形状：任意维度，允许标量 Tensor 和空 Tensor
- 输出：
  - `y`: `float32` Tensor
  - 形状：与 `x` 完全一致
  - 值：`y[i] = |x[i]|`

示例：
- `x = [1.5, -2.0, -0.0, NaN, +Inf, -Inf]`
- `y = [1.5, 2.0, +0.0, NaN, +Inf, +Inf]`

## 7. Compatibility Requirements / 兼容性需求
- 适配 NPU ARCH `dav-2201` 的算子实现与调度要求。
- 输入输出接口需符合 Mozi/NPU 图编排的基础逐元素算子约定。
- 结果 Tensor 的 `shape` 和 `dtype` 必须与输入完全一致，便于与现有图算子链路兼容。
- 对 `NaN`、`Inf`、`-0.0` 等 IEEE 754 特殊值的处理应与 `float32` 语义一致。

## 8. Performance Expectations / 性能期望
- 该算子属于基础逐元素算子，应优先保证正确性与稳定性。
- 在 `dav-2201` 上应采用适合向量化/并行化的实现路径，避免不必要的标量逐元素软件循环。
- 本需求未提供具体吞吐、延迟或带宽指标，因此不写入数值型性能门槛。

## 9. Accuracy and Numerical Expectations / 精度期望
- 输出精度与 `float32` 绝对值语义一致。
- 对有限值，结果应与数学绝对值一致，允许正常的 `float32` 表达误差范围内的实现差异，但不得改变符号约定和特殊值类别。
- `-0.0` 必须输出为 `+0.0`。
- `NaN` 必须保持为 `NaN`。
- `+Inf` 与 `-Inf` 的输出必须为 `+Inf`。

## 10. Constraints / 约束
- 仅支持 `float32` 输入。
- 算子语义为单输入逐元素变换，不涉及广播、reduce 或 shape 变换。
- 输出必须与输入保持完全相同的 `shape`。
- 不引入额外的数值截断、饱和或类型转换。
- 不破坏空 Tensor、标量 Tensor 和特殊浮点值的行为。

## 11. Acceptance Criteria / 验收标准
- 对任意合法 `float32` Tensor 输入，输出 Tensor 的 `shape` 与 `dtype` 与输入完全一致。
- 对以下样例必须得到预期结果：
  - 正数输入保持不变
  - 负数输入取相反数
  - `+0.0` 输出为 `+0.0`
  - `-0.0` 输出为 `+0.0`
  - `NaN` 仍为 `NaN`
  - `+Inf` 输出为 `+Inf`
  - `-Inf` 输出为 `+Inf`
  - 空 Tensor 输出为空 Tensor，且 shape 不变
  - 标量 Tensor 输出标量 Tensor，且值正确
- 任意维度 Tensor 均可正常执行，不发生维度相关错误。
- 不修改输入 Tensor 的数据内容。

## 12. Open Questions / 待澄清问题
none

## 13. References / 参考资料
- 用户需求：`Abs` 算子，`float32` 输入输出保持一致，支持任意维度 Tensor 和边界值处理。
- IEEE 754 `float32` 浮点语义：`NaN`、`Inf`、`-0.0` 的标准行为。
