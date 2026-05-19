# Abs PRD

## 1. Background / 背景
Abs 算子用于对输入 Tensor 的每个 float32 元素取绝对值，是数值计算、损失函数、归一化、误差统计等流程中的基础逐元素算子。当前需求面向 NPU ARCH dav-2201 开发 Abs 算子，要求覆盖常规数值与 IEEE 754 float32 特殊值边界场景。

## 2. Goal / 目标
- 提供 Abs 算子能力：输入一个 float32 类型 Tensor `x`，输出一个与输入 shape 相同、dtype 相同的 Tensor `y`。
- 对每个元素满足 `y[i] = |x[i]|`。
- 支持任意维度 Tensor，包括标量 Tensor 与空 Tensor。
- 正确处理正数、负数、`+0.0`、`-0.0`、`NaN`、`+Inf`、`-Inf` 等边界输入。
- 明确该算子在 NPU ARCH dav-2201 上的需求范围，为后续 SPEC、设计、实现和测试提供依据。

## 3. Non-Goals / 非目标
- 不要求支持 float32 以外的 dtype。
- 不要求改变输入 Tensor 的 shape、rank、layout 或 dtype。
- 不要求定义梯度、反向传播或训练框架自动微分行为。
- 不要求支持复数、量化、稀疏 Tensor 或非 Tensor 输入。
- 不在 PRD 中规定 kernel 设计、tiling 策略、内存规划、调度方式或硬件指令选择。

## 4. User Scenarios / 使用场景
- 用户在 NPU dav-2201 上运行包含绝对值计算的推理或数值处理图。
- 用户需要对任意 shape 的 float32 Tensor 进行逐元素绝对值变换，且输出 shape 与输入完全一致。
- 用户需要在模型或算子测试中验证特殊浮点值的 Abs 语义，例如 `-0.0`、`NaN` 和 `-Inf`。
- 用户需要对空 Tensor 或标量 Tensor 调用 Abs，并获得与输入 Tensor 元数据一致的输出。

## 5. Functional Requirements / 功能需求
- 输入 `x` 必须是 float32 类型 Tensor。
- 输出 `y` 必须是 float32 类型 Tensor。
- 输出 `y` 的 shape 必须与输入 `x` 完全相同。
- Abs 必须对输入 Tensor 的每个元素独立执行绝对值计算，不进行广播、规约或跨元素依赖计算。
- 对任意有效元素索引 `i`，输出必须满足 `y[i] = |x[i]|`。
- 算子必须支持任意维度 Tensor，包括 0 维标量 Tensor、1 维及以上普通 Tensor，以及任一维长度为 0 的空 Tensor。
- 对空 Tensor，输出必须为空 Tensor，shape 与 dtype 与输入一致，且不产生元素级计算结果。
- 对标量 Tensor，输出必须仍为标量 Tensor。
- 边界数值语义必须满足：正有限数输出其自身数值；负有限数输出对应正数；`+0.0` 输出 `+0.0`；`-0.0` 输出数值为 `+0.0`；`+Inf` 输出 `+Inf`；`-Inf` 输出 `+Inf`；`NaN` 输出为 `NaN`。

## 6. Input and Output Overview / 输入输出概述
- 输入：`x`，float32 Tensor，shape 为任意合法 Tensor shape。
- 输出：`y`，float32 Tensor，shape 与 `x` 相同。
- 输入输出关系：`y = abs(x)`，且 `y[i] = |x[i]|`。
- 输出 Tensor 不改变输入 Tensor 的维度数量、各维大小或 dtype。

## 7. NPU ARCH
目标架构范围为 NPU ARCH dav-2201。该 PRD 仅定义 Abs 算子在 dav-2201 上需要支持的产品行为、输入输出接口、数值语义和验收要求，不包含具体硬件执行设计。

## 8. 算子原型
算子原型必须使用 PyTorch ATen IR 形式描述。

```text
aten::abs(Tensor self) -> Tensor
```

其中 `self` 对应输入 Tensor `x`，返回值对应输出 Tensor `y`。本 PRD 约束 `self` 的 dtype 为 float32。

## 9. Compatibility Requirements / 兼容性需求
- 行为语义应与 PyTorch ATen `aten::abs` 在 float32 Tensor 上的逐元素绝对值语义保持一致。
- 输入为 `NaN` 时，输出必须为 `NaN`；本 PRD 不要求规定 NaN payload 或 NaN 符号位保持策略。
- 输入为 `-0.0` 时，输出数值必须为 `+0.0`。
- 对合法的空 Tensor 与标量 Tensor，算子调用应成功并返回符合 shape 与 dtype 要求的输出。
- 不支持的 dtype 或非 Tensor 输入不属于本 PRD 的功能范围。

## 10. Accuracy and Numerical Expectations / 精度期望
- 对所有 float32 有限数输入，输出应为对应 float32 绝对值结果。
- 对正有限数，输出与输入数值相同。
- 对负有限数，输出为其相反数的正值。
- 对 `+0.0` 和 `-0.0`，输出数值为 `+0.0`。
- 对 `+Inf` 和 `-Inf`，输出为 `+Inf`。
- 对 `NaN`，输出为 `NaN`。
- 由于 Abs 是逐元素精确符号处理类算子，本 PRD 不定义额外近似误差容忍度。

## 11. Constraints / 约束
- 仅要求支持 float32 Tensor。
- 算子必须保持输入输出 shape 一致。
- 算子必须保持输入输出 dtype 一致。
- 算子不得修改输入 Tensor 的值。
- 算子需求范围限定在 NPU ARCH dav-2201。
- 本 PRD 不引入性能指标、吞吐指标或延迟指标；如后续需要，应由独立需求明确给出可度量目标。

## 12. Acceptance Criteria / 验收标准
- 给定 float32 Tensor `x = [1.0, 2.5, +0.0]`，输出为 `[1.0, 2.5, +0.0]`。
- 给定 float32 Tensor `x = [-1.0, -2.5, -0.0]`，输出数值为 `[1.0, 2.5, +0.0]`。
- 给定包含 `NaN` 的 float32 Tensor，输出对应位置为 `NaN`。
- 给定包含 `+Inf` 和 `-Inf` 的 float32 Tensor，输出对应位置均为 `+Inf`。
- 给定任意维度的非空 float32 Tensor，输出 shape 与输入 shape 完全一致，dtype 为 float32。
- 给定 0 维 float32 标量 Tensor，输出为 0 维 float32 标量 Tensor，数值为输入绝对值。
- 给定空 float32 Tensor，输出为空 Tensor，shape 与 dtype 与输入一致。
- 对不属于 float32 Tensor 的输入，验收不要求通过本算子功能测试。

## 13. Open Questions / 待澄清问题
None

## 14. References / 参考资料
- 用户需求：开发 Abs 算子，输入 float32 Tensor `x`，输出同 shape、同 dtype Tensor `y`，满足 `y[i] = |x[i]|`，支持任意维度与指定边界场景，目标 NPU ARCH dav-2201。
- PyTorch ATen operator schema: `aten::abs(Tensor self) -> Tensor`。
