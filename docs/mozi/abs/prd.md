# Abs PRD

## 1. Background / 背景
`Abs` 是基础逐元素算子，用于对输入 Tensor 的每个元素取绝对值。该算子是数值计算、特征变换和后处理中的常用基础能力，需要在 NPU ARCH `dav-2201` 上提供一致的算子支持。

## 2. Goal / 目标
- 支持 `float32` 类型输入 Tensor 的逐元素绝对值计算。
- 输出 Tensor 与输入 Tensor 保持相同 shape 和相同 dtype。
- 支持任意维度 Tensor，包括标量 Tensor 和空 Tensor。
- 正确处理正数、负数、`+0.0`、`-0.0`、`NaN`、`+Inf`、`-Inf` 等边界值。

## 3. Non-Goals / 非目标
- 不扩展到除 `float32` 以外的其他 dtype。
- 不定义广播、拼接、维度变换或归约行为。
- 不在本 PRD 中描述 kernel 设计、调度策略、tiling、缓存规划或其他实现细节。

## 4. User Scenarios / 使用场景
- 用户将任意形状的 `float32` Tensor 输入 `Abs`，获得逐元素绝对值输出。
- 用户输入标量 Tensor 时，仍能返回一个标量 Tensor。
- 用户输入空 Tensor 时，算子应返回同 shape、同 dtype 的空 Tensor。
- 用户输入包含 `NaN` 或无穷值的 Tensor 时，输出应符合绝对值语义。

## 5. Functional Requirements / 功能需求
- 算子对输入 Tensor `x` 的每个元素执行 `y[i] = |x[i]|`。
- 输入仅支持 `float32` Tensor。
- 输出 Tensor 的 shape 必须与输入完全一致。
- 输出 Tensor 的 dtype 必须与输入完全一致。
- 算子必须支持任意维度 Tensor。
- 算子必须支持标量 Tensor。
- 算子必须支持空 Tensor，并保持空 Tensor 语义不变。
- 对于 `x < 0` 的元素，输出其相反数。
- 对于 `x > 0` 的元素，输出原值。
- 对于 `+0.0` 和 `-0.0`，输出应为绝对值结果，即非负零语义。
- 对于 `NaN`，输出仍应为 `NaN`。
- 对于 `+Inf` 和 `-Inf`，输出应为 `+Inf`。

## 6. Input and Output Overview / 输入输出概述
- 输入：
  - `x`: `float32` Tensor，shape 任意，可为标量或空 Tensor。
- 输出：
  - `y`: `float32` Tensor，shape 与 `x` 相同，逐元素结果满足 `y[i] = |x[i]|`。

## 7. NPU ARCH
目标架构为 `dav-2201`。本算子 PRD 仅约束该架构上的功能支持范围与语义一致性，不定义具体硬件实现方式。

## 8. 算子原型
算子原型必须使用 PyTorch ATen IR 形式描述。

```text
aten::abs(Tensor x) -> Tensor y
```

说明：
- `x`: `float32` Tensor
- `y`: 与 `x` shape 相同、dtype 相同的 Tensor

## 9. Compatibility Requirements / 兼容性需求
- 语义应与 PyTorch `abs` 的逐元素绝对值行为保持一致。
- 对于输入 shape、dtype、空 Tensor、标量 Tensor 的处理，应与基础 Tensor 语义一致。
- 不引入额外的数值变换、shape 变换或 dtype 变换。

## 10. Accuracy and Numerical Expectations / 精度期望
- 结果应满足逐元素绝对值定义。
- 正常有限值的结果应与数学绝对值一致。
- `NaN` 输入应保持为 `NaN`。
- `+Inf` 和 `-Inf` 输入应输出 `+Inf`。
- `+0.0` 与 `-0.0` 的输出应满足绝对值语义，不应产生负零结果。

## 11. Constraints / 约束
- 仅支持 `float32` 输入与输出。
- 输出 shape 必须与输入完全一致。
- 输出 dtype 必须与输入完全一致。
- 必须支持任意维度、空 Tensor、标量 Tensor。
- 本 PRD 不包含性能指标要求。

## 12. Acceptance Criteria / 验收标准
- 输入任意合法 `float32` Tensor，输出 Tensor 的 shape 与 dtype 均与输入一致。
- 输入正数、负数、`+0.0`、`-0.0`、`NaN`、`+Inf`、`-Inf` 时，输出满足绝对值语义。
- 输入标量 Tensor 时，输出仍为标量 Tensor。
- 输入空 Tensor 时，输出仍为空 Tensor，且 shape 与 dtype 保持一致。
- 输入任意维度 Tensor 时，算子均可正常执行并返回正确结果。

## 13. Open Questions / 待澄清问题
n/a

## 14. References / 参考资料
- 用户需求：开发一个 `Abs` 算子，输入 `float32` Tensor，输出与输入 shape 和 dtype 相同的 Tensor，满足逐元素绝对值语义，面向 NPU ARCH `dav-2201`。
