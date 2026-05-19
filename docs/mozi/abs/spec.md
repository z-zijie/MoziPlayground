# Abs SPEC

## 1. Overview / 概述
`Abs` 是基础逐元素算子，用于对输入 Tensor 的每个元素取实数绝对值。它面向 NPU ARCH `dav-2201` 提供与 PRD 一致的可观察行为：单输入、单输出、shape 不变、dtype 不变、逐元素计算，且支持标量 Tensor 与空 Tensor。

## 2. Scope / 范围
本 SPEC 仅定义 `Abs` 的行为契约。

包含内容:
- 单输入、单输出的逐元素绝对值语义
- 仅 `float32` 输入与输出
- 任意维度 Tensor、标量 Tensor、空 Tensor 的行为
- 对 `+0.0`、`-0.0`、`NaN`、`+Inf`、`-Inf` 的可观察语义
- 输出 shape 与输入 shape 完全一致，输出 dtype 与输入 dtype 完全一致

不包含内容:
- `float32` 之外的 dtype 支持
- 广播、拼接、维度变换、归约
- kernel 设计、调度策略、tiling、缓存规划、内存规划或其他实现细节

## 3. Supported Platforms / 支持的NPU平台
`dav-2201`

## 4. Operator Interface / 算子接口

### PyTorch ATen IR
```text
aten::abs(Tensor x) -> Tensor y
```

### Pure Python Signature
```python
def abs(x: Tensor) -> Tensor:
    """Return the elementwise absolute value of x.

    The operator computes y[i] = |x[i]| for every element of the input tensor.
    It preserves the input tensor's shape and dtype, and it does not modify x
    in place. The operator supports scalar tensors and empty tensors.

    Args:
        x (Tensor): Input tensor. Must be float32. May have any rank, including
            a scalar tensor or an empty tensor. The tensor is read-only with
            respect to this operator.

    Returns:
        Tensor: A new float32 tensor with the same shape as x. Each element is
        the absolute value of the corresponding input element. Negative zero is
        mapped to non-negative zero. NaN remains NaN. +Inf and -Inf map to +Inf.
    """
    ...
```

### Pure C++ Signature
```cpp
/**
 * @brief Compute the elementwise absolute value of an input tensor.
 *
 * The operator returns a new tensor y such that y[i] = |x[i]| for every
 * element index i. The output preserves the input shape and float32 dtype.
 * Scalar tensors and empty tensors are supported. The input tensor is not
 * mutated, and the returned tensor does not alias the input for writable
 * semantics.
 *
 * Numeric semantics:
 * - Negative finite values map to their non-negative magnitude.
 * - Positive finite values are preserved.
 * - +0.0 and -0.0 both produce non-negative zero.
 * - NaN remains NaN.
 * - +Inf and -Inf both produce +Inf.
 *
 * @param x Input tensor. Must be float32. May be any rank, including a scalar
 *          tensor or an empty tensor.
 * @return A new float32 tensor with the same shape as x and elementwise
 *         absolute-value results.
 */
Tensor abs(const Tensor& x);
```

## 5. Input Specification / 输入规格
- `x` 是单个输入 Tensor。
- `x` 的 dtype 必须是 `float32`。
- `x` 可以是任意 rank Tensor。
- `x` 可以是标量 Tensor。
- `x` 可以是空 Tensor，包括具有零元素的张量形状。
- `x` 在本算子中只读，不得被原地修改。

## 6. Output Specification / 输出规格
- `y` 是单个输出 Tensor。
- `y` 的 dtype 必须与 `x` 完全一致，且为 `float32`。
- `y` 的 shape 必须与 `x` 完全一致。
- `y` 是新结果 Tensor，不得作为输入 `x` 的可写别名。
- `y[i] = |x[i]|` 对所有有效元素索引成立。

## 7. Attribute Specification / 属性规格
本算子无额外属性。

## 8. Mathematical Semantics / 数学语义
设输入张量 `x` 的索引集合为 `I`。定义输出张量 `y` 为：

```text
∀ i ∈ I,  y[i] = |x[i]|
```

其中 `|·|` 表示实数绝对值函数。该映射逐元素独立应用于每个有效索引，不进行广播、重排、归约或维度变换。若 `I` 为空，则 `y` 也是空张量，且上述定义在空索引集合上成立。

## 9. Functional Semantics / 功能语义
- 对每个输入元素独立计算绝对值。
- 当 `x[i] < 0` 时，`y[i] = -x[i]`。
- 当 `x[i] > 0` 时，`y[i] = x[i]`。
- 当 `x[i] = +0.0` 或 `x[i] = -0.0` 时，`y[i]` 为非负零语义。
- 当 `x[i]` 为 `NaN` 时，`y[i]` 仍为 `NaN`。
- 当 `x[i]` 为 `+Inf` 或 `-Inf` 时，`y[i] = +Inf`。
- 算子不执行广播、归约、转置、重排、切片或 dtype 转换。

## 10. Numeric Semantics / 数值语义
- 对有限 `float32` 值，结果应与数学绝对值一致。
- 算子不引入近似计算要求。
- `NaN` 输入保持为 `NaN`。
- `+Inf` 和 `-Inf` 的输出为 `+Inf`。
- `-0.0` 的输出应转换为非负零语义。
- PRD 未定义额外的舍入、溢出、下溢或容差要求。

## 11. Shape Semantics / Shape 语义
- 输出 shape 必须与输入 shape 完全一致。
- 标量 Tensor 输入对应标量 Tensor 输出。
- 空 Tensor 输入对应空 Tensor 输出，且空 Tensor 的 shape 保持不变。
- 不发生广播、降维、升维或维度重排。

## 12. Data Type Support / 数据类型支持
- 仅支持 `float32`。
- 输入和输出 dtype 必须一致。
- 其他 dtype 不在本 SPEC 的支持范围内。

## 13. Layout and Format Constraints / Layout 与 Format 约束
- PRD 未定义额外的 layout 或 format 转换要求。
- 本 SPEC 仅约束 Tensor 的逻辑元素语义、shape 和 dtype 一致性。
- 若输入满足算子接口的 Tensor 要求，则布局差异不改变本 SPEC 定义的逐元素语义。

## 14. Boundary Cases / 边界场景
- 标量 Tensor：按单元素 Tensor 处理，输出仍为标量 Tensor。
- 空 Tensor：保持为空 Tensor，shape 与 dtype 不变。
- `+0.0` / `-0.0`：输出为非负零语义。
- `NaN`：输出为 `NaN`。
- `+Inf` / `-Inf`：输出为 `+Inf`。
- 任意 rank 的合法 `float32` Tensor：应正常执行并返回同 shape、同 dtype 结果。

## 15. Error Handling / 错误处理
- 非 `float32` 输入不在支持范围内，应拒绝执行。
- 非 Tensor 输入不在算子接口定义内，应由前置类型检查拒绝。
- 除上述情况外，PRD 未定义其他错误条件。

## 16. Compatibility / 兼容性说明
- 语义应与 PyTorch `abs` 的逐元素绝对值行为一致。
- 对输入 shape、标量 Tensor、空 Tensor、`NaN`、无穷大和符号零的处理，应与基础 Tensor 语义一致。
- 不引入额外的 dtype 变换、shape 变换或数值变换。

## 17. Performance Requirements / 性能要求
PRD 未指定额外性能要求。

## 18. Acceptance Criteria / 验收标准
- 对任意合法 `float32` Tensor 输入，输出 Tensor 的 shape 与 dtype 均与输入一致。
- 对每个元素，输出满足 `y[i] = |x[i]|`。
- 对 `NaN`、`+Inf`、`-Inf`、`+0.0`、`-0.0` 的输入，输出满足本 SPEC 定义的数值语义。
- 标量 Tensor 和空 Tensor 输入均可按本 SPEC 正确处理。
- 非 `float32` 输入应被明确拒绝，不得被当作支持输入处理。
- 本 SPEC 不包含未被 PRD 支持的额外行为声明。

## 19. Open Issues / 待确认问题
n/a
