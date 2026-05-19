# Abs SPEC

## 1. Overview / 概述
Abs 算子根据 PRD 定义，对输入 float32 Tensor `x` 的每个元素执行绝对值变换，返回同 shape、同 dtype 的 float32 Tensor `y`。对每个有效元素索引 `i`，结果满足 `y[i] = |x[i]|`。

本 SPEC 是从 `/Users/eureka/Workspace/MoziPlayground/docs/mozi/abs/prd.md` 生成的行为契约，用于约束后续 DESIGN、实现和测试阶段。本文档只描述外部可观察的接口、输入输出、数学语义、数值语义、shape/dtype 规则、边界行为和验收标准，不规定 kernel 设计、tiling、内存规划、调度方式或硬件指令选择。

## 2. Scope / 范围
本 SPEC 覆盖 NPU ARCH dav-2201 上的 Abs 正向算子行为，功能范围包括：

- 输入一个 float32 Tensor `x`，输出一个 float32 Tensor `y`。
- 输出 shape 与输入 shape 完全相同。
- 对所有元素独立计算绝对值，不进行广播、规约、reshape 或跨元素依赖计算。
- 支持任意维度 Tensor，包括 0 维标量 Tensor、1 维及以上 Tensor、任一维长度为 0 的空 Tensor。
- 覆盖正有限数、负有限数、`+0.0`、`-0.0`、`NaN`、`+Inf`、`-Inf` 的 float32 数值语义。

本 SPEC 明确排除以下行为：

- float32 以外的 dtype。
- 复数、量化、稀疏 Tensor 或非 Tensor 输入。
- 梯度、反向传播或训练框架自动微分行为。
- 输入 shape、rank、layout 或 dtype 的改变。
- 性能、吞吐或延迟目标。

## 3. Supported Platforms / 支持的NPU平台
目标平台范围为 NPU ARCH dav-2201。平台相关约束仅限于 PRD 指定的产品行为、接口、数值语义和验收要求；本 SPEC 不规定任何硬件执行设计。

## 4. Operator Interface / 算子接口
### PyTorch ATen IR

```text
aten::abs(Tensor self) -> Tensor
```

`self` 对应本 SPEC 中的输入 Tensor `x`，返回值对应输出 Tensor `y`。本 SPEC 仅要求 `self` 为 float32 Tensor。

### Pure Python Signature

```python
def abs(x):
    """
    Compute elementwise absolute value for a float32 Tensor.

    Args:
        x: Required input Tensor. It must be a float32 Tensor with any legal
            Tensor rank and shape, including rank-0 scalar tensors and tensors
            with zero-sized dimensions. The input participates as a read-only
            operand: Abs must not mutate its values, shape, rank, layout, or
            dtype. No default value is defined, and the parameter is not
            optional. Non-Tensor inputs, sparse tensors, quantized tensors,
            complex tensors, and dtypes other than float32 are outside the
            supported functional scope.

    Returns:
        A float32 Tensor with exactly the same shape and rank as x. For every
        valid element index i, the returned tensor y satisfies y[i] = |x[i]|.
        Empty inputs return empty outputs with matching metadata, and scalar
        inputs return scalar outputs. The returned value is the output tensor
        of the operator; this API does not define an in-place aliasing form.

    Mathematical and numeric semantics:
        The operator is elementwise and has no broadcasting, reduction, reshape,
        or cross-element dependency. Positive finite float32 values are returned
        unchanged, negative finite values return the corresponding positive
        magnitude, +0.0 returns +0.0, -0.0 returns numeric +0.0, +Inf returns
        +Inf, -Inf returns +Inf, and NaN returns NaN. The PRD does not require
        preservation of NaN payload or NaN sign bit.
    """
    pass
```

### Pure C++ Signature

```cpp
/**
 * @brief Compute elementwise absolute value for a float32 Tensor.
 *
 * Constraints: x must be a Tensor with dtype float32. Any legal tensor rank and
 * shape are supported, including rank-0 scalar tensors and tensors with
 * zero-sized dimensions. Dtypes other than float32, non-Tensor inputs, sparse
 * tensors, quantized tensors, and complex tensors are outside the supported
 * functional scope. Abs does not broadcast, reduce, reshape, or otherwise
 * change rank, shape, layout, or dtype.
 *
 * Numeric semantics: for every element, positive finite values are returned
 * unchanged, negative finite values return their positive magnitude, +0.0
 * returns +0.0, -0.0 returns numeric +0.0, +Inf returns +Inf, -Inf returns
 * +Inf, and NaN returns NaN. No additional approximation tolerance is defined.
 * The PRD does not require preservation of NaN payload or NaN sign bit.
 *
 * Memory semantics: x is a read-only operand and must not be mutated. The
 * returned Tensor is the operator output. This signature does not define an
 * in-place aliasing form.
 *
 * @param x Required input Tensor. It provides the element values and metadata
 *     for Abs. x must have dtype float32, may have any legal rank and shape,
 *     may be rank-0, and may contain zero-sized dimensions. x has no default
 *     value and is not optional.
 * @return Tensor A float32 Tensor y with exactly the same rank and shape as x,
 *     where each element satisfies y[i] = |x[i]|.
 */
Tensor Abs(const Tensor& x);
```

## 5. Input Specification / 输入规格
`x` 是唯一输入操作数：

- 类型：Tensor。
- dtype：必须为 float32。
- rank：任意合法 Tensor rank，包括 0 维标量 Tensor。
- shape：任意合法 Tensor shape；每个维度大小为合法非负整数；允许任一维长度为 0。
- value domain：任意 float32 值，包括有限数、`+0.0`、`-0.0`、`NaN`、`+Inf`、`-Inf`。
- layout/format：PRD 不引入额外 layout 或 format 限制；Abs 不应改变输入输出可观察的 shape、rank、layout 或 dtype。
- optionality：必选输入，不支持省略。
- aliasing/mutability：`x` 是只读输入；算子不得修改输入 Tensor 的值或元数据。
- unsupported inputs：float32 以外 dtype、复数、量化、稀疏 Tensor、非 Tensor 输入不属于本 SPEC 的支持范围。

## 6. Output Specification / 输出规格
`y` 是唯一输出值：

- 类型：Tensor。
- dtype：float32，与输入 `x` 相同。
- rank：与 `x` 完全相同。
- shape：与 `x` 完全相同，包括标量 shape `()` 与包含零维长度的空 Tensor shape。
- value domain：float32 值；每个输出元素为对应输入元素的绝对值。
- layout/format：不得改变 PRD 约束下可观察的输入 layout/format 语义；本 SPEC 不定义 layout 转换行为。
- aliasing/mutability：返回值是 Abs 的输出 Tensor；本 SPEC 不定义 in-place 形式，且不得通过输出行为修改输入 `x`。
- determinism：对同一合法输入，输出 shape、dtype 和每个元素的数值语义确定。

## 7. Attribute Specification / 属性规格
Abs 不定义任何 operator attribute。算子行为完全由输入 Tensor `x` 决定。

## 8. Mathematical Semantics / 数学语义
令输入 Tensor 为：

\[
x \in \mathbb{F}_{32}^{S}
\]

其中 \(S = (d_0, d_1, \ldots, d_{r-1})\) 是合法 Tensor shape，\(r \ge 0\)，每个 \(d_k \in \mathbb{N}\)。当 \(r = 0\) 时，\(S = ()\) 表示标量 Tensor。当存在 \(d_k = 0\) 时，索引集合为空。

输出 Tensor 为：

\[
y = \operatorname{Abs}(x) \in \mathbb{F}_{32}^{S}
\]

对所有有效索引：

\[
i \in I(S) = \{(i_0,\ldots,i_{r-1}) \mid 0 \le i_k < d_k\}
\]

逐元素定义为：

\[
y_i = |x_i|
\]

其中：

\[
|a| =
\begin{cases}
a, & a \ge 0 \\
-a, & a < 0
\end{cases}
\]

该映射不改变 Tensor 的 shape、rank 或 dtype，不引入广播、规约、重排或跨元素依赖。对空索引集合 \(I(S)=\varnothing\)，输出仍具有 shape \(S\)，且没有元素级结果需要计算。

## 9. Functional Semantics / 功能语义
Abs 的可观察功能语义如下：

- 对输入 Tensor 的每个元素独立执行绝对值计算。
- 输出 Tensor 与输入 Tensor shape 完全相同。
- 输出 Tensor 与输入 Tensor dtype 完全相同，且本 SPEC 仅支持 float32。
- 不执行广播、规约、reshape、rank 变更或 layout 变更。
- 不读取其他输入，不使用 attributes，不依赖元素之间的顺序或相邻关系。
- 对空 Tensor，返回同 shape、同 dtype 的空 Tensor。
- 对标量 Tensor，返回标量 Tensor，值为输入标量的绝对值。
- 行为语义应与 PyTorch ATen `aten::abs` 在 float32 Tensor 上的逐元素绝对值语义保持一致。

## 10. Numeric Semantics / 数值语义
Abs 的 float32 数值语义如下：

- 正有限数输出其自身数值。
- 负有限数输出对应正数。
- `+0.0` 输出 `+0.0`。
- `-0.0` 输出数值为 `+0.0`。
- `+Inf` 输出 `+Inf`。
- `-Inf` 输出 `+Inf`。
- `NaN` 输出为 `NaN`。
- PRD 不要求规定 NaN payload 或 NaN 符号位保持策略。
- 对所有 float32 有限数输入，绝对值结果仍为 float32，并且不定义额外近似误差容忍度。
- 对 `+Inf`、`-Inf` 和 `NaN`，遵循上述特殊值传播规则。
- 本 SPEC 不定义 float32 以外 dtype 的舍入、溢出、下溢或提升行为。

## 11. Shape Semantics / Shape 语义
输出 shape 直接等于输入 `x` 的 shape。Abs 不进行广播、规约、reshape、flatten、transpose 或 rank 改变。0 维标量 Tensor 的 shape 保持为 `()`；任一维长度为 0 的空 Tensor 保持原 shape。

```python
import numpy as np

def abs(x):
    """
    Infer output shape metadata for Abs.

    Args:
        x: Required tensor-like metadata object for the input Tensor. It must
            expose a shape attribute representing any legal Tensor shape,
            including () for a rank-0 scalar tensor and shapes containing
            zero-sized dimensions for empty tensors. x is required and is not
            optional. Shape inference does not inspect or compute element
            values.

    Returns:
        tuple: The exact output shape. The returned shape is identical to
        x.shape. Rank-0 scalar tensors return (), and empty tensors preserve all
        zero-sized dimensions.

    Raises:
        TypeError: If x does not expose tensor shape metadata.
        ValueError: If a dimension is negative.

    Shape-rule notes:
        Abs is an elementwise identity-shape operator. It has no broadcasting,
        reduction, reshape, or symbolic dimension transformation rule.
    """
    shape = getattr(x, "shape", None)
    if shape is None:
        raise TypeError("Abs shape inference requires tensor shape metadata for x.")

    normalized_shape = tuple(int(dim) for dim in tuple(shape))
    if any(dim < 0 for dim in normalized_shape):
        raise ValueError("Abs shape inference requires non-negative dimensions.")

    shape_rules = {
        "identity": lambda input_shape: input_shape,
    }
    return shape_rules["identity"](normalized_shape)
```

## 12. Data Type Support / 数据类型支持
Abs 仅支持输入 dtype 为 float32。输出 dtype 与输入 dtype 相同，仍为 float32。不支持 dtype promotion，不支持 float32 以外 dtype，不支持复数、量化或非 Tensor dtype 语义。

```python
import numpy as np

def abs(x):
    """
    Infer output dtype metadata for Abs.

    Args:
        x: Required tensor-like metadata object for the input Tensor. It must
            expose dtype metadata. The only supported dtype is float32. x is
            required and is not optional. Dtype inference does not inspect or
            compute element values.

    Returns:
        numpy.dtype: np.dtype("float32") when x.dtype is float32.

    Raises:
        TypeError: If x does not expose dtype metadata or if x.dtype is not
        float32.

    Promotion-rule notes:
        Abs has no dtype promotion rule in this SPEC. The table maps the single
        supported input dtype directly to the output dtype.
    """
    dtype = getattr(x, "dtype", None)
    if dtype is None:
        raise TypeError("Abs dtype inference requires tensor dtype metadata for x.")

    dtype_rules = {
        np.dtype("float32"): np.dtype("float32"),
    }

    normalized_dtype = np.dtype(dtype)
    if normalized_dtype not in dtype_rules:
        raise TypeError("Abs supports only float32 input tensors.")

    return dtype_rules[normalized_dtype]
```

## 13. Layout and Format Constraints / Layout 与 Format 约束
PRD 不要求改变输入 Tensor 的 shape、rank、layout 或 dtype。因此，Abs 的输出必须保持与输入一致的可观察 Tensor shape/rank/dtype 语义，并且本 SPEC 不引入 layout 转换、format 转换或 contiguous-only 要求。

稀疏 Tensor、量化 Tensor 和非 Tensor 输入不属于支持范围。除 PRD 明确排除项外，本 SPEC 不添加额外 layout 或 format 条件。

## 14. Boundary Cases / 边界场景
Abs 必须覆盖以下边界场景：

- 正有限 float32：输出与输入数值相同。
- 负有限 float32：输出对应正数。
- `+0.0`：输出 `+0.0`。
- `-0.0`：输出数值为 `+0.0`。
- `+Inf`：输出 `+Inf`。
- `-Inf`：输出 `+Inf`。
- `NaN`：输出为 `NaN`；NaN payload 与符号位保持策略不在 PRD 要求范围内。
- 0 维标量 Tensor：输出仍为 0 维标量 Tensor，数值为输入绝对值。
- 空 Tensor：输出为空 Tensor，shape 与 dtype 与输入一致，不产生元素级计算结果。
- 任意维度非空 Tensor：输出 shape 与输入 shape 完全一致，dtype 为 float32。

## 15. Error Handling / 错误处理
合法调用必须满足：输入 `x` 是 float32 Tensor，且其 shape 是合法 Tensor shape。对合法的空 Tensor 与标量 Tensor，算子调用应成功。

以下输入不属于本 SPEC 的成功执行范围：

- float32 以外 dtype。
- 非 Tensor 输入。
- 复数 Tensor。
- 量化 Tensor。
- 稀疏 Tensor。

对这些不支持输入，算子不得返回一个被视为本 SPEC 合法 Abs 结果的成功输出。具体异常类型、错误码和诊断文本由调用框架或上层接口约定；本 SPEC 仅要求错误行为能够区分其不属于 float32 Tensor Abs 的支持范围。

## 16. Compatibility / 兼容性说明
Abs 的行为语义应与 PyTorch ATen `aten::abs` 在 float32 Tensor 上的逐元素绝对值语义保持一致。

兼容性要求包括：

- ATen schema：`aten::abs(Tensor self) -> Tensor`。
- `self` 对应输入 `x`，返回值对应输出 `y`。
- 输入为 `NaN` 时，输出必须为 `NaN`；不要求 NaN payload 或 NaN 符号位保持策略。
- 输入为 `-0.0` 时，输出数值必须为 `+0.0`。
- 合法空 Tensor 与标量 Tensor 调用应成功并返回符合 shape 与 dtype 要求的输出。
- 不支持的 dtype 或非 Tensor 输入不属于 PRD 功能范围。

## 17. Performance Requirements / 性能要求
PRD 不引入性能指标、吞吐指标或延迟指标。因此，本 SPEC 不定义额外可度量性能要求。

## 18. Acceptance Criteria / 验收标准
Abs 的验收标准如下：

- 给定 float32 Tensor `x = [1.0, 2.5, +0.0]`，输出为 `[1.0, 2.5, +0.0]`。
- 给定 float32 Tensor `x = [-1.0, -2.5, -0.0]`，输出数值为 `[1.0, 2.5, +0.0]`。
- 给定包含 `NaN` 的 float32 Tensor，输出对应位置为 `NaN`。
- 给定包含 `+Inf` 和 `-Inf` 的 float32 Tensor，输出对应位置均为 `+Inf`。
- 给定任意维度的非空 float32 Tensor，输出 shape 与输入 shape 完全一致，dtype 为 float32。
- 给定 0 维 float32 标量 Tensor，输出为 0 维 float32 标量 Tensor，数值为输入绝对值。
- 给定空 float32 Tensor，输出为空 Tensor，shape 与 dtype 与输入一致。
- Shape 推导必须返回与输入 shape 完全相同的 shape metadata。
- Dtype 推导必须对 float32 输入返回 float32，且不执行 dtype promotion。
- 算子不得修改输入 Tensor 的值或元数据。
- 对不属于 float32 Tensor 的输入，验收不要求通过本算子功能测试，并且不得被视为本 SPEC 的合法成功输出。

## 19. Open Issues / 待确认问题
无
