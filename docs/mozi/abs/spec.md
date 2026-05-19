# Abs SPEC

## 1. Overview / 概述
`Abs` 是一个逐元素绝对值算子。对于输入 Tensor `x` 中的每个元素，算子返回同位置元素的绝对值结果 `y[i] = |x[i]|`。

本 SPEC 来源于 `Abs PRD`，约束 `Abs` 在 NPU ARCH `dav-2201` 上的外部可观察行为：仅支持 `float32` Tensor，输出 Tensor 与输入 Tensor 保持完全相同的 shape 和 dtype，并正确处理标量 Tensor、空 Tensor、`+0.0`、`-0.0`、`NaN`、`+Inf` 与 `-Inf`。

## 2. Scope / 范围
本 SPEC 覆盖以下功能范围：

- 支持 `float32` Tensor 输入 `x`。
- 对 `x` 执行逐元素绝对值计算。
- 输出 Tensor `y` 的 shape 与 `x` 完全一致。
- 输出 Tensor `y` 的 dtype 与 `x` 完全一致，即 `float32`。
- 支持任意维度 Tensor，包括标量 Tensor 和空 Tensor。
- 数值语义与 PyTorch `abs` 的逐元素绝对值行为保持一致。

本 SPEC 明确不覆盖以下行为：

- 不支持 `float32` 以外的 dtype。
- 不定义广播、拼接、维度变换、reshape 或归约行为。
- 不定义额外数值变换、shape 变换或 dtype 变换。
- 不包含 kernel 设计、调度策略、tiling、缓存规划、硬件指令选择或其他实现细节。

## 3. Supported Platforms / 支持的NPU平台
目标 NPU 平台为 `dav-2201`。本 SPEC 仅约束该平台上的功能支持范围与行为语义一致性，不规定任何硬件实现方式。

## 4. Operator Interface / 算子接口
### PyTorch ATen IR

```text
aten::abs(Tensor x) -> Tensor y
```

### Pure Python Signature

```python
import numpy as np


def abs(x):
    """
    Compute the element-wise absolute value of a float32 Tensor.

    Args:
        x: Required input Tensor. `x` must be a float32 Tensor with any rank,
            including rank-0 scalar Tensor and tensors with zero-sized
            dimensions. The logical shape is unrestricted and is preserved in
            the output. No broadcasting, reduction, reshape, or dtype promotion
            is applied. The PRD does not require `x` to be contiguous or to use
            a specific layout. `x` is read-only from the operator contract
            perspective and is not mutated or aliased as the output.

    Returns:
        A float32 Tensor `y` with exactly the same shape as `x`. For every valid
        element index `i`, `y[i] = |x[i]|`. Positive finite values are preserved,
        negative finite values are negated, `+0.0` and `-0.0` produce non-negative
        zero semantics, `NaN` remains `NaN`, and both `+Inf` and `-Inf` produce
        `+Inf`. Empty inputs produce an empty Tensor with the same shape and
        dtype. The returned Tensor is a distinct output value and does not
        mutate `x`.
    """
```

### Pure C++ Signature

```cpp
/**
 * @brief Compute the element-wise absolute value of a float32 Tensor.
 *
 * Constraints: x must be a float32 Tensor with any rank, including scalar
 * tensors and tensors with zero-sized dimensions. The output has exactly the
 * same logical shape and dtype as x. The operator does not perform broadcasting,
 * reduction, reshape, dtype promotion, or dtype conversion. The PRD does not
 * require a specific tensor layout or contiguity property.
 *
 * Numeric semantics: for each element, positive finite values are preserved,
 * negative finite values are negated, +0.0 and -0.0 produce non-negative zero
 * semantics, NaN remains NaN, and +Inf or -Inf produce +Inf.
 *
 * Memory semantics: x is an input-only Tensor. The returned Tensor is a distinct
 * output value and the operator contract does not mutate x or require output
 * aliasing with x.
 *
 * @param x Required float32 input Tensor. Any rank and any shape are valid,
 *          including scalar shape and shapes with zero-sized dimensions.
 * @return A float32 Tensor with the same shape as x and element-wise absolute
 *         values of x.
 */
Tensor abs(const Tensor& x);
```

## 5. Input Specification / 输入规格
| Input | Required | Dtype | Rank / Shape | Value domain | Layout / Format | Aliasing |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | Yes | `float32` only | Any rank and any shape, including scalar Tensor and empty Tensor | All `float32` values, including finite values, `+0.0`, `-0.0`, `NaN`, `+Inf`, and `-Inf` | PRD does not specify a concrete layout or format requirement | Input-only; must not be mutated by the operator contract |

Unsupported input:

- Any dtype other than `float32` is outside the supported scope.
- Non-Tensor inputs are outside the supported operator interface.

## 6. Output Specification / 输出规格
| Output | Dtype | Shape | Value semantics | Layout / Format | Aliasing |
| --- | --- | --- | --- | --- | --- |
| `y` | Same as `x`, therefore `float32` | Exactly same shape as `x` | For every element index `i`, `y[i] = |x[i]|` | No additional layout or format conversion semantics are specified by the PRD | Distinct output value; no output aliasing requirement is specified |

The output is deterministic for a fixed input Tensor under the numeric semantics in this SPEC.

## 7. Attribute Specification / 属性规格
`Abs` has no attributes. All behavior is determined solely by the required input Tensor `x`.

## 8. Mathematical Semantics / 数学语义
Let `X` be a tensor over the domain `Float32` with shape `S = (d_0, d_1, ..., d_{n-1})`, where `n >= 0`. The valid index set is:

\[
I(S) = \{(i_0, i_1, ..., i_{n-1}) \mid 0 \le i_k < d_k,\ 0 \le k < n\}
\]

For scalar Tensor, `n = 0` and the index set contains the single scalar index. For an empty Tensor, at least one dimension size is zero and `I(S)` is empty.

`Abs` is the mapping:

\[
\operatorname{Abs}: Float32^S \rightarrow Float32^S
\]

such that for every valid index \(i \in I(S)\):

\[
Y_i = |X_i| =
\begin{cases}
X_i, & X_i \ge 0 \\
-X_i, & X_i < 0
\end{cases}
\]

Special floating-point values follow the numeric semantics section: `NaN` maps to `NaN`, `+Inf` maps to `+Inf`, `-Inf` maps to `+Inf`, and signed zero maps to non-negative zero semantics.

## 9. Functional Semantics / 功能语义
`Abs` applies the absolute value operation independently to each element of `x`.

- No element depends on any other element.
- The input and output have identical rank and dimension sizes.
- No broadcasting is performed.
- No reduction is performed.
- No reshape, flatten, transpose, or other dimension transformation is performed.
- No dtype promotion or dtype conversion is performed.
- Empty Tensor inputs return empty Tensor outputs with the same shape and dtype.
- Scalar Tensor inputs return scalar Tensor outputs.
- Observable behavior must be compatible with PyTorch `abs` element-wise absolute value semantics for the supported `float32` input scope.

## 10. Numeric Semantics / 数值语义
The operator follows `float32` element-wise absolute value semantics.

| Input value category | Output behavior |
| --- | --- |
| `x[i] > 0` finite value | `y[i] = x[i]` |
| `x[i] < 0` finite value | `y[i] = -x[i]` |
| `+0.0` | Output satisfies non-negative zero absolute value semantics |
| `-0.0` | Output satisfies non-negative zero absolute value semantics |
| `NaN` | Output remains `NaN` |
| `+Inf` | Output is `+Inf` |
| `-Inf` | Output is `+Inf` |

No additional rounding, tolerance, accumulation, overflow handling, underflow handling, or precision relaxation is specified by the PRD beyond the exact element-wise absolute value definition for `float32`.

## 11. Shape Semantics / Shape 语义
The output shape is exactly the input shape. Rank is preserved for all valid inputs, including rank-0 scalar Tensor. Zero-sized dimensions are preserved, so an empty input Tensor returns an empty output Tensor with the same shape. `Abs` does not support or apply broadcasting, reduction, reshape, concatenation, or dimension insertion/removal.

```python
import numpy as np


def abs(x):
    """
    Infer the output shape metadata for Abs.

    Args:
        x: Input Tensor-like object or shape metadata. The value must represent
            a float32 Tensor with any rank, including scalar Tensor shape `()`
            and shapes with zero-sized dimensions. The PRD defines no
            broadcasting, reduction, reshape, or dimension transformation.

    Returns:
        A tuple representing the output shape. The returned shape is exactly the
        same as the input shape. Scalar Tensor input returns `()`.

    Unsupported/error cases:
        Non-Tensor inputs or metadata without a recoverable shape are invalid
        for this operator interface. Dtype validation is handled by dtype
        inference and operator input validation, not by shape inference.

    Shape-rule notes:
        Abs is element-wise and shape-preserving. Empty Tensor shapes are valid
        and remain unchanged. NumPy is imported to keep the reference shape
        metadata handling explicit; this function returns only shape metadata
        and does not compute tensor values.
    """
    shape_rules = (
        ("shape_attribute", lambda value: tuple(value.shape)),
        ("shape_metadata", lambda value: tuple(value)),
    )

    if hasattr(x, "shape"):
        return shape_rules[0][1](x)

    try:
        return shape_rules[1][1](x)
    except TypeError as exc:
        raise TypeError("Abs shape inference requires Tensor shape metadata") from exc
```

## 12. Data Type Support / 数据类型支持
`Abs` supports only `float32` input. The output dtype is exactly the input dtype and is therefore `float32`. No dtype promotion, dtype conversion, mixed dtype behavior, integer support, boolean support, or complex support is defined.

```python
def abs(x):
    """
    Infer the output dtype metadata for Abs.

    Args:
        x: Input Tensor-like object or dtype metadata. The input must represent
            a float32 Tensor. Shape may be arbitrary, including scalar and empty
            Tensor shapes. The operator has no attributes and no optional inputs.

    Returns:
        The output dtype metadata. For the only supported input dtype `float32`,
        the returned dtype is `float32`.

    Unsupported/error cases:
        Any dtype other than `float32` is unsupported. Inputs without recoverable
        dtype metadata are invalid for dtype inference.

    Promotion-rule notes:
        Abs does not apply dtype promotion or conversion. The table-driven rule
        maps `float32` to `float32` and rejects all other dtypes.
    """
    dtype_rules = {
        "float32": "float32",
        np.dtype("float32"): "float32",
    }

    dtype = getattr(x, "dtype", x)
    if dtype in dtype_rules:
        return dtype_rules[dtype]

    dtype_name = str(dtype)
    if dtype_name in dtype_rules:
        return dtype_rules[dtype_name]

    raise TypeError("Abs supports only float32 input dtype")
```

## 13. Layout and Format Constraints / Layout 与 Format 约束
The PRD does not specify a concrete memory layout, tensor format, or contiguity requirement. The behavioral contract is layout-independent at the logical Tensor level:

- The logical element order is defined by Tensor indices.
- The output preserves the input logical shape and dtype.
- No layout conversion behavior is part of the operator semantics.
- No additional format-specific behavior is specified.

## 14. Boundary Cases / 边界场景
| Boundary case | Required behavior |
| --- | --- |
| Scalar Tensor | Return scalar Tensor with value `|x|` and dtype `float32` |
| Empty Tensor | Return empty Tensor with exactly the same shape and dtype |
| Tensor with zero-sized dimension | Preserve the same shape and return an empty Tensor |
| Arbitrary rank Tensor | Apply element-wise absolute value and preserve rank |
| Positive finite values | Preserve the input value |
| Negative finite values | Return the negated value |
| `+0.0` | Return non-negative zero absolute value semantics |
| `-0.0` | Return non-negative zero absolute value semantics |
| `NaN` | Return `NaN` |
| `+Inf` | Return `+Inf` |
| `-Inf` | Return `+Inf` |

## 15. Error Handling / 错误处理
The supported input dtype is only `float32`. Inputs with dtype other than `float32` are unsupported and must be rejected by operator validation.

The operator interface requires one Tensor input `x`. Missing input, non-Tensor input, or an input whose dtype is not `float32` is outside the supported contract. The PRD does not define a specific diagnostic string or exception class.

All ranks and shapes are valid for `float32` Tensor input, including scalar and empty Tensor shapes, so rank, dimension count, and zero-sized dimensions are not error conditions.

## 16. Compatibility / 兼容性说明
For the supported `float32` Tensor input scope, `Abs` must be semantically compatible with PyTorch `abs` element-wise absolute value behavior.

Compatibility requirements:

- Match PyTorch `abs` for positive values, negative values, signed zero, `NaN`, `+Inf`, and `-Inf` within the supported `float32` scope.
- Preserve input shape and dtype, matching standard Tensor element-wise operator behavior.
- Do not introduce extra numeric transformation, shape transformation, dtype transformation, broadcasting, or reduction behavior.

No aliases, additional overloads, or backward compatibility requirements are specified by the PRD.

## 17. Performance Requirements / 性能要求
No additional measurable performance requirement is specified by the PRD.

## 18. Acceptance Criteria / 验收标准
The SPEC is satisfied when all of the following are true:

- Given any legal `float32` Tensor input, the output Tensor shape exactly equals the input Tensor shape.
- Given any legal `float32` Tensor input, the output Tensor dtype exactly equals the input Tensor dtype.
- For positive finite input elements, output elements equal the input values.
- For negative finite input elements, output elements equal the negated input values.
- For `+0.0` and `-0.0`, output elements satisfy non-negative zero absolute value semantics.
- For `NaN` input elements, output elements remain `NaN`.
- For `+Inf` and `-Inf` input elements, output elements are `+Inf`.
- Scalar Tensor input returns scalar Tensor output.
- Empty Tensor input returns empty Tensor output with unchanged shape and dtype.
- Arbitrary-rank `float32` Tensor input executes as an element-wise shape-preserving operator.
- Inputs with dtype other than `float32` are rejected as unsupported.
- The documented PyTorch ATen IR, pure Python signature, and pure C++ signature all describe the same operator contract.

## 19. Open Issues / 待确认问题
No open issues.
