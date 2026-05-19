# Abs SPEC

## 1. Overview / 概述
Abs 是逐元素绝对值算子。它接收一个 float32 Tensor `x`，返回一个与输入 shape、rank、layout 和 dtype 保持一致的 float32 Tensor `y`，并对每个有效元素索引 `i` 满足 `y[i] = |x[i]|`。

本 SPEC 基于 `docs/mozi/abs/prd.md` 中的 Abs PRD 生成，用于定义后续 DESIGN 和实现阶段可依赖的外部可观察行为契约，不包含 kernel 设计、tiling 策略、内存规划、调度方式或硬件指令选择。

## 2. Scope / 范围
本 SPEC 覆盖 NPU ARCH dav-2201 上 float32 Tensor 的 Abs 行为，包括任意维度 Tensor、0 维标量 Tensor、空 Tensor，以及 `+0.0`、`-0.0`、`NaN`、`+Inf`、`-Inf` 等 float32 边界数值。

本 SPEC 不覆盖 float32 以外 dtype、复数、量化 Tensor、稀疏 Tensor、非 Tensor 输入、梯度、反向传播、训练框架自动微分行为，也不要求定义性能、吞吐或延迟指标。

## 3. Supported Platforms / 支持的NPU平台
目标平台范围为 NPU ARCH dav-2201。Abs 在该平台上的产品行为、输入输出接口、数值语义和验收要求必须符合本 SPEC。

本 SPEC 不定义 dav-2201 以外平台的行为兼容性要求。

## 4. Operator Interface / 算子接口
### PyTorch ATen IR

Schema:

```text
aten::abs(Tensor self) -> Tensor
```

### Pure Python Signature

Signature:

```python
def abs(self):
    """Compute elementwise absolute value for a float32 tensor.

    Args:
        self: Required input tensor `x`. It must be a Tensor with dtype float32,
            any valid rank including rank 0, and any valid shape including shapes
            with zero-sized dimensions. The tensor layout is preserved by the
            result. `self` is read-only for this operator and must not be
            modified by the call. No default value is defined and `None` is not
            a valid input.

    Returns:
        A float32 Tensor `y` with the same shape, rank, and layout as `self`.
        For every valid element index `i`, `y[i] = |self[i]|`.

    Raises:
        TypeError: If `self` is not a Tensor or its dtype is not float32.
    """
```

### Pure C++ Signature

Framework-independent signature:

```cpp
/**
 * @brief Computes elementwise absolute value for a float32 tensor.
 *
 * @param self Required input tensor x. The tensor element type is float, which
 * represents float32. The tensor may have any valid rank including rank 0 and
 * any valid shape including shapes with zero-sized dimensions. The input layout
 * is preserved by the returned tensor. The input is read-only and must not be
 * modified. No optional input form or default value is defined.
 *
 * @return Tensor<float> Output tensor y with the same shape, rank, layout, and
 * dtype as self. For every valid element index i, y[i] is the absolute value of
 * self[i].
 *
 * @throws TypeError If the caller provides a non-tensor value or a tensor whose
 * dtype is not float32.
 */
Tensor<float> Abs(const Tensor<float>& self);
```

## 5. Input Specification / 输入规格
`self` / `x` 是唯一输入。

| Property / 属性 | Specification / 规格 |
| --- | --- |
| Operand kind | Required Tensor input |
| Dtype | float32 only |
| Rank | Any valid Tensor rank, including rank 0 scalar Tensor |
| Shape | Any valid Tensor shape; zero-sized dimensions are supported |
| Value domain | All float32 values, including finite values, `+0.0`, `-0.0`, `NaN`, `+Inf`, and `-Inf` |
| Layout | Input layout is not changed by this operator |
| Optionality | Required; `None` is not valid |
| Aliasing and mutability | The input is read-only and must not be modified by the operator |

## 6. Output Specification / 输出规格
The operator returns one Tensor `y`.

| Property / 属性 | Specification / 规格 |
| --- | --- |
| Operand kind | Tensor output |
| Dtype | float32, identical to input dtype |
| Rank | Identical to input rank |
| Shape | Identical to input shape |
| Layout | Identical to input layout |
| Value domain | float32 absolute-value results, including `+0.0`, `NaN`, and `+Inf` boundary results |
| Aliasing and mutability | The operator is functional: it must not mutate `self`; the returned value is the observable Abs result |
| Determinism | For the same input Tensor values and metadata, output values and metadata are deterministic |

For an empty Tensor, `y` is an empty Tensor with the same shape, dtype, rank, and layout as `self`.

## 7. Attribute Specification / 属性规格
Abs defines no operator attributes. There is no attribute type, default value, valid range, or attribute-input interaction.

## 8. Mathematical Semantics / 数学语义
Let the input tensor be \(x \in \mathbb{F}_{32}^{S}\), where \(S = (s_0, s_1, \ldots, s_{r-1})\) is any valid tensor shape with rank \(r \ge 0\), and \(\mathbb{F}_{32}\) is the set of IEEE 754 float32 values relevant to the PRD, including finite values, signed zeros, infinities, and NaN.

Abs is the mapping:

\[
\operatorname{Abs}: \mathbb{F}_{32}^{S} \rightarrow \mathbb{F}_{32}^{S}
\]

For every valid index \(i \in S\):

\[
y_i = \operatorname{Abs}(x)_i = |x_i|
\]

For finite real-valued float32 inputs:

\[
|a| =
\begin{cases}
a, & a \ge 0 \\
-a, & a < 0
\end{cases}
\]

The mapping is elementwise. It does not perform broadcasting, reduction, reshape, permutation, or any cross-element computation.

For an empty shape domain with no valid element indices, the mapping returns an empty tensor over the same shape \(S\) without elementwise values to compute.

## 9. Functional Semantics / 功能语义
Abs applies the float32 absolute-value operation independently to each element of `self`.

- Positive finite values return the same numeric value.
- Negative finite values return the corresponding positive value.
- `+0.0` returns `+0.0`.
- `-0.0` returns numeric `+0.0`.
- `+Inf` returns `+Inf`.
- `-Inf` returns `+Inf`.
- `NaN` returns `NaN`.

The output shape, rank, dtype, and layout are the same as the input. The operator has no broadcasting, no reduction, no reshape, and no cross-element dependency. The call must not modify the input Tensor value or metadata.

Behavior must be compatible with PyTorch ATen `aten::abs(Tensor self) -> Tensor` for float32 Tensor elementwise absolute value, within the constraints stated by the PRD.

## 10. Numeric Semantics / 数值语义
For all finite float32 values, the output is the exact float32 absolute-value result. Abs is a sign-processing elementwise operator, and the PRD defines no additional approximate error tolerance.

Signed zero behavior is fixed by the PRD: `+0.0` produces `+0.0`, and `-0.0` produces numeric `+0.0`.

Infinity behavior is fixed by the PRD: `+Inf` and `-Inf` both produce `+Inf`.

NaN behavior is fixed by the PRD at the observable value level: an input `NaN` produces an output `NaN`. The PRD does not require preservation of NaN payload or NaN sign bit, so this SPEC does not impose a payload or sign-bit preservation rule for NaN.

No rounding, overflow, or underflow rule beyond float32 absolute-value behavior is introduced by this SPEC.

## 11. Shape Semantics / Shape 语义
The output shape is exactly the input shape. The output rank is exactly the input rank. Scalar Tensor input with rank 0 returns scalar Tensor output with rank 0. Empty Tensor input, including any input shape with at least one zero-sized dimension, returns an empty Tensor with the same shape. No broadcasting, reduction, reshape, squeeze, unsqueeze, or symbolic dimension transformation is performed.

```python
import numpy as np

def abs(self):
    """InferShape reference for Abs.

    Args:
        self: Tensor metadata object for input `x`. It must expose shape
            metadata either as a `shape` attribute or as a dictionary entry named
            `shape`. The shape may be rank 0 `()`, any non-empty tuple/list of
            non-negative integer dimensions, or a shape containing one or more
            zero-sized dimensions.

    Returns:
        Tuple[int, ...]: Output shape metadata. The returned shape is exactly
        the input shape.

    Raises:
        TypeError: If shape metadata is unavailable or is not a sequence of
            dimensions for non-scalar tensors.
        ValueError: If any concrete dimension is negative.

    Shape-rule notes:
        Abs uses an identity shape rule. It performs no broadcasting, reduction,
        reshape, or rank change. Empty and scalar shapes are preserved.
    """
    shape = self.get("shape") if isinstance(self, dict) else getattr(self, "shape", None)
    if shape is None:
        raise TypeError("Abs InferShape requires tensor shape metadata")

    normalized_shape = tuple(np.atleast_1d(shape).tolist()) if shape != () else ()
    for dim in normalized_shape:
        if int(dim) < 0:
            raise ValueError("Abs shape dimensions must be non-negative")

    shape_rules = (
        ("identity", lambda input_shape: tuple(int(dim) for dim in input_shape)),
    )
    return shape_rules[0][1](normalized_shape)
```

## 12. Data Type Support / 数据类型支持
Abs supports float32 input Tensor only. The output dtype is float32 and is identical to the input dtype. The operator performs no dtype promotion, demotion, casting, or mixed-dtype behavior.

Inputs with dtype other than float32 are outside the supported functional scope of this SPEC. Non-Tensor inputs are also outside the supported functional scope.

```python
import numpy as np

def abs(self):
    """InferDtype reference for Abs.

    Args:
        self: Tensor metadata object for input `x`. It must expose dtype
            metadata either as a `dtype` attribute or as a dictionary entry named
            `dtype`. The only supported dtype is float32.

    Returns:
        numpy.dtype: Output dtype metadata. The returned dtype is `np.float32`.

    Raises:
        TypeError: If dtype metadata is unavailable or if the input dtype is not
            float32.

    Promotion-rule notes:
        Abs has no dtype promotion, no dtype conversion, and no mixed-dtype
        behavior. The single supported dtype rule maps float32 input to float32
        output.
    """
    dtype = self.get("dtype") if isinstance(self, dict) else getattr(self, "dtype", None)
    if dtype is None:
        raise TypeError("Abs InferDtype requires tensor dtype metadata")

    normalized_dtype = np.dtype(dtype)
    dtype_rules = (
        (np.dtype("float32"), np.dtype("float32")),
    )
    for input_dtype, output_dtype in dtype_rules:
        if normalized_dtype == input_dtype:
            return output_dtype

    raise TypeError("Abs supports only float32 input dtype")
```

## 13. Layout and Format Constraints / Layout 与 Format 约束
Abs does not change input layout or tensor format. The output layout is identical to the input layout.

The PRD does not introduce a contiguity requirement, a layout conversion requirement, or a special tensor format requirement beyond operating on a valid float32 Tensor in the target environment.

## 14. Boundary Cases / 边界场景
- Rank 0 scalar Tensor: output remains rank 0 scalar Tensor, with value equal to the scalar absolute value.
- Empty Tensor: output is empty, and shape, rank, dtype, and layout are identical to input.
- Tensor with one or more zero-sized dimensions: output preserves the same zero-sized dimensions.
- Positive finite values: output equals input numeric value.
- Negative finite values: output equals the corresponding positive value.
- `+0.0`: output is `+0.0`.
- `-0.0`: output numeric value is `+0.0`.
- `+Inf`: output is `+Inf`.
- `-Inf`: output is `+Inf`.
- `NaN`: output is `NaN`; NaN payload and sign-bit preservation are not required by the PRD.

## 15. Error Handling / 错误处理
Conforming calls require `self` to be a Tensor with dtype float32.

If the caller provides a non-Tensor input or a Tensor whose dtype is not float32, the input is unsupported by this SPEC. Such calls must not be reported as successful Abs computations under this SPEC. The exact diagnostic text and exception class are not specified by the PRD, but validation should be able to distinguish unsupported dtype or non-Tensor input from a successful float32 Tensor call.

Valid scalar Tensor and empty Tensor inputs must not be rejected solely because of rank 0 or zero element count.

## 16. Compatibility / 兼容性说明
Abs must be behaviorally compatible with PyTorch ATen `aten::abs(Tensor self) -> Tensor` for float32 Tensor elementwise absolute value.

The supported schema is:

```text
aten::abs(Tensor self) -> Tensor
```

The PRD does not require compatibility for dtype other than float32, complex values, quantized tensors, sparse tensors, non-Tensor inputs, gradients, reverse-mode behavior, or automatic differentiation behavior.

For NaN input, output must be NaN. The PRD explicitly does not require NaN payload or sign-bit preservation.

## 17. Performance Requirements / 性能要求
No additional measurable performance, throughput, or latency requirement is established by the PRD.

## 18. Acceptance Criteria / 验收标准
- Given float32 Tensor `x = [1.0, 2.5, +0.0]`, output `y` is `[1.0, 2.5, +0.0]`.
- Given float32 Tensor `x = [-1.0, -2.5, -0.0]`, output `y` has numeric values `[1.0, 2.5, +0.0]`.
- Given a float32 Tensor containing `NaN`, each corresponding output position is `NaN`.
- Given a float32 Tensor containing `+Inf` and `-Inf`, each corresponding output position is `+Inf`.
- Given any non-empty float32 Tensor with arbitrary valid rank and shape, output shape equals input shape exactly and output dtype is float32.
- Given a rank 0 float32 scalar Tensor, output is a rank 0 float32 scalar Tensor whose value is the input absolute value.
- Given an empty float32 Tensor, output is an empty Tensor with the same shape, rank, layout, and dtype as input.
- Given a valid float32 input Tensor, the operator does not mutate the input Tensor values or metadata.
- Shape inference returns exactly the input shape for scalar, empty, and non-empty Tensor shapes.
- Dtype inference accepts float32 input and returns float32 output without promotion or casting.
- Inputs outside float32 Tensor scope are not required to pass functional Abs tests for this SPEC.

## 19. Open Issues / 待确认问题
None
