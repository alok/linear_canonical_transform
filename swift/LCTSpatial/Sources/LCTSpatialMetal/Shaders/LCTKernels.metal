#include <metal_stdlib>
using namespace metal;

inline float2 complex_add(float2 lhs, float2 rhs) {
    return lhs + rhs;
}

inline float2 complex_mul(float2 lhs, float2 rhs) {
    return float2(
        lhs.x * rhs.x - lhs.y * rhs.y,
        lhs.x * rhs.y + lhs.y * rhs.x
    );
}

inline float2 complex_div(float2 lhs, float2 rhs) {
    float denominator = dot(rhs, rhs);
    return float2(
        (lhs.x * rhs.x + lhs.y * rhs.y) / denominator,
        (lhs.y * rhs.x - lhs.x * rhs.y) / denominator
    );
}

inline float2 complex_exp_bounded(float2 value, float maximumReal) {
    float scale = exp(clamp(value.x, -maximumReal, maximumReal));
    return scale * float2(cos(value.y), sin(value.y));
}

/// Applies one separable LCT axis to a row-major complex field.
kernel void lct_axis_pass(
    device const float2 *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    constant float2 &a [[buffer(2)]],
    constant float2 &b [[buffer(3)]],
    constant float2 &d [[buffer(4)]],
    constant uint4 &dimensions [[buffer(5)]],
    constant float &maximumExponent [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    const uint length = dimensions.x;
    const uint elementStride = dimensions.y;
    const uint totalCount = dimensions.w;
    if (gid >= totalCount) {
        return;
    }

    const uint outputCoordinate = (gid / elementStride) % length;
    const uint lineStart = gid - outputCoordinate * elementStride;
    const float center = float(length - 1) * 0.5f;
    const float y = float(outputCoordinate) - center;
    const float2 denominator = b * float(length);
    const float2 piI = float2(0.0f, M_PI_F);
    float2 sum = 0.0f;

    for (uint inputCoordinate = 0; inputCoordinate < length; ++inputCoordinate) {
        const float x = float(inputCoordinate) - center;
        float2 numerator = a * (x * x);
        numerator += float2(-2.0f * x * y, 0.0f);
        numerator += d * (y * y);
        const float2 phase = complex_mul(piI, complex_div(numerator, denominator));
        const float2 kernelValue = complex_exp_bounded(phase, maximumExponent);
        const uint sourceIndex = lineStart + inputCoordinate * elementStride;
        sum = complex_add(sum, complex_mul(input[sourceIndex], kernelValue));
    }

    output[gid] = sum * rsqrt(float(length));
}
