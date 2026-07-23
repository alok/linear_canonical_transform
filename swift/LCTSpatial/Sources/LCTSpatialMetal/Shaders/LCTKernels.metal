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

inline float2 complex_sqrt(float2 value) {
    const float radius = length(value);
    const float realPart = sqrt(max((radius + value.x) * 0.5f, 0.0f));
    const float imaginaryMagnitude = sqrt(max((radius - value.x) * 0.5f, 0.0f));
    const float imaginaryPart = value.y < 0.0f ? -imaginaryMagnitude : imaginaryMagnitude;
    return float2(realPart, imaginaryPart);
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

/// Handles the finite-grid b = 0 branch as a chirped real-axis scaling:
/// sqrt(d) exp(iπ c d y² / n) f(dy), with linear interpolation.
kernel void lct_singular_axis_pass(
    device const float2 *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    constant float2 &c [[buffer(2)]],
    constant float2 &d [[buffer(3)]],
    constant uint4 &dimensions [[buffer(4)]],
    constant float &maximumExponent [[buffer(5)]],
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
    const float sourceCoordinate = d.x * y + center;
    if (sourceCoordinate < 0.0f || sourceCoordinate > float(length - 1)) {
        output[gid] = 0.0f;
        return;
    }

    const uint lower = uint(floor(sourceCoordinate));
    const uint upper = min(lower + 1, length - 1);
    const float fraction = sourceCoordinate - float(lower);
    const float2 lowerValue = input[lineStart + lower * elementStride];
    const float2 upperValue = input[lineStart + upper * elementStride];
    const float2 sample = mix(lowerValue, upperValue, fraction);
    const float2 piI = float2(0.0f, M_PI_F);
    const float2 phase = complex_mul(
        piI,
        complex_mul(c, d) * (y * y / float(length))
    );
    const float2 chirp = complex_exp_bounded(phase, maximumExponent);
    output[gid] = complex_mul(complex_mul(complex_sqrt(d), chirp), sample);
}

struct PreviewSourceVertex {
    float3 position;
    float3 normal;
};

struct PreviewVertex {
    float3 position;
    float3 normal;
    float4 color;
};

inline float3 phase_rgb(float phase, float luminance) {
    float hue = fract((phase + M_PI_F) / (2.0f * M_PI_F));
    float3 shifted = abs(fract(hue + float3(0.0f, 2.0f / 3.0f, 1.0f / 3.0f)) * 6.0f - 3.0f);
    float3 rgb = clamp(shifted - 1.0f, 0.0f, 1.0f);
    return mix(float3(luminance * 0.18f), rgb * luminance, 0.86f);
}

/// Updates a duplicated RealityKit mesh with the real and imaginary parts of
/// q' = a q + b p, where q is position and p is a scaled normal field.
kernel void canonical_preview_mesh(
    device const PreviewSourceVertex *source [[buffer(0)]],
    device PreviewVertex *output [[buffer(1)]],
    constant float2 &a [[buffer(2)]],
    constant float2 &b [[buffer(3)]],
    constant float4 &parameters [[buffer(4)]],
    constant uint &vertexCount [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= vertexCount) {
        return;
    }

    const PreviewSourceVertex sourceVertex = source[gid];
    const float3 companion = sourceVertex.normal * parameters.x;
    const float3 realPosition = a.x * sourceVertex.position + b.x * companion;
    const float3 imaginaryPosition = a.y * sourceVertex.position + b.y * companion;
    const float3 phaseProbe = normalize(float3(1.0f, 1.6180339f, 2.6180339f));
    const float realProbe = dot(realPosition, phaseProbe);
    const float imaginaryProbe = dot(imaginaryPosition, phaseProbe);
    const float phase = atan2(imaginaryProbe, realProbe);
    const float magnitude = length(float2(realProbe, imaginaryProbe));
    const float luminance = 0.25f + 0.75f * (1.0f - exp(-magnitude));
    const float3 rgb = phase_rgb(phase, luminance);

    output[gid].position = realPosition;
    output[gid].normal = sourceVertex.normal;
    output[gid].color = float4(rgb, 1.0f);

    output[gid + vertexCount].position = imaginaryPosition;
    output[gid + vertexCount].normal = sourceVertex.normal;
    output[gid + vertexCount].color = float4(rgb, 0.28f * parameters.y);
}
