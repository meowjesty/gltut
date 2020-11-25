#version 460

layout(location = 0) out vec4 out_color;

void main() {
    float lerp_value = gl_FragCoord.y / 768.0;

    out_color = mix(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.2, 0.2, 0.2, 1.0), lerp_value);
}