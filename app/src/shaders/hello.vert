#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec4 out_color;

layout(set = 1, binding = 0)
uniform vec2 offset;

void main() {
    vec4 total_offset = vec4(offset.x, offset.y, 0.0, 0.0);
    gl_Position = vec4(position + total_offset, 1.0);
    out_color = vec4(color, 1.0);
}