#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
// layout(location = 1) in uvec3 color;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0)
uniform Uniforms {
    vec2 offset;
};

/*
struct Vertex {
    float vertex_x, vertex_y, vertex_z;
    float normal_x, normal_y, normal_z;
    float texture_u, texture_v;
};

layout(set = 1, binding = 0)
buffer Vertices {
    Vertex vertices[];
}
*/

void main() {
    // Vertex vertex = vertices[gl_VertexIndex];
    vec3 total_offset = vec3(offset.x, offset.y, 0.0);
    // TODO(alex): this crashes the app
    // gl_Position = vec4(position + total_offset, 1.0);
    gl_Position = vec4(position, 1.0);
    out_color = vec4(color, 1.0);
}