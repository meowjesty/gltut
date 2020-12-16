#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
// layout(location = 1) in uvec3 color;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0)
uniform Uniforms {
    // TODO(alex): If we multiply the `view_position`, we lose our screen for some reason that I
    // don't understand yet. It seems like this `view_position` is something to multiply the models
    // by, but not every single vertice?
    vec4 view_position;
    mat4 view_projection;
};
// uniform Uniforms {
//     mat4 model;
//     mat4 view;
//     mat4 projection;
// };


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
    // TODO(alex): this crashes the app
    // gl_Position = vec4(position + total_offset, 1.0);

    // vec4 position_with_camera = projection * view * model * vec4(position, 1.0);
    vec4 position_with_camera = view_projection * vec4(position, 1.0);
    // vec4 position_with_camera = vec4(position, 1.0);

    gl_Position = position_with_camera;
    // out_color = view_projection * vec4(color, 1.0);
    out_color = vec4(color, 1.0);
}