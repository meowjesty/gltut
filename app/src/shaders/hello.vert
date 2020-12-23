#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 texture_coordinates;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_texture_coordinates;

layout(set = 0, binding = 0)
uniform Uniforms {
    // TODO(alex): If we multiply the `view_position`, we lose our screen for some reason that I
    // don't understand yet. It seems like this `view_position` is something to multiply the models
    // by, but not every single vertice?
    vec4 view_position;
    mat4 view_projection;
};

/*
// TODO(alex): How do we use this? Also, figure out #includes.
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
    vec4 position_with_camera = view_projection * vec4(position, 1.0);

    gl_Position = position_with_camera;
    out_color = vec4(color, 1.0);
    out_texture_coordinates = texture_coordinates;
}