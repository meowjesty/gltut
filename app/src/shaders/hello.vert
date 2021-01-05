#version 460

layout(location = 0) in vec3 position;
// layout(location = 2) in vec2 texture_coordinates;
// NOTE(alex): This is a `Transform` thingy that we have in Unity, the vertex `position` is being
// loaded directly from a huge glTF buffer, so we don't have control over inidividual vertices on
// the CPU-side. To change the current vertex (move it around, scale, rotate) we can only do it
// on the GPU-side (here), this means that for each vertex, we multiply its `position` by this
// transform matrix that is passed via instance buffer (so each instance of the model may end up
// with different `gl_Position` values).
layout(location = 3) in mat4 model_matrix;
// layout(location = 7) in vec3 normal;


layout(location = 0) out vec4 out_color;
// layout(location = 1) out vec2 out_texture_coordinates;
// layout(location = 2) out vec2 out_offset;

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
    // vec4 offset_position = vec4(position.x + offsetter.x, position.y + offsetter.y, position.z, 1.0);
    // vec4 position_with_camera = view_projection * model_matrix * offset_position;
    vec4 position_with_camera = view_projection * model_matrix * vec4(position, 1.0);
    // vec4 position_with_camera = view_projection * vec4(position, 1.0);

    gl_Position = position_with_camera;
    // out_color = vec4(color, 1.0);
    // out_texture_coordinates = texture_coordinates;
    // out_offset = offset;
    // out_offset = vec2(0.0, 0.0);
    out_color = vec4(model_matrix[0][1], model_matrix[1][2], model_matrix[2][2], 1.0);
}