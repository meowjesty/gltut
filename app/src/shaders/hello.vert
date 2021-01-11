#version 460

// NOTE(alex): `position` is in **model space**.
layout(location = 0) in vec3 position;
// NOTE(alex): You must pay very close attention to the `slot` when setting vertex buffers, we were
// having a weird strecthing issue because the buffers' slots were swapped, giving incorrect values.
layout(location = 1) in vec2 texture_coordinates;
// NOTE(alex): This is a `Transform` thingy that we have in Unity, the vertex `position` is being
// loaded directly from a huge glTF buffer, so we don't have control over inidividual vertices on
// the CPU-side. To change the current vertex (move it around, scale, rotate) we can only do it
// on the GPU-side (here), this means that for each vertex, we multiply its `position` by this
// transform matrix that is passed via instance buffer (so each instance of the model may end up
// with different `gl_Position` values).
layout(location = 10) in mat4 model;

layout(location = 0) out vec2 out_texture_coordinates;

layout(set = 0, binding = 0)
uniform Uniforms {
    // TODO(alex): If we multiply the `view_position`, we lose our screen for some reason that I
    // don't understand yet. It seems like this `view_position` is something to multiply the models
    // by, but not every single vertice?
    // NOTE(alex): Camera matrix:
    // [0] -> vec4 with camera position;
    // [1] -> vec4 looks at;
    // [2] -> vec4 up direction;
    mat4 view;
    // NOTE(alex): Projection matrix with FoV, aspect ratio, display range.
    mat4 projection;
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
    mat4 model_view_projection = projection * view * model;
    vec4 position_with_camera = model_view_projection * vec4(position, 1.0);

    gl_Position = position_with_camera;
    out_texture_coordinates = texture_coordinates;
}