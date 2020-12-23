#version 460

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 texture_coordinates;

layout(location = 0) out vec4 out_color;

// NOTE(alex): This is the second set of bind groups, the first set belongs to the camera.
// The index is related to how these are created in the pipeline (the binding group layout order),
// and they're shared betwen shaders, so even though `set = 0` is only used by the vertex shader,
// it's an occupied number that cannot be reused here.
layout(set = 1, binding = 0) uniform texture2D texture_2d;
layout(set = 1, binding = 1) uniform sampler texture_sampler;

void main() {
    vec4 textured_color = texture(sampler2D(texture_2d, texture_sampler), texture_coordinates);
    // out_color = color;
    out_color = textured_color;
}