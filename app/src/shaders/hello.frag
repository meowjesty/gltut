#version 460

// layout(location = 0) in vec4 color;
// layout(location = 1) in vec2 texture_coordinates;
layout(location = 2) in vec2 offset;

layout(location = 0) out vec4 out_color;

// NOTE(alex): This is the second set of bind groups, the first set belongs to the camera.
// The index is related to how these are created in the pipeline (the binding group layout order),
// and they're shared betwen shaders, so even though `set = 0` is only used by the vertex shader,
// it's an occupied number that cannot be reused here.
layout(set = 1, binding = 0) uniform texture2D texture_2d;
layout(set = 1, binding = 1) uniform sampler texture_sampler;

void main() {
    // vec4 textured_color = texture(sampler2D(texture_2d, texture_sampler), texture_coordinates);
    // out_color = color;
    // out_color = textured_color * color;
    // float x = 0.0;
    // if (offset.x > 0.0) {
    //     x = 1.0;
    // } else if (offset.x < 0.0) {
    //     x = 0.3;
    // }
    // float y = 0.0;
    // if (offset.y > 0.0) {
    //     y = 1.0;
    // } else if (offset.y < 0.0) {
    //     y = 0.3;
    // }

    // out_color = vec4(0.0 + offset.x, 0.0 + offset.y, 1.0, 1.0);
    // TODO(alex): Think I've finally figured it out! The reason we don't see the changes in the
    // kittens, but we see the vertex movement + change of color in the weird line, is because
    // we only write to the kitten vertex buffer once. Even if we don't change the buffer values
    // every frame (the whole model doesn't need to change like that), we still need to copy the
    // buffer to a new draw call so the kitten is able to load the values here.
    // Currently the kitten starts with `offset = vec3(0)`, that's why they're black (0, 0, 0).
    // It's not so simple though, check `let model_size =` in `renderer.rs`.
    // TODO(alex): The problem is actually that we have only 1 offset value, it gets passed to the
    // first vertex index, so every other vertex won't have access to it. That's why we're getting
    // a weird line that changes color and moves around, this line is the single vertex position
    // that receives the offset value and adds it. This is not the way to pass data to a model,
    // we need an uniform, just like the camera and projection matrices, uniforms are passed to
    // every vertex (hence the global aspect to them).
    // TODO(alex): But then, how would we move different models by different amounts? Having 1
    // uniform transform for each model doesn't seem like the proper way to do it. Should I
    // append a vector (or matrix) at the end of the model buffer (the model vertex buffer), and
    // just say something like "location 3 starts at buffer.len()"? This is where a `Transform`
    // abstraction will come in handy, as we can use it to treat the model as 1 vector (or matrix),
    // as if we're dealing only with the origin of the model, move it around in CPU within game
    // logic, and then pass this transform at the end of model buffer. This way we don't ever need
    // to parse the mesh vertices like so many tutorials do, we treat the model as a **thing** and
    // use the transform matrix to act on it on the CPU side.
    // NOTE(alex): Moving a model around can be done in many ways:
    // - uniform buffer objects (not a good solution);
    // - instance buffer objects;
    out_color = vec4(1.0, 0.2, 0.5, 1.0);
}