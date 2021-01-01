# GPU Buffers

My understanding so far (01/01/2020) for each kind of buffer (`vertex, uniform, index, instance`)
is as follows:

- `vertex buffer` is supposed to be treated as an object with its vertices,
`position, colors, normals, texture coordinates`, these are some of the components you want
this object to have. You don't want to change its values on the CPU-side, these buffers will be
big (mesh loading), so changing things inside it, in cpu code, is finnicky at best, especially when
we're dealing with `glTF` binary buffer data directly (load it directly into a GPU buffer). The
best way to make changes, let's say to the object position, is by using other kinds of buffers that
interact with vertices (vertex by vertex for each `gl_VertexID`);

- `uniform buffer` are globals that don't have different values for individual objects, that's why
they're a perfect candidate for holding the `view, projection` matrices, as every single object in
a scene will going through the same camera space transformation. Whenever we want every object to
suffer the same transformation, uniforms come to the rescue.

- `instance buffer` handle instances of objects, say we have 10 characters in a scene, they may all
share the same model (generic robot villain), but can't have all 10 occupying the same position,
so we store transformations in an `Instance` and apply these transformations to the characters we
want (`gl_InstanceID`). These transformations will happen to each vertex, and we select which
instance we want (on the CPU-side) with some form of indexing, and the GPU-side with an instance id.
This is how we change colors of these robotic villains, while also moving them independently of
each other, something that would require having 10 different `vertex buffers` with the fully loaded
(same) model, and we would also need to identify in this model the positions, colors and so on,
change them (on the CPU-side), and then pack it back into the `vertex buffer` to see the changes on
the screen.

For more on the `glsl` variables mentioned read the
[wiki](https://www.khronos.org/opengl/wiki/Vertex_Shader/Defined_Inputs).
