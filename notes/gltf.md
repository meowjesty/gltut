# glTF

This format is supposed to be used directly, it was created to avoid having importers, loaders and
parsers for your 3D data. The traditional way of doing things was:

1. Make an `.obj` file (or other 3D file format);

2. Have an importer/converter transform the data into what the user application wants;

3. Use this converted data in the runtime.

The new idea with glTF is to convert (if needed) whatever 3D files you have into it, and load the
glTF directly into the runtime application.

## Basic structure of glTF

The file describes a whole 3D scene in json:

- `scene` that contains a hierarchy of `nodes` that define a `scene graph`;

- The 3D objects in teh scene are defined using `meshes` that are attached to the `nodes`;

- `materials` define the appearance of the objects;

- `animations` describe how the 3D objects are transformed (rotated to translated) over time;

- `skins` define how the geometry of the objects is deformed based on a skeleton pose;

- `cameras` describe the view configuration for the renderer.

![glTF json structure](assets/gltfJsonStructure.png)

- [`scene`](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0/#reference-scene)
is the entry point for the description of the scene that is stored in the glTF. It refers to the
`node`s that define the scene graph.

- [`node`](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0/#reference-node)
is one node in the scene graph hierarchy. It can contain a transformation
(e.g., rotation or translation), and it may refer to further (child) nodes. Additionally, it may
refer to `mesh` or `camera` instances that are "attached" to the node, or to a `skin` that
describes a mesh deformation.

- [`camera`](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0/#reference-camera)
defines the view configuration for rendering the scene.

- [`mesh`](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0/#reference-mesh)
describes a geometric object that appears in the scene. It refers to `accessor` objects that are
used for accessing the actual geometry data, and to `material`s that define the appearance of the
object when it is rendered.

- [`skin`](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0/#reference-skin)
defines parameters that are required for vertex skinning, which allows the deformation of a mesh
based on the pose of a virtual character.
The values of these parameters are obtained from an `accessor`.

- [`animation`](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0/#reference-animation)
describes how transformations of certain nodes (e.g., rotation or translation) change over time.

- [`accessor`](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0/#reference-accessor)
is used as an abstract source of arbitrary data. It is used by the `mesh`, `skin`, and `animation`,
and provides the geometry data, the skinning parameters and the time-dependent animation values.
It refers to a
[`bufferView`](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0/#reference-bufferView),
which is a part of a
[`buffer`](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0/#reference-buffer)
that contains the actual raw binary data.

- [`material`](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0/#reference-material)
contains the parameters that define the appearance of an object. It usually refers to `texture`
objects that will be applied to the rendered geometry.

- [`texture`](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0/#reference-texture)
is defined by a
[`sampler`](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0/#reference-sampler)
and an
[`image`](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0/#reference-image).
The `sampler` defines how the texture `image` should be placed on the object.

## External data

Binary data is stored in a separate file (`model.bin`) and it contains the geometry (and other
binary friendly data, like textures), this data can be stored in a render-friendly format, skipping
the need to parse, decode, or preprocess it.

### [Minimal glTF](https://github.com/KhronosGroup/glTF-Tutorials/blob/master/gltfTutorial/gltfTutorial_003_MinimalGltfFile.md)
