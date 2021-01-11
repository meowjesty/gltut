use std::path;

use log::info;

#[derive(Debug)]
pub(crate) struct Model {
    pub(crate) meshes: Vec<Mesh>,
}

/// TODO(alex): Some of these fields are optional (or might be sets, instead of single elements).
/// This breaks the `kitten.gltf` model, as it doesn't have `TEXCOORD_0` nor related buffer.
/// Some models may have additional `TEXCOORD_1`, `TEXCOORD_2`.
#[derive(Debug)]
pub(crate) struct Mesh {
    pub(crate) positions: wgpu::Buffer,
    pub(crate) texture_coordinates: wgpu::Buffer,
    /// NOTE(alex): Indices can be thought of as pointers into the vertex buffer, they take care of
    /// duplicated vertices, by essentially treating each vertex as a "thing", that's why
    /// the pointer analogy is so fitting here.
    pub(crate) indices: wgpu::Buffer,
    pub(crate) indices_count: usize,
}

/// NOTE(alex): This is a higher level approach to loading the data, it uses the glTF-crate
/// iterator API and we get types in rust-analyzer. The other approach of directly reading from the
/// glTF-json API ends up accomplishing the same things, except that by using `include_bytes`
/// instead of `gltf::import` we get a `'static` lifetime for our buffers, which allows us to
/// easily return slices, meanwhile in the iterator API we're loading the buffer locally, meaning
/// that we can't just return slices into it (can't borrow something allocated inside a
/// non-static lifetime function).
///
/// This function is a bit more complete than the direct json manipulation one, as it loops through
/// the whole scene looking for data, while the other just goes straight to meshes.
///
/// [Meshes](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#meshes)
///
/// How to deal with texture coordinate sets (TEXCOORD_0, _1, ...):
/// https://gamedev.stackexchange.com/questions/132001/multiple-texture-coordinates-per-vertex-directx11
///
/// TODO(alex): This issue might be easy to solve, but I don't want to deal with lifetimes
/// right now, so we're just allocating vectors and returning them. Must revisit this later.
///
/// TODO(alex): We correctly load the `ship_light.gltf` file, a more complex model than the
/// `kitten.gltf`. Now I have to read the other data:
///
/// - `NORMAL`
/// - `TANGENT`
///
/// TODO(alex): Load material data and bind the textures to the drawn model.
/// TODO(alex): Load texture coordinate sets correctly, we're currently loading every set in 1
/// buffer, which probably breaks when the model actually has a `TEXCOORD_0, TEXCOORD_1`.
/// This doesn't work even when we have only 1 set! Why? Check RenderDoc.
/// TODO(alex): We need to get the correct types for some of these buffers, `TEXCOORD_0` may be
/// a `Vec2<f32>`, or `Vec2<u16>`, or `Vec2<u8>`.
/// TODO(alex): To get the mesh data, we don't need to walk through the `scenes`, this would only
/// be neccessary if we wanted to render the whole scene as it's described in the glTF file.
/// I'm not interested in this right now, but it doesn't affect us in any way to **keep things as
/// they are**.
// TODO(alex): Load the model texture (`textures` folder for `scene.gltf`).
pub(crate) fn load_model<'x>(path: &path::Path, device: &wgpu::Device) -> Model {
    use core::mem::*;
    let (document, buffers, _images) = gltf::import(path).expect("Could not open gltf file.");

    // NOTE(alex): So apparently there is no need to translate the positions, normals and so on
    // into our custom structures, it seems like a waste of effort when glTF already has everything
    // nicely packed into its buffers.
    // To access we index into each buffer by type, grab from offset to length + offset and look
    // up what type of buffer data this is (`target`).
    // It looks like there are more aspects to this whole process, like how to handle the whole
    // scene, maybe we can navigate from scene->meshes->buffers? Would this be neccessary though?
    // And finally, how would we change the positions, when all we have are buffers of data, but no
    // way do stuff like `vertex.x * some_rotation * delta_time`? I'll start with this direct
    // approach of buffer->gpu, and later I see how to efficiently load individual vertices into
    // a `Vertex`, then `Mesh` and whatever else is needed.
    // I'm not even sure right now what transform movement entails.
    // Well, thinking about it a bit better, to change position we could just pass the
    // transformation value (vector or matrix) to the shader, and change the vertices there, as
    // the shader will have complete access to each vertex.
    // NOTE(alex): The glTF primitives and buffer view relation works as follows:
    // - `POSITION: 0` means that the buffer view 0 holds every position `vec3`;
    // - `NORMAL: 1` means that the buffer view 1 holds every normal `vec3`;
    // - `TANGENT: 2` means that the buffer view 2 holds every tangent `vec4`;
    // - `TEXCOORD_0: 3` means that the buffer view 3 holds every texture coordinates `vec2`;
    // The main difference will be seen on `indices`, which will be different for each `Primitive`,
    // so `primitives[0] -> indices: 4`, while `primitives[1] -> indices: 5`, this same behaviour
    // is seen on `primitives -> material.
    // let mut indices = Vec::with_capacity(4);
    // let mut positions = Vec::with_capacity(4);
    // let mut counts = Vec::with_capacity(4);
    // let mut texture_coordinates = Vec::with_capacity(4);
    use wgpu::util::{BufferInitDescriptor, DeviceExt};

    // for scene in document.scenes() {
    // for node in scene.nodes() {
    // if let Some(mesh) = node.mesh() {
    let mut meshes = Vec::with_capacity(4);
    info!("Loading geometry from file {:?}", path);
    for mesh in document.meshes() {
        info!("Loading mesh {:?}", mesh.name());

        let mut indices = None;
        let mut indices_count = 0;
        let mut positions = None;
        let mut texture_coordinates = None;

        for primitive in mesh.primitives() {
            info!("primitive {:?}", primitive.index());

            if let Some(indices_accessor) = primitive.indices() {
                let label = indices_accessor.name();
                let count = indices_accessor.count();
                let view = indices_accessor.view().unwrap();
                let index = view.buffer().index();
                let offset = view.offset();
                let length = view.length();
                let buffer = buffers.get(index).unwrap();
                let indices_buffer = device.create_buffer_init(&BufferInitDescriptor {
                    label,
                    contents: &buffer[offset..offset + length],
                    // NOTE(alex): We don't need `COPY_DST` here because this buffer won't
                    // be changing value, if we think about these indices as being
                    // 1 geometric figure, they'll remain the same, unless you wanted to quickly
                    // change it from a rectangle to some other polygon.
                    // Right now I don't see why you would need this, as when I think about
                    // 3D models, they're not supposed to be deformed in this way,
                    // what we could do is apply transformations to the vertices themselves,
                    // but the indices stay constant.
                    usage: wgpu::BufferUsage::INDEX,
                });
                info!(
                    "indices -> count {:?}, bufferView {:?}, byteLength {:?}, byteOffset {:?}",
                    count,
                    view.index(),
                    length,
                    offset
                );
                indices = Some(indices_buffer);
                indices_count = count;
            }

            for (semantic, accessor) in primitive.attributes() {
                info!("attributes {:?}", accessor.index());
                let label = accessor.name();
                // NOTE(alex): Number of components, if we have VEC3 as the data type, then to get
                // the number of bytes would be something like `count * size_of(VEC3)`.
                let data_type = accessor.data_type();
                let data_size = match data_type {
                    gltf::accessor::DataType::I8 => size_of::<i8>(),
                    gltf::accessor::DataType::U8 => size_of::<u8>(),
                    gltf::accessor::DataType::I16 => size_of::<i16>(),
                    gltf::accessor::DataType::U16 => size_of::<u16>(),
                    gltf::accessor::DataType::U32 => size_of::<u32>(),
                    gltf::accessor::DataType::F32 => size_of::<f32>(),
                };
                let dimensions = accessor.dimensions();
                let size_bytes = match dimensions {
                    gltf::accessor::Dimensions::Scalar => data_size,
                    gltf::accessor::Dimensions::Vec2 => data_size * 2,
                    gltf::accessor::Dimensions::Vec3 => data_size * 3,
                    gltf::accessor::Dimensions::Vec4 => data_size * 4,
                    gltf::accessor::Dimensions::Mat2 => data_size * 2 * 2,
                    gltf::accessor::Dimensions::Mat3 => data_size * 3 * 3,
                    gltf::accessor::Dimensions::Mat4 => data_size * 4 * 4,
                };
                let count = accessor.count();
                // FIXME(alex): This doesn't always agree with the actual length of the buffer.
                let length = size_bytes * count;
                let view = accessor.view().unwrap();
                let byte_offset = view.offset();
                let byte_length = view.length();
                // NOTE(alex): `bufferView: 2` this is the buffer view index in the accessors.
                // let view_index = view.index();
                let index = view.buffer().index();
                // TODO(alex): This will always be `0` for our `scene.gltf` model.
                assert_eq!(index, 0);
                let buffer = buffers.get(index).unwrap();
                let attributes = &buffer[byte_offset..byte_offset + byte_length];
                let attributes_buffer = device.create_buffer_init(&BufferInitDescriptor {
                    label,
                    contents: attributes,
                    // NOTE(alex): `usage: COPY_DST` is related to the staging buffers idea.
                    // This means that this buffer will be used as the destination for some data.
                    // The kind of buffer must also be specified, so you need the
                    // `VERTEX` usage here.
                    usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
                });

                match semantic {
                    gltf::Semantic::Positions => {
                        info!("attributes -> {:?}", semantic);
                        info!(
                            "count {:?}, bufferView {:?}, byteLength {:?}, byteOffset {:?}",
                            count,
                            view.index(),
                            byte_length,
                            byte_offset
                        );
                        info!("attributes -> buffer len {:?}", attributes.len());
                        positions = Some(attributes_buffer);
                    }
                    gltf::Semantic::TexCoords(set_index) => {
                        info!("attributes -> {:?}", semantic);
                        info!(
                            "count {:?}, bufferView {:?}, byteLength {:?}, byteOffset {:?}",
                            count,
                            view.index(),
                            byte_length,
                            byte_offset
                        );
                        info!("attributes -> buffer len {:?}", attributes.len());
                        texture_coordinates = Some(attributes_buffer);
                    }
                    _ => (),
                }
            }
        }

        meshes.push(Mesh {
            positions: positions.unwrap(),
            indices: indices.unwrap(),
            indices_count,
            texture_coordinates: texture_coordinates.unwrap(),
        });
    }
    // }
    // }
    let model = Model { meshes };

    model
}

pub fn debug_accessor(accessor: &gltf::Accessor, mesh_name: &str) -> String {
    let count = accessor.count();
    // "bufferView": 3,
    let index = accessor.index();
    // "max": "Some(Array([Number(375.3972473144531), Number(326.95660400390625)]))",
    let max = accessor.max();
    // "min": "Some(Array([Number(-375.3972473144531), Number(-519.8281860351563)]))",
    let min = accessor.min();
    let name = accessor.name();
    let normalized = accessor.normalized();
    let offset = accessor.offset();
    let size = accessor.size();

    let view = accessor.view().unwrap();
    let byte_offset = view.offset();

    format!(
        r#"
        {:?}: {{
            "count": "{:?}",
            "index": "{:?}",
            "max": "{:?}",
            "min": "{:?}",
            "name": "{:?}",
            "normalized": "{:?}",
            "offset": "{:?}",
            "size": "{:?}"
            "byte_offset": "{:?}"
        }}
    "#,
        mesh_name, count, index, max, min, name, normalized, offset, size, byte_offset
    )
}
