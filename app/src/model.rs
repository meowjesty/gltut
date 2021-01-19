use std::path;

use image::GenericImageView;
use log::{error, info};

use crate::texture::Texture;

#[derive(Debug)]
pub(crate) struct Model {
    pub(crate) meshes: Vec<Mesh>,
    // pub(crate) textures: Vec<Mesh>,
    // pub(crate) materials: Vec<Mesh>,
}

#[derive(Debug)]
pub(crate) struct Image {
    pub(crate) size: wgpu::Extent3d,
    pub(crate) pixels: Vec<u8>,
    pub(crate) format: wgpu::TextureFormat,
}

/// TODO(alex): Fill out this struct.
/// ADD(alex): These are supposed to be passed as uniforms, so we need bind group.
/// ADD(alex): The materials should hold a pipeline reference, as they're the main driving force
/// between having multiple vertex and fragment shaders. The mesh only contains data that easily
/// belongs to a single shader combo, but the some materials will have textures, while others only
/// have colors, and so on. This is where the biggest difference exists in the models' metadata,
/// the optional and variable fields that require specialized shaders (`scene.gltf` vs `kitten.gltf`
/// is an example of this).
///
/// The pipeline here is non-owning (only a reference) because we'll have common materials that are
/// sharing the same shaders (they differ by value, not by kinds of data).
pub struct Material {
    /// TODO(alex): The texture in a glTF file is related to the `tangent` attribute and is named
    /// `normalTexture`, is this the same thing as an actual texture?
    pub(crate) texture: Texture,
    pub(crate) pbr_metallic_roughness: glam::Vec4,
    /// `RGBA`
    ///
    /// NOTE(alex): This is a multiplier for the fragment shader final color value, it's a base
    /// color of this material, a blue object would have `[0.0, 0.0, 1.0, 1.0]`, while a
    /// red object would be `[1.0, 0.0, 0.0, 1.0]`.
    pub(crate) base_color_factor: glam::Vec4,

    /// NOTE(alex): This determines how metallic the material is in a range of
    /// `0.0` being non-metal, and `1.0` being metal.
    ///
    /// Check the `assets/metallicRoughnessSpheres.png`.
    pub(crate) metallic_factor: f32,

    /// NOTE(alex): Mirror-like properties, `0.0` reflects, and `1.0` is completely rough.
    ///
    /// Check the `assets/metallicRoughnessSpheres.png`.
    pub(crate) roughness_factor: f32,
}

/// TODO(alex): Some of these fields are optional (or might be sets, instead of single elements).
/// This breaks the `kitten.gltf` model, as it doesn't have `TEXCOORD_0` nor related buffer.
/// Some models may have additional `TEXCOORD_1`, `TEXCOORD_2`.
///
/// ADD(alex): Even when handling optional values like this, it's not enough, as the validation
/// layers will complain that no value is passed to the shader (no texture coordinates in the model,
/// but a `in vec2 tex_coords` in the shader). This means that to solve this problem, either we
/// need an uber shader and use a single `in` with array of some type (which sounds like a
/// horrible solution tbh), or have different pipelines for different model cases (the better
/// approach, as we also need to specify the `IndexFormat` during pipeline creation).
///
/// The rest of what we're doing here seems good enough.
#[derive(Debug)]
pub(crate) struct Mesh {
    /// NOTE(alex): This is not used right now, but it could be useful when loading a whole `scene`
    /// in the future (reference the mesh `nodes` by id).
    pub(crate) id: usize,
    /// NOTE(alex): `Vec3<f32>`
    pub(crate) normals: Option<wgpu::Buffer>,
    /// NOTE(alex): `Vec3<f32>`
    pub(crate) positions: wgpu::Buffer,
    /// NOTE(alex): `Vec4<f32>`
    pub(crate) tangents: Option<wgpu::Buffer>,
    pub(crate) texture_coordinates: Option<wgpu::Buffer>,

    /// NOTE(alex): Indices can be thought of as pointers into the vertex buffer, they take care of
    /// duplicated vertices, by essentially treating each vertex as a "thing", that's why
    /// the pointer analogy is so fitting here.
    pub(crate) indices: Option<Indices>,
    // TODO(alex): Add this to the mesh, the big issue here will be how to use the material as a
    // reference, as `materials_buffer[0]` may be used by multiple meshes.
    // pub(crate) material: Option<&Material>,
}

#[derive(Debug)]
pub(crate) struct Indices {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) count: usize,
    pub(crate) format: wgpu::IndexFormat,
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
pub(crate) fn load_model<'x>(
    path: &path::Path,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Model {
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
    let mut images = Vec::with_capacity(4);
    info!("Loading geometry from file {:?}", path);

    for gltf_image in document.images() {
        info!("Loading image {:?}", gltf_image.name());
        let source = gltf_image.source();
        info!("{:?}", source);
        match source {
            gltf::image::Source::View { view, mime_type } => {}
            gltf::image::Source::Uri { uri, mime_type } => {
                let path_str = format!("./assets/{}", uri);
                let path = path::Path::new(&path_str);
                info!("image path {:?}", path);
                let dynamic_image = image::open(path).unwrap();
                let dimensions: (u32, u32) = dynamic_image.dimensions();
                let size = wgpu::Extent3d {
                    width: dimensions.0,
                    height: dimensions.1,
                    depth: 1,
                };

                // TODO(alex): `wgpu` doesn't support most of these formats, so we're going to
                // convert the image, instead of trying to use its correct format.
                // There's also an issue with size `16` image formats, as they become `Vec<u16>`,
                // instead of `Vec<u8>` which causes conflict. I'm out of ideas for solving this.
                /*
                let format = match dynamic_image {
                    image::DynamicImage::ImageLuma8(_) => {
                        error!("Image format not supported ImageLuma8, converting to rgba8.");
                        wgpu::TextureFormat::Rgba8UnormSrgb
                    }
                    image::DynamicImage::ImageLumaA8(_) => {
                        error!("Image format not supported ImageLumaA8, converting to rgba8.");
                        wgpu::TextureFormat::Rgba8UnormSrgb
                    }
                    image::DynamicImage::ImageRgb8(_) => {
                        error!("Image format not supported ImageRgb8, converting to rgba8.");
                        wgpu::TextureFormat::Rgba8UnormSrgb
                    }
                    image::DynamicImage::ImageRgba8(_) => wgpu::TextureFormat::Rgba8UnormSrgb,
                    image::DynamicImage::ImageBgr8(_) => {
                        error!("Image format not supported ImageBgr8, converting to rgba8.");
                        wgpu::TextureFormat::Rgba8UnormSrgb
                    }
                    image::DynamicImage::ImageBgra8(_) => wgpu::TextureFormat::Bgra8UnormSrgb,
                    image::DynamicImage::ImageLuma16(_) => {
                        error!("Image format not supported ImageLuma16, converting to rgba16.");
                        wgpu::TextureFormat::Rgba16Float
                    }
                    image::DynamicImage::ImageLumaA16(_) => {
                        error!("Image format not supported ImageLumaA16, converting to rgba16.");
                        wgpu::TextureFormat::Rgba16Float
                    }
                    image::DynamicImage::ImageRgb16(_) => {
                        error!("Image format not supported ImageRgb16, converting to rgba16.");
                        wgpu::TextureFormat::Rgba16Float
                    }
                    image::DynamicImage::ImageRgba16(_) => wgpu::TextureFormat::Rgba16Float,
                };
                */

                // NOTE(alex): We ignore the actual format and just perform a conversion here.
                let pixels = dynamic_image.to_rgba8().to_vec();
                let format = wgpu::TextureFormat::Rgba8UnormSrgb;

                let image = Image {
                    size,
                    pixels,
                    format,
                };

                images.push(image);
            }
        }
    }

    let mut textures = Vec::with_capacity(4);
    for gltf_texture in document.textures() {
        info!("Loading texture {:?}", gltf_texture.name());

        let image_index = gltf_texture.source().index();
        let image = images.get(image_index).unwrap();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: gltf_texture.name(),
            size: image.size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: image.format,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });

        queue.write_texture(
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &image.pixels,
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: 4 * image.size.width,
                rows_per_image: image.size.height,
            },
            image.size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: gltf_texture.name(),
            format: Some(image.format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            level_count: core::num::NonZeroU32::new(1),
            base_array_layer: 0,
            array_layer_count: core::num::NonZeroU32::new(1),
        });

        let gltf_sampler = gltf_texture.sampler();
        let address_mode_u = match gltf_sampler.wrap_s() {
            gltf::texture::WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            gltf::texture::WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
            gltf::texture::WrappingMode::Repeat => wgpu::AddressMode::Repeat,
        };
        let address_mode_v = match gltf_sampler.wrap_t() {
            gltf::texture::WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            gltf::texture::WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
            gltf::texture::WrappingMode::Repeat => wgpu::AddressMode::Repeat,
        };
        let mag_filter = match gltf_sampler.mag_filter() {
            Some(mag_filter) => match mag_filter {
                gltf::texture::MagFilter::Nearest => wgpu::FilterMode::Nearest,
                gltf::texture::MagFilter::Linear => wgpu::FilterMode::Linear,
            },
            None => wgpu::FilterMode::Linear,
        };
        let min_filter = match gltf_sampler.min_filter() {
            Some(min_filter) => match min_filter {
                gltf::texture::MinFilter::Nearest => wgpu::FilterMode::Nearest,
                gltf::texture::MinFilter::Linear => wgpu::FilterMode::Linear,
                gltf::texture::MinFilter::NearestMipmapNearest => wgpu::FilterMode::Nearest,
                gltf::texture::MinFilter::LinearMipmapNearest => wgpu::FilterMode::Linear,
                gltf::texture::MinFilter::NearestMipmapLinear => wgpu::FilterMode::Nearest,
                gltf::texture::MinFilter::LinearMipmapLinear => wgpu::FilterMode::Linear,
            },
            None => wgpu::FilterMode::Nearest,
        };

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: gltf_sampler.name(),
            address_mode_u,
            address_mode_v,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter,
            min_filter,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: core::f32::MAX,
            compare: None,
            anisotropy_clamp: core::num::NonZeroU8::new(16),
        });

        let texture = Texture {
            texture,
            view,
            sampler,
        };
        textures.push(texture);
    }

    // TODO(alex): Now I have everything needed to load the `materials` into wgpu-style data.
    // The issue becomes: `material.normalTexture.texCoord` is a link to a `Mesh`, but the
    // `mesh.material` is a link to a `Material`, we have cycle references :(.

    for mesh in document.meshes() {
        info!("Loading mesh {:?}", mesh.name());

        let id = mesh.index();
        let mut indices = None;
        let mut normals = None;
        let mut positions = None;
        let mut tangents = None;
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

                let index_format = match indices_accessor.data_type() {
                    gltf::accessor::DataType::U16 => wgpu::IndexFormat::Uint16,
                    gltf::accessor::DataType::U32 => wgpu::IndexFormat::Uint32,
                    invalid_type => panic!(
                        "Invalid index format type {:?} for mesh {:?}.",
                        invalid_type,
                        mesh.name()
                    ),
                };

                indices = Some(Indices {
                    buffer: indices_buffer,
                    count,
                    format: index_format,
                });
            }

            for (semantic, accessor) in primitive.attributes() {
                info!("attributes {:?}", accessor.index());
                let label = accessor.name();
                // NOTE(alex): Number of components, if we have VEC3 as the data type, then to get
                // the number of bytes would be something like `count * size_of(VEC3)`.
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

                info!("attributes -> {:?}", semantic);
                info!(
                    "count {:?}, bufferView {:?}, byteLength {:?}, byteOffset {:?}",
                    accessor.count(),
                    view.index(),
                    byte_length,
                    byte_offset
                );
                info!("attributes -> buffer len {:?}", attributes.len());

                match semantic {
                    gltf::Semantic::Positions => {
                        positions = Some(attributes_buffer);
                    }
                    gltf::Semantic::TexCoords(_) => {
                        texture_coordinates = Some(attributes_buffer);
                    }
                    gltf::Semantic::Normals => {
                        normals = Some(attributes_buffer);
                    }
                    gltf::Semantic::Tangents => {
                        tangents = Some(attributes_buffer);
                    }
                    gltf::Semantic::Colors(_) => {}
                    gltf::Semantic::Joints(_) => {}
                    gltf::Semantic::Weights(_) => {}
                }
            }
        }

        meshes.push(Mesh {
            id,
            normals,
            positions: positions.unwrap(),
            indices,
            tangents,
            texture_coordinates,
        });
    }

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
