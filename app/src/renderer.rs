use std::{
    io, iter,
    num::{NonZeroU32, NonZeroU8},
    path,
};

use bytemuck::{Pod, Zeroable};
use futures::{task, task::LocalSpawnExt};
use glam::swizzles::*;
use image::{imageops::index_colors, GenericImageView};
use log::info;
use wgpu::{util::DeviceExt, BindGroupLayoutEntry, SwapChainDescriptor, TextureAspect};
use wgpu_glyph::{ab_glyph, GlyphBrushBuilder};
use winit::{dpi, window};

use crate::{
    camera::{Camera, Projection},
    debug_gltf_json,
    texture::Texture,
    vertex::{DebugVertex, Vertex},
    world::World,
    CameraController,
};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    pub view_position: glam::Vec4,
    pub view_projection: glam::Mat4,
}

// unsafe impl Zeroable for Uniforms {}
// unsafe impl Pod for Uniforms {}

impl Uniforms {
    pub const SIZE: wgpu::BufferAddress = core::mem::size_of::<Self>() as wgpu::BufferAddress;

    pub fn update_view_projection(&mut self, camera: &Camera, projection: &Projection) {
        self.view_position =
            glam::Vec4::new(camera.position.x, camera.position.y, camera.position.z, 1.0);
        self.view_projection = projection.perspective() * camera.view_matrix();
    }
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Pod, Zeroable)]
pub struct Instance {
    position: glam::Vec4,
    rotation: glam::Quat,
}

impl Instance {
    pub const SIZE: wgpu::BufferAddress = core::mem::size_of::<glam::Mat4>() as wgpu::BufferAddress;
    /// This descriptor is a bit on the long side because we're doing a sort of manual conversion
    /// into shader `mat4` type, if you red this as being [Vec4; 4] it becomes clearer that this
    /// is a 4x4 matrix. Is there a simpler way of doing this? As it stands, having to specify
    /// each `shader_location` is error prone, and incovenient.
    ///
    /// The big difference between this and a `VertexBufferDescriptor` is the `step_mode`.
    ///
    /// TODO(alex): Improving this probably ties in with using `struct Instance` in the `.vert`.
    pub const DESCRIPTOR: wgpu::VertexBufferDescriptor<'static> = wgpu::VertexBufferDescriptor {
        stride: Self::SIZE,
        step_mode: wgpu::InputStepMode::Instance,
        attributes: &[
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                shader_location: 3,
                format: wgpu::VertexFormat::Float4,
            },
            wgpu::VertexAttributeDescriptor {
                offset: core::mem::size_of::<glam::Vec4>() as wgpu::BufferAddress,
                shader_location: 4,
                format: wgpu::VertexFormat::Float4,
            },
            wgpu::VertexAttributeDescriptor {
                offset: (core::mem::size_of::<glam::Vec4>() * 2) as wgpu::BufferAddress,
                shader_location: 5,
                format: wgpu::VertexFormat::Float4,
            },
            wgpu::VertexAttributeDescriptor {
                offset: (core::mem::size_of::<glam::Vec4>() * 3) as wgpu::BufferAddress,
                shader_location: 6,
                format: wgpu::VertexFormat::Float4,
            },
        ],
    };

    pub fn model_matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_rotation_translation(self.rotation, self.position.xyz())
    }
}

const NUM_INSTANCES_PER_ROW: u32 = 5;

/// TODO(alex): There needs to be a separation between things here and an actual `Pipeline`.
/// As it stands, `Renderer::new()` will create a pipeline with very specific details, that
/// can't be easily changed (I could set them in the `renderer` instance, but not a good solution),
/// so I think it's time to refactor this massive struct into smaller, configurable pieces that
/// belong together (be careful not to end up wrapping each thing individually).
/// This neccessity is being triggered by the `bind_group_layouts` requiring a uniform buffer
/// to be present during `Renderer` creation, but we don't actually have only a single uniform
/// buffer (we do now, but will need more later).
///
/// This same issue arises with the vertex buffer, it kinda looks like these buffers are tightly
/// coupled to their pipelines (1 vertex shader to 1 vertex buffer), and the renderer here has
/// the multiple pipelines.
///
/// I've tried doing it but ended up just in wrap-land, nothing good came out of it...
pub struct Renderer {
    /// The _window_ (canvas, surface to draw to, ...).
    surface: wgpu::Surface,
    /// The logical device (a handle to the GPU).
    device: wgpu::Device,
    /// How we send the commands due to the asynchronous nature of `wgpu` (vulkan).
    queue: wgpu::Queue,
    swap_chain_descriptor: wgpu::SwapChainDescriptor,
    /// NOTE(alex): Infrastructure for the queue of images waiting to be presented to the screen,
    /// in OpenGL there is a default `FrameBuffer`, but wgpu requires an infrastructure of
    /// that will own the buffers. That's why to render something, you access the next frame
    /// to be presented via `swap_chain.get_current_frame()`, fills this frame buffer,
    /// and then render it.
    swap_chain: wgpu::SwapChain,
    /// NOTE(alex): Different shaders require different pipelines, try to understand each pipeline
    /// as a different set of things. Shaders are programs so pipelines act as proccesses for each.
    /// I've seen the notion of having a single _uber shader_, but it doesn't seem like the best
    /// strategy. Each shader is an optmized program, and you need a pipeline for each.
    render_pipelines: Vec<wgpu::RenderPipeline>,
    vertex_buffers: Vec<wgpu::Buffer>,
    /// NOTE(alex): Indices can be thought of as pointers into the vertex buffer, they take care of
    /// duplicated vertices, by essentially treating each vertex as a "thing", that's why
    /// the pointer analogy is so fitting here.
    index_buffers: Vec<wgpu::Buffer>,
    /// NOTE(alex): Uniforms in wgpu (and vulkan) are different from uniforms in OpenGL.
    /// Here we can't dynamically set uniforms with a call like `glUniform2f(id, x, y);`, as
    /// it is set in stone at pipeline creation (`bind_group`).
    ///
    /// This does not mean that you can't change the values passed though (this restriction is in
    /// relation to the bind groups), you can still modify uniforms by writing to the buffer like
    /// we're doing here with the `staging_belt.copy_from_slice`.
    uniform_buffer: wgpu::Buffer,
    /// NOTE(alex): Trying to figure out how to move vertices without changing the model buffer,
    /// basically change them in the shader, instead of CPU.
    offset_buffer: wgpu::Buffer,
    /// NOTE(alex): wgpu recommends putting binding groups accordingly to usage, so bindings that
    // are run per-frame (change the least) be bind group 0, per-pass bind group 1, and
    /// per-material bind group 2.
    ///
    /// These can be thought of as resources in CPU that you want to pass to the GPU.
    /// They're linked to the shader stages (the `binding` in a vertex shader), so binding an
    /// uniform would be like `(layout = 0, binding = 1)`, where 1 is the index.
    bind_groups: Vec<wgpu::BindGroup>,
    // glyph_brush: wgpu_glyph::GlyphBrush<()>,
    /// Higher level API for handling something similar to staging buffers.
    ///
    /// It creates a threaded mechanism to handle copying our data (vertices, normals, ...) into
    /// a GPU buffer by using multiple buffers (chunks) with `Sender, Receiver` thread
    /// synchronization.
    ///
    /// After finding (or creating) a GPU buffer of appropriate size, it handles writing by
    /// calling `encoder.copy_buffer_to_buffer`. Which we would have to use in the case of
    /// staging buffers.
    /// [Encoder details](https://github.com/gfx-rs/wgpu-rs/blob/master/src/util/belt.rs#L96).
    ///
    /// `finish` closes the buffers (chunks) to writing until the GPU is done using them.
    ///
    /// `recall` does the heavy synchronization work, checking which buffers are done and moving
    /// those into a _free_ state.
    ///
    /// Vulkan requires specifying the memory type to be used in buffers when you create a buffer,
    /// same as in wgpu, with the big difference being that Vulkan has the flag
    /// `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` which makes the buffer not accessible by the CPU.
    /// Staging buffers in vulkan will have the GPU buffer with this flag, meanwhile we don't need
    /// it here in wgpu-land (doesn't exist), we just need to specify `wgpu::BufferUsage::COPY_DST`.
    staging_belt: wgpu::util::StagingBelt,
    size: dpi::PhysicalSize<u32>,
    /// NOTE(alex): If you think of the vertex buffer as having 1 object (in our bunch of squares
    /// to form a cube example, squares come together to form 1 object, the cube), then instancing
    /// is the creation of instances of this object. This is why transformations being in the
    /// instance buffer makes sense, these transformations are supposed to happen for specific
    /// objects (instances of objects).
    ///
    /// The workflow goes like this:
    /// 1. have an object (mesh with vertices, normals, texture coordinates);
    /// 2. put this object in a vertex buffer;
    /// 3. decide how many instances of this object you want;
    /// 4. have transformations for each instance (instance 1 color is red, 2 is blue, move-x 3);
    /// 5. put these transformations in an instance buffer;
    /// 6. draw instanced, with the desired instances.
    ///
    /// Keep in mind that, even though the vertex buffer (and the shader vectors, matrices) will be
    /// the same for each instance of an object, the instance buffer may contain whatever
    /// transformations are neccessary to do what you want. It may contain arbitrary data to, for
    /// example, indetify an instance (`instance_id` field) of an object. This would allow us to
    /// change selected instances by other fields than some index in an array.
    ///
    /// Thinking ahead, we could have a `HashMap<String, Instance>` of instances and have an object
    /// be rendered as "Aunt May", and another be "Mary Jane", sharing the vertex buffer object
    /// (a woman's body), but having a hair color transformation.
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    depth_texture: Texture,
    num_indices: usize,
    positions: Vec<u8>,
}

impl Renderer {
    /// The pipeline describes the configurable state of the GPU.
    ///
    /// Modern pipeline APIs (wgpu, vulkan) will require that most of the state description be
    /// configured in advance (at initialization), this means that if you need a slightly
    /// different vertex shader (or just different vertex layout), you'll need another pipeline.
    ///
    /// All of this means to me that my initial idea of an abstracted and highly configurable
    /// everything is kinda bust, we need an understanding of every layout before the app even
    /// starts, so dynamic configuration is more like dynamically selecting the correct pipeline,
    /// instead of changing some values in some struct.
    ///
    /// The `BindGroupLayout` and (its companions) is a way to tackle global variables in the GPU.
    /// This is closely related to uniforms (which are the shader globals), and the alternative to
    /// not using them, would be to keep updating the vertex buffers for small changes, that
    /// happen every single frame (or have a happen at a high-frequency).
    ///
    /// The `BindGroupLayout` specifies what the types of resources that are going to be accessed
    /// by the pipeline, it's a handle to the GPU-side of a binding group.
    ///
    /// The `BindGroupLayoutDescriptor` specifies the actual buffer or image resources that will
    /// be bound via each of its `entries`. It's bound to the drawing commands, just like the
    /// vertex buffers.
    ///
    /// The `entries` are similar to how vulkan has a set of descriptors that can be bound, in wgpu
    /// you don't need a descriptor pool, the `BindGroupLayoutDescriptor` handles this pool for us.
    pub fn create_render_pipeline(
        label: Option<&str>,
        device: &wgpu::Device,
        swap_chain_descriptor: &wgpu::SwapChainDescriptor,
        vertex_shader: wgpu::ShaderModuleSource,
        fragment_shader: wgpu::ShaderModuleSource,
        bind_group_layouts: &[&wgpu::BindGroupLayout],
        vertex_buffer_descriptors: &[wgpu::VertexBufferDescriptor],
    ) -> wgpu::RenderPipeline {
        let vs_module = device.create_shader_module(vertex_shader);
        let fs_module = device.create_shader_module(fragment_shader);

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts,
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label,
            layout: Some(&render_pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
                // polygon_mode: wgpu::PolygonMode::Fill,
            }),
            color_states: &[wgpu::ColorStateDescriptor {
                format: swap_chain_descriptor.format,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            // NOTE(alex): Part of the `Input Assembly`, what kind of geometry will be drawn.
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            // NOTE(alex): This does the depth testing.
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                // NOTE(alex): How the depth testing will compare z-ordering, tells when to discard
                // a new pixel, `Less` means pixels will be drawn front-to-back.
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilStateDescriptor::default(),
            }),
            vertex_state: wgpu::VertexStateDescriptor {
                // NOTE(alex): `IndexFormat` for 3D models in glTF will be specified in the json
                // file, so we can't really create the pipeline before we've looked through our
                // model data.
                // TODO(alex): This brings a new question of how do we have glTF models with
                // different `IndexFormat`s? Is this even desirable? Probably not, but is there a
                // way to convert the files to use the desired format?
                // index_format: wgpu::IndexFormat::Uint32,
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &vertex_buffer_descriptors,
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        render_pipeline
    }

    pub async fn new(window: &window::Window, world: &mut World) -> Self {
        let window_size = window.inner_size();

        // NOTE(alex): Handle to wgpu.
        // This is very similar to how vulkan is initialized, you first get a connection between
        // the application and wgpu.
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();
        info!("{:?} supported features {:?}.", adapter, adapter.features());

        // NOTE(alex): `device` is the logical device (a handle to the GPU).
        // `queue` is how we send the commands due to the asynchronous nature of `wgpu` (vulkan).
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    // label: Some("Main device descriptor"),
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    shader_validation: true,
                },
                None,
            )
            .await
            .unwrap();

        let render_format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let swap_chain_descriptor = wgpu::SwapChainDescriptor {
            // usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: render_format,
            width: window_size.width,
            height: window_size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        let swap_chain = device.create_swap_chain(&surface, &swap_chain_descriptor);

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind group for uniforms (shader globals)"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    // ty: wgpu::BindingType::Buffer {
                    //     min_binding_size: None,
                    //     ty: wgpu::BufferBindingType::Uniform,
                    //     has_dynamic_offset: false,
                    // },
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },

                    count: None,
                }],
            });
        // TODO(alex): This initialization is redundant, there has to be a way to create the
        // uniforms using the camera and projection values correctly.
        world.projection.aspect_ratio =
            swap_chain_descriptor.width as f32 / swap_chain_descriptor.height as f32;
        world
            .uniforms
            .update_view_projection(&world.camera, &world.projection);
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform buffer"),
            contents: bytemuck::cast_slice(&[world.uniforms]),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(uniform_buffer.slice(..)),
                // resource: wgpu::BindingResource::Buffer {
                //     buffer: &uniform_buffer,
                //     offset: 0,
                //     size: None,
                // },
            }],
        });

        let texture_bytes = include_bytes!("../../assets/tree.png");
        let texture = Texture::from_bytes(&device, &queue, texture_bytes).unwrap();
        // NOTE(alex): Similar to how using uniforms require specifying the layout, before you can
        // actually bind the data into the GPU (tell the GPU how this data should be used, how
        // it's structured, which shaders use it). Similar to vulkan's
        // `VkDescriptorSetLayoutBinding` and the `entries` are like `VkWriteDescriptorSet`.
        //
        // Keep in mind that this doesn't mean we can't transfer the texture data, the buffer
        // doesn't care about these details, this is how we connect bytes to shaders.
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Texture bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::SampledTexture {
                            dimension: wgpu::TextureViewDimension::D2,
                            component_type: wgpu::TextureComponentType::Uint,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        // TODO(alex): `comparison` has to do with linear filtering?
                        ty: wgpu::BindingType::Sampler { comparison: false },
                        count: None,
                    },
                ],
            });
        // NOTE(alex): `BindGroup`s are their own thing (separate from `BindGroupLayout`) as this
        // allows us to swap out bind groups, as long as they share the same layout. So we could
        // have different samplers here (with a different `anisotropy_clamp` value for example),
        // and swap between them at runtime. The `RenderPipeline` only depends on the layout, not
        // on the bind group data itself, so we don't even need to create a new pipeline if we
        // just wanted to change the things here.
        //
        // Notice that we pass the `TextureView` and `Sampler` in the bind group creation,
        // connecting (well, binding) the view and sampler we created for our texture, into the
        // shader. It uses the layout to determine what goes where (in our case both go into
        // fragment shader), and attaches the resources (view and sampler) to the binding.
        //
        // The tl;dr of these graphics APIs seems to be: have some resources, tell the API how
        // they're structured (their purpose, where they go), link this idea of resources
        // (the meta-resources, or descriptors) in CPU -> shaders, then copy the buffer with these
        // resources to the GPU.
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture bind group"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
            ],
        });

        // NOTE(alex): Instancing follows pretty much the same pattern as passing vertex data:
        // - have a `VertexBufferDescriptor` detailing how the data is structured for the shader;
        // - create a Buffer to put the instace data in;
        // - use it in the shader;
        // A good chunk of the code here is just about changing where each copy goes.
        const SPACE_BETWEEN: f32 = 5.0;
        let instances = (0..NUM_INSTANCES_PER_ROW)
            .flat_map(|z| {
                (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                    let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                    let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                    let position: glam::Vec3 = glam::Vec3::new(x as f32, 0.0, z as f32);

                    let rotation = if position == glam::Vec3::zero() {
                        // NOTE(alex): Quaternions can affect scale if they're not correct.
                        glam::Quat::from_axis_angle(glam::Vec3::unit_z(), f32::to_radians(0.0))
                    } else {
                        glam::Quat::from_axis_angle(
                            position.clone().normalize(),
                            f32::to_radians(45.0),
                        )
                    };

                    Instance {
                        position: glam::Vec4::new(position.x, position.y, position.z, 1.0),
                        rotation,
                    }
                })
            })
            .collect::<Vec<_>>();
        let instance_data = instances
            .iter()
            .map(Instance::model_matrix)
            .collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        });

        let offset_descriptor = wgpu::VertexBufferDescriptor {
            stride: core::mem::size_of::<glam::Vec2>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[wgpu::VertexAttributeDescriptor {
                offset: 0,
                format: wgpu::VertexFormat::Float2,
                shader_location: 10,
            }],
        };
        let offset_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Offset x y buffer"),
            contents: bytemuck::cast_slice(&[world.offset]),
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        });

        let staging_belt = wgpu::util::StagingBelt::new(1024);

        // TODO(alex): Try out the higher level API, now that I have a better understanding of the
        // glTF formats, and we know the renderer is working.
        let (positions, (indices, indices_count)) = debug_gltf_json();

        // TODO(alex): The shaders and descriptors are tightly coupled (for obvious reasons),
        // so it makes sense to handle every kind of possible `VertexBufferDescriptor` during
        // initialization (it doesn't seems to be a configurable feature). It might be refactored
        // out of here into a more clean `Pipeline` struct, but I guess that's about it.
        let hello_vs = wgpu::include_spirv!("./shaders/hello.vert.spv");
        let hello_fs = wgpu::include_spirv!("./shaders/hello.frag.spv");
        // NOTE(alex): The pipeline holds the buffer descriptors, it has to understand what kind of
        // data will be passed to the shaders, so the descriptors must be created before the
        // pipeline.
        // TODO(alex): This is a good candidate to be refactored out, as it doesn't need to be
        // initialized here. We need it initialized only at `present()`, as the pipeline requires
        // only the buffer descriptor, not the buffer itself. But where do we move it to?
        // The above also applies to uniform buffers (any kind of buffer really, only descriptors
        // are actually important at initialization).
        // NOTE(alex): The buffer is our way of passing data (vertex in this case) to the GPU,
        // the pipeline requires buffer descriptors (for each kind of buffer), but it doesn't hold
        // the buffers themselves, that's why you can create and fill these buffers anywhere
        // (before they're used, of course).
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            // contents: bytemuck::cast_slice(&world.vertices),
            contents: positions,
            // NOTE(alex): `usage: COPY_DST` is related to the staging buffers idea. This means that
            // this buffer will be used as the destination for some data.
            // The kind of buffer must also be specified, so you need the `VERTEX` usage here.
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            // contents: bytemuck::cast_slice(&world.indices),
            contents: indices,
            // NOTE(alex): We don't need `COPY_DST` here because this buffer won't be changing
            // value, if we think about these indices as being 1 geometric figure, they'll remain
            // the same, unless you wanted to quickly change it from a rectangle to some other
            // polygon.
            // Right now I don't see why you would need this, as when I think about 3D models,
            // they're not supposed to be deformed in this way, what we could do is apply
            // transformations to the vertices themselves, but the indices stay constant.
            usage: wgpu::BufferUsage::INDEX,
        });

        let depth_texture = Texture::create_depth_texture(&device, &swap_chain_descriptor);

        let hello_render_pipeline = Renderer::create_render_pipeline(
            Some("Pipeline: Hello"),
            &device,
            &swap_chain_descriptor,
            hello_vs,
            hello_fs,
            &[&uniform_bind_group_layout, &texture_bind_group_layout],
            &[Vertex::DESCRIPTOR, Instance::DESCRIPTOR, offset_descriptor],
        );

        let render_pipelines = vec![hello_render_pipeline];

        let font = ab_glyph::FontArc::try_from_slice(include_bytes!(
            "../../assets/Kosugi_maru/KosugiMaru-Regular.ttf"
        ))
        .unwrap();
        // let glyph_brush = GlyphBrushBuilder::using_font(font).build(&device, render_format);

        Self {
            surface,
            device,
            queue,
            swap_chain_descriptor,
            swap_chain,
            render_pipelines,
            vertex_buffers: vec![vertex_buffer],
            index_buffers: vec![index_buffer],
            uniform_buffer,
            offset_buffer,
            bind_groups: vec![uniform_bind_group, texture_bind_group],
            // glyph_brush,
            staging_belt,
            size: window_size,
            instances,
            instance_buffer,
            depth_texture,
            positions: positions.to_vec(),
            // NOTE(alex): When dealing with buffers directly, we want to pass the number of index
            // elements, not the length of the buffer itself.
            num_indices: indices_count as usize,
        }
    }

    /// WARNING(alex): This breaks if `new_size.width == 0 || new_size.height == 0` (minimized).
    pub fn resize(&mut self, new_size: dpi::PhysicalSize<u32>, world: &mut World) {
        self.size = new_size;
        self.swap_chain_descriptor.width = new_size.width;
        self.swap_chain_descriptor.height = new_size.height;
        world.projection.resize(new_size.width, new_size.height);
        self.swap_chain = self
            .device
            .create_swap_chain(&self.surface, &self.swap_chain_descriptor);
        // NOTE(alex): Remember that the depth texture must remain the same size as the swap chain
        // image.
        self.depth_texture =
            Texture::create_depth_texture(&self.device, &self.swap_chain_descriptor);
    }

    pub fn present(&mut self, world: &World, spawner: &impl task::LocalSpawn) {
        // NOTE(alex): To draw a triangle, the encoder sequence of commands is:
        // 1 - Begin the render pass;
        // 2 - Bind the pipeline;
        // 3 - Draw 3 vertices;
        // 4 - End the render pass.

        // NOTE(alex): The command encoder is equivalent to the vulkan `CommandBuffer`, where
        // you record the operations that are submitted to the queue.
        // We can have multiple encoders, each in a separate thread, so we set all the state
        // needed, and at the end just submit (`finish()`) it to the queue. Thanks to this
        // approach I don't think memory barriers are needed in the GPU side of things, unlike
        // in vulkan which would require this setup to be done in a thread-safe way, we get
        // thread safety for "free". Everything is ordered by the calling order of `finish`
        // to the queue.
        // NOTE(alex): Encoders are also neccessary for memory commands, that's why the
        // `StagingBelt.write_buffer` function has an `&mut Encoder` argument. These memory mapping
        // operations count as commands, and we could put them in different Encoders.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // for (index, vertices) in world.vertices.as_slice().iter().enumerate() {
        // NOTE(alex): My understanding of the `StagingBelt` so far is that, it's an easier
        // API to use than having staging buffers.
        // The staging buffers usage (for a vertex buffer) would be:
        // 1. Have 2 buffers, 1-CPU and 1-GPU bound;
        // 2. The GPU buffer has the copy attribute;
        // 3. We write the data into the CPU buffer whenever we want;
        // 4. Copy the CPU buffer to the GPU buffer when it's time to render the results;
        // 5. This is achieved with the `queue`;
        // Meanwhile the staging belt will handle this double buffering for us (including
        // the synchronization neccessary), and we just call `copy_from_slice(vertices)`.
        // The only handling we have to do, is to `recall` the buffers after we submitted
        // every change (including render pass) to the queue. This is needed to "unblock" the
        // buffers.
        // It's okay to create only 1 big vertex buffer (any kind of buffer), and we keep
        // writing and recalling from it (this buffer is a GPU buffer of the staging buffer
        // way), and any other vertex buffer we have, may be used for the different layouts
        // required by other vertex shaders. In this example, so far, we only need 1 vertex
        // buffer, and 1 uniform buffer (there's only 1 buffer layout for each).
        // self.staging_belt
        //     .write_buffer(
        //         &mut encoder,
        //         self.vertex_buffers.first().unwrap(),
        //         index as u64 * Vertex::SIZE,
        //         wgpu::BufferSize::new(Vertex::SIZE).unwrap(),
        //         &self.device,
        //     )
        //     .copy_from_slice(bytemuck::bytes_of(vertices));

        // NOTE(alex): This is how you would use a more traditional staging buffer.
        // encoder.copy_buffer_to_buffer(......)
        // }

        self.staging_belt
            .write_buffer(
                &mut encoder,
                &self.uniform_buffer,
                0,
                wgpu::BufferSize::new(Uniforms::SIZE).unwrap(),
                &self.device,
            )
            .copy_from_slice(bytemuck::bytes_of(&world.uniforms));

        // TODO(alex): Research how to change vertices values in GPU.
        // self.staging_belt
        //     .write_buffer(
        //         &mut encoder,
        //         &self.offset_buffer,
        //         0,
        //         wgpu::BufferSize::new(core::mem::size_of::<glam::Vec2>() as u64).unwrap(),
        //         &self.device,
        //     )
        //     .copy_from_slice(bytemuck::bytes_of(&[world.offset]));

        // TODO(alex): This gives an error:
        // "copy would end up overruning the bounds of one of the buffers or textures"
        // let model_size = Vertex::SIZE * self.positions.len() as u64;
        // TODO(alex): This doesn't make the kitten move either.
        // let model_size = self.positions.len() as u64;
        // self.staging_belt
        //     .write_buffer(
        //         &mut encoder,
        //         &self.vertex_buffers.get(0).unwrap(),
        //         0,
        //         wgpu::BufferSize::new(model_size).unwrap(),
        //         &self.device,
        //     )
        //     .copy_from_slice(&self.positions);

        let mut debug_count = 0;
        // for instance in self.instances.iter_mut() {
        //     instance.position.x += world.offset.x;
        //     instance.position.y += world.offset.y;
        //     if debug_count > 3 {
        //         break;
        //     }
        //     debug_count += 1;
        // }
        // NOTE(alex): Moving a model around can be done in many ways:
        // - uniform buffer objects (not a good solution);
        // - instance buffer objects;
        // TODO(alex): Take a look in renderdoc to see how this instance buffer is handled in the
        // shaders.
        self.instances.get_mut(0).unwrap().position.x += world.offset.x;
        self.instances.get_mut(0).unwrap().position.y += world.offset.x;
        self.instances.get_mut(5).unwrap().position.x += world.offset.x;
        self.instances.get_mut(5).unwrap().position.y += world.offset.x;
        self.instances.get_mut(10).unwrap().position.x += world.offset.x;
        self.instances.get_mut(10).unwrap().position.y += world.offset.x;
        self.instances.get_mut(16).unwrap().position.x += world.offset.x;
        self.instances.get_mut(16).unwrap().position.y += world.offset.x;
        self.instances.get_mut(22).unwrap().position.x += world.offset.x;
        self.instances.get_mut(22).unwrap().position.y += world.offset.x;
        self.instances.get_mut(24).unwrap().position.x += world.offset.x;
        self.instances.get_mut(24).unwrap().position.y += world.offset.x;
        let instance_data = self
            .instances
            .iter()
            .map(Instance::model_matrix)
            .collect::<Vec<_>>();
        self.staging_belt
            .write_buffer(
                &mut encoder,
                &self.instance_buffer,
                0,
                wgpu::BufferSize::new(Instance::SIZE * instance_data.len() as u64).unwrap(),
                &self.device,
            )
            .copy_from_slice(bytemuck::cast_slice(&instance_data));

        let _render_result = self
            .swap_chain
            .get_current_frame()
            .and_then(|frame| {
                {
                    // NOTE(alex): We may have multiple render passes, each using a different
                    // pipeline (or the same), to handle different contents (shaders, vertices).
                    // An example would be having a first render pass that draws diffuse geometry,
                    // by using a `diffuse_shader` and a second pass that does metal with
                    // `metal_shader`.
                    let mut first_render_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                                attachment: &frame.output.view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 0.1,
                                        g: 0.2,
                                        b: 0.3,
                                        a: 1.0,
                                    }),
                                    store: true,
                                },
                            }],
                            // NOTE(alex): Depth testing, notice that we use the
                            // `depth_texture.view`, and not the texture itself.
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachmentDescriptor {
                                    attachment: &self.depth_texture.view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: true,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                        });

                    first_render_pass.set_pipeline(&self.render_pipelines.first().unwrap());

                    // NOTE(alex): 3D Model index buffer.
                    first_render_pass
                        .set_index_buffer(self.index_buffers.first().unwrap().slice(..));

                    for (i, bind_group) in self.bind_groups.iter().enumerate() {
                        // NOTE(alex): The bind group index must match the `set` value in the
                        // shader, so:
                        // `layout(set = 0, ...)`
                        // Requires:
                        // `set_bind_group(0, ...)`.
                        first_render_pass.set_bind_group(i as u32, bind_group, &[]);
                    }

                    // NOTE(alex): 3D Model vertex buffer (and related).
                    first_render_pass
                        .set_vertex_buffer(0, self.vertex_buffers.get(0).unwrap().slice(..));

                    first_render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                    first_render_pass.set_vertex_buffer(2, self.offset_buffer.slice(..));

                    // NOTE(alex): wgpu API takes advantage of ranges to specify the offset
                    // into the vertex buffer. `0..len()` means that the `gl_VertexIndex`
                    // starts at `0`.
                    // `instances` range is the same, but for the `gl_InstanceIndex` used in
                    // instanced rendering.
                    // In vulkan, this function call would look like `(0, 0)`.
                    // first_render_pass.draw(0..world.vertices.len() as u32, 0..1);
                    first_render_pass.draw_indexed(
                        // 0..world.indices.len() as u32,
                        // 0..self.indices as u32,
                        0..self.num_indices as u32,
                        0,
                        // NOTE(alex): The main advantage of having this be a `Range<u32>` over
                        // just a number, is that with ranges it becomes possible to skip/select
                        // which instances to draw (how many copies).
                        0..self.instances.len() as _,
                    );
                }

                self.staging_belt.finish();

                // NOTE(alex): `encoder.finish` is called after all our operations are configured,
                // and the this sequence of commands is ready to be sent to the queue.
                // NOTE(alex): We could have multiple commands being submitted to the `queue`,
                // that's why `encoder.finish()` is put into an iterator (`Once<T>` is a single
                // element iterator).
                self.queue.submit(iter::once(encoder.finish()));

                let belt_future = self.staging_belt.recall();
                // TODO(alex): Why do we need this spawner?
                // Removing it, the triangle keeps rotating, no valdation errors, everything seems
                // just fine.
                // NOTE(alex): Maybe this is similar to the `vkQueueWaitIdle(graphicsQueue)` idea
                // from vulkan? Vulkan also allows using fences to deal with this same issue, they
                // allow for scheduling multiple transfers (memory) simultaneously.
                spawner.spawn_local(belt_future).unwrap();

                Ok(())
            })
            .or_else(|fail| match fail {
                wgpu::SwapChainError::Outdated => {
                    self.swap_chain = self
                        .device
                        .create_swap_chain(&self.surface, &self.swap_chain_descriptor);
                    Ok(())
                }
                fail => {
                    eprintln!("Render.present failed with -> {:?}", fail);
                    Err(fail)
                }
            });
    }
}
