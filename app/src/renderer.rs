use std::{
    iter,
    num::{NonZeroU32, NonZeroU8},
};

use bytemuck::{Pod, Zeroable};
use futures::{task, task::LocalSpawnExt};
use glam::swizzles::*;
use log::info;
use wgpu::{util::DeviceExt, BindGroupLayoutEntry, SwapChainDescriptor, TextureAspect};
use wgpu_glyph::{ab_glyph, GlyphBrushBuilder};
use winit::{dpi, window};

use crate::{vertex::Vertex, CameraController};

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

    pub fn model(&self) -> glam::Mat4 {
        glam::Mat4::from_rotation_translation(self.rotation, self.position.xyz())
    }
}

const NUM_INSTANCES_PER_ROW: u32 = 10;

pub type Radians = f32;

#[derive(Debug, Default)]
pub struct Camera {
    // pub eye: glam::Vec3,
    // target: glam::Vec3,
    // up: glam::Vec3,
    // aspect_ratio: f32,
    // fov_y: f32,
    // z_near: f32,
    // z_far: f32,
    /// X-axis moves the thumb towards the scren +;
    /// Y-axis moves the index towards the screen +;
    /// Z-axis moves the hand towards your nose +;
    pub position: glam::Vec3,
    pub yaw: Radians,
    pub pitch: Radians,
}

// TODO(alex): Is this correct for right-handed coordinates?
pub fn look_at_dir(eye: glam::Vec3, dir: glam::Vec3, up: glam::Vec3) -> glam::Mat4 {
    // NOTE(alex): Normalizing a vector that is already at length 1.0 changes nothing.
    let f = dir.normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);

    #[cfg_attr(rustfmt, rustfmt_skip)]
        glam::Mat4::from_cols_array_2d(&[
            [s.x.clone(), u.x.clone(), -f.x.clone(), 0.0],
            [s.y.clone(), u.y.clone(), -f.y.clone(), 0.0],
            [s.z.clone(), u.z.clone(), -f.z.clone(), 0.0],
            [-eye.dot(s), -eye.dot(u), eye.dot(f), 1.0],
        ])
}

impl Camera {
    pub fn view_matrix(&self) -> glam::Mat4 {
        // NOTE(alex): glam doesn't have a public version of look at direction
        // (`cgmath::loot_at_dir), that's why it was rotating around the center point, as this
        // `glam::look_at_rh` looks at the center (locks the center, not a direction).
        // TODO(alex): Is there a way to use `let view_matrix = glam::Mat4::look_at_rh(` correctly
        // by using look_at_rh formula with the correct center (check the math)?
        // TODO(alex): We don't need the `OPENGL_TO_WGPU_MATRIX`, things still look okay so far
        // without it.
        let view_matrix = look_at_dir(
            self.position,
            glam::Vec3::new(self.yaw.cos(), self.pitch.sin(), self.yaw.sin()).normalize(),
            glam::Vec3::unit_y(),
        );

        view_matrix
    }

    // pub fn view_projection_matrix(&self) -> glam::Mat4 {
    //     let view = glam::Mat4::look_at_rh(self.eye, self.target, self.up);
    //     let projection = glam::Mat4::perspective_rh(
    //         self.fov_y.to_radians(),
    //         self.aspect_ratio,
    //         self.z_near,
    //         self.z_far,
    //     );

    //     projection * view
    // }
}

// #[rustfmt::skip]
// pub const OPENGL_TO_WGPU_MATRIX: glam::Mat4 = glam::const_mat4!([
//     1.0, 0.0, 0.0, 0.0,
//     0.0, 1.0, 0.0, 0.0,
//     0.0, 0.0, 0.5, 0.0,
//     0.0, 0.0, 0.5, 1.0,
// ]);

#[derive(Debug, Default)]
pub struct Projection {
    pub aspect_ratio: f32,
    pub fov_y: f32,
    pub z_near: f32,
    pub z_far: f32,
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: glam::Mat4 = glam::const_mat4!([
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
]);
impl Projection {
    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect_ratio = width as f32 / height as f32;
    }

    pub fn perspective(&self) -> glam::Mat4 {
        let perspective =
            glam::Mat4::perspective_rh(self.fov_y, self.aspect_ratio, self.z_near, self.z_far);

        perspective
    }
}

#[derive(Debug, Default)]
pub struct World {
    pub vertices: Vec<Vertex>,
    pub(crate) indices: Vec<u32>,
    pub(crate) camera_controller: CameraController,
    pub(crate) camera: Camera,
    pub(crate) projection: Projection,
    pub(crate) uniforms: Uniforms,
}

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
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
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
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint32,
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
        let texture_image = image::load_from_memory(texture_bytes).unwrap();
        let texture_rgba = texture_image.as_rgba8().unwrap();
        let texture_dimensions = image::GenericImageView::dimensions(&texture_image);
        let texture_size = wgpu::Extent3d {
            width: texture_dimensions.0,
            height: texture_dimensions.1,
            depth: 1,
        };
        // NOTE(alex): This is somewhat equivalent to the vulkan `VkImage`, the main difference is
        // that memory handling is easier in wgpu.
        // NOTE(alex): 1D images can be used to store an array of data or gradient, 2D are mainly
        // used for textures (here), while 3D images can be used to store voxel volumes.
        // NOTE(alex): You could use the shader to access the buffer of pixels directly, but
        // it isn't optimal.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // TODO(alex): zeux said that this is the modern format that should be used, why?
            // format: wgpu::TextureFormat::Depth32Float,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            // NOTE(alex): `TextureUsage::SAMPLED` means optimal for shader, while
            // `TextureUsage::COPY_DST` is optimal as the destination in a transfer op (copy to).
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });
        // NOTE(alex): We only have to write this image to a buffer once (if we're not changing the
        // image), so copying the data during the renderer initialization is fine.
        // There is a legacy way of doing it, that involves copying it as a buffer, and doing the
        // `encoder`, `staging_belt` dance (this is how vulkan works with its staging buffers).
        // Notice that we can treat the texture as a buffer of bytes here, and later think about
        // how we want to apply modifications to it. The `TextureView` is how the data will be
        // actually used, so before we just need to specify what kind of bytes we have, how are
        // they formatted, size, and usage.
        // Later the `TextureView` will be created based on the `Texture`, but it doesn't really
        // care about buffer stuff.
        // And the `Sampler`s are even more independent, as they have no connection to neither
        // `Texture` or `TextureView`.
        queue.write_texture(
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            texture_rgba,
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: 4 * texture_dimensions.0,
                rows_per_image: texture_dimensions.1,
            },
            texture_size,
        );
        // NOTE(alex): Images are accessed by views rather than directly.
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Texture view"),
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            level_count: NonZeroU32::new(1),
            base_array_layer: 0,
            array_layer_count: NonZeroU32::new(1),
        });
        // NOTE(alex): How the texture (the texels) will be mapped into geometry, what kind of
        // filters should it apply.
        // The sampler is independent of the image (texture), it's a descriptor that can be used
        // for any image we want and it will apply these properties (filters) to it. It's an
        // interface to extract colors from a texture.
        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Texture sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: std::f32::MAX,
            // NOTE(alex): Texels will be compared to a value, and then the result will be used in
            // filtering operations (useful for shadow maps).
            compare: None,
            // TODO(alex): Is there an equivalent to vulkan's
            // `VkPhysicalDeviceProperties.limits.maxSamplerAnisotropy`?
            anisotropy_clamp: NonZeroU8::new(16),
        });
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
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
            ],
        });

        const SPACE_BETWEEN: f32 = 5.0;
        let instances = (0..NUM_INSTANCES_PER_ROW)
            .flat_map(|z| {
                (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                    let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                    let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                    let position: glam::Vec3 =
                        glam::Vec3::new(x as f32, 0.0, z as f32);

                    let rotation = if position == glam::Vec3::zero() {
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
        let instance_data = instances.iter().map(Instance::model).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsage::VERTEX,
        });

        let staging_belt = wgpu::util::StagingBelt::new(1024);

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
            contents: bytemuck::cast_slice(&world.vertices),
            // NOTE(alex): `usage: COPY_DST` is related to the staging buffers idea. This means that
            // this buffer will be used as the destination for some data.
            // The kind of buffer must also be specified, so you need the `VERTEX` usage here.
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&world.indices),
            // NOTE(alex): We don't need `COPY_DST` here because this buffer won't be changing
            // value, if we think about these indices as being 1 geometric figure, they'll remain
            // the same, unless you wanted to quickly change it from a rectangle to some other
            // polygon.
            // Right now I don't see why you would need this, as when I think about 3D models,
            // they're not supposed to be deformed in this way, what we could do is apply
            // transformations to the vertices themselves, but the indices stay constant.
            usage: wgpu::BufferUsage::INDEX,
        });
        let hello_render_pipeline = Renderer::create_render_pipeline(
            Some("Pipeline: Hello"),
            &device,
            &swap_chain_descriptor,
            hello_vs,
            hello_fs,
            &[&uniform_bind_group_layout, &texture_bind_group_layout],
            &[Vertex::DESCRIPTOR, Instance::DESCRIPTOR],
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
            bind_groups: vec![uniform_bind_group, texture_bind_group],
            // glyph_brush,
            staging_belt,
            size: window_size,
            instances,
            instance_buffer,
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
    }

    pub fn present(&mut self, world: &World, spawner: &impl task::LocalSpawn) {
        // for vertex_buffer in self.vertex_buffers.as_slice() {
        //     self.queue
        //         .write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&world.vertices));
        // }

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

        for (index, vertices) in world.vertices.as_slice().iter().enumerate() {
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
            self.staging_belt
                .write_buffer(
                    &mut encoder,
                    self.vertex_buffers.first().unwrap(),
                    index as u64 * Vertex::SIZE,
                    wgpu::BufferSize::new(Vertex::SIZE).unwrap(),
                    &self.device,
                )
                .copy_from_slice(bytemuck::bytes_of(vertices));

            // NOTE(alex): This is how you would use a more traditional staging buffer.
            // encoder.copy_buffer_to_buffer(......)
        }

        self.staging_belt
            .write_buffer(
                &mut encoder,
                &self.uniform_buffer,
                0,
                wgpu::BufferSize::new(Uniforms::SIZE).unwrap(),
                &self.device,
            )
            .copy_from_slice(bytemuck::bytes_of(&world.uniforms));

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
                                        g: 0.1,
                                        b: 0.1,
                                        a: 1.0,
                                    }),
                                    store: true,
                                },
                            }],
                            depth_stencil_attachment: None,
                        });

                    first_render_pass.set_pipeline(&self.render_pipelines.first().unwrap());
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

                    for (i, vertex_buffer) in self.vertex_buffers.iter().enumerate() {
                        first_render_pass.set_vertex_buffer(i as u32, vertex_buffer.slice(..));
                    }

                    first_render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

                    // NOTE(alex): wgpu API takes advantage of ranges to specify the offset
                    // into the vertex buffer. `0..len()` means that the `gl_VertexIndex`
                    // starts at `0`.
                    // `instances` range is the same, but for the `gl_InstanceIndex` used in
                    // instanced rendering.
                    // In vulkan, this function call would look like `(0, 0)`.
                    // first_render_pass.draw(0..world.vertices.len() as u32, 0..1);
                    first_render_pass.draw_indexed(
                        0..world.indices.len() as u32,
                        0,
                        0..self.instances.len() as _,
                    );
                }

                // self.glyph_brush
                //     .draw_queued(
                //         &self.device,
                //         &mut self.staging_belt,
                //         &mut encoder,
                //         &frame.output.view,
                //         self.swap_chain_descriptor.width,
                //         self.swap_chain_descriptor.height,
                //     )
                //     .unwrap();

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
