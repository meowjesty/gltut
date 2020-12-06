use std::iter;

use bytemuck::{Pod, Zeroable};
use futures::{task, task::LocalSpawnExt};
use log::info;
use wgpu::util::DeviceExt;
use wgpu_glyph::{ab_glyph, GlyphBrushBuilder};
use winit::{dpi, window};

use crate::vertex::Vertex;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct Uniforms {
    offset: glam::Vec2,
}

unsafe impl Zeroable for Uniforms {}
unsafe impl Pod for Uniforms {}

#[derive(Debug, Default)]
pub struct World {
    pub vertices: Vec<Vertex>,
}

const VERTICES: &[Vertex] = &[
    // NOTE(alex): Position is done in counter-clockwise fashion, starting from the middle point
    // in this case.
    Vertex {
        position: glam::const_vec3!([0.0, 0.9, 0.0]), // middle point
        color: glam::const_vec3!([1.0, 0.0, 0.0]),
        texture_coordinates: glam::const_vec2!([0.0, 0.0]),
    },
    Vertex {
        position: glam::const_vec3!([-0.9, -0.9, 0.0]), // left-most point
        color: glam::const_vec3!([0.0, 1.0, 0.0]),
        texture_coordinates: glam::const_vec2!([0.0, 0.0]),
    },
    Vertex {
        position: glam::const_vec3!([0.9, -0.9, 0.0]), // right-most point
        color: glam::const_vec3!([0.0, 0.0, 1.0]),
        texture_coordinates: glam::const_vec2!([0.0, 0.0]),
    },
];

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
    /// TODO(alex): Think about double buffering this.
    vertex_buffers: Vec<wgpu::Buffer>,
    /// NOTE(alex): Uniforms in wgpu (and vulkan) are different from uniforms in OpenGL.
    /// Here we can't dynamically set uniforms with a call like `glUniform2f(id, x, y);`, as
    /// it is set in stone at pipeline creation (`bind_group`).
    uniform_buffers: Vec<wgpu::Buffer>,
    bind_groups: Vec<wgpu::BindGroup>,
    // glyph_brush: wgpu_glyph::GlyphBrush<()>,
    staging_belt: wgpu::util::StagingBelt,
    size: dpi::PhysicalSize<u32>,
}

impl Renderer {
    /// The pipeline describes the configurable state of the GPU.
    ///
    /// Modern pipeline APIs (wgpu, vulkan) will require that most of the state description be
    /// configured in advance (at initialization), this means that if you need a slightly
    /// different vertex shader (or just different vertex layout), you'll need another pipeline.
    ///
    /// All of this means to me that my initial idea of abstracted and highly configurable
    /// everything is kinda bust, we need an understanding of every layout before the app even
    /// starts, so dynamic configuration is more like dynamically selecting the correct pipeline,
    /// instead of changing some values in some struct.
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
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &vertex_buffer_descriptors,
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        render_pipeline
    }

    pub async fn new(window: &window::Window, world: &World) -> Self {
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

        let sample_uniform = Uniforms {
            offset: glam::const_vec2!([0.01, 0.01]),
        };

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
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform buffer"),
            contents: bytemuck::cast_slice(&[sample_uniform]),
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
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        });
        let hello_render_pipeline = Renderer::create_render_pipeline(
            Some("Pipeline: Hello"),
            &device,
            &swap_chain_descriptor,
            hello_vs,
            hello_fs,
            &[&uniform_bind_group_layout],
            &[Vertex::DESCRIPTOR],
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
            uniform_buffers: vec![uniform_buffer],
            bind_groups: vec![uniform_bind_group],
            // glyph_brush,
            staging_belt,
            size: window_size,
        }
    }

    /// WARNING(alex): This breaks if `new_size.width == 0 || new_size.height == 0` (minimized).
    pub fn resize(&mut self, new_size: dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.swap_chain_descriptor.width = new_size.width;
        self.swap_chain_descriptor.height = new_size.height;
        self.swap_chain = self
            .device
            .create_swap_chain(&self.surface, &self.swap_chain_descriptor);
    }

    // TODO(alex): Staging buffer!
    // Figure out how to use the `StagingBelt` to copy data from CPU memory to GPU memory, the
    // way we're doing right now (`queue.write_buffer`) is inneficient, as the CPU has access to
    // this memory, and we want to have 2 distinct buffers (memory regions), one that's used by
    // the CPU (where we change things, `World`), and one where we just copy the changes into
    // (a GPU buffer).
    // This is done in vulkan via staging buffers, wgpu also supports this way, but it has this
    // staging belt concept that seems more high level (try it first).
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
        }

        self.staging_belt.finish();

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

                    for (index, bind_group) in self.bind_groups.iter().enumerate() {
                        // NOTE(alex): The bind group index must match the `set` value in the
                        // shader, so:
                        // `layout(set = 0, ...)`
                        // Requires:
                        // `set_bind_group(0, ...)`.
                        first_render_pass.set_bind_group(index as u32, bind_group, &[]);
                    }

                    for (index, vertex_buffer) in self.vertex_buffers.iter().enumerate() {
                        first_render_pass.set_vertex_buffer(index as u32, vertex_buffer.slice(..));
                        // NOTE(alex): wgpu API takes advantage of ranges to specify the offset
                        // into the vertex buffer. `0..len()` means that the `gl_VertexIndex`
                        // starts at `0`.
                        // `instances` range is the same, but for the `gl_InstanceIndex` used in
                        // instanced rendering.
                        // In vulkan, this function call would look like `(0, 0)`.
                        first_render_pass.draw(0..VERTICES.len() as u32, 0..1);
                    }
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

                // NOTE(alex): `encoder.finish` is called after all our operations are configured,
                // and the this sequence of commands is ready to be sent to the queue.
                // NOTE(alex): We could have multiple commands being submitted to the `queue`,
                // that's why `encoder.finish()` is put into an iterator (`Once<T>` is a single
                // element iterator).
                self.queue.submit(iter::once(encoder.finish()));

                let belt_future = self.staging_belt.recall();
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
