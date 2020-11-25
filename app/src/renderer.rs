use std::iter;

use wgpu::{util::DeviceExt, Texture};
use wgpu_glyph::{ab_glyph, GlyphBrushBuilder};
use winit::{dpi, window};

use crate::vertex::Vertex;

const VERTICES: &[Vertex] = &[
    // NOTE(alex): Position is done in counter-clockwise fashion, starting from the middle point
    // in this case.
    Vertex {
        position: glam::const_vec3!([0.0, 0.9, 0.0]), // middle point
        color: glam::const_vec3!([0.0, 0.0, 0.0]),
        texture_coordinates: glam::const_vec2!([0.0, 0.0]),
    },
    Vertex {
        position: glam::const_vec3!([-0.9, -0.9, 0.0]), // left-most point
        color: glam::const_vec3!([0.0, 0.0, 0.0]),
        texture_coordinates: glam::const_vec2!([0.0, 0.0]),
    },
    Vertex {
        position: glam::const_vec3!([0.9, -0.9, 0.0]), // right-most point
        color: glam::const_vec3!([0.0, 0.0, 0.0]),
        texture_coordinates: glam::const_vec2!([0.0, 0.0]),
    },
];

pub struct Renderer {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    swap_chain_descriptor: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
    // NOTE(alex): Different shaders require different pipelines, try to understand each pipeline
    // as a different set of things. Shaders are programs so pipelines act as proccesses for each.
    // I've seen the notion of having a single _uber shader_, but it doesn't seem like the best
    // strategy. Each shader is an optmized program, and you need a pipeline for each.
    render_pipelines: Vec<wgpu::RenderPipeline>,
    vertex_buffer: wgpu::Buffer,
    glyph_brush: wgpu_glyph::GlyphBrush<()>,
    staging_belt: wgpu::util::StagingBelt,
    size: dpi::PhysicalSize<u32>,
}

impl Renderer {
    pub fn create_render_pipeline(
        label: Option<&str>,
        device: &wgpu::Device,
        swap_chain_descriptor: &wgpu::SwapChainDescriptor,
        vertex_shader: wgpu::ShaderModuleSource,
        fragment_shader: wgpu::ShaderModuleSource,
        bind_group_layouts: Vec<&wgpu::BindGroupLayout>,
        vertex_buffer_descriptors: &[wgpu::VertexBufferDescriptor],
    ) -> wgpu::RenderPipeline {
        let vs_module = device.create_shader_module(vertex_shader);
        let fs_module = device.create_shader_module(fragment_shader);

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &bind_group_layouts,
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
            }),
            color_states: &[wgpu::ColorStateDescriptor {
                format: swap_chain_descriptor.format,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
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

    pub async fn new(window: &window::Window) -> Self {
        let window_size = window.inner_size();

        // Handle to the GPU
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        // device = logical device
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    shader_validation: true,
                },
                None,
            )
            .await
            .unwrap();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&VERTICES),
            usage: wgpu::BufferUsage::VERTEX,
        });

        let render_format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let swap_chain_descriptor = wgpu::SwapChainDescriptor {
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
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let staging_belt = wgpu::util::StagingBelt::new(1024);

        let hello_vs = wgpu::include_spirv!("./shaders/hello.vert.spv");
        let hello_fs = wgpu::include_spirv!("./shaders/hello.frag.spv");
        let hello_render_pipeline = Renderer::create_render_pipeline(
            Some("Pipeline: Hello"),
            &device,
            &swap_chain_descriptor,
            hello_vs,
            hello_fs,
            Vec::new(),
            &[Vertex::descriptor_3d()],
        );

        let render_pipelines = vec![hello_render_pipeline];

        let font = ab_glyph::FontArc::try_from_slice(include_bytes!(
            "../../assets/Kosugi_maru/KosugiMaru-Regular.ttf"
        ))
        .unwrap();
        let glyph_brush = GlyphBrushBuilder::using_font(font).build(&device, render_format);

        Self {
            surface,
            device,
            queue,
            swap_chain_descriptor,
            swap_chain,
            render_pipelines,
            vertex_buffer,
            glyph_brush,
            staging_belt,
            size: window_size,
        }
    }

    pub fn resize(&mut self, new_size: dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.swap_chain_descriptor.width = new_size.width;
        self.swap_chain_descriptor.height = new_size.height;
        self.swap_chain = self
            .device
            .create_swap_chain(&self.surface, &self.swap_chain_descriptor);
    }

    pub fn present(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let _render_result = self
            .swap_chain
            .get_current_frame()
            .and_then(|frame| {
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                            attachment: &frame.output.view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.8,
                                    g: 0.1,
                                    b: 0.4,
                                    a: 1.0,
                                }),
                                store: true,
                            },
                        }],
                        depth_stencil_attachment: None,
                    });

                    render_pass.set_pipeline(&self.render_pipelines.first().unwrap());
                    // TODO(alex): These are good candidates for refactoring into configurable
                    // functions (if we use multiple sources for vertices, we can't have only one
                    // vertex_buffer being the solely source of everything can we?).
                    render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                    render_pass.draw(0..3, 0..1);
                }

                self.glyph_brush
                    .draw_queued(
                        &self.device,
                        &mut self.staging_belt,
                        &mut encoder,
                        &frame.output.view,
                        self.swap_chain_descriptor.width,
                        self.swap_chain_descriptor.height,
                    )
                    .unwrap();

                self.staging_belt.finish();
                self.queue.submit(iter::once(encoder.finish()));

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
