# Scratchpad

```rust
// NOTE(alex): We need buffers for the depth passes.
        let depth_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Depth Pass Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        dimension: wgpu::TextureViewDimension::D2,
                        component_type: wgpu::TextureComponentType::Float,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: true },
                    count: None,
                },
            ],
        });
        let depth_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Depth Pass Bind Group"),
            layout: &depth_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&depth_texture.sampler),
                },
            ],
        });
        let depth_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Depth Pass Vertex Buffer"),
            contents: bytemuck::cast_slice(DEPTH_VERTICES),
            usage: wgpu::BufferUsage::VERTEX,
        });
        let depth_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Depth Pass Index Buffer"),
            contents: bytemuck::cast_slice(DEPTH_INDICES),
            usage: wgpu::BufferUsage::INDEX,
        });
        let depth_pipeline_layout = device.create_pipeline_layout(&wgpu::)


#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq, Pod, Zeroable)]
pub struct DepthVertex {
    pub position: glam::Vec3,
    pub texture_coordinates: glam::Vec2,
}

impl DepthVertex {
    pub const SIZE: wgpu::BufferAddress = size_of::<Self>() as wgpu::BufferAddress;
    pub const DESCRIPTOR: wgpu::VertexBufferDescriptor<'static> = wgpu::VertexBufferDescriptor {
        stride: Self::SIZE,
        step_mode: wgpu::InputStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float3,
            },
            wgpu::VertexAttributeDescriptor {
                offset: VEC3_SIZE as wgpu::BufferAddress,
                shader_location: 1,
                format: wgpu::VertexFormat::Float2,
            },
        ],
    };
}



const DEPTH_VERTICES: &[DepthVertex] = &[
    DepthVertex {
        position: glam::const_vec3!([0.0, 0.0, 0.0]),
        texture_coordinates: glam::const_vec2!([0.0, 1.0]),
    },
    DepthVertex {
        position: glam::const_vec3!([1.0, 0.0, 0.0]),
        texture_coordinates: glam::const_vec2!([1.0, 1.0]),
    },
    DepthVertex {
        position: glam::const_vec3!([1.0, 1.0, 0.0]),
        texture_coordinates: glam::const_vec2!([0.0, 1.0]),
    },
    DepthVertex {
        position: glam::const_vec3!([0.0, 1.0, 0.0]),
        texture_coordinates: glam::const_vec2!([0.0, 0.0]),
    },
];

const DEPTH_INDICES: &[u16] = &[0, 1, 2, 0, 2, 3];

struct DepthPass {
    texture: Texture,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_depth_indices: u32,
    render_pipeline: wgpu::RenderPipeline,
}

vertex.rs
pub const DESCRIPTOR_CUSTOM: wgpu::VertexBufferDescriptor<'static> =
        wgpu::VertexBufferDescriptor {
            stride: Self::SIZE,
            step_mode: wgpu::InputStepMode::Vertex,
            // attributes: &wgpu::vertex_attr_array![0 => Float3, 1 => Float3],
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float3,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: VEC3_SIZE as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float3,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: (VEC3_SIZE * 2) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float2,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: (VEC3_SIZE * 2 + VEC2_SIZE) as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float2,
                },
            ],
        };


renderer.rs

/*
        let path = path::Path::new("./assets/kitten.gltf");
        let (document, buffers, images) = gltf::import(path).expect("Could not open gltf file.");
        let mut positions = None;
        let mut normals = None;
        let mut indices = None;
        for mesh in document.meshes() {
            for primitive in mesh.primitives() {
                if let Some(accessor) = primitive.indices() {
                    let view = accessor.view().unwrap();
                    let index = view.buffer().index();
                    let offset = view.offset();
                    let length = view.length();
                    indices = Some(&buffers.get(index).unwrap()[offset..offset + length]);

                    println!(
                        "primitive: offset {:?} length {:?} index {:?}",
                        offset, length, index
                    );
                }

                for (semantic, accessor) in primitive.attributes() {
                    let view = accessor.view().unwrap();
                    let offset = view.offset();
                    let length = view.length();
                    // let stride = view.stride().unwrap_or(1);
                    let buffer_view = view.buffer();
                    let index = buffer_view.index();
                    let buffer = &buffers.get(index).unwrap()[offset..offset + length];

                    let offset = accessor.offset() * accessor.data_type().size();
                    let length = accessor.count() * accessor.data_type().size();

                    /*
                    semantic: offset 0 length 59424 index 0
                    semantic: offset 713088 length 59424 index 0
                                        */

                    println!(
                        "semantic {:?}: offset {:?} length {:?} index {:?}",
                        semantic, offset, length, index
                    );

                    match semantic {
                        gltf::Semantic::Positions => {
                            positions = Some(&buffer[offset..offset + length]);
                        }
                        gltf::Semantic::Normals => {
                            normals = Some(&buffer[offset..offset + length]);
                        }
                        gltf::Semantic::Tangents => {}
                        gltf::Semantic::Colors(_) => {}
                        gltf::Semantic::TexCoords(_) => {}
                        gltf::Semantic::Joints(_) => {}
                        gltf::Semantic::Weights(_) => {}
                    }
                }
            }
        }
        */

{
                    let mut debug_render_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                                attachment: &frame.output.view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 1.0,
                                        g: 0.1,
                                        b: 0.1,
                                        a: 1.0,
                                    }),
                                    store: true,
                                },
                            }],
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

                    debug_render_pass.set_pipeline(&self.render_pipelines.first().unwrap());

                    // NOTE(alex): 3D Model index buffer.
                    debug_render_pass
                        .set_index_buffer(self.index_buffers.get(1).unwrap().slice(..));

                    for (i, bind_group) in self.bind_groups.iter().enumerate() {
                        // NOTE(alex): The bind group index must match the `set` value in the
                        // shader, so:
                        // `layout(set = 0, ...)`
                        // Requires:
                        // `set_bind_group(0, ...)`.
                        debug_render_pass.set_bind_group(i as u32, bind_group, &[]);
                    }

                    debug_render_pass
                        .set_vertex_buffer(2 as u32, self.vertex_buffers.get(1).unwrap().slice(..));

                    debug_render_pass.set_vertex_buffer(2, self.offset_buffer.slice(..));
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

 // NOTE(alex): Copy the debug vertices.
        let debug_buffer_len = world.debug_vertices.len() as u64 * DebugVertex::SIZE;
        self.staging_belt
            .write_buffer(
                &mut encoder,
                &self.vertex_buffers.get(1).unwrap(),
                0,
                wgpu::BufferSize::new(debug_buffer_len).unwrap(),
                &self.device,
            )
            .copy_from_slice(bytemuck::cast_slice(&world.debug_vertices));
layout(location = 24) in vec3 debug_position;
layout(location = 25) in vec3 debug_color;
layout(location = 26) in vec2 debug_texture_coordinates;

let debug_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Debug Vertex Buffer"),
            contents: bytemuck::cast_slice(&world.debug_vertices),
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        });
        let debug_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Debug Index Buffer"),
            contents: bytemuck::cast_slice(&world.debug_indices),
            usage: wgpu::BufferUsage::INDEX | wgpu::BufferUsage::COPY_DST,
        });


&[

                Vertex::DESCRIPTOR,
                Instance::DESCRIPTOR,
                offset_descriptor,
                DebugVertex::DESCRIPTOR,
            ],

```
