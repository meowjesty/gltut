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


```
