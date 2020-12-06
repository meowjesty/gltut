use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    pub position: glam::Vec3,
    pub color: glam::Vec3,
    pub texture_coordinates: glam::Vec2,
}

impl Vertex {
    pub const SIZE: wgpu::BufferAddress = core::mem::size_of::<Self>() as wgpu::BufferAddress;
    pub const DESCRIPTOR: wgpu::VertexBufferDescriptor<'static> = wgpu::VertexBufferDescriptor {
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
                offset: core::mem::size_of::<glam::Vec3>() as wgpu::BufferAddress,
                shader_location: 1,
                format: wgpu::VertexFormat::Float3,
            },
        ],
    };

    // pub fn descriptor_3d<'x>() -> Box<wgpu::VertexBufferDescriptor<'x>> {
    //     let attributes = wgpu::vertex_attr_array![0 => Float3, 1 => Float3];
    //     Box::new(wgpu::VertexBufferDescriptor {
    //         stride: core::mem::size_of::<Self>() as wgpu::BufferAddress,
    //         step_mode: wgpu::InputStepMode::Vertex,
    //         attributes: &attributes,
    //     })
    // }

    // pub fn descriptor_2d<'x>() -> wgpu::VertexBufferDescriptor<'x> {
    //     wgpu::VertexBufferDescriptor {
    //         stride: core::mem::size_of::<Self>() as wgpu::BufferAddress,
    //         step_mode: wgpu::InputStepMode::Vertex,
    //         attributes: &wgpu::vertex_attr_array![0 => Float3],
    //     }
    // }
}
