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
    //     let mut attributes = wgpu::vertex_attr_array![0 => Float3, 1 => Float3];
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

pub fn cube(
    origin: glam::Vec3,
    size: f32,
    index: u32,
    color: glam::Vec3,
) -> (Vec<Vertex>, Vec<u32>) {
    let mut back = vec![
        Vertex {
            position: glam::Vec3::new(origin.x, origin.y, origin.z),
            color: glam::Vec3::new(1.0, 0.0, 0.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x + size, origin.y, origin.z),
            color: glam::Vec3::new(1.0, 0.0, 0.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x, origin.y + size, origin.z),
            color: glam::Vec3::new(1.0, 0.0, 0.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x + size, origin.y + size, origin.z),
            color: glam::Vec3::new(1.0, 0.0, 0.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
    ];
    let mut back_indices = vec![
        0 + index,
        1 + index,
        2 + index,
        3 + index,
        2 + index,
        1 + index,
    ];

    let mut front = vec![
        Vertex {
            position: glam::Vec3::new(origin.x, origin.y, origin.z + size),
            color: glam::Vec3::new(0.0, 0.0, 1.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x + size, origin.y, origin.z + size),
            color: glam::Vec3::new(0.0, 0.0, 1.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x, origin.y + size, origin.z + size),
            color: glam::Vec3::new(0.0, 0.0, 1.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x + size, origin.y + size, origin.z + size),
            color: glam::Vec3::new(0.0, 0.0, 1.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
    ];
    let mut front_indices = vec![
        0 + 4 + index,
        1 + 4 + index,
        2 + 4 + index,
        3 + 4 + index,
        2 + 4 + index,
        1 + 4 + index,
    ];

    let mut left = vec![
        Vertex {
            position: glam::Vec3::new(origin.x, origin.y, origin.z),
            color: glam::Vec3::new(color.x, color.y, color.z + size),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x, origin.y, origin.z + size),
            color: glam::Vec3::new(color.x, color.y, color.z + size),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x, origin.y + size, origin.z),
            color: glam::Vec3::new(color.x, color.y, color.z + size),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x, origin.y + size, origin.z + size),
            color: glam::Vec3::new(color.x, color.y, color.z + size),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
    ];
    let mut left_indices = vec![
        0 + 8 + index,
        1 + 8 + index,
        2 + 8 + index,
        3 + 8 + index,
        2 + 8 + index,
        1 + 8 + index,
    ];

    let mut right = vec![
        Vertex {
            position: glam::Vec3::new(origin.x + size, origin.y, origin.z),
            color: glam::Vec3::new(1.0, 0.0, 0.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x + size, origin.y, origin.z + size),
            color: glam::Vec3::new(0.0, 1.0, 0.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x + size, origin.y + size, origin.z),
            color: glam::Vec3::new(0.0, 0.0, 1.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x + size, origin.y + size, origin.z + size),
            color: glam::Vec3::new(0.0, 1.0, 1.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
    ];
    let mut right_indices = vec![
        0 + 12 + index,
        1 + 12 + index,
        2 + 12 + index,
        3 + 12 + index,
        2 + 12 + index,
        1 + 12 + index,
    ];

    let mut top = vec![
        Vertex {
            position: glam::Vec3::new(origin.x, origin.y + size, origin.z + size),
            color: glam::Vec3::new(0.0, 0.0, 0.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x + size, origin.y + size, origin.z + size),
            color: glam::Vec3::new(0.0, 0.0, 0.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x, origin.y + size, origin.z),
            color: glam::Vec3::new(0.0, 0.0, 0.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x + size, origin.y + size, origin.z),
            color: glam::Vec3::new(0.0, 0.0, 0.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
    ];
    let mut top_indices = vec![
        0 + 16 + index,
        1 + 16 + index,
        2 + 16 + index,
        3 + 16 + index,
        2 + 16 + index,
        1 + 16 + index,
    ];

    let mut bottom = vec![
        Vertex {
            position: glam::Vec3::new(origin.x, origin.y, origin.z + size),
            color: glam::Vec3::new(1.0, 1.0, 1.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x + size, origin.y, origin.z + size),
            color: glam::Vec3::new(1.0, 1.0, 1.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x, origin.y, origin.z),
            color: glam::Vec3::new(1.0, 1.0, 1.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
        Vertex {
            position: glam::Vec3::new(origin.x + size, origin.y, origin.z),
            color: glam::Vec3::new(1.0, 1.0, 1.0),
            texture_coordinates: glam::const_vec2!([0.0, 0.0]),
        },
    ];
    let mut bottom_indices = vec![
        0 + 20 + index,
        1 + 20 + index,
        2 + 20 + index,
        3 + 20 + index,
        2 + 20 + index,
        1 + 20 + index,
    ];

    back.append(&mut front);
    back.append(&mut left);
    back.append(&mut right);
    back.append(&mut top);
    back.append(&mut bottom);

    back_indices.append(&mut front_indices);
    back_indices.append(&mut left_indices);
    back_indices.append(&mut right_indices);
    back_indices.append(&mut top_indices);
    back_indices.append(&mut bottom_indices);

    (back, back_indices)
}

/*
vertices: vec![
            // Back square
            Vertex {
                position: glam::const_vec3!([-0.6, 0.6, 0.0]),
                color: glam::const_vec3!([1.0, 1.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            Vertex {
                position: glam::const_vec3!([-0.6, -0.6, 0.0]),
                color: glam::const_vec3!([1.0, 1.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            Vertex {
                position: glam::const_vec3!([0.6, -0.6, 0.0]),
                color: glam::const_vec3!([1.0, 1.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            Vertex {
                position: glam::const_vec3!([0.6, 0.6, 0.0]),
                color: glam::const_vec3!([1.0, 1.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            // Front square
            Vertex {
                position: glam::const_vec3!([-0.6, 0.6, 0.6]),
                color: glam::const_vec3!([0.0, 0.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            Vertex {
                position: glam::const_vec3!([-0.6, -0.6, 0.6]),
                color: glam::const_vec3!([0.0, 0.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            Vertex {
                position: glam::const_vec3!([0.6, -0.6, 0.6]),
                color: glam::const_vec3!([0.0, 0.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            Vertex {
                position: glam::const_vec3!([0.6, 0.6, 0.6]),
                color: glam::const_vec3!([0.0, 0.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            // Left square
            Vertex {
                position: glam::const_vec3!([-0.61, 0.6, 0.0]),
                color: glam::const_vec3!([1.0, 0.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            Vertex {
                position: glam::const_vec3!([-0.61, -0.6, 0.0]),
                color: glam::const_vec3!([1.0, 0.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            Vertex {
                position: glam::const_vec3!([-0.61, -0.6, 0.6]),
                color: glam::const_vec3!([1.0, 0.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            Vertex {
                position: glam::const_vec3!([-0.61, 0.6, 0.6]),
                color: glam::const_vec3!([1.0, 0.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            // Right square
            Vertex {
                position: glam::const_vec3!([0.61, 0.6, 0.0]),
                color: glam::const_vec3!([0.0, 1.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            Vertex {
                position: glam::const_vec3!([0.61, -0.6, 0.0]),
                color: glam::const_vec3!([0.0, 1.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            Vertex {
                position: glam::const_vec3!([0.61, -0.6, 0.6]),
                color: glam::const_vec3!([0.0, 1.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            Vertex {
                position: glam::const_vec3!([0.61, 0.6, 0.6]),
                color: glam::const_vec3!([0.0, 1.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
        ],

#[rustfmt::skip]
        indices: vec![
            0, 1, 2, 3, 0, 2, // Rotating square
            4, 5, 6, 7, 4, 6, // Rotating square
            8, 9, 10, 11, 8, 10, // Rotating square
            12, 13, 14, 15, 12, 14, // Rotating square
            16, 17, 18,
        ],
*/
