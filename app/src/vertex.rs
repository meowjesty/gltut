#![feature(const_fn)]
#![feature(const_trait_impl)]

use core::mem::*;
use std::{io, path};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::texture::Texture;

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    pub position: glam::Vec3,
    pub color: glam::Vec3,
    pub texture_coordinates: glam::Vec2,
    pub normal: glam::Vec3,
}

// NOTE(alex): `const Traits` are not in Rust yet.
// https://github.com/rust-lang/rust/pull/79287
// pub trait SizeOf {
//     const fn size_of() -> usize;
// }

pub const VEC2_SIZE: usize = size_of::<glam::Vec2>();
pub const VEC3_SIZE: usize = size_of::<glam::Vec3>();

impl Vertex {
    // pub const SIZE: wgpu::BufferAddress = core::mem::size_of::<Self>() as wgpu::BufferAddress;
    pub const SIZE: wgpu::BufferAddress = size_of::<[f32; 3]>() as wgpu::BufferAddress;
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
            // NOTE(alex): This is not the way to change the data contained in this buffer, we must
            // pass changing values (to do the circular movement, for example) in a separate buffer,
            // like we don for the `model_matrix` in the vertex shader.
            // wgpu::VertexAttributeDescriptor {
            //     offset: size_of::<[f32; 3]>() as wgpu::BufferAddress,
            //     shader_location: 1,
            //     format: wgpu::VertexFormat::Float2,
            // },
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

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct DebugVertex {
    pub(crate) position: glam::Vec3,
    pub(crate) color: glam::Vec3,
    pub(crate) texture_coordinates: glam::Vec2,
}

impl DebugVertex {
    pub const SIZE: wgpu::BufferAddress = size_of::<Self>() as wgpu::BufferAddress;
    pub const DESCRIPTOR: wgpu::VertexBufferDescriptor<'static> = wgpu::VertexBufferDescriptor {
        stride: Self::SIZE,
        step_mode: wgpu::InputStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // NOTE(alex): This must be less than 32.
                shader_location: 24,
                format: wgpu::VertexFormat::Float3,
            },
            wgpu::VertexAttributeDescriptor {
                offset: size_of::<glam::Vec3>() as wgpu::BufferAddress,
                shader_location: 25,
                format: wgpu::VertexFormat::Float3,
            },
            wgpu::VertexAttributeDescriptor {
                offset: (size_of::<glam::Vec3>() * 2) as wgpu::BufferAddress,
                shader_location: 26,
                format: wgpu::VertexFormat::Float2,
            },
        ],
    };
}

/// Back -> red
/// Front -> blue
/// Left -> red + blue
/// Right -> green + blue
/// Top -> black
/// Bottom -> white
pub(crate) fn cube(
    origin: glam::Vec3,
    size: f32,
    index: u32,
    color: glam::Vec3,
) -> (Vec<DebugVertex>, Vec<u32>) {
    // WARNING(alex): Texture coordinates are Y-inverted!
    // [0,0]------------------[1,0]
    // ----------------------------
    // ----------------------------
    // [0,1]------------------[1,1]
    // So we must invert them here (our origin in camera is [0,0], but texture origin is [0,1]).
    let tex_o = glam::const_vec2!([0.0, 1.0]);
    let tex_rd = glam::const_vec2!([1.0, 1.0]);
    let tex_lu = glam::const_vec2!([0.0, 0.0]);
    let tex_ru = glam::const_vec2!([1.0, 0.0]);

    let mut back = vec![
        DebugVertex {
            position: glam::Vec3::new(origin.x, origin.y, origin.z),
            color: glam::Vec3::new(1.0, 0.0, 0.0),
            texture_coordinates: tex_o,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x + size, origin.y, origin.z),
            color: glam::Vec3::new(1.0, 0.0, 0.0),
            texture_coordinates: tex_rd,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x, origin.y + size, origin.z),
            color: glam::Vec3::new(1.0, 0.0, 0.0),
            texture_coordinates: tex_lu,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x + size, origin.y + size, origin.z),
            color: glam::Vec3::new(1.0, 0.0, 0.0),
            texture_coordinates: tex_ru,
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
        DebugVertex {
            position: glam::Vec3::new(origin.x, origin.y, origin.z + size),
            color: glam::Vec3::new(0.0, 0.0, 1.0),
            texture_coordinates: tex_o,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x + size, origin.y, origin.z + size),
            color: glam::Vec3::new(0.0, 0.0, 1.0),
            texture_coordinates: tex_rd,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x, origin.y + size, origin.z + size),
            color: glam::Vec3::new(0.0, 0.0, 1.0),
            texture_coordinates: tex_lu,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x + size, origin.y + size, origin.z + size),
            color: glam::Vec3::new(0.0, 0.0, 1.0),
            texture_coordinates: tex_ru,
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
        DebugVertex {
            position: glam::Vec3::new(origin.x, origin.y, origin.z),
            color: glam::Vec3::new(1.0, 0.0, 1.0),
            texture_coordinates: tex_o,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x, origin.y, origin.z + size),
            color: glam::Vec3::new(1.0, 0.0, 1.0),
            texture_coordinates: tex_rd,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x, origin.y + size, origin.z),
            color: glam::Vec3::new(1.0, 0.0, 1.0),
            texture_coordinates: tex_lu,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x, origin.y + size, origin.z + size),
            color: glam::Vec3::new(1.0, 0.0, 1.0),
            texture_coordinates: tex_ru,
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
        DebugVertex {
            position: glam::Vec3::new(origin.x + size, origin.y, origin.z),
            color: glam::Vec3::new(1.0, 0.0, 0.0),
            texture_coordinates: tex_o,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x + size, origin.y, origin.z + size),
            color: glam::Vec3::new(0.0, 1.0, 0.0),
            texture_coordinates: tex_rd,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x + size, origin.y + size, origin.z),
            color: glam::Vec3::new(0.0, 0.0, 1.0),
            texture_coordinates: tex_lu,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x + size, origin.y + size, origin.z + size),
            color: glam::Vec3::new(0.0, 0.0, 1.0),
            texture_coordinates: tex_ru,
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
        DebugVertex {
            position: glam::Vec3::new(origin.x, origin.y + size, origin.z + size),
            color: glam::Vec3::new(0.0, 0.0, 0.0),
            texture_coordinates: tex_o,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x + size, origin.y + size, origin.z + size),
            color: glam::Vec3::new(0.0, 0.0, 0.0),
            texture_coordinates: tex_rd,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x, origin.y + size, origin.z),
            color: glam::Vec3::new(0.0, 0.0, 0.0),
            texture_coordinates: tex_lu,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x + size, origin.y + size, origin.z),
            color: glam::Vec3::new(0.0, 0.0, 0.0),
            texture_coordinates: tex_ru,
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
        DebugVertex {
            position: glam::Vec3::new(origin.x, origin.y, origin.z + size),
            color: glam::Vec3::new(1.0, 1.0, 1.0),
            texture_coordinates: tex_o,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x + size, origin.y, origin.z + size),
            color: glam::Vec3::new(1.0, 1.0, 1.0),
            texture_coordinates: tex_rd,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x, origin.y, origin.z),
            color: glam::Vec3::new(1.0, 1.0, 1.0),
            texture_coordinates: tex_lu,
        },
        DebugVertex {
            position: glam::Vec3::new(origin.x + size, origin.y, origin.z),
            color: glam::Vec3::new(1.0, 1.0, 1.0),
            texture_coordinates: tex_ru,
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

#[derive(Debug)]
pub struct Mesh {
    name: String,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_elements: u32,
    material: usize,
}

impl Mesh {
    pub fn new(file: String, device: &wgpu::Device) -> Mesh {
        let (document, buffers, images) = gltf::import(file).unwrap();

        // TODO(alex): Improve this quick hack to get a name.
        let name = if let Some(mesh) = document.meshes().find(|mesh| mesh.name().is_some()) {
            mesh.name().unwrap()
        } else {
            "Generic Mesh Name"
        };

        // TODO(alex): How do we improve this whole mess of cycling through the mesh primitives?
        // Multiple `for` loops looks cleaner.
        let primitives = document.meshes().flat_map(|mesh| mesh.primitives());

        let positions = primitives
            .clone()
            .flat_map(|primitive| {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                reader.read_positions()
            })
            .flat_map(|positions| positions.map(|position| position.into()))
            .collect::<Vec<glam::Vec3>>();

        /*
        let colors = primitives
            .flat_map(|primitive| {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                reader.read_colors(what is set?)
            })
            .flat_map(|colors| colors.map(|color| color.into()))
            .collect::<Vec<glam::Vec3>>();

        let positions_colors = positions.into_iter().zip(colors.into_iter());
        */

        let vertices = positions
            .into_iter()
            .map(|position| Vertex {
                position,
                ..Default::default()
            })
            .collect::<Vec<Vertex>>();

        let indices = primitives
            .clone()
            .flat_map(|primitive| {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                reader.read_indices()
            })
            .flat_map(|index| index.into_u32())
            .collect::<Vec<u32>>();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh vertex buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh index buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsage::INDEX | wgpu::BufferUsage::COPY_DST,
        });

        Mesh {
            name: name.to_string(),
            vertex_buffer,
            index_buffer,
            num_elements: todo!(),
            material: todo!(),
        }
    }

    pub fn upload(
        &self,
        staging_belt: &mut wgpu::util::StagingBelt,
        mut encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
    ) {
        // for (index, vertices) in self.vertices.as_slice().iter().enumerate() {
        //     staging_belt
        //         .write_buffer(
        //             &mut encoder,
        //             &self.vertex_buffer,
        //             index as u64 * Vertex::SIZE,
        //             wgpu::BufferSize::new(Vertex::SIZE).unwrap(),
        //             &device,
        //         )
        //         .copy_from_slice(bytemuck::bytes_of(vertices));
        // }
    }
}

#[derive(Debug)]
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

// impl Model {
//     pub fn load<P: AsRef<path::Path>>(
//         device: &wgpu::Device,
//         queue: &wgpu::Queue,
//         layout: &wgpu::BindGroupLayout,
//         path: P,
//     ) -> Result<Self, String> {
//         let (document, buffers, images) = gltf::import(path).map_err(|err| err.to_string())?;

//         // TODO(alex): This is the loop format I was talking about above.
//         let mut positions: Vec<glam::Vec3> = Vec::with_capacity(32 * 1024);
//         let mut indices: Vec<u32> = Vec::with_capacity(32 * 1024);
//         for mesh in document.meshes() {
//             for primitive in mesh.primitives() {
//                 let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
//                 if let Some(iter) = reader.read_positions() {
//                     for position in iter {
//                         positions.push(position.into());
//                     }
//                 };
//                 if let Some(read_indices) = reader.read_indices() {
//                     for index in read_indices.into_u32() {
//                         indices.push(index);
//                     }
//                 }
//             }
//         }

//         Ok(Model {
//             meshes: (),
//             materials: (),
//         })
//     }
// }

#[derive(Debug)]
pub struct Material {
    pub name: String,
    pub texture: Texture,
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
