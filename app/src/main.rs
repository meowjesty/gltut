use log::info;
use std::{
    collections::HashMap,
    f32::consts::{FRAC_PI_2, PI},
    path,
    time::Instant,
};
use world::World;

use wgpu::util::DeviceExt;
use winit::{
    dpi,
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, MouseScrollDelta, VirtualKeyCode,
        WindowEvent,
    },
    event_loop, window,
};

pub(crate) mod camera;
pub(crate) mod renderer;
pub(crate) mod texture;
pub(crate) mod vertex;
pub(crate) mod world;

use camera::{Camera, Projection};
use renderer::Uniforms;
use vertex::cube;

pub struct StagingBuffer {
    buffer: wgpu::Buffer,
    size: wgpu::BufferAddress,
}

impl StagingBuffer {
    pub fn new<T: bytemuck::Pod + Sized>(device: &wgpu::Device, data: &[T]) -> StagingBuffer {
        let size = core::mem::size_of::<T>() * data.len();

        StagingBuffer {
            buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsage::COPY_SRC,
                label: Some("Staging Buffer"),
            }),
            size: size as wgpu::BufferAddress,
        }
    }

    pub fn copy_to_buffer(&self, encoder: &mut wgpu::CommandEncoder, other: &wgpu::Buffer) {
        encoder.copy_buffer_to_buffer(&self.buffer, 0, other, 0, self.size)
    }
}

fn setup_logger() -> Result<(), fern::InitError> {
    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        .level(log::LevelFilter::Info)
        .chain(std::io::stdout())
        .chain(fern::log_file("output.log")?)
        .apply()?;
    Ok(())
}

fn compute_position_offsets(elapsed: f32) -> (f32, f32) {
    let loop_duration = 5.0;
    let scale = PI * 2.0 / loop_duration;
    let current_time_through = elapsed % loop_duration;
    let x_offset = f32::cos(current_time_through * scale) * 0.055;
    let y_offset = f32::sin(current_time_through * scale) * 0.055;
    (x_offset, y_offset)
}

#[derive(Debug, Default)]
pub struct CameraController {
    mouse_pressed: bool,
    left: f32,
    right: f32,
    forward: f32,
    backward: f32,
    up: f32,
    down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    scroll: f32,
    speed: f32,
    sensitivity: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> CameraController {
        CameraController {
            speed,
            sensitivity,
            ..Default::default()
        }
    }

    pub fn process_keyboard(&mut self, key: VirtualKeyCode, state: ElementState) {
        let amount = if state == ElementState::Pressed {
            1.0
        } else {
            0.0
        };

        match key {
            VirtualKeyCode::W | VirtualKeyCode::Up => {
                self.forward = amount;
            }
            VirtualKeyCode::S | VirtualKeyCode::Down => {
                self.backward = amount;
            }
            VirtualKeyCode::A | VirtualKeyCode::Left => {
                self.left = amount;
            }
            VirtualKeyCode::D | VirtualKeyCode::Right => {
                self.right = amount;
            }
            VirtualKeyCode::Space => {
                self.up = amount;
            }
            VirtualKeyCode::LShift => {
                self.down = amount;
            }
            _ => (),
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        if self.mouse_pressed {
            self.rotate_horizontal = mouse_dx as f32;
            self.rotate_vertical = mouse_dy as f32;
        }
    }

    pub fn process_scroll(&mut self, scroll_delta: &MouseScrollDelta) {
        self.scroll = match scroll_delta {
            // I'm assuming a line is about 100 pixels
            MouseScrollDelta::LineDelta(_, scroll) => scroll * 100.0,
            MouseScrollDelta::PixelDelta(dpi::PhysicalPosition { y: scroll, .. }) => *scroll as f32,
        };
    }

    pub fn update_camera(&mut self, camera: &mut Camera, delta_time: f32) {
        // forward/backward, left/right
        let (yaw_sin, yaw_cos) = camera.yaw.sin_cos();
        let forward = glam::Vec3::new(yaw_cos, 0.0, yaw_sin).normalize();
        let right = glam::Vec3::new(-yaw_sin, 0.0, yaw_cos).normalize();
        camera.position += forward * (self.forward - self.backward) * self.speed * delta_time;
        camera.position += right * (self.right - self.left) * self.speed * delta_time;

        // Move in/out (aka. "zoom")
        // Note: this isn't an actual zoom. The camera's position
        // changes when zooming. I've added this to make it easier
        // to get closer to an object you want to focus on.
        let (pitch_sin, pitch_cos) = camera.pitch.sin_cos();
        let scrollward =
            glam::Vec3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize();
        camera.position += scrollward * self.scroll * self.speed * self.sensitivity * delta_time;
        self.scroll = 0.0;

        // Move up/down. Since we don't use roll, we can just
        // modify the y coordinate directly.
        camera.position.y += (self.up - self.down) * self.speed * delta_time;

        // Rotate
        camera.yaw += self.rotate_horizontal * self.sensitivity * delta_time;
        camera.pitch += -self.rotate_vertical * self.sensitivity * delta_time;

        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        if camera.pitch < -FRAC_PI_2 {
            camera.pitch = -FRAC_PI_2;
        } else if camera.pitch > FRAC_PI_2 {
            camera.pitch = FRAC_PI_2;
        }
    }
}

fn handle_input(event: &DeviceEvent, world: &mut World) {
    match event {
        DeviceEvent::Key(KeyboardInput {
            virtual_keycode: Some(key),
            state,
            ..
        }) => {
            world.camera_controller.process_keyboard(*key, *state);
        }
        DeviceEvent::MouseMotion { delta } => {
            world.camera_controller.process_mouse(delta.0, delta.1);
        }
        DeviceEvent::MouseWheel { delta } => {
            world.camera_controller.process_scroll(delta);
        }
        DeviceEvent::Motion { .. } => {}
        DeviceEvent::Button { button, state } => {
            if *button == 1 && *state == ElementState::Pressed {
                world.camera_controller.mouse_pressed = true;
            } else {
                world.camera_controller.mouse_pressed = false;
            }
        }
        DeviceEvent::Text { .. } => {}
        _ => (),
    }
}

struct Indices {
    buffer: Vec<u8>,
    count: usize,
}

pub struct Geometry {
    positions: Vec<u8>,
    indices: Vec<u8>,
    indices_count: usize,
    texture_coordinates: Vec<u8>,
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
pub fn load_geometry<'x>(path: &path::Path) -> Geometry {
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
    let mut indices = Vec::with_capacity(4);
    let mut positions = Vec::with_capacity(4);
    let mut counts = Vec::with_capacity(4);
    let mut texture_coordinates = Vec::with_capacity(4);
    let mut num_primitives = 0;
    use std::fs::*;
    use std::io::prelude::*;
    let mut debug_file = File::create("debug_file.json").unwrap();
    let mut num_positions = 0;
    let mut num_texture_coordinates = 0;
    // for scene in document.scenes() {
    // for node in scene.nodes() {
    // if let Some(mesh) = node.mesh() {
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            if let Some(indices_accessor) = primitive.indices() {
                let count = indices_accessor.count();
                let view = indices_accessor.view().unwrap();
                let index = view.buffer().index();
                let offset = view.offset();
                let length = view.length();
                let buffer = buffers.get(index).unwrap();
                let indices_buffer = &buffer[offset..offset + length];
                // indices_buf = Some((indices_buffer.to_vec(), count));
                indices.push(indices_buffer.to_vec());
                counts.push(count);
            }

            for (semantic, accessor) in primitive.attributes() {
                debug_file.write_all(
                    debug_accessor(&accessor, &mesh.name().unwrap_or("nameless")).as_bytes(),
                );
                // NOTE(alex): Number of components, if we have VEC3 as the data type, then to get
                // the number of bytes would be something like `count * size_of(VEC3)`.
                let data_type = accessor.data_type();
                let data_size = match data_type {
                    gltf::accessor::DataType::I8 => size_of::<i8>(),
                    gltf::accessor::DataType::U8 => size_of::<u8>(),
                    gltf::accessor::DataType::I16 => size_of::<i16>(),
                    gltf::accessor::DataType::U16 => size_of::<u16>(),
                    gltf::accessor::DataType::U32 => size_of::<u32>(),
                    gltf::accessor::DataType::F32 => size_of::<f32>(),
                };
                let dimensions = accessor.dimensions();
                let size_bytes = match dimensions {
                    gltf::accessor::Dimensions::Scalar => data_size,
                    gltf::accessor::Dimensions::Vec2 => data_size * 2,
                    gltf::accessor::Dimensions::Vec3 => data_size * 3,
                    gltf::accessor::Dimensions::Vec4 => data_size * 4,
                    gltf::accessor::Dimensions::Mat2 => data_size * 2 * 2,
                    gltf::accessor::Dimensions::Mat3 => data_size * 3 * 3,
                    gltf::accessor::Dimensions::Mat4 => data_size * 4 * 4,
                };
                let count = accessor.count();
                let length = size_bytes * count;
                let view = accessor.view().unwrap();
                let byte_offset = view.offset();
                // NOTE(alex): `bufferView: 2` this is the buffer view index in the accessors.
                // let view_index = view.index();
                let index = view.buffer().index();
                // TODO(alex): This will always be `0` for our `scene.gltf` model.
                assert_eq!(index, 0);
                let buffer = buffers.get(index).unwrap();
                let attributes = &buffer[byte_offset..byte_offset + length];

                match semantic {
                    gltf::Semantic::Positions => {
                        // positions_buf = Some(positions_buffer.to_vec())
                        positions.push(attributes.to_vec());
                        num_positions += 1;
                        println!(
                            "positions view {:?} accessor {:?} buffer {:?} byte_offset {:?}",
                            view.index(),
                            accessor.index(),
                            view.buffer().index(),
                            byte_offset,
                        );
                        println!(
                            "positions size_bytes {:?} * count {:?} = {:?}",
                            size_bytes,
                            count,
                            size_bytes * count
                        );
                        // FIXME(alex): The main problem appears to be here.
                        // TODO(alex): This assertion fails, we're reading only half the vertices
                        // `55872`, instead of `111744`!
                        assert_eq!(view.length(), attributes.len());
                    }
                    gltf::Semantic::TexCoords(set_index) => {
                        texture_coordinates.push(attributes.to_vec());
                        num_texture_coordinates += 1;
                        println!(
                            "textures view {:?} accessor {:?} buffer {:?} byte_offset {:?}",
                            view.index(),
                            accessor.index(),
                            view.buffer().index(),
                            byte_offset,
                        );
                        println!(
                            "textures size_bytes {:?} * count {:?} = {:?}",
                            size_bytes,
                            count,
                            size_bytes * count
                        );
                        assert_eq!(view.length(), attributes.len());
                    }
                    _ => (),
                }

                num_primitives += 1;
                println!(
                    "num primitives {:?} -> num_positions {:?} -> num_textures {:?}",
                    num_primitives, num_positions, num_texture_coordinates,
                );
            }
        }
    }
    // }
    // }
    // TODO(alex): We're getting correct values for indexing, count.
    // `positions view 2 accessor 0 buffer 0`
    // `textures view 1 accessor 3 buffer 0`
    // `positions count 4656 -> textures_count 4656`
    let geometry_positions = positions.into_iter().flatten().collect();
    let geometry_indices = indices.into_iter().flatten().collect();
    let count = counts.iter().sum();
    let geometry_texture_coordinates = texture_coordinates.into_iter().flatten().collect();

    let geometry = Geometry {
        positions: geometry_positions,
        indices: geometry_indices,
        indices_count: count,
        texture_coordinates: geometry_texture_coordinates,
    };

    let positions_count = geometry.positions.len() / size_of::<[f32; 3]>();
    let textures_count = geometry.texture_coordinates.len() / size_of::<[f32; 2]>();
    println!(
        "positions count {:?} -> textures_count {:?}",
        positions_count, textures_count
    );

    geometry
}

#[derive(Debug, Eq, PartialEq, Hash)]
enum PrimitiveKind {
    Position,
    Normal,
    Tangent,
    TextureCoordinates,
    Indices(usize),
}

/*
fn load_model_gltf<'x>() -> HashMap<PrimitiveKind, Vec<Vec<u8>>> {
    use gltf::json::{accessor::*, mesh::*, *};
    // let kitten = include_bytes!("../../assets/kitten.gltf");
    // let binary = include_bytes!("../../assets/kitten_data.bin");
    let gltf_model =
        include_bytes!("../../assets/kenney_piratekit_1.1/Models/glTF format/ship_light.gltf");
    let binary =
        include_bytes!("../../assets/kenney_piratekit_1.1/Models/glTF format/ship_light.bin");
    let mut indices = Vec::with_capacity(32);
    let mut positions = Vec::with_capacity(32);
    let mut normals = Vec::with_capacity(32);
    let mut meshes: HashMap<PrimitiveKind, Vec<Vec<u8>>> = HashMap::with_capacity(32);
    // let root: Root = Root::from_slice(kitten).unwrap();
    let root: Root = Root::from_slice(gltf_model).unwrap();
    for mesh in root.meshes {
        for primitive in mesh.primitives {
            // NOTE(alex): Load indices buffer.
            if let Some(indices_index) = primitive.indices {
                if let Some(accessor) = root.accessors.get(indices_index.value()) {
                    let count = accessor.count;
                    let view_index = accessor.buffer_view.unwrap().value();
                    let view = root.buffer_views.get(view_index).unwrap();
                    let offset = view.byte_offset.unwrap() as usize;
                    let length = view.byte_length as usize;
                    let indices_buffer = &binary[offset..offset + length];
                    indices.push((indices_buffer, count));
                }
            }

            for (semantic, accessor) in primitive.attributes {
                if let Some(accessor) = root.accessors.get(accessor.value()) {
                    // let view_index = accessor.buffer_view.unwrap().value();
                    let offset = accessor.byte_offset as usize;
                    let count = accessor.count as usize;
                    /*
                    let GenericComponentType(component_type) = accessor.component_type.unwrap();
                    match component_type {
                        I8 => (),
                        U8 => (),
                        I16 => (),
                        U16 => (),
                        U32 => (),
                        F32 => (),
                    }
                    */
                    let type_ = accessor.type_.unwrap();
                    let size_bytes = match type_ {
                        Type::Scalar => core::mem::size_of::<f32>(),
                        Type::Vec3 => core::mem::size_of::<[f32; 3]>(),
                        _ => core::mem::size_of::<u32>(),
                    };
                    let length = count * size_bytes;
                    // let view = root.buffer_views.get(view_index).unwrap();
                    let buffer = &binary[offset..offset + length];

                    match semantic.unwrap() {
                        Semantic::Positions => positions.push(buffer.to_vec()),
                        Semantic::Normals => normals.push(buffer.to_vec()),
                        _ => (),
                    }
                }
            }
        }
    }
    meshes.insert(PrimitiveKind::Position, positions);
    meshes.insert(PrimitiveKind::Normal, normals);
    // TODO(alex): Deal with loading multiple different model primitives, think of them as
    // submodels, like the doors of a car.
    // TODO(alex): Separate the count from buffer, put count in Indices(count).
    // indices.into_iter().map(|buffer, count| ());
    // meshes.insert(PrimitiveKind::Indices, indices);

    meshes
}
*/

fn main() {
    let _logger = setup_logger().unwrap();

    // debug_glb();
    // debug_gltf_json();
    // panic!("Debugging.");

    let (mut pool, spawner) = {
        let local_pool = futures::executor::LocalPool::new();
        let spawner = local_pool.spawner();
        (local_pool, spawner)
    };

    let timer = Instant::now();
    let event_loop = event_loop::EventLoop::new();
    let window = window::WindowBuilder::new()
        .with_resizable(true)
        .with_title("GL Tut")
        .with_inner_size(dpi::PhysicalSize::new(1024, 768))
        .with_min_inner_size(dpi::PhysicalSize::new(480, 720))
        .build(&event_loop)
        .unwrap();

    /*
    let (mut vertices, mut indices) = cube(
        glam::Vec3::new(5.0, 5.0, 0.0),
        5.0,
        0,
        glam::Vec3::new(1.0, 0.0, 0.0),
    );

    let (mut high_z_vertices, mut high_z_indices) = cube(
        glam::Vec3::new(-10.0, -10.0, 10.0),
        5.0,
        24,
        glam::Vec3::new(0.0, 1.0, 1.0),
    );

    let (mut neg_z_vertices, mut neg_z_indices) = cube(
        glam::Vec3::new(10.0, 10.0, -10.0),
        5.0,
        48,
        glam::Vec3::new(0.0, 1.0, 1.0),
    );

    let (mut zeroed_vertices, mut zeroed_indices) = cube(
        glam::Vec3::new(0.0, 0.0, 0.0),
        10.0,
        72,
        glam::Vec3::new(0.0, 1.0, 1.0),
    );

    vertices.append(&mut high_z_vertices);
    vertices.append(&mut neg_z_vertices);
    vertices.append(&mut zeroed_vertices);
    indices.append(&mut high_z_indices);
    indices.append(&mut neg_z_indices);
    indices.append(&mut zeroed_indices);

    for i in 1..12 {
        let (mut vs, mut is) = cube(
            glam::Vec3::new(
                5.0 + (i as f32 * 5.0),
                5.0 + (i as f32 * 5.0),
                5.0 + (i as f32 * 5.0),
            ),
            5.0,
            72 + (i * 24),
            glam::Vec3::new(0.0, 1.0, 1.0),
        );

        vertices.append(&mut vs);
        indices.append(&mut is);
    }
    */

    let (debug_vertices, indices) = cube(
        glam::Vec3::new(0.0, 0.0, 0.0),
        5.0,
        0,
        glam::Vec3::new(0.0, 1.0, 1.0),
    );

    let mut world = World {
        // NOTE(alex): Position is done in counter-clockwise fashion, starting from the middle point
        // in this case.
        debug_vertices,
        // NOTE(alex): These indices will work from:
        // middle->left->right (implicit back to middle);
        // middle->right->up left (implicit back to middle);
        debug_indices: indices,
        camera: Camera {
            // eye: (0.0, 1.0, 2.0).into(),
            // target: (0.0, 0.0, 0.0).into(),
            // up: glam::Vec3::unit_y(),
            // aspect_ratio: window.inner_size().width as f32 / window.inner_size().height as f32,
            // fov_y: 45.0,
            // z_near: 0.1,
            // z_far: 100.0,
            position: glam::const_vec3!([0.0, 5.0, 10.0]),
            // NOTE(alex): Yaw "rotates" around the Y-axis, positive rotates the right hand to the
            // right (meaning the thumb points away from the nose), negative rotates the thumb
            // towards your nose (-90 degrees makes your thumb point to your nose, the Z-axis
            // points left).
            yaw: f32::to_radians(-90.0),
            // NOTE(alex): Pitch "rotates" around the X-axis, positive rotates the right hand
            // downwards (index points towards your nose) with the Z-axis pointing down, negative
            // the index points away from your nose, with the Z-axis pointing upwards.
            pitch: f32::to_radians(-20.0),
        },
        projection: Projection {
            aspect_ratio: window.inner_size().width as f32 / window.inner_size().height as f32,
            fov_y: f32::to_radians(45.0),
            z_near: 0.1,
            z_far: 100.0,
        },
        // TODO(alex): Mouse is moving too fast, maybe something wrong with the math?
        camera_controller: CameraController::new(1.0, 0.2),
        uniforms: Uniforms::default(),
        offset: glam::const_vec2!([1.0, 1.0]),
    };

    let mut renderer = futures::executor::block_on(renderer::Renderer::new(&window, &mut world));
    let mut step_timer = fixedstep::FixedStep::start(30.0).limit(5);
    let mut has_focus = false;
    window.request_redraw();

    event_loop.run(move |event, _target_window, control_flow| {
        *control_flow = event_loop::ControlFlow::Poll;

        match event {
            Event::DeviceEvent { ref event, .. } => {
                if has_focus {
                    handle_input(event, &mut world);
                }
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(new_size) => {
                    renderer.resize(new_size, &mut world);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    renderer.resize(*new_inner_size, &mut world);
                }
                WindowEvent::CloseRequested => *control_flow = event_loop::ControlFlow::Exit,
                WindowEvent::KeyboardInput { input, .. } => match input {
                    winit::event::KeyboardInput {
                        state: winit::event::ElementState::Pressed,
                        virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                        ..
                    } => {
                        *control_flow = event_loop::ControlFlow::Exit;
                    }
                    _ => (),
                },
                WindowEvent::Focused(focused) => {
                    has_focus = focused;
                }
                _ => (),
            },
            Event::MainEventsCleared => {
                // TODO(alex): How to properly interpolate the values after update?
                while step_timer.update() {
                    let (x_offset, y_offset) =
                        compute_position_offsets(timer.elapsed().as_secs_f32());
                    world.offset = glam::Vec2::new(x_offset, y_offset);
                }
                world
                    .camera_controller
                    .update_camera(&mut world.camera, step_timer.render_delta() as f32);
                world
                    .uniforms
                    .update_view_projection(&world.camera, &world.projection);
                let delta = step_timer.render_delta();
                // TODO(alex): Do the rotation and setup the MVP.
                // world.camera.model = glam::const_mat4!([1.0; 16]);
                window.request_redraw();
                // TODO(alex): Why do we need this pool?
                // Removing it, the triangle keeps rotating, no valdation errors, everything seems
                // just fine (added this quite late actually).
                pool.run_until_stalled();
            }
            Event::RedrawRequested { .. } => {
                renderer.present(&world, &spawner);
            }
            _ => (),
        }
    })
}
