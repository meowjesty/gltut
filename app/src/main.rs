use std::{f32::consts::PI, time::Instant};

use renderer::{Camera, Projection, Uniforms, World};
use vertex::Vertex;
use wgpu::util::DeviceExt;
use winit::{
    dpi,
    event::{Event, WindowEvent},
    event_loop, window,
};

mod renderer;
mod vertex;

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
    let x_offset = f32::cos(current_time_through * scale) * 0.005;
    let y_offset = f32::sin(current_time_through * scale) * 0.005;
    (x_offset, y_offset)
}

fn main() {
    let _logger = setup_logger().unwrap();

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

    let mut world = World {
        // NOTE(alex): Position is done in counter-clockwise fashion, starting from the middle point
        // in this case.
        vertices: vec![
            // Rotating triangle
            // 0
            Vertex {
                position: glam::const_vec3!([0.4, 0.4, 0.5]), // middle point
                color: glam::const_vec3!([1.0, 0.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            // 1
            Vertex {
                position: glam::const_vec3!([-0.4, 0.4, 0.5]), // left-most point
                color: glam::const_vec3!([0.0, 1.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            // 2
            Vertex {
                position: glam::const_vec3!([0.4, -0.4, 0.5]), // right-most point
                color: glam::const_vec3!([0.0, 0.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            // 3
            Vertex {
                position: glam::const_vec3!([-0.4, -0.4, 0.5]),
                color: glam::const_vec3!([1.0, 0.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            // 4
            Vertex {
                position: glam::const_vec3!([-0.7, 0.9, 0.0]),
                color: glam::const_vec3!([1.0, 0.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.4131759, 0.00759614]),
            },
            // 5
            Vertex {
                position: glam::const_vec3!([-0.9, 0.7, 0.0]),
                color: glam::const_vec3!([1.0, 0.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0048659444, 0.43041354]),
            },
            // 6
            Vertex {
                position: glam::const_vec3!([-0.5, 0.7, 0.0]),
                color: glam::const_vec3!([1.0, 0.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.28081453, 0.949397057]),
            },
            // 7
            Vertex {
                position: glam::const_vec3!([-0.7, -0.7, 0.0]),
                color: glam::const_vec3!([0.0, 1.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.85967, 0.84732911]),
            },
            // 8
            Vertex {
                position: glam::const_vec3!([-0.9, -0.9, 0.0]),
                color: glam::const_vec3!([0.0, 1.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.9414737, 0.2652641]),
            },
            // 9
            Vertex {
                position: glam::const_vec3!([-0.5, -0.9, 0.0]),
                color: glam::const_vec3!([0.0, 1.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.9414737, 0.2652641]),
            },
            // 10
            Vertex {
                position: glam::const_vec3!([0.7, 0.9, 1.0]),
                color: glam::const_vec3!([0.0, 0.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.4131759, 0.00759614]),
            },
            // 11
            Vertex {
                position: glam::const_vec3!([0.5, 0.7, 1.0]),
                color: glam::const_vec3!([0.0, 0.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.0048659444, 0.43041354]),
            },
            // 12
            Vertex {
                position: glam::const_vec3!([0.9, 0.7, 1.0]),
                color: glam::const_vec3!([0.0, 0.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.28081453, 0.949397057]),
            },
            // 13
            Vertex {
                position: glam::const_vec3!([0.7, -0.7, 0.5]),
                color: glam::const_vec3!([1.0, 0.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.85967, 0.84732911]),
            },
            // 14
            Vertex {
                position: glam::const_vec3!([0.5, -0.9, 0.5]),
                color: glam::const_vec3!([1.0, 0.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.9414737, 0.2652641]),
            },
            // 15
            Vertex {
                position: glam::const_vec3!([0.9, -0.9, 0.5]),
                color: glam::const_vec3!([1.0, 0.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.9414737, 0.2652641]),
            },
            // 16
            Vertex {
                position: glam::const_vec3!([-0.086, 0.492, 0.0]),
                color: glam::const_vec3!([1.0, 1.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.85967, 0.84732911]),
            },
            // 17
            Vertex {
                position: glam::const_vec3!([-0.495, 0.0695, 0.0]),
                color: glam::const_vec3!([1.0, 1.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.9414737, 0.2652641]),
            },
            // 18
            Vertex {
                position: glam::const_vec3!([0.441, 0.234, 0.0]),
                color: glam::const_vec3!([1.0, 0.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.9414737, 0.2652641]),
            },
            // 19
            Vertex {
                position: glam::const_vec3!([0.0, 0.9, 0.0]),
                color: glam::const_vec3!([1.0, 1.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.85967, 0.84732911]),
            },
            // 20
            Vertex {
                position: glam::const_vec3!([-0.9, -0.9, 0.0]),
                color: glam::const_vec3!([1.0, 1.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.9414737, 0.2652641]),
            },
            // 21
            Vertex {
                position: glam::const_vec3!([0.9, -0.9, 0.0]),
                color: glam::const_vec3!([1.0, 0.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.9414737, 0.2652641]),
            },
            // 22
            Vertex {
                position: glam::const_vec3!([1.0, 1.0, 0.0]),
                color: glam::const_vec3!([0.0, 0.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.85967, 0.84732911]),
            },
            // 23
            Vertex {
                position: glam::const_vec3!([-1.0, -1.0, 0.0]),
                color: glam::const_vec3!([0.0, 0.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.9414737, 0.2652641]),
            },
            // 24
            Vertex {
                position: glam::const_vec3!([1.0, -1.0, 0.0]),
                color: glam::const_vec3!([1.0, 1.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.9414737, 0.2652641]),
            },
            // 25
            Vertex {
                position: glam::const_vec3!([-1.0, 1.0, 0.0]),
                color: glam::const_vec3!([1.0, 0.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.9414737, 0.2652641]),
            },
        ],
        // NOTE(alex): These indices will work from:
        // middle->left->right (implicit back to middle);
        // middle->right->up left (implicit back to middle);
        #[rustfmt::skip]
        indices: vec![
            // Rotating triangle
            0, 1, 2, 3, 2, 1, // Rotating square
            4, 5, 6, // Top-left triangle
            7, 8, 9, // Bottom-left triangle
            10, 11, 12, // Top-right triangle
            13, 14, 15, // Bottom-right triangle
            16, 17, 18, // Where
            // 19, 20, 21, // GIANT
            // 22, 23, 24, 25, 23, 22, // SQUARE
        ],
        // TODO(alex): Get a better understanding of this camera, triangles disappeared because
        // I don't truly understand the values here.
        camera: Camera {
            // eye: (0.0, 1.0, 2.0).into(),
            // target: (0.0, 0.0, 0.0).into(),
            // up: glam::Vec3::unit_y(),
            // aspect_ratio: window.inner_size().width as f32 / window.inner_size().height as f32,
            // fov_y: 45.0,
            // z_near: 0.1,
            // z_far: 100.0,
            position: glam::const_vec3!([0.0, 0.0, 5.0]),
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
    };

    let mut renderer = futures::executor::block_on(renderer::Renderer::new(&window, &mut world));
    let mut step_timer = fixedstep::FixedStep::start(75.0).limit(5);
    window.request_redraw();

    event_loop.run(move |event, _target_window, control_flow| {
        *control_flow = event_loop::ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(new_size) => {
                    renderer.resize(new_size, &mut world);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    renderer.resize(*new_inner_size, &mut world);
                }
                WindowEvent::CloseRequested => *control_flow = event_loop::ControlFlow::Exit,
                _ => (),
            },
            Event::MainEventsCleared => {
                while step_timer.update() {}
                let delta = step_timer.render_delta();
                let (x_offset, y_offset) = compute_position_offsets(timer.elapsed().as_secs_f32());
                for mut vertex in world.vertices.iter_mut() {
                    vertex.position.x += x_offset * delta as f32;
                    vertex.position.y += y_offset * delta as f32;
                }
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
