use std::{f32::consts::PI, time::Instant};

use renderer::World;
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
        vertices: vec![
            Vertex {
                position: glam::const_vec3!([0.0, 0.3, 0.0]), // middle point
                color: glam::const_vec3!([1.0, 0.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            Vertex {
                position: glam::const_vec3!([-0.3, -0.3, 0.0]), // left-most point
                color: glam::const_vec3!([0.0, 1.0, 0.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
            Vertex {
                position: glam::const_vec3!([0.3, -0.3, 0.0]), // right-most point
                color: glam::const_vec3!([0.0, 0.0, 1.0]),
                texture_coordinates: glam::const_vec2!([0.0, 0.0]),
            },
        ],
    };

    let mut renderer = futures::executor::block_on(renderer::Renderer::new(&window, &world));
    let mut step_timer = fixedstep::FixedStep::start(60.0).limit(5);
    window.request_redraw();

    event_loop.run(move |event, _target_window, control_flow| {
        *control_flow = event_loop::ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(new_size) => {
                    renderer.resize(new_size);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    renderer.resize(*new_inner_size);
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
                window.request_redraw();
            }
            Event::RedrawRequested { .. } => {
                renderer.present(&world, &spawner);
            }
            _ => (),
        }
    })
}
