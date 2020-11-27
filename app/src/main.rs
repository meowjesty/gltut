use std::{f32::consts::PI, f64::consts::FRAC_PI_6, time::Instant};

use renderer::World;
use vertex::Vertex;
use winit::{
    dpi,
    event::{Event, WindowEvent},
    event_loop, window,
};

mod renderer;
mod vertex;

fn compute_position_offsets(elapsed: f32) -> (f32, f32) {
    let loop_duration = 5.0;
    let scale = PI * 2.0 / loop_duration;
    let current_time_through = elapsed % loop_duration;
    let x_offset = f32::cos(current_time_through * scale) * 0.005;
    let y_offset = f32::sin(current_time_through * scale) * 0.005;
    (x_offset, y_offset)
}

fn main() {
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

    let mut renderer = futures::executor::block_on(renderer::Renderer::new(&window));
    renderer.push_vertex_buffer(bytemuck::cast_slice(&world.vertices));
    let mut step_timer = fixedstep::FixedStep::start(60.0).limit(5);
    window.request_redraw();

    event_loop.run(move |event, _, control_flow| match event {
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
            renderer.present(&world);
        }
        _ => (),
    })
}
