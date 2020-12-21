use std::{
    f32::consts::{FRAC_PI_2, PI},
    time::Instant,
};

use log::info;
use renderer::{Camera, Projection, Uniforms, World};
use vertex::{cube, Vertex};
use wgpu::util::DeviceExt;
use winit::{
    dpi,
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, MouseScrollDelta, VirtualKeyCode,
        WindowEvent,
    },
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

fn handle_input(event: &DeviceEvent, world: &mut World, delta_time: f32) {
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
        DeviceEvent::Motion { axis, value } => {}
        DeviceEvent::Button { button, state } => {
            if *button == 1 && *state == ElementState::Pressed {
                world.camera_controller.mouse_pressed = true;
            } else {
                world.camera_controller.mouse_pressed = false;
            }
        }
        DeviceEvent::Text { codepoint } => {}
        _ => (),
    }
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

    let mut world = World {
        // NOTE(alex): Position is done in counter-clockwise fashion, starting from the middle point
        // in this case.
        vertices,
        // NOTE(alex): These indices will work from:
        // middle->left->right (implicit back to middle);
        // middle->right->up left (implicit back to middle);
        indices,
        // TODO(alex): Get a better understanding of this camera, triangles disappeared because
        // I don't truly understand the values here.
        // TODO(alex): Camera is still wrong, mouse rotation is busted, it jiggles like crazy,
        // without really rotating. The math is still off, that's why WASD are rotating the camera,
        // instead of moving it sideways.
        // The camera position itself is working as intended regarding WASD:
        // - (AD) changes x;
        // - (Shift/Space) changes y;
        // - (WS) changes z (the more forward you go, the lower z gets, until it flips negative);
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
    };

    let mut renderer = futures::executor::block_on(renderer::Renderer::new(&window, &mut world));
    let mut step_timer = fixedstep::FixedStep::start(30.0).limit(5);
    window.request_redraw();

    event_loop.run(move |event, _target_window, control_flow| {
        *control_flow = event_loop::ControlFlow::Poll;

        match event {
            Event::DeviceEvent { ref event, .. } => {
                handle_input(event, &mut world, step_timer.render_delta() as f32);
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
                _ => (),
            },
            Event::MainEventsCleared => {
                // TODO(alex): How to properly interpolate the values after update?
                while step_timer.update() {
                    let (x_offset, y_offset) =
                        compute_position_offsets(timer.elapsed().as_secs_f32());
                    for mut vertex in world.vertices.iter_mut() {
                        // vertex.position.x += x_offset;
                        // vertex.position.y += y_offset;
                    }
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
