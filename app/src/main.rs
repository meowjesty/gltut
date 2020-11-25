use winit::{
    dpi,
    event::{Event, WindowEvent},
    event_loop, window,
};

mod renderer;
mod vertex;

fn main() {
    let event_loop = event_loop::EventLoop::new();
    let window = window::WindowBuilder::new()
        .with_resizable(true)
        .with_title("GL Tut")
        .with_inner_size(dpi::PhysicalSize::new(1024, 768))
        .with_min_inner_size(dpi::PhysicalSize::new(480, 720))
        .build(&event_loop)
        .unwrap();

    let mut renderer = futures::executor::block_on(renderer::Renderer::new(&window));
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
            let _delta = step_timer.render_delta();
            window.request_redraw();
        }
        Event::RedrawRequested { .. } => {
            renderer.present();
        }
        _ => (),
    })
}
