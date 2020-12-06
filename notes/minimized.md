# Winit minimized

Apparently `winit` doesn't offer a simple solution to the problem of minimizing the window, the
swap chain in `wgpu` (and vulkan) doesn't work when `width == 0 || height == 0`, so we get
validation errors.

I've tried putting the following in `WindowEvent::Resized(new_size)`, `Event::MainEventsCleared`,
and `Event::RedrawRequested { .. }`, but none of these alternatives worked, neither with `Poll` or
`Wait`.

```rust
while new_size.width == 0 || new_size.height == 0 {
    *control_flow = event_loop::ControlFlow::Poll;
    // *control_flow = event_loop::ControlFlow::Wait;
}
```

In `glfw` you would simply do:

```c++
glfwGetFramebufferSize(window, &width, &height);
while (width == 0 || height == 0) {
    glfwGetFramebufferSize(window, &width, &height);
    glfwWaitEvents();
}
```

There is no good solution right now, so I'm ignoring it and waiting.

This is being tracked by:

[208](https://github.com/rust-windowing/winit/issues/208)

[1026](https://github.com/gfx-rs/wgpu/issues/1026)
