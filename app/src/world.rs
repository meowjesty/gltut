use crate::{
    camera::{Camera, Projection},
    renderer::Uniforms,
    vertex::DebugVertex,
    CameraController,
};

#[derive(Debug, Default)]
pub struct World {
    pub(crate) debug_vertices: Vec<DebugVertex>,
    pub(crate) debug_indices: Vec<u32>,
    pub(crate) camera_controller: CameraController,
    pub(crate) camera: Camera,
    pub(crate) projection: Projection,
    pub(crate) uniforms: Uniforms,
    pub(crate) offset: glam::Vec2,
}
