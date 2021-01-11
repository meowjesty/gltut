pub type Radians = f32;

#[derive(Debug, Default)]
pub struct Projection {
    pub aspect_ratio: f32,
    pub fov_y: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Projection {
    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect_ratio = width as f32 / height as f32;
    }

    pub fn perspective(&self) -> glam::Mat4 {
        let perspective =
            glam::Mat4::perspective_rh(self.fov_y, self.aspect_ratio, self.z_near, self.z_far);

        perspective
    }
}

/// http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/
#[derive(Debug, Default)]
pub struct Camera {
    /// X-axis moves the thumb towards the scren +;
    /// Y-axis moves the index towards the screen +;
    /// Z-axis moves the hand towards your nose +;
    pub position: glam::Vec3,
    pub yaw: Radians,
    pub pitch: Radians,
}

// TODO(alex): Is this correct for right-handed coordinates?
pub fn look_at_dir(eye: glam::Vec3, dir: glam::Vec3, up: glam::Vec3) -> glam::Mat4 {
    // NOTE(alex): Normalizing a vector that is already at length 1.0 changes nothing.
    let f = dir.normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);

    #[cfg_attr(rustfmt, rustfmt_skip)]
        glam::Mat4::from_cols_array_2d(&[
            [s.x.clone(), u.x.clone(), -f.x.clone(), 0.0],
            [s.y.clone(), u.y.clone(), -f.y.clone(), 0.0],
            [s.z.clone(), u.z.clone(), -f.z.clone(), 0.0],
            [-eye.dot(s), -eye.dot(u), eye.dot(f), 1.0],
        ])
}

impl Camera {
    // NOTE(alex): This link gives a clearer explanation of how this works.
    /// https://stackoverflow.com/questions/21830340/understanding-glmlookat
    pub fn view(&self) -> glam::Mat4 {
        // NOTE(alex): glam doesn't have a public version of look at direction
        // (`cgmath::loot_at_dir), that's why it was rotating around the center point, as this
        // `glam::look_at_rh` looks at the center (locks the center, not a direction).
        // TODO(alex): Is there a way to use `let view_matrix = glam::Mat4::look_at_rh(` correctly
        // by using look_at_rh formula with the correct center (check the math)?
        // TODO(alex): We don't need the `OPENGL_TO_WGPU_MATRIX`, things still look okay so far
        // without it.
        let center = glam::Vec3::new(self.yaw.cos(), self.pitch.sin(), self.yaw.sin()).normalize();
        let view_matrix =
            glam::Mat4::look_at_rh(self.position, self.position + center, glam::Vec3::unit_y());

        view_matrix
    }
}

// #[rustfmt::skip]
// pub const OPENGL_TO_WGPU_MATRIX: glam::Mat4 = glam::const_mat4!([
//     1.0, 0.0, 0.0, 0.0,
//     0.0, 1.0, 0.0, 0.0,
//     0.0, 0.0, 0.5, 0.0,
//     0.0, 0.0, 0.5, 1.0,
// ]);
