use super::*;

#[derive(Default)]
pub struct RenderSdf {
    program: GLuint,

    u_camera_inverse: GLint,
    u_camera_pos: GLint,

    u_lemons:     GLint,
    u_lemons_len: GLint,
    lemons_for_shader: Box<[f32]>,
    lemons_for_shader_dirty: bool,

    u_selection_instance_id: GLint,
    u_selection_glow: GLint,

    u_hover_instance_id: GLint,
    u_hover_glow: GLint,

    vao: GLuint,
}

impl RenderSdf {
    #[inline]
    pub fn init(max_bodies: usize) -> Self {
        let mut init = RenderSdf::default();
        unsafe {
            init.program = gl::link_shaders(&[
                gl::compile_shader(include_str!("shader/lemon_sdf.vert.glsl"), gl::VERTEX_SHADER),
                gl::compile_shader(include_str!("shader/lemon_sdf.frag.glsl"), gl::FRAGMENT_SHADER),
            ]);
            gl::UseProgram(init.program);

            init.u_lemons = gl::GetUniformLocation(init.program, cstr!("u_lemons"));
            init.u_lemons_len = gl::GetUniformLocation(init.program, cstr!("u_lemons_len"));
            init.lemons_for_shader = vec![0.0; FORCE_MAX_BODIES * 8].into_boxed_slice();
            init.lemons_for_shader_dirty = true;

            init.u_camera_inverse= gl::GetUniformLocation(init.program, cstr!("u_camera_inverse"));
            init.u_camera_pos= gl::GetUniformLocation(init.program, cstr!("u_camera_pos"));

            init.u_hover_glow = gl::GetUniformLocation(init.program, cstr!("u_hover_glow"));
            gl::Uniform1f(init.u_hover_glow, 0.0);

            init.u_selection_glow = gl::GetUniformLocation(init.program, cstr!("u_selection_glow"));
            gl::Uniform1f(init.u_selection_glow, 0.0);

            init.u_hover_instance_id = gl::GetUniformLocation(init.program, cstr!("u_hover_instance_id"));
            gl::Uniform1i(init.u_hover_instance_id, !0);

            init.u_hover_glow = gl::GetUniformLocation(init.program, cstr!("u_hover_glow"));
            gl::Uniform1f(init.u_hover_glow, 0.0);

            init.vao = gl::gen_object(gl::GenVertexArrays);
            gl::BindVertexArray(init.vao);

            let a_ndc = gl::GetAttribLocation(init.program, cstr!("a_ndc")) as GLuint;
            let vbo_ndc = gl::gen_object(gl::GenBuffers);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo_ndc);
            let verts: [Point2; 4] = [
                point2!(1.0, 1.0),
                point2!(-1.0, 1.0),
                point2!(1.0, -1.0),
                point2!(-1.0, -1.0),
            ];
            gl::buffer_data(gl::ARRAY_BUFFER, &verts, gl::STATIC_DRAW);

            gl::VertexAttribPointer(a_ndc, 2, gl::FLOAT, gl::FALSE, 0, ptr::null());
            gl::EnableVertexAttribArray(a_ndc);
        }
        init
    }

    #[inline]
    pub fn render_lemons(&mut self, n: usize) {
        unsafe {
            gl::UseProgram(self.program);
            gl::UniformMatrix2x4fv(
                self.u_lemons,
                n as GLsizei, gl::FALSE, 
                self.lemons_for_shader.as_ptr()
            );
            gl::Uniform1ui(self.u_lemons_len, n as _);

            gl::BindVertexArray(self.vao);
            gl::DrawArrays(
                gl::TRIANGLE_STRIP,
                0,
                4,
            );
        }
        self.lemons_for_shader_dirty = false;
    }

    #[inline]
    pub fn update_camera(&mut self, camera: &Camera) {
        let camera_inverse = (camera.projection * camera.view).invert().unwrap();
        unsafe {
            gl::UseProgram(self.program);
            gl::UniformMatrix4fv(self.u_camera_inverse, 1, gl::FALSE, camera_inverse.as_ptr());
            gl::Uniform3fv(self.u_camera_pos, 1, camera.position.as_ptr());
        }
    }

    #[inline]
    pub fn update_lemon(&mut self, id: usize, 
        lemon: Option<&Lemon>,
    ) {
        if let Some(lemon) = lemon {
            let mut dest = &mut self.lemons_for_shader[8*id..8*(id+1)];
            write_lemon_for_shader(lemon, &mut dest);
        };
        self.lemons_for_shader_dirty = true;
    }

    #[inline]
    pub fn update_lemons(&mut self, lemons: &[Lemon]) {
        write_lemons_for_shader(lemons, &mut self.lemons_for_shader);
        self.lemons_for_shader_dirty = true;
    }

    #[inline]
    pub fn update_hover(&mut self,
        id:   Option<usize>,
        glow: Option<Real>,
    ) {
        return;
        unsafe {
            gl::UseProgram(self.program);
            if let Some(id) = id {
                gl::Uniform1i(self.u_hover_instance_id, id as _);
            }
            if let Some(glow) = glow {
                gl::Uniform1f(self.u_hover_glow, glow);
            }
        }
    }

    #[inline]
    pub fn update_selection(&mut self,
        id:   Option<usize>,
        glow: Option<Real>,
    ) {
        return;
        unsafe {
            gl::UseProgram(self.program);
            if let Some(id) = id {
                gl::Uniform1i(self.u_selection_instance_id, id as _);
            }
            if let Some(glow) = glow {
                gl::Uniform1f(self.u_selection_glow, glow);
            }
        }
    }
}

pub fn write_lemon_for_shader(lemon: &Lemon, dest: &mut [f32]) {
    if dest.len() != 8 { 
        panic!("incorrect slice length for write_lemon_for_sdf_shader");
    }

    dest[0+0] = lemon.phys.position.x;
    dest[0+1] = lemon.phys.position.y;
    dest[0+2] = lemon.phys.position.z;

    let lemon_vertical = lemon.get_vertical() * lemon.scale;
    dest[4+0] = lemon_vertical.x;
    dest[4+1] = lemon_vertical.y;
    dest[4+2] = lemon_vertical.z;

    dest[0+3] = lemon.radius;

    dest[4+3] = lemon.focal_radius();
}

pub fn write_lemons_for_shader(lemons: &[Lemon], dest: &mut [f32]) {
    if dest.len() < 8*lemons.len() { 
        panic!("insufficient slice length for write_lemons_for_sdf_shader");
    }
    for (i, lemon) in lemons.iter().enumerate() {
        write_lemon_for_shader(lemon, &mut dest[8*i..8*(i+1)]);
    }
}