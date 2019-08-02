use super::*;

#[repr(C)]
pub struct Camera {
    pub view:       Mat4,
    pub projection: Mat4,

    pub position: Point3,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            view:       mat4!(1.0),
            projection: mat4!(1.0),

            position: point3!(),
        }
    }
}

#[derive(Default)]
pub struct Render {
    program: GLuint,

    u_camera_view: GLint,
    u_camera_projection: GLint,
    u_camera_position: GLint,

    u_selection_instance_id: GLint,
    u_selection_glow: GLint,

    vao: GLuint,

    ebo: GLuint,
    ebo_len: usize,

    vbo_transform: GLuint,
    vbo_lemon_s: GLuint,
    vbo_lemon_color: GLuint,
    vbo_lemon_glow: GLuint,

    hover_id:       usize,
    hover_glow:     Real,
    selection_id:   usize,
    selection_glow: Real,
}

impl Render {
    #[inline]
    pub fn init(max_bodies: usize) -> Self {
        let mut init = Render::default();
        unsafe {
            init.program = gl::link_shaders(&[
                gl::compile_shader(include_str!("shader/lemon.vert.glsl"), gl::VERTEX_SHADER),
                gl::compile_shader(include_str!("shader/lemon.frag.glsl"), gl::FRAGMENT_SHADER),
            ]);
            gl::UseProgram(init.program);

            init.u_camera_view = gl::GetUniformLocation(init.program, cstr!("u_camera_view"));
            init.u_camera_projection = gl::GetUniformLocation(init.program, cstr!("u_camera_projection"));
            init.u_camera_position = gl::GetUniformLocation(init.program, cstr!("u_camera_position"));

            let u_ambient_color = gl::GetUniformLocation(init.program, cstr!("u_ambient_color"));
            gl::Uniform4fv(u_ambient_color, 1, as_ptr(&BACK_COLOR));

            let u_direction_light = gl::GetUniformLocation(init.program, cstr!("u_direction_light"));
            gl::Uniform3fv(u_direction_light, 1, vec3!(2.5, 0.5, 1.5).as_ptr());

            let txo_radius_normal_z_atlas = gl::gen_object(gl::GenTextures);
            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, txo_radius_normal_z_atlas);

            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as _);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as _);

            let radius_normal_z_map = lemon::make_radius_normal_z_map();
            gl::TexImage2D(
                gl::TEXTURE_2D, 0,
                gl::RGB as _, lemon::MAP_RESOLUTION as _, lemon::MAP_RESOLUTION as _, 0,
                gl::RGB as _, gl::UNSIGNED_BYTE, radius_normal_z_map.as_ptr() as *const GLvoid,
            );
            gl::GenerateMipmap(gl::TEXTURE_2D);
            let u_radius_normal_z_map = gl::GetUniformLocation(init.program,
                cstr!("u_radius_normal_z_map")
            );
            gl::Uniform1i(u_radius_normal_z_map, 0);

            init.vao = gl::gen_object(gl::GenVertexArrays);
            gl::BindVertexArray(init.vao);

            // PER-INSTANCE ATTRIBUTES
            let a_lemon_glow   = gl::GetAttribLocation(init.program, cstr!("a_lemon_glow")) as GLuint;
            init.vbo_lemon_glow = gl::gen_object(gl::GenBuffers);
            gl::BindBuffer(gl::ARRAY_BUFFER, init.vbo_lemon_glow);
            gl::buffer_init::<GLfloat>(gl::ARRAY_BUFFER, max_bodies, gl::DYNAMIC_DRAW);
            gl::EnableVertexAttribArray(a_lemon_glow);
            gl::VertexAttribPointer(a_lemon_glow, 1, gl::FLOAT, gl::FALSE, 0, 0 as *const GLvoid);
            gl::VertexAttribDivisor(a_lemon_glow, 1);

            let a_transform    = gl::GetAttribLocation(init.program, cstr!("a_transform")) as GLuint;
            init.vbo_transform = gl::gen_object(gl::GenBuffers);
            gl::BindBuffer(gl::ARRAY_BUFFER, init.vbo_transform);
            gl::buffer_init::<Mat4>(gl::ARRAY_BUFFER, max_bodies, gl::STREAM_DRAW);
            for i in 0..4 { // all 4 column vectors
                let a_transform_i = a_transform + i as GLuint;
                gl::EnableVertexAttribArray(a_transform_i);
                gl::VertexAttribPointer(
                    a_transform_i , 4, gl::FLOAT, gl::FALSE,
                    mem::size_of::<Mat4>() as GLsizei,
                    ptr::null::<f32>().offset(4 * i) as *const GLvoid,
                );
                gl::VertexAttribDivisor(a_transform_i, 1);
            }

            let a_lemon_s   = gl::GetAttribLocation(init.program, cstr!("a_lemon_s")) as GLuint;
            init.vbo_lemon_s = gl::gen_object(gl::GenBuffers);
            gl::BindBuffer(gl::ARRAY_BUFFER, init.vbo_lemon_s);
            gl::buffer_init::<f32>(gl::ARRAY_BUFFER, max_bodies, gl::DYNAMIC_DRAW);
            gl::EnableVertexAttribArray(a_lemon_s);
            gl::VertexAttribPointer(a_lemon_s, 1, gl::FLOAT, gl::FALSE, 0, 0 as *const GLvoid);
            gl::VertexAttribDivisor(a_lemon_s, 1);

            let a_lemon_color   = gl::GetAttribLocation(init.program, cstr!("a_lemon_color")) as GLuint;
            init.vbo_lemon_color = gl::gen_object(gl::GenBuffers);
            gl::BindBuffer(gl::ARRAY_BUFFER, init.vbo_lemon_color);
            gl::buffer_init::<Vec3>(gl::ARRAY_BUFFER, max_bodies, gl::DYNAMIC_DRAW);
            gl::EnableVertexAttribArray(a_lemon_color);
            gl::VertexAttribPointer(a_lemon_color, 3, gl::FLOAT, gl::FALSE, 0, 0 as *const GLvoid);
            gl::VertexAttribDivisor(a_lemon_color, 1);

            // PER-VERTEX ATTRIBUTES
            let base_mesh = lemon::make_base_mesh();

            let a_position   = gl::GetAttribLocation(init.program, cstr!("a_position")) as GLuint;
            let vbo_position = gl::gen_object(gl::GenBuffers);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo_position);
            gl::buffer_data(gl::ARRAY_BUFFER, &base_mesh.points, gl::STATIC_DRAW);
            gl::EnableVertexAttribArray(a_position);
            gl::VertexAttribPointer(a_position, 3, gl::FLOAT, gl::FALSE, 0, 0 as *const GLvoid);

            init.ebo = gl::gen_object(gl::GenBuffers);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, init.ebo);
            gl::buffer_data(gl::ELEMENT_ARRAY_BUFFER, &base_mesh.indices, gl::STATIC_DRAW);
            init.ebo_len = base_mesh.indices.len();
        }
        init
    }

    #[inline]
    pub fn render_lemons(&mut self, n: usize) {
        unsafe {
            gl::UseProgram(self.program);
            gl::BindVertexArray(self.vao);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, self.ebo);
            gl::DrawElementsInstanced(
                gl::TRIANGLES,
                self.ebo_len as GLsizei,
                ELEMENT_INDEX_TYPE,
                ptr::null(),
                n as GLsizei,
            );
        }
    }

    #[inline]
    pub fn update_camera(&mut self, camera: &Camera) {
        unsafe {
            gl::UseProgram(self.program);
            gl::UniformMatrix4fv(self.u_camera_view, 1, gl::FALSE,  camera.view.as_ptr());
            gl::UniformMatrix4fv(self.u_camera_projection, 1, gl::FALSE, camera.projection.as_ptr());
            gl::Uniform3fv(self.u_camera_position, 1, camera.position.as_ptr());
        }
    }

    #[inline]
    pub fn update_lemon(&mut self, id: usize, 
        lemon_s:         Option<&Real>,
        lemon_color:     Option<&Vec3>,
        lemon_transform: Option<&Mat4>,
    ) {
        unsafe {
            gl::UseProgram(self.program);
            if let Some(lemon_s) = lemon_s {
                gl::BindBuffer(gl::ARRAY_BUFFER, self.vbo_lemon_s);
                gl::buffer_sub_data(gl::ARRAY_BUFFER, id, slice::from_ref(lemon_s));
            }
            if let Some(lemon_color) = lemon_color {
                gl::BindBuffer(gl::ARRAY_BUFFER, self.vbo_lemon_color);
                gl::buffer_sub_data(gl::ARRAY_BUFFER, id, slice::from_ref(lemon_color));
            }
            if let Some(lemon_transform) = lemon_transform {
                gl::BindBuffer(gl::ARRAY_BUFFER, self.vbo_transform);
                gl::buffer_sub_data(gl::ARRAY_BUFFER, id, slice::from_ref(lemon_transform));
            }
        }
    }

    #[inline]
    pub fn update_lemon_transforms(&mut self, transforms: &[Mat4]) {
        unsafe {
            gl::BindBuffer(gl::ARRAY_BUFFER, self.vbo_transform);
            gl::buffer_sub_data(gl::ARRAY_BUFFER, 0, transforms);
        }
    }

    #[inline]
    pub fn update_hover(&mut self,
        id:   Option<usize>,
        glow: Option<Real>,
    ) {
        unsafe {
            gl::BindBuffer(gl::ARRAY_BUFFER, self.vbo_lemon_glow);
            if let Some(id) = id {
                if self.hover_id != !0 && self.hover_id != id {
                    let mut remove_glow = 0.0;
                    if self.selection_id == self.hover_id {
                        remove_glow += self.selection_glow;
                    }
                    gl::buffer_sub_data(
                        gl::ARRAY_BUFFER, self.hover_id, 
                        slice::from_ref(&remove_glow),
                    );
                }
                self.hover_id = id;
            }
            if let Some(glow) = glow {
                self.hover_glow = glow;
            }
            if self.hover_id != !0 {
                let mut set_glow = self.hover_glow;
                if self.selection_id == self.hover_id {
                    set_glow += self.selection_glow;
                }
                gl::buffer_sub_data(
                    gl::ARRAY_BUFFER, self.hover_id, 
                    slice::from_ref(&set_glow),
                );
            }
        }
    }

    #[inline]
    pub fn update_selection(&mut self,
        id:   Option<usize>,
        glow: Option<Real>,
    ) {
        unsafe {
            gl::BindBuffer(gl::ARRAY_BUFFER, self.vbo_lemon_glow);
            if let Some(id) = id {
                if self.selection_id != !0 && self.selection_id != id {
                    let mut remove_glow = 0.0;
                    if self.hover_id == self.selection_id {
                        remove_glow += self.hover_glow;
                    }
                    gl::buffer_sub_data(
                        gl::ARRAY_BUFFER, self.selection_id, 
                        slice::from_ref(&remove_glow),
                    );
                }
                self.selection_id = id;
            }
            if let Some(glow) = glow {
                self.selection_glow = glow;
            }

            if self.selection_id != !0 {
                let mut set_glow = self.selection_glow;
                if self.hover_id == self.selection_id {
                    set_glow += self.hover_glow;
                }
                gl::buffer_sub_data(
                    gl::ARRAY_BUFFER, self.selection_id, 
                    slice::from_ref(&set_glow),
                );
            }
        }
    }
}