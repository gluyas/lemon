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

    u_selection_instance_id: GLint,
    u_selection_glow: GLint,

    u_hover_instance_id: GLint,
    u_hover_glow: GLint,

    ubo_camera: GLuint,

    vao: GLuint,

    ebo: GLuint,
    ebo_len: usize,

    vbo_transform: GLuint,
    vbo_lemon_s: GLuint,
    vbo_lemon_color: GLuint,
}

impl Render {
    #[inline]
    pub fn init(max_bodies: usize) -> Self {
        let mut init = Render::default();
        unsafe {
            init.program = gl::link_shaders(&[
                gl::compile_shader(
                    include_bytes!("shader/lemon.vert.glsl"), 
                    gl::VERTEX_SHADER,
                ).unwrap(),
                gl::compile_shader(
                    include_bytes!("shader/lemon.frag.glsl"), 
                    gl::FRAGMENT_SHADER,
                ).unwrap(),
            ]).unwrap();
            gl::UseProgram(init.program);

            init.ubo_camera = gl::gen_object(gl::GenBuffers);
            gl::BindBuffer(gl::UNIFORM_BUFFER, init.ubo_camera);
            gl::buffer_init::<Camera>(gl::UNIFORM_BUFFER, 1, gl::DYNAMIC_DRAW);

            let camera_binding_index: GLuint = 1;
            gl::BindBufferBase(gl::UNIFORM_BUFFER, camera_binding_index, init.ubo_camera);

            let camera_index = gl::GetUniformBlockIndex(init.program, cstr!("Camera").as_ptr() as *const GLchar);
            gl::UniformBlockBinding(init.program, camera_index, camera_binding_index);

            init.u_selection_instance_id = gl::get_uniform_location(init.program, cstr!("u_selection_instance_id")).unwrap();
            gl::Uniform1i(init.u_selection_instance_id, !0);

            init.u_selection_glow = gl::get_uniform_location(init.program, cstr!("u_selection_glow")).unwrap();
            gl::Uniform1f(init.u_selection_glow, 0.0);

            init.u_hover_instance_id = gl::get_uniform_location(init.program, cstr!("u_hover_instance_id")).unwrap();
            gl::Uniform1i(init.u_hover_instance_id, !0);

            init.u_hover_glow = gl::get_uniform_location(init.program, cstr!("u_hover_glow")).unwrap();
            gl::Uniform1f(init.u_hover_glow, 0.0);

            let txo_radius_normal_z_atlas = gl::gen_object(gl::GenTextures);
            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, txo_radius_normal_z_atlas);

            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as _);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as _);

            let radius_normal_z_map = lemon::make_radius_normal_z_map();
            gl::TexImage2D(
                gl::TEXTURE_2D, 0,
                gl::RG32F as _, lemon::MAP_RESOLUTION as _, lemon::MAP_RESOLUTION as _, 0,
                gl::RG, gl::FLOAT, radius_normal_z_map.as_ptr() as *const GLvoid,
            );
            gl::GenerateMipmap(gl::TEXTURE_2D);
            let u_radius_normal_z_map = gl::get_uniform_location(init.program, cstr!("u_radius_normal_z_map")).unwrap();
            gl::Uniform1i(u_radius_normal_z_map, 0);

            init.vao = gl::gen_object(gl::GenVertexArrays);
            gl::BindVertexArray(init.vao);

            // PER-INSTANCE ATTRIBUTES
            let a_transform    = gl::get_attrib_location(init.program, cstr!("a_transform")).unwrap();
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

            let a_lemon_s   = gl::get_attrib_location(init.program, cstr!("a_lemon_s")).unwrap();
            init.vbo_lemon_s = gl::gen_object(gl::GenBuffers);
            gl::BindBuffer(gl::ARRAY_BUFFER, init.vbo_lemon_s);
            gl::buffer_init::<f32>(gl::ARRAY_BUFFER, max_bodies, gl::DYNAMIC_DRAW);
            gl::EnableVertexAttribArray(a_lemon_s);
            gl::VertexAttribPointer(a_lemon_s, 1, gl::FLOAT, gl::FALSE, 0, 0 as *const GLvoid);
            gl::VertexAttribDivisor(a_lemon_s, 1);

            let a_lemon_color   = gl::get_attrib_location(init.program, cstr!("a_lemon_color")).unwrap();
            init.vbo_lemon_color = gl::gen_object(gl::GenBuffers);
            gl::BindBuffer(gl::ARRAY_BUFFER, init.vbo_lemon_color);
            gl::buffer_init::<Vec3>(gl::ARRAY_BUFFER, max_bodies, gl::DYNAMIC_DRAW);
            gl::EnableVertexAttribArray(a_lemon_color);
            gl::VertexAttribPointer(a_lemon_color, 3, gl::FLOAT, gl::FALSE, 0, 0 as *const GLvoid);
            gl::VertexAttribDivisor(a_lemon_color, 1);

            // PER-VERTEX ATTRIBUTES
            let base_mesh = lemon::make_base_mesh();

            let a_position   = gl::get_attrib_location(init.program, cstr!("a_position")).unwrap();
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
            gl::BindBuffer(gl::UNIFORM_BUFFER, self.ubo_camera);
            gl::buffer_data(gl::UNIFORM_BUFFER, slice::from_ref(camera), gl::DYNAMIC_DRAW);
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