use super::*;

#[derive(Copy, Clone, Debug, Default)]
struct DrawCmd {
    vao: GLuint,
    vbo: GLuint,
    len: usize,
    frames: usize,
}

#[derive(Debug)]
pub struct DebugRender {
    program:    GLuint,
    u_camera:   GLuint,
    a_color:    GLuint,
    a_position: GLuint,

    draw_cmds: Vec<DrawCmd>,
}

impl DebugRender {
    pub fn new() -> Self {
        unsafe {
            let program = gl_link_shaders(&[
                gl_compile_shader(include_str!("shader/debug.vert.glsl"), gl::VERTEX_SHADER),
                gl_compile_shader(include_str!("shader/debug.frag.glsl"), gl::FRAGMENT_SHADER),
            ]);
            let u_camera = gl::GetUniformLocation(program, cstr!("u_camera")) as GLuint;

            let a_color    = gl::GetAttribLocation(program, cstr!("a_color")) as GLuint;
            let a_position = gl::GetAttribLocation(program, cstr!("a_position")) as GLuint;

            DebugRender {
                program, u_camera, a_color, a_position,
                draw_cmds: Vec::new(),
            }
        }
    }

    pub fn with_shared_context(share: &DebugRender) -> Self {
        unsafe {
            let mut new: DebugRender = mem::transmute_copy(share);
            new.draw_cmds = Vec::new();
            new
        }
    }

    pub fn update_camera(&mut self, camera: &Mat4) {
        unsafe {
            let program = gl_get(gl::CURRENT_PROGRAM);
            {
                gl::UseProgram(self.program);
                gl::UniformMatrix4fv(self.u_camera as GLint, 1, gl::FALSE, camera.as_ptr());
            }
            gl::UseProgram(program);
        }
    }

    pub fn draw_ray(&mut self, color: &Vec3, frames: usize, origin: &Point3, direction: &Vec3) {
        self.draw_line(color, frames, &[*origin, origin + direction]);
    }

    pub fn draw_line(&mut self, color: &Vec3, frames: usize, points: &[Point3]) {
        if points.len() < 2 { panic!("attempted to draw debug line with <2 points"); }
        unsafe {
            let vao = gl_get(gl::VERTEX_ARRAY_BINDING);
            let vbo = gl_get(gl::ARRAY_BUFFER_BINDING);
            {
                let mut draw_cmd = mem::uninitialized::<DrawCmd>();
                draw_cmd.len     = points.len();
                draw_cmd.frames  = frames;

                draw_cmd.vao = gl_gen_object(gl::GenVertexArrays);
                gl::BindVertexArray(draw_cmd.vao);

                draw_cmd.vbo = gl_gen_object(gl::GenBuffers);
                gl::BindBuffer(gl::ARRAY_BUFFER, draw_cmd.vbo);
                gl_buffer_init::<Vec3>(gl::ARRAY_BUFFER, points.len() + 1, gl::STATIC_DRAW);

                gl_buffer_sub_data(gl::ARRAY_BUFFER, 0, slice::from_ref(color));
                gl::EnableVertexAttribArray(self.a_color);
                gl::VertexAttribPointer(
                    self.a_color, 3, gl::FLOAT,
                    gl::FALSE, 0, mem::size_of::<[Vec3; 0]>() as *const GLvoid
                );
                gl::VertexAttribDivisor(self.a_color, 1);

                gl_buffer_sub_data(gl::ARRAY_BUFFER, 1, points);
                gl::EnableVertexAttribArray(self.a_position);
                gl::VertexAttribPointer(
                    self.a_position, 3, gl::FLOAT,
                    gl::FALSE, 0, mem::size_of::<[Vec3; 1]>() as *const GLvoid
                );

                self.draw_cmds.push(draw_cmd);
            }
            gl::BindVertexArray(vao);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        }
    }

    pub fn draw_axes(&mut self, scale: f32, frames: usize, transform: &Mat4) {
        unsafe {
            let vao = gl_get(gl::VERTEX_ARRAY_BINDING);
            let vbo = gl_get(gl::ARRAY_BUFFER_BINDING);
            {
                let mut draw_cmd = mem::uninitialized::<DrawCmd>();
                draw_cmd.len     = 6;
                draw_cmd.frames  = frames;

                draw_cmd.vao = gl_gen_object(gl::GenVertexArrays);
                gl::BindVertexArray(draw_cmd.vao);

                draw_cmd.vbo = gl_gen_object(gl::GenBuffers);
                gl::BindBuffer(gl::ARRAY_BUFFER, draw_cmd.vbo);

                let origin    = transform.transform_point(point3!()).to_vec();
                let transform = |v: Vec3| transform.transform_point(point3!(scale*v)).to_vec();
                gl_buffer_data(gl::ARRAY_BUFFER, &[
                    VEC3_X, transform(VEC3_X), VEC3_X, origin,
                    VEC3_Y, transform(VEC3_Y), VEC3_Y, origin,
                    VEC3_Z, transform(VEC3_Z), VEC3_Z, origin,
                ], gl::STATIC_DRAW);

                gl::EnableVertexAttribArray(self.a_color);
                gl::VertexAttribPointer(
                    self.a_color, 3, gl::FLOAT, gl::FALSE,
                    mem::size_of::<[Vec3; 2]>() as GLsizei,
                    mem::size_of::<[Vec3; 0]>() as *const GLvoid,
                );

                gl::EnableVertexAttribArray(self.a_position);
                gl::VertexAttribPointer(
                    self.a_position, 3, gl::FLOAT, gl::FALSE,
                    mem::size_of::<[Vec3; 2]>() as GLsizei,
                    mem::size_of::<[Vec3; 1]>() as *const GLvoid
                );

                self.draw_cmds.push(draw_cmd);
            }
            gl::BindVertexArray(vao);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        }
    }

    pub fn render_frame(&mut self) {
        if self.draw_cmds.is_empty() { return; }
        unsafe {
            let program = gl_get(gl::CURRENT_PROGRAM);
            let vao     = gl_get(gl::VERTEX_ARRAY_BINDING);
            let vbo     = gl_get(gl::ARRAY_BUFFER_BINDING);
            {
                gl::UseProgram(self.program);

                let mut index = 0;
                while index < self.draw_cmds.len() {
                    let ref mut draw_cmd = self.draw_cmds[index];

                    gl::BindVertexArray(draw_cmd.vao);
                    gl::DrawArraysInstanced(gl::LINE_STRIP, 0, draw_cmd.len as GLint, 1);

                    if draw_cmd.frames > 1 {
                        draw_cmd.frames -= 1;
                        index += 1;
                    } else {
                        gl::DeleteBuffers(1, &draw_cmd.vbo);
                        gl::DeleteVertexArrays(1, &draw_cmd.vao);
                        self.draw_cmds.swap_remove(index);
                    }
                }
            }
            gl::UseProgram(program);
            gl::BindVertexArray(vao);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        }
    }
}
