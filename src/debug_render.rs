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
    program:     GLuint,
    u_camera:    GLint,
    a_color:     GLuint,
    a_position:  GLuint,
    a_transform: GLuint,

    camera: Camera,

    reverse_index: usize,
    draw_cmds: Vec<DrawCmd>,

    transform_stack: Vec<Mat4>,
}

impl DebugRender {
    pub fn new() -> Self {
        let program = gl::link_shaders(&[
            gl::compile_shader(
                include_bytes!("shader/debug.vert.glsl"), 
                gl::VERTEX_SHADER,
            ).unwrap(),
            gl::compile_shader(
                include_bytes!("shader/debug.frag.glsl"), 
                gl::FRAGMENT_SHADER,
            ).unwrap(),
        ]).unwrap();
        let u_camera = gl::get_uniform_location(program, cstr!("u_camera")).unwrap();

        let a_color     = gl::get_attrib_location(program, cstr!("a_color")).unwrap();
        let a_position  = gl::get_attrib_location(program, cstr!("a_position")).unwrap();
        let a_transform = gl::get_attrib_location(program, cstr!("a_transform")).unwrap();

        DebugRender {
            program, u_camera, a_color, a_position, a_transform,
            camera: Camera::default(),
            draw_cmds: Vec::new(), reverse_index: !0,
            transform_stack: vec![Mat4::identity()],
        }
    }

    pub fn with_shared_context(share: &DebugRender) -> Self {
        DebugRender {
            program: share.program, 
            u_camera: share.u_camera,
            a_color: share.a_color, 
            a_position: share.a_position,
            a_transform: share.a_transform,
            camera: share.camera,

            draw_cmds: Vec::new(),
            reverse_index: !0,
            transform_stack: vec![Mat4::identity()],
        }
    }

    pub fn update_camera(&mut self, camera: &Camera) {
        unsafe { gl_with_temp_state!(
            CURRENT_PROGRAM,
        {
            gl::UseProgram(self.program);
            gl::UniformMatrix4fv(self.u_camera, 1, gl::FALSE, (camera.projection * camera.view).as_ptr());
        }); }

        self.camera = camera.clone();
    }

    pub fn get_camera(&self) -> &Camera {
        &self.camera
    }

    pub fn draw_ray(&mut self, color: &Vec3, frames: usize, origin: &Point3, direction: &Vec3) {
        self.draw_line(color, frames, &[*origin, origin + direction]);
    }

    pub fn draw_line(&mut self, color: &Vec3, frames: usize, points: &[Point3]) {
        if points.len() < 2 { panic!("attempted to draw debug line with <2 points"); }
        unsafe { gl_with_temp_state!(
            VERTEX_ARRAY_BINDING,
            ARRAY_BUFFER_BINDING,
        {
            let mut draw_cmd = mem::uninitialized::<DrawCmd>();
            draw_cmd.len     = points.len();
            draw_cmd.frames  = frames;

            draw_cmd.vao = gl::gen_object(gl::GenVertexArrays);
            gl::BindVertexArray(draw_cmd.vao);

            draw_cmd.vbo = gl::gen_object(gl::GenBuffers);
            gl::BindBuffer(gl::ARRAY_BUFFER, draw_cmd.vbo);

            gl::buffer_init::<f32>(gl::ARRAY_BUFFER, 3*points.len() + 3 + 16, gl::STATIC_DRAW);

            gl::buffer_sub_data(gl::ARRAY_BUFFER, 0, points);
            gl::EnableVertexAttribArray(self.a_position);
            gl::VertexAttribPointer(
                self.a_position, 3, gl::FLOAT, gl::FALSE,
                0, ptr::null::<Vec3>().offset(0) as *const GLvoid,
            );

            gl::buffer_sub_data(gl::ARRAY_BUFFER, points.len(), slice::from_ref(color));
            gl::EnableVertexAttribArray(self.a_color);
            gl::VertexAttribPointer(
                self.a_color, 3, gl::FLOAT, gl::FALSE,
                0, ptr::null::<Vec3>().offset(points.len() as isize) as *const GLvoid
            );
            gl::VertexAttribDivisor(self.a_color, 1);

            let transform = self.transform_stack.last().unwrap();
            gl::buffer_sub_data::<f32>(gl::ARRAY_BUFFER, 
                3*points.len() + 3, 
                AsRef::<[f32; 16]>::as_ref(&transform)
            );
            for i in 0..4 {
                let a_transform_i = self.a_transform + i as GLuint;
                gl::EnableVertexAttribArray(a_transform_i);
                gl::VertexAttribPointer(
                    a_transform_i, 4, gl::FLOAT, gl::FALSE,
                    0, ptr::null::<f32>().offset(3*points.len() as isize + 3 + 4*i) as *const GLvoid
                );
                gl::VertexAttribDivisor(a_transform_i, 1);
            }

            self.draw_cmds.push(draw_cmd);
        }); }
    }

    pub fn draw_axes(&mut self, scale: f32, frames: usize, transform: &Mat4) {
        unsafe { gl_with_temp_state!(
            VERTEX_ARRAY_BINDING,
            ARRAY_BUFFER_BINDING,
        {
            let mut draw_cmd = mem::uninitialized::<DrawCmd>();
            draw_cmd.len     = 6;
            draw_cmd.frames  = frames;

            draw_cmd.vao = gl::gen_object(gl::GenVertexArrays);
            gl::BindVertexArray(draw_cmd.vao);

            draw_cmd.vbo = gl::gen_object(gl::GenBuffers);
            gl::BindBuffer(gl::ARRAY_BUFFER, draw_cmd.vbo);

            const AXES: &[Vec3] = &[
                VEC3_X, VEC3_X, VEC3_X, VEC3_0,
                VEC3_Y, VEC3_Y, VEC3_Y, VEC3_0,
                VEC3_Z, VEC3_Z, VEC3_Z, VEC3_0,
            ];

            // TODO: use a shared array buffer
            gl::buffer_init::<f32>(gl::ARRAY_BUFFER, 3*AXES.len() + 16, gl::STATIC_DRAW);
            gl::buffer_sub_data(gl::ARRAY_BUFFER, 0, AXES);

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

            let transform = self.transform_stack.last().unwrap() 
                          * transform 
                          * Mat4::from_diagonal(vec4!(scale*VEC3_1, 1.0));
            gl::buffer_sub_data::<f32>(gl::ARRAY_BUFFER, 
                3*AXES.len(), 
                AsRef::<[f32; 16]>::as_ref(&transform)
            );
            for i in 0..4 {
                let a_transform_i = self.a_transform + i as GLuint;
                gl::EnableVertexAttribArray(a_transform_i);
                gl::VertexAttribPointer(
                    a_transform_i, 4, gl::FLOAT, gl::FALSE,
                    0, ptr::null::<f32>().offset(3*AXES.len() as isize + 4*i) as *const GLvoid
                );
                gl::VertexAttribDivisor(a_transform_i, 1);
            }

            self.draw_cmds.push(draw_cmd);
        }); }
    }

    pub fn reverse_draw_order_begin(&mut self) {
        if self.reverse_index != !0 {
            panic!("DebugRender attemped reverse_draw_order_begin without ending previous sequence");
        }
        self.reverse_index = self.draw_cmds.len() - 1;
    }

    pub fn reverse_draw_order_end(&mut self) {
        if self.reverse_index == !0 {
            panic!("DebugRender attemped reverse_draw_order_end without reverse_draw_order_begin");
        }
        let reverse_range = self.reverse_index..self.draw_cmds.len();
        self.draw_cmds[reverse_range].reverse();
        self.reverse_index = !0;
    }

    pub fn push_transform(&mut self, transform: &Mat4) {
        self.transform_stack.push(self.transform_stack.last().unwrap() * transform);
    }

    pub fn pop_transform(&mut self) {
        if self.transform_stack.len() <= 1 { 
            panic!("DebugRender attempted to pop empty transform stack"); 
        }
        self.transform_stack.pop();
    }

    pub fn render_frame(&mut self) {
        if self.reverse_index != !0 {
            panic!("DebugRender attemped render_frame without ending reversed sequence");
        }
        if self.transform_stack.len() != 1 {
            panic!("DebugRender pushed a transform that was not popped");
        }
        if self.draw_cmds.is_empty() { return; }
        unsafe { gl_with_temp_state!(
            CURRENT_PROGRAM,
            VERTEX_ARRAY_BINDING,
            ARRAY_BUFFER_BINDING,
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
                    self.draw_cmds.remove(index);
                }
            }
        }); }
    }
}
