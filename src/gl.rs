use super::*;

mod bindings {
    include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));
}
pub use crate::gl::bindings::*;

#[inline]
pub unsafe fn buffer_data<T>(target: GLenum, data: &[T], usage: GLenum) {
    gl::BufferData(
        target,
        mem::size_of_val(data) as GLsizeiptr,
        data.as_ptr() as *const GLvoid,
        usage,
    );
}

#[inline]
pub unsafe fn buffer_sub_data<T>(target: GLenum, offset: usize, data: &[T]) {
    gl::BufferSubData(
        target,
        (offset  * mem::size_of::<T>()) as GLintptr,
        mem::size_of_val(data) as GLsizeiptr,
        data.as_ptr() as *const GLvoid,
    );
}

#[inline]
pub unsafe fn buffer_init<T>(target: GLenum, count: usize, usage: GLenum) {
    gl::BufferData(
        target,
        (mem::size_of::<T>() * count) as GLsizeiptr,
        0 as *const GLvoid,
        usage,
    );
}

#[inline]
pub unsafe fn gen_object(gen_callback: unsafe fn (GLsizei, *mut GLuint)) -> GLuint {
    let mut name = GLuint::default();
    gen_callback(1, &mut name);
    name
}

pub unsafe trait GlGet {
    const GL_GET: unsafe fn (GLenum, *mut Self);
}

unsafe impl GlGet for GLboolean {
    const GL_GET: unsafe fn (GLenum, *mut Self) = gl::GetBooleanv;
}
unsafe impl GlGet for GLint {
    const GL_GET: unsafe fn (GLenum, *mut Self) = gl::GetIntegerv;
}
unsafe impl GlGet for GLuint {
    const GL_GET: unsafe fn (GLenum, *mut Self) = {
        unsafe fn get(p: GLenum, r: *mut GLuint) { gl::GetIntegerv(p, r as *mut GLint); }
        get
    };
}
unsafe impl GlGet for GLint64 {
    const GL_GET: unsafe fn (GLenum, *mut Self) = gl::GetInteger64v;
}
unsafe impl GlGet for GLuint64 {
    const GL_GET: unsafe fn (GLenum, *mut Self) = {
        unsafe fn get(p: GLenum, r: *mut GLuint64) { gl::GetInteger64v(p, r as *mut GLint64); }
        get
    };
}
unsafe impl GlGet for GLfloat {
    const GL_GET: unsafe fn (GLenum, *mut Self) = gl::GetFloatv;
}
unsafe impl GlGet for GLdouble {
    const GL_GET: unsafe fn (GLenum, *mut Self) = gl::GetDoublev;
}

#[inline]
pub unsafe fn get<T: GlGet>(parameter: GLenum) -> T {
    let mut result: T = mem::uninitialized(); T::GL_GET(parameter, &mut result);
    result
}

pub fn compile_shader(src: &str, ty: GLenum) -> GLuint {
    unsafe {
        let shader = gl::CreateShader(ty);
        gl::ShaderSource(
            shader, 1,
            &(src.as_ptr() as *const GLchar) as *const *const _,
            &(src.len() as GLint) as *const _
        );
        gl::CompileShader(shader);

        // check shader compile errors
        let mut status = gl::FALSE as GLint;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut status);

        if status != (gl::TRUE as GLint) {
            use std::str::from_utf8;

            let mut len = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = Vec::with_capacity(len as usize);
            buf.set_len((len as usize) - 1); // subtract 1 to skip the trailing null character
            gl::GetShaderInfoLog(
                shader,
                len,
                ptr::null_mut(),
                buf.as_mut_ptr() as *mut GLchar,
            );
            panic!(
                "GLSL compile error:\n{}",
                from_utf8(&buf).ok().expect("ShaderInfoLog not valid utf8")
            );
        }
        shader
    }
}

pub fn link_shaders(shaders: &[GLuint]) -> GLuint {
    unsafe {
        let program = gl::CreateProgram();
        for &shader in shaders { gl::AttachShader(program, shader); }
        gl::LinkProgram(program);

        {   // check link status
            let mut status = gl::FALSE as GLint;
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut status);
            if status != (gl::TRUE as GLint) {
                use std::str::from_utf8;

                let mut len = 0;
                gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
                let mut buf = Vec::with_capacity(len as usize);
                buf.set_len((len as usize) - 1); // subtract 1 to skip the trailing null character
                gl::GetProgramInfoLog(
                    program,
                    len,
                    ptr::null_mut(),
                    buf.as_mut_ptr() as *mut GLchar,
                );
                panic!(
                    "GLSL link error:\n{}",
                    from_utf8(&buf).ok().expect("ProgramInfoLog not valid utf8")
                );
            }
        }
        program
    }
}
