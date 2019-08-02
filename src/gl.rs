use super::*;

mod bindings {
    include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));
}
pub use crate::gl::bindings::*;

#[macro_export]
macro_rules! gl_with_temp_state {
    ($($get:ident),+, $block:block) => { {
        $(
            #[allow(non_snake_case)]
            let $get = gl::get(gl::$get);
        )+
        let result = $block;
        $(
            gl_pop_temp_state!($get, $get);
        )+
        result
    } }
}

macro_rules! gl_pop_temp_state {
    ($temp:ident, CURRENT_PROGRAM)       => { gl::UseProgram($temp) };
    ($temp:ident, VERTEX_ARRAY_BINDING)  => { gl::BindVertexArray($temp) };
    ($temp:ident, ARRAY_BUFFER_BINDING)  => { gl::BindBuffer(gl::ARRAY_BUFFER, $temp) };
    ($temp:ident, ACTIVE_TEXTURE)        => { gl::ActiveTexture($temp) };
    ($temp:ident, TEXTURE_BINDING_2D)    => { gl::BindTexture(gl::TEXTURE_2D, $temp) };
    ($temp:ident, TEXTURE_BINDING_RECTANGLE) => {
        gl::BindTexture(gl::TEXTURE_RECTANGLE, $temp)
    };
}

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

#[inline]
pub unsafe fn get<T: GlGet>(parameter: GLenum) -> T {
    let mut result: T = mem::uninitialized(); T::GL_GET(parameter, &mut result);
    result
}

bitflags! {
    pub struct Error: GLenum {
        const NO_ERROR                      = gl::NO_ERROR;
        const INVALID_ENUM                  = gl::INVALID_ENUM;
        const INVALID_VALUE                 = gl::INVALID_VALUE;
        const INVALID_OPERATION             = gl::INVALID_OPERATION;
        const INVALID_FRAMEBUFFER_OPERATION = gl::INVALID_FRAMEBUFFER_OPERATION;
        const OUT_OF_MEMORY                 = gl::OUT_OF_MEMORY;
    }
}

pub unsafe fn get_error() -> Result<(), Error> {
    let mut error = Error::NO_ERROR;
    loop {
        let next_error = gl::GetError();
        if next_error == gl::NO_ERROR {
            break;
        } else {
            error.bits |= next_error;
        }
    }
    if error == Error::NO_ERROR {
        Ok(())
    } else {
        Err(error)
    }
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
