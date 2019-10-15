use super::*;

mod bindings {
    include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));
}
pub use crate::gl::bindings::*;

mod macros;
#[macro_export]
#[macro_use]
pub use crate::gl::macros::*;

mod error;
pub use crate::gl::error::*;

mod get;
pub use crate::gl::get::*;

mod shader;
pub use crate::gl::shader::*;

#[inline]
pub fn get_uniform_location(program: GLuint, name: &CStr) -> Result<GLint, &CStr> {
    let location = unsafe { gl::GetUniformLocation(program, name.as_ptr() as *const GLchar) };

    #[cfg(debug)] gl::get_error().unwrap();

    if location != -1 {
        Ok(location)
    } else {
        Err(name)
    }
}

#[inline]
pub fn get_attrib_location(program: GLuint, name: &CStr) -> Result<GLuint, &CStr> {
    let location = unsafe { gl::GetAttribLocation(program, name.as_ptr() as *const GLchar) };

    #[cfg(debug)] gl::get_error().unwrap();

    if location != -1 {
        Ok(location as GLuint)
    } else {
        Err(name)
    }
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
