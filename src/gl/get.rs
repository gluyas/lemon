use super::*;

#[inline]
pub unsafe fn get<T: GlGet>(parameter: GLenum) -> T {
    let mut result: T = mem::uninitialized(); 
    T::GL_GET(parameter, &mut result);
    result
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