use super::*;

bitflags! {
    pub struct GlError: GLenum {
        const NO_ERROR                      = gl::NO_ERROR;
        const INVALID_ENUM                  = gl::INVALID_ENUM;
        const INVALID_VALUE                 = gl::INVALID_VALUE;
        const INVALID_OPERATION             = gl::INVALID_OPERATION;
        const INVALID_FRAMEBUFFER_OPERATION = gl::INVALID_FRAMEBUFFER_OPERATION;
        const OUT_OF_MEMORY                 = gl::OUT_OF_MEMORY;
    }
}

impl fmt::Display for GlError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt::Debug::fmt(self, f)
    }
}

impl Error for GlError { }

pub fn get_error() -> Result<(), GlError> {
    let mut error = GlError::NO_ERROR;
    loop {
        let next_error = unsafe { gl::GetError() };
        if next_error == gl::NO_ERROR {
            break;
        } else {
            error.bits |= next_error;
        }
    }
    if error == GlError::NO_ERROR {
        Ok(())
    } else {
        Err(error)
    }
}
