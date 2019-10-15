use super::*;

#[derive(Clone)]
pub struct GlslError {
    description: String, 
}

impl<T: Into<String>> From<T> for GlslError {
    fn from(t: T) -> Self {
        GlslError { description: t.into() }
    }
}

impl fmt::Display for GlslError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt::Display::fmt(&self.description, f)
    }
}

impl fmt::Debug for GlslError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt::Display::fmt(self, f)
    }
}

impl Error for GlslError { }

pub fn compile_and_link_shaders_from_path(
    srcs_and_types: &[(&dyn AsRef<Path>, GLenum)],
) -> Result<GLuint, GlslError> {
    unsafe {
        let program = gl::CreateProgram();
        let mut result = Ok(program);

        for &(src, ty) in srcs_and_types { 
            match gl::compile_shader_from_path(src, ty) {
                Ok(shader) => {
                    if result.is_ok() {
                        gl::AttachShader(program, shader);
                    }
                },
                Err(error) => {
                    let info_log = error.to_string();
                    result = Err(match result {
                        Err(existing_info_log) => String::from(format!("{}\n{}", &existing_info_log, &info_log)),
                        Ok(_no_info_log)       => info_log,
                    });
                },
            }
        }

        if result.is_ok() {
            gl::LinkProgram(program);

            // check program link errors
            let mut status: GLint = 0;
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut status);

            if status != gl::TRUE as GLint {
                let info_log = get_program_info_log(program);
                result = Err(String::from("shader program link error:\n\r") + &info_log);
            }
        }

        if result.is_err() {
            gl::delete_program_and_attached_shaders(program, Some(srcs_and_types.len()));
        }
        result.map_err(GlslError::from)
    }
}

pub fn compile_shader(src: &[u8], ty: GLenum) -> Result<GLuint, GlslError> {
    unsafe {
        let shader = gl::CreateShader(ty);
        gl::ShaderSource(
            shader, 
            1,
            &(src.as_ptr() as *const GLchar) as *const *const GLchar,
            &(src.len() as GLint) as *const GLint,
        );
        gl::CompileShader(shader);

        // check shader compile errors
        let mut status: GLint = 0;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut status);

        if status == gl::TRUE as GLint {
            Ok(shader)
        } else {
            let info_log = get_shader_info_log(shader);
            gl::DeleteShader(shader);

            let shader_name = match ty {
                gl::VERTEX_SHADER   => "Vertex Shader",
                gl::FRAGMENT_SHADER => "Fragment Shader",
                gl::GEOMETRY_SHADER => "Geometry Shader",
                _                   => "(unknown shader type)",
            };
            Err(GlslError::from(format!("{} compile error:\n\r{}", shader_name, &info_log)))
        }
    }
}

pub fn compile_shader_from_path(src: impl AsRef<Path>, ty: GLenum) -> Result<GLuint, GlslError> {
    compile_shader(&fs::read(&src).unwrap(), ty)
        .map_err(|error| GlslError::from(format!("{} {}",
            src.as_ref().to_string_lossy(),
            error.to_string(),
        )))
}

pub fn link_shaders(shaders: &[GLuint]) -> Result<GLuint, GlslError> {
    unsafe {
        let program = gl::CreateProgram();
        for &shader in shaders { 
            gl::AttachShader(program, shader);
        }
        gl::LinkProgram(program);

        // check program link errors
        let mut status: GLint = 0;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut status);

        if status == gl::TRUE as GLint {
            Ok(program)
        } else {
            let info_log = get_program_info_log(program);
            gl::DeleteProgram(program);

            Err(GlslError::from(format!("Shader Program link error:\n{}", &info_log)))
        }
    }
}

pub fn delete_program_and_attached_shaders(program: GLuint, max_shaders: Option<usize>) {
    unsafe {
        let max_shaders = match max_shaders {
            Some(max_shaders) => max_shaders as GLint,
            None => {
                let mut num_shaders = 0;
                gl::GetProgramiv(program, gl::ATTACHED_SHADERS, &mut num_shaders as *mut GLint);
                num_shaders
            },
        };
        let mut shaders_to_delete     = Vec::with_capacity(max_shaders as usize);
        let mut shaders_to_delete_len = 0 as GLsizei;

        gl::GetAttachedShaders(
            program, max_shaders, 
            &mut shaders_to_delete_len as *mut GLsizei, 
            shaders_to_delete.as_mut_ptr(),
        );
        shaders_to_delete.set_len(shaders_to_delete_len as usize);

        for shader in shaders_to_delete {
            gl::DetachShader(program, shader);
            gl::DeleteShader(shader);
        }
        gl::DeleteProgram(program);   
    }
}

pub fn get_program_info_log(program: GLuint) -> String {
    get_generic_info_log(program, gl::GetProgramiv, gl::GetProgramInfoLog)
}

pub fn get_shader_info_log(shader: GLuint) -> String {
    get_generic_info_log(shader, gl::GetShaderiv, gl::GetShaderInfoLog)
}

fn get_generic_info_log(
    name:         GLuint,
    get_param:    unsafe fn(GLuint, GLenum, *mut GLint),
    get_info_log: unsafe fn(GLuint, GLsizei, *mut GLsizei, *mut GLchar),
) -> String {
    unsafe {
        let mut info_log_len: GLsizei = 0;
        get_param(name, gl::INFO_LOG_LENGTH, &mut info_log_len as *mut GLsizei);

        if info_log_len == 0 { return String::new(); }

        let mut buffer = vec![0_u8; info_log_len as usize];
        get_info_log(name, info_log_len, ptr::null_mut(), buffer.as_mut_ptr() as *mut GLchar);
        buffer.truncate(info_log_len as usize - 1);

        String::from_utf8(buffer).unwrap()
    }
}
