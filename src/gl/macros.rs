use super::*;

#[macro_export]
macro_rules! gl_with_temp_state {
    ($($get:ident),+, $block:block) => { {
        macro_rules! gl_pop_temp_state {
            ($temp:ident, CURRENT_PROGRAM)       => { gl::UseProgram($temp) };
            ($temp:ident, VERTEX_ARRAY_BINDING)  => { gl::BindVertexArray($temp) };
            ($temp:ident, ARRAY_BUFFER_BINDING)  => { gl::BindBuffer(gl::ARRAY_BUFFER, $temp) };
            ($temp:ident, ACTIVE_TEXTURE)        => { gl::ActiveTexture($temp) };
            ($temp:ident, TEXTURE_BINDING_2D)    => { gl::BindTexture(gl::TEXTURE_2D, $temp) };
            ($temp:ident, TEXTURE_BINDING_RECTANGLE) => {
                gl::BindTexture(gl::TEXTURE_RECTANGLE, $temp)
            };
        };

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