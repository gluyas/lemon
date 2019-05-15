use super::*;

pub struct DebugUi {
    width:  usize,
    height: usize,

    program:        GLuint,
    vao:            GLuint,
    texture_unit:   GLuint,
    texture_object: GLuint,

    pixels: Vec<u16>,
    dirty:  bool,
    clear:  bool,
}

impl DebugUi {
    pub fn new(width: usize, height: usize, texture_unit: GLuint) -> Self {
        unsafe { gl_with_temp_state!(
            CURRENT_PROGRAM,
            VERTEX_ARRAY_BINDING,
            ARRAY_BUFFER_BINDING,
            ACTIVE_TEXTURE,
            TEXTURE_BINDING_RECTANGLE,
        {
            gl::ActiveTexture(texture_unit);
            let texture_object = gl::gen_object(gl::GenTextures);
            gl::BindTexture(gl::TEXTURE_RECTANGLE, texture_object);

            let program = gl::link_shaders(&[
                gl::compile_shader(include_str!("shader/ui.vert.glsl"), gl::VERTEX_SHADER),
                gl::compile_shader(include_str!("shader/ui.frag.glsl"), gl::FRAGMENT_SHADER),
            ]);
            gl::UseProgram(program);

            let u_pixels = gl::GetUniformLocation(program, cstr!("u_pixels"));
            gl::Uniform1i(u_pixels, texture_unit as GLint - gl::TEXTURE0 as GLint);

            let u_depth = gl::GetUniformLocation(program, cstr!("u_depth"));
            gl::Uniform1f(u_depth, 1.0);

            let vao = gl::gen_object(gl::GenVertexArrays);
            gl::BindVertexArray(vao);

            let vbo = gl::gen_object(gl::GenBuffers);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);

            gl::buffer_data(gl::ARRAY_BUFFER, &[
                vec2!(-1.0, -1.0), vec2!(1.0, -1.0), vec2!(1.0, 1.0), vec2!(-1.0, 1.0),
            ], gl::STATIC_DRAW);

            let a_position = gl::GetAttribLocation(program, cstr!("a_position")) as GLuint;
            gl::EnableVertexAttribArray(a_position);
            gl::VertexAttribPointer(
                a_position, 2, gl::FLOAT, gl::FALSE,
                0, ptr::null(),
            );

            DebugUi {
                width, height,
                program, vao, texture_unit, texture_object,
                pixels: vec![0; width * height], dirty: true, clear: true,
            }
        }) }
    }

    pub fn copy_pixels(&mut self,
        x: isize, y: isize,
        w: usize, h: usize,
        pixels: &[u16],
    ) {
        self.copy_pixels_cropped(
            x, y,
            0, 0,
            w, h,
            pixels, w,
        );
    }

    pub fn copy_pixels_cropped(&mut self,
        dst_x: isize, dst_y: isize,
        src_x: usize, src_y: usize,
        mut img_w: usize, mut img_h: usize,
        pixels: &[u16], pixels_w: usize,
    ) {
        unsafe {
            if src_x + img_w > pixels_w {
                panic!("insufficent columns in pixel data: {} > {}",
                    if src_x == 0 { format!("{}", img_w) }
                    else          { format!("{} ({} + {})", src_x+img_w, src_x, img_w) },
                    pixels_w,
                );
            }
            if (src_y + img_h) * pixels_w > pixels.len() {
                panic!("insufficient rows in pixel data: {} > {} ({} > {})",
                    if src_y == 0 { format!("{}", img_h) }
                    else          { format!("{} ({} + {})", src_y+img_h, src_y, img_h) },
                    pixels.len() / pixels_w,

                    (src_y + img_h) * pixels_w,
                    pixels.len(),
                );
            }
            if img_w == 0                     || img_h == 0
            || dst_x >= (self.width as isize) || dst_y >= (self.height as isize)
            || dst_x + (img_w as isize) < 0   || dst_y + (img_h as isize) < 0 {
                return;
            }

            let mut src = pixels.as_ptr().add(src_x + src_y * pixels_w);
            let mut dst = self.pixels.as_mut_ptr().offset(dst_x + dst_y * self.width as isize);

            if dst_x + img_w as isize >= self.width as isize {
                img_w = (self.width  as isize - dst_x) as usize;
            }
            if dst_y + img_h as isize >= self.height as isize {
                img_h = (self.height as isize - dst_y) as usize;
            }

            if dst_x < 0 {
                src    = src.offset(-dst_x);
                dst    = dst.offset(-dst_x);
                img_w -= dst_x.neg() as usize;
            }
            if dst_y < 0 {
                src    = src.offset(-dst_y * pixels_w   as isize);
                dst    = dst.offset(-dst_y * self.width as isize);
                img_h -= dst_y.neg() as usize;
            }

            copy_pixels_raw(
                src, pixels_w,
                dst, self.width,
                img_w, img_h,
            );
        }
        self.set_dirty();
    }

    pub fn clear_pixels(&mut self) {
        if self.clear { return; }
        unsafe { ptr::write_bytes(self.pixels.as_mut_ptr(), 0, self.width * self.height); }
        self.set_dirty();
        self.clear = true;
    }

    pub fn clear_pixels_cropped(&mut self,
        x: isize, y: isize,
        mut w: usize, mut h: usize,
    ) {
        if self.clear { return; }
        unsafe {
            if w == 0                     || h == 0
            || x >= (self.width as isize) || y >= (self.height as isize)
            || x + (w as isize) < 0       || y + (h as isize) < 0 {
                return;
            }
            let mut dst = self.pixels.as_mut_ptr().offset(x + y * self.width as isize);

            if x + w as isize >= self.width as isize {
                w = (self.width  as isize - x) as usize;
            }
            if y + h as isize >= self.height as isize {
                h = (self.height as isize - y) as usize;
            }

            if x < 0 {
                dst    = dst.offset(-x);
                w -= x.neg() as usize;
            }
            if y < 0 {
                dst    = dst.offset(-y * self.width as isize);
                h -= y.neg() as usize;
            }

            clear_pixels_raw(
                dst, self.width,
                w, h,
            );
        }
        self.set_dirty();
    }

    pub fn render(&mut self) {
        if self.clear { return; }

        unsafe { gl_with_temp_state!(
            CURRENT_PROGRAM,
            VERTEX_ARRAY_BINDING,
        {
            if self.dirty { gl_with_temp_state!(
                ACTIVE_TEXTURE,
                TEXTURE_BINDING_RECTANGLE,
            {
                gl::ActiveTexture(self.texture_unit);
                gl::BindTexture(gl::TEXTURE_RECTANGLE, self.texture_object);
                gl::TexImage2D(
                    gl::TEXTURE_RECTANGLE, 0,
                    gl::RGBA as _, self.width as _, self.height as _, 0,
                    gl::RGBA, gl::UNSIGNED_SHORT_5_5_5_1, self.pixels.as_ptr() as *const _,
                );
                self.dirty = false;
            }); }

            gl::UseProgram(self.program);
            gl::BindVertexArray(self.vao);
            gl::DrawArrays(gl::TRIANGLE_FAN, 0, 4);
        }); }
    }

    pub fn pixels(&self) -> &[u16] {
        &self.pixels
    }

    pub fn pixels_mut(&mut self) -> &mut [u16] {
        self.set_dirty();
        &mut self.pixels
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    fn set_dirty(&mut self) {
        self.dirty = true;
        self.clear = false;
    }
}

pub unsafe fn copy_pixels_raw(
    src: *const u16, src_w: usize,
    dst: *mut   u16, dst_w: usize,
    img_w: usize, img_h: usize,
) {
    for i in 0..img_h {
        ptr::copy_nonoverlapping(
            src.add(i * src_w),
            dst.add(i * dst_w),
            img_w,
        );
    }
}

pub unsafe fn clear_pixels_raw(
    dst: *mut u16, dst_w: usize,
    img_w: usize, img_h: usize,
) {
    for i in 0..img_h {
        ptr::write_bytes(dst.add(i * dst_w), 0, img_w);
    }
}

pub struct Histogram {
    pub x: isize, pub y: isize, w: usize, h: usize,
    scale_level: usize, scale_counter: usize,
    pixels: Vec<u16>, col_index: usize,
    bar_segments: Vec<HistogramMark>,
    lines:        Vec<HistogramMark>,
}

struct HistogramMark {
    pub name:   &'static str,
    pub color:  u16,
    pub height: usize,
}

impl Histogram {
    pub fn new(x: isize, y: isize, w: usize, h: usize) -> Self {
        Histogram {
            x, y, w, h,
            scale_level: 1, scale_counter: w,
            pixels: vec![0; w * h], col_index: 0,
            bar_segments: Vec::new(),
            lines:        Vec::new(),
        }
    }

    pub fn add_bar_segment(&mut self, name: &'static str, color: u16, height: usize) {
        if name != "" && self.bar_segments.iter().find(|other| name == other.name).is_some() {
            panic!("duplicate histogram bar entry: {}", name);
        }
        self.bar_segments.push(HistogramMark { name, color, height });
    }

    pub fn add_line(&mut self, name: &'static str, color: u16, height: usize) {
        if name != "" && self.lines.iter().find(|other| name == other.name).is_some() {
            panic!("duplicate histogram line entry: {}", name);
        }
        self.lines.push(HistogramMark { name, color, height });
    }

    pub fn render_and_flush_buffer(&mut self, bar_width: usize, bar_space: usize) {
        // ensure sufficient scaling factor to keep marks within height
        let high_mark = self.lines.iter()
                            .map(|mark| mark.height)
                 .chain(self.bar_segments.iter()
                            .map(|mark| mark.height.saturating_sub(1)))
                 .max().unwrap_or(0);
        let scale_level = (high_mark / self.h + 1).next_power_of_two();

        if self.scale_counter == 0 || scale_level >= self.scale_level {
            self.scale_level   = scale_level;
            self.scale_counter = self.w;
        } else {
            self.scale_counter = self.scale_counter.saturating_sub(bar_width + bar_space);
        }

        // draw bar segments
        let mut base_mark = 0;
        for bar_segment in self.bar_segments.iter() {
            let mark = bar_segment.height / self.scale_level;
            for i_row in base_mark..mark {
                // fill bar width
                for mut i_col in 0..bar_width {
                    i_col += self.col_index;
                    i_col %= self.w;
                    self.pixels[i_row * self.w + i_col] = bar_segment.color;
                }
                // clear bar spacing
                for mut i_col in 0..bar_space {
                    i_col += self.col_index + bar_width;
                    i_col %= self.w;
                    self.pixels[i_row * self.w + i_col] = 0;
                }
            }
            base_mark = base_mark.max(mark);
        }
        // clear pixels above bar
        for i_row in base_mark..HISTOGRAM_HEIGHT {
            for mut i_col in 0..(bar_width + bar_space) {
                i_col += self.col_index;
                i_col %= self.w;
                self.pixels[i_row * self.w + i_col] = 0;
            }
        }
        // draw lines
        for line in self.lines.iter() {
            let mark = line.height / self.scale_level;
            for mut i_col in 0..(bar_width + bar_space) {
                i_col += self.col_index;
                i_col %= self.w;
                self.pixels[mark * self.w + i_col] = line.color;
            }
        }
        // advance column index
        self.col_index += bar_width + bar_space;
        self.col_index %= self.w;
        self.clear_buffer();
    }

    pub fn clear_buffer(&mut self) {
        self.bar_segments.clear();
        self.lines.clear();
    }

    pub fn is_buffer_empty(&self) -> bool {
        self.bar_segments.len() == 0 && self.lines.len() == 0
    }

    pub fn clear_pixels(&mut self) {
        unsafe { ptr::write_bytes(
            self.pixels.as_mut_ptr() as *mut u16,
            0, self.w * self.h,
        ); }
        self.scale_level   = 1;
        self.scale_counter = self.w;
    }

    pub fn copy_pixels_to_ui(&self, ui: &mut DebugUi) {
        ui.copy_pixels_cropped(
            self.x, self.y,
            self.col_index, 0,
            self.w - self.col_index, self.h,
            &self.pixels, self.w,
        );
        ui.copy_pixels_cropped(
            self.x + (self.w - self.col_index) as isize, self.y,
            0, 0,
            self.col_index, self.h,
            &self.pixels, self.w,
        );
    }

    pub fn clear_pixels_from_ui(&self, ui: &mut DebugUi) {
        ui.clear_pixels_cropped(self.x, self.y, self.w, self.h);
    }
}
