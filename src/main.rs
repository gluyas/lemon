#[macro_use]
extern crate lazy_static;
extern crate glutin;
extern crate cgmath;
extern crate png;
extern crate rand;
//extern crate image;

macro_rules! vec4 {
    () => {
        Vec4 { x:zero(), y:zero(), z:zero(), w:zero() }
    };
    ($xyzw:expr) => {
        vec4!($xyzw.x, $xyzw.y, $xyzw.z, $xyzw.w)
    };
    ($x:expr, $y:expr, $z:expr, $w:expr) => (
        Vec4 { x:$x, y:$y, z:$z, w:$w }
    );
}

macro_rules! vec3 {
    () => {
        Vec3 { x:zero(), y:zero(), z:zero() }
    };
    ($xyz:expr) => {
        vec3!($xyz.x, $xyz.y, $xyz.z)
    };
    ($x:expr, $y:expr, $z:expr) => (
        Vec3 { x:$x, y:$y, z:$z }
    );
}

macro_rules! vec2 {
    () => {
        Vec2 { x:zero(), y:zero() }
    };
    ($xy:expr) => {
        vec2!($xy.x, $xy.y)
    };
    ($x:expr, $y:expr) => (
        Vec2 { x:$x, y:$y }
    );
}

macro_rules! point3 {
    () => {
        Point3 { x:zero(), y:zero(), z:zero() }
    };
    ($xyz:expr) => {
        point3!($xyz.x, $xyz.y, $xyz.z)
    };
    ($x:expr, $y:expr, $z:expr) => (
        Point3 { x:$x, y:$y, z:$z }
    );
}

macro_rules! point2 {
    () => {
        Point2 { x:zero(), y:zero() }
    };
    ($xy:expr) => {
        point2!($xy.x, $xy.y)
    };
    ($x:expr, $y:expr) => (
        Point2 { x:$x, y:$y }
    );
}

macro_rules! mat4 {
    () => (
        mat4!(1.0)
    );
    ($n:expr) => (
        Mat4 {
            x: vec4!($n, 0.0, 0.0, 0.0),
            y: vec4!(0.0, $n, 0.0, 0.0),
            z: vec4!(0.0, 0.0, $n, 0.0),
            w: vec4!(0.0, 0.0, 0.0, $n),
        }
    );
}

macro_rules! mat3 {
    () => (
        mat3!(1.0)
    );
    ($n:expr) => (
        Mat3 {
            x: vec3!($n, 0.0, 0.0),
            y: vec3!(0.0, $n, 0.0),
            z: vec3!(0.0, 0.0, $n),
        }
    );
}

macro_rules! cstr {
    ($s:expr) => (
        concat!($s, "\0") as *const str as *const [c_char] as *const c_char
    );
}

macro_rules! color {
    ($hex:expr) => (
        Vec4 {
            x: (($hex as u32 >> 0x18) & 0xFF) as f32 / 255.0,
            y: (($hex as u32 >> 0x10) & 0xFF) as f32 / 255.0,
            z: (($hex as u32 >> 0x08) & 0xFF) as f32 / 255.0,
            w: (($hex as u32 >> 0x00) & 0xFF) as f32 / 255.0,
        }
    );
}

mod gl;
use crate::gl::types::*;

mod jagged;
use crate::jagged::Jagged;

mod lemon;
use crate::lemon::{Lemon, NormalizedLemon};

mod debug_render;
use crate::debug_render::DebugRender;

use glutin::*;

use rand::random;

use cgmath::{*, num_traits::{zero, one}};
type Vec4   = cgmath::Vector4<f32>;
    const VEC4_X: Vec4 = vec4!(1.0, 0.0, 0.0, 0.0);
    const VEC4_Y: Vec4 = vec4!(0.0, 1.0, 0.0, 0.0);
    const VEC4_Z: Vec4 = vec4!(0.0, 0.0, 1.0, 0.0);
    const VEC4_W: Vec4 = vec4!(0.0, 0.0, 0.0, 1.0);
type Vec3   = cgmath::Vector3<f32>;
    const VEC3_X: Vec3 = vec3!(1.0, 0.0, 0.0);
    const VEC3_Y: Vec3 = vec3!(0.0, 1.0, 0.0);
    const VEC3_Z: Vec3 = vec3!(0.0, 0.0, 1.0);
type Vec2   = cgmath::Vector2<f32>;
    const VEC2_X: Vec2 = vec2!(1.0, 0.0);
    const VEC2_Y: Vec2 = vec2!(0.0, 1.0);
type Point3 = cgmath::Point3<f32>;
type Point2 = cgmath::Point2<f32>;
type Mat4   = cgmath::Matrix4<f32>;
type Mat3   = cgmath::Matrix3<f32>;
type Quat   = cgmath::Quaternion<f32>;

use std::{
    default::Default,
    f32::{self, consts::PI},
    mem,
    ops::*,
    os::raw::c_char,
    ptr,
    slice,
    time::{Duration, Instant},
    thread,
};

pub const TAU: f32 = PI * 2.0;

type ElementIndex = u16;
const ELEMENT_INDEX_TYPE: GLenum = gl::UNSIGNED_SHORT;

#[derive(Copy, Clone, Debug)]
pub struct Collision {
    /// Point of collision. If overlapping, the midpoint between the two objects.
    pub point:  Point3,

    /// Surface normal of collision. Points towards the first object.
    pub normal: Vec3,

    /// Minimum displacment along `normal` required to separate the objects.
    /// The objects' surfaces will kiss at `point` when translated by half `depth`
    /// along `normal` and `-normal` respectively.
    pub depth:  f32,
}

impl Neg for Collision {
    type Output = Collision;
    fn neg(mut self) -> Collision {
        self.normal = self.normal.neg();
        self
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Rigidbody {
    pub mass:     f32,
    pub position: Point3,
    pub velocity: Vec3,

    pub inertia_local:    Vec3,
    pub orientation:      Quat,
    pub angular_momentum: Vec3,
}

impl Rigidbody {
    pub fn get_transform(&self) -> Mat4 {
        Mat4::from_translation(vec3!(self.position))
            * Mat4::from(Mat3::from(self.orientation))
    }

    pub fn get_transform_inverse(&self) -> Mat4 {
        Mat4::from(Mat3::from(self.orientation).transpose())
            * Mat4::from_translation(-vec3!(self.position))
    }

    pub fn get_inertia(&self) -> Mat3 {
        let rotation = Mat3::from(self.orientation);
        rotation * Mat3::from_diagonal(self.inertia_local) * rotation.transpose()
    }

    pub fn get_inertia_inverse(&self) -> Mat3 {
        let rotation = Mat3::from(self.orientation);
        rotation * Mat3::from_diagonal(1.0 / self.inertia_local) * rotation.transpose()
    }
}

impl Default for Rigidbody {
    fn default() -> Self {
        Rigidbody {
            mass: 0.0,
            position: point3!(),
            velocity: vec3!(),
            inertia_local: vec3!(),
            orientation: Quat::one(),
            angular_momentum: vec3!(),
        }
    }
}

const LEMON_COLOR: Vec4 = color!(0xFFF44F_FF);
const BACK_COLOR:  Vec4 = color!(0xA2EFEF_00);

const LEMON_TEX_SIZE: usize = 1;

const FRAME_RATE:       usize    = 60;
const FRAME_DELTA_TIME: f32      = 1.0 / FRAME_RATE as f32;
const FRAME_DURATION:   Duration = Duration::from_nanos((1.0e+9 / FRAME_RATE as f64) as u64);

/// Constants for converting from human-readable SI units into per-frame units
mod si {
    use super::*;

    pub const METER:    f32 = 1.0;
    pub const SECOND:   f32 = FRAME_RATE as f32;
    pub const KILOGRAM: f32 = 1.0;

    pub const NEWTON:   f32 = KILOGRAM * METER / SECOND / SECOND;
    pub const PASCAL:   f32 = NEWTON / (METER*METER);
    pub const JOULE:    f32 = NEWTON * METER;
}
use crate::si::*;

const MAX_BODIES: usize = 256;

// TODO: 9.81 gravity and scale lemons to a resonable size?
const PHYS_GRAVITY: Vec3 = vec3!(0.0, 0.0, -23.0 * METER / SECOND / SECOND);

const LEMON_COLLISION_ELASTICITY: f32 = 0.15;

const LEMON_FRICTION:     f32 = 1.55 * NEWTON / NEWTON;
const LEMON_ANGULAR_DRAG: f32 = 1.0 * NEWTON * METER;

const LEMON_SCALE_MAX:    f32 = 1.25;
const LEMON_SCALE_MIN:    f32 = 0.75;
const LEMON_S_MIN:        f32 = 0.50;
const LEMON_S_MAX:        f32 = 0.75;
const LEMON_PARTY_S_MIN:  f32 = 0.30;
const LEMON_PARTY_S_MAX:  f32 = 0.95;

const WIDTH:  usize = 1280;
const HEIGHT: usize = 720;

fn main() {
    let mut events_loop = EventsLoop::new();
    let window = WindowBuilder::new()
        .with_title("lemon")
        .with_dimensions(dpi::LogicalSize::new(WIDTH as _, HEIGHT as _))
        .with_resizable(false);
    let context = ContextBuilder::new()
        .with_multisampling(2);
    let gl_window = GlWindow::new(window, context, &events_loop).unwrap();

    unsafe { gl_window.make_current().unwrap(); }
    gl::load_with(|symbol| gl_window.get_proc_address(symbol) as *const _);

    #[repr(C)]
    struct Camera {
        view:       Mat4,
        projection: Mat4,

        position: Point3,
    };

    let (mut camera, camera_ubo, camera_binding_index) = unsafe {
        let mut camera = Camera {
            view:       mat4!(1.0),
            projection: mat4!(1.0),

            position: point3!(),
        };

        let camera_ubo = gl::gen_object(gl::GenBuffers);
        gl::BindBuffer(gl::UNIFORM_BUFFER, camera_ubo);
        gl::buffer_init::<Camera>(gl::UNIFORM_BUFFER, 1, gl::DYNAMIC_DRAW);

        let camera_binding_index: GLuint = 1;
        gl::BindBufferBase(gl::UNIFORM_BUFFER, camera_binding_index, camera_ubo);

        (camera, camera_ubo, camera_binding_index)
    };

    let (vao, base_mesh, vbo_transform, vbo_lemon_s) = unsafe {
        let program = gl::link_shaders(&[
            gl::compile_shader(include_str!("shader/lemon.vert.glsl"), gl::VERTEX_SHADER),
            gl::compile_shader(include_str!("shader/lemon.frag.glsl"), gl::FRAGMENT_SHADER),
        ]);
        gl::UseProgram(program);

        let camera_index = gl::GetUniformBlockIndex(program, cstr!("Camera"));
        gl::UniformBlockBinding(program, camera_index, camera_binding_index);

        let u_lemon_color = gl::GetUniformLocation(program, cstr!("u_lemon_color"));
        gl::Uniform4fv(u_lemon_color, 1, as_ptr(&LEMON_COLOR));

        let u_ambient_color = gl::GetUniformLocation(program, cstr!("u_ambient_color"));
        gl::Uniform4fv(u_ambient_color, 1, as_ptr(&BACK_COLOR));
/*
        let txo_normal_map = gl::gen_object(gl::GenTextures);
        let normal_map = vec![vec3!(); 1024];
        gl::ActiveTexture(gl::TEXTURE0);
        gl::BindTexture(gl::TEXTURE_2D, txo_normal_map);

        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR_MIPMAP_LINEAR as _);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as _);

        let mut max_anisotropy = 0.0;
        gl::GetFloatv(gl::MAX_TEXTURE_MAX_ANISOTROPY_EXT, &mut max_anisotropy);
        gl::TexParameterf(gl::TEXTURE_2D, gl::TEXTURE_MAX_ANISOTROPY_EXT, max_anisotropy);

        gl::TexImage2D(
            gl::TEXTURE_2D, 0,
            gl::RGB8_SNORM as _, LEMON_TEX_SIZE as GLsizei, LEMON_TEX_SIZE as GLsizei, 0,
            gl::RGB, gl::FLOAT, normal_map.as_ptr() as *const GLvoid,
        );
        gl::GenerateMipmap(gl::TEXTURE_2D);
        let u_normal_map = gl::GetUniformLocation(program, cstr!("u_normal_map"));
        gl::Uniform1i(u_normal_map, 0);
*/
        let txo_radius_normal_z_atlas = gl::gen_object(gl::GenTextures);
        gl::ActiveTexture(gl::TEXTURE0);
        gl::BindTexture(gl::TEXTURE_2D, txo_radius_normal_z_atlas);

        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as _);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as _);

        let radius_normal_z_map = lemon::make_radius_normal_z_map();
        gl::TexImage2D(
            gl::TEXTURE_2D, 0,
            gl::RG as _, lemon::MAP_RESOLUTION as _, lemon::MAP_RESOLUTION as _, 0,
            gl::RG, gl::FLOAT, radius_normal_z_map.as_ptr() as *const GLvoid,
        );
        gl::GenerateMipmap(gl::TEXTURE_2D);
        let u_radius_normal_z_map = gl::GetUniformLocation(program,
            cstr!("u_radius_normal_z_map")
        );
        gl::Uniform1i(u_radius_normal_z_map, 0);

        let vao = gl::gen_object(gl::GenVertexArrays);
        gl::BindVertexArray(vao);

        let base_mesh = lemon::make_base_mesh();

        // PER-INSTANCE ATTRIBUTES
        let a_transform    = gl::GetAttribLocation(program, cstr!("a_transform")) as GLuint;
        let vbo_transform = gl::gen_object(gl::GenBuffers);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo_transform);
        gl::buffer_init::<Mat4>(gl::ARRAY_BUFFER, MAX_BODIES, gl::STREAM_DRAW);
        for i in 0..4 { // all 4 column vectors
            let a_transform_i = a_transform + i as GLuint;
            gl::EnableVertexAttribArray(a_transform_i);
            gl::VertexAttribPointer(
                a_transform_i , 4, gl::FLOAT, gl::FALSE,
                mem::size_of::<Mat4>() as GLsizei,
                ptr::null::<f32>().offset(4 * i) as *const GLvoid,
            );
            gl::VertexAttribDivisor(a_transform_i, 1);
        }

        let a_lemon_s   = gl::GetAttribLocation(program, cstr!("a_lemon_s")) as GLuint;
        let vbo_lemon_s = gl::gen_object(gl::GenBuffers);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo_lemon_s);
        gl::buffer_init::<f32>(gl::ARRAY_BUFFER, MAX_BODIES, gl::DYNAMIC_DRAW);
        gl::EnableVertexAttribArray(a_lemon_s);
        gl::VertexAttribPointer(a_lemon_s, 1, gl::FLOAT, gl::FALSE, 0, 0 as *const GLvoid);
        gl::VertexAttribDivisor(a_lemon_s, 1);

        // PER-VERTEX ATTRIBUTES
        let a_position   = gl::GetAttribLocation(program, cstr!("a_position")) as GLuint;
        let vbo_position = gl::gen_object(gl::GenBuffers);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo_position);
        gl::buffer_data(gl::ARRAY_BUFFER, &base_mesh.points, gl::STATIC_DRAW);
        gl::EnableVertexAttribArray(a_position);
        gl::VertexAttribPointer(a_position, 3, gl::FLOAT, gl::FALSE, 0, 0 as *const GLvoid);

        let ebo = gl::gen_object(gl::GenBuffers);
        gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
        gl::buffer_data(gl::ELEMENT_ARRAY_BUFFER, &base_mesh.indices, gl::STATIC_DRAW);

        (vao, base_mesh, vbo_transform, vbo_lemon_s)
    };

    let mut lemons = Vec::with_capacity(MAX_BODIES);
    fn spawn_lemon(lemons: &mut Vec<Lemon>, vbo_lemon_s: GLuint) {
        if lemons.len() >= MAX_BODIES { return; }

        let scale = LEMON_SCALE_MIN + (LEMON_SCALE_MAX-LEMON_SCALE_MIN) * random::<f32>();
        let s     = LEMON_S_MIN + (LEMON_S_MAX-LEMON_S_MIN) * random::<f32>();
        let mut new_lemon = Lemon::new(s, scale);
        unsafe {
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo_lemon_s);
            gl::buffer_sub_data(
                gl::ARRAY_BUFFER,
                lemons.len(),
                slice::from_ref(&s),
            );
        }

        reset_lemon(&mut new_lemon);
        lemons.push(new_lemon);
    }
    fn reset_lemon(lemon: &mut Lemon) {
        lemon.phys.position         = point3!(0.0, 0.0, (2.0+3.0*random::<f32>())*METER);
        lemon.phys.orientation      = Quat::from_axis_angle(
                                      random::<Vec3>().normalize(),
                                      Deg(30.0 * (random::<f32>() - 0.5)));
        lemon.phys.velocity         = vec3!() * METER / SECOND;
        lemon.phys.angular_momentum = lemon.phys.get_inertia()
                                    * (random::<Vec3>() - vec3!(0.5, 0.5, 0.5)) * 2.0
                                    * TAU / 5.0 / SECOND;
    }
    macro_rules! current_lemon       { () => { lemons.last_mut().unwrap() } }
    macro_rules! current_lemon_index { () => { lemons.len() - 1 } }
    spawn_lemon(&mut lemons, vbo_lemon_s);

    let mut debug            = DebugRender::new();
    let mut debug_depth_test = DebugRender::with_shared_context(&debug);
    {
        let grid = make_line_strip_grid(point3!(), (VEC3_X*1.5, VEC3_Y*1.5), 20);
        debug_depth_test.draw_line(&color!(0xFFFFFFFF).truncate(), !0, &grid);

        debug.draw_axes(1.5, !0, &Mat4::identity());
    }
    let mut debug_draw_axes               = false;
    let mut debug_draw_torus_section      = false;
    let mut debug_draw_colliders_floor    = false;
    let mut debug_draw_colliders_lemon    = false;
    let mut debug_draw_bounding_volumes   = false;
    let mut debug_draw_motion_vectors     = false;
    let mut debug_draw_collision_response = false;
    let mut debug_lemon_party             = false;

    let mut debug_frame_store             = Jagged::new();
    let mut debug_frame_current           = 0;
    macro_rules! debug_frame_store_reset { () => {
        debug_frame_store.clear();
        debug_frame_current = 0;
        debug_frame_store.push_copy(&lemons);
    }; }
    debug_frame_store.push_copy(&lemons);

    let mut debug_pause                   = false;
    let mut debug_pause_next_frame        = false;

    let mut camera_fovy: f32 = 30.0_f32.to_radians();
    let mut camera_distance: f32 = 9.0_f32;
    let mut camera_elevation: f32 = 0.0_f32.to_radians();
    let mut camera_azimuth: f32 = 0.0_f32.to_radians();

    let mut mouse_pos: Vec2 = vec2!();
    let mut mouse_drag: Option<Vec2> = Some(vec2!());
    let mut mouse_down = false;

    let mut exit = false;
    while !exit {
        let frame_start_time = Instant::now();
        // POLL INPUT
        events_loop.poll_events(|event| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => { exit = true; },
                WindowEvent::MouseInput { button: MouseButton::Left, state, .. } => match state {
                    ElementState::Pressed  => { mouse_down = true; },
                    ElementState::Released => { mouse_down = false; },
                },
                WindowEvent::CursorMoved { position: pos, .. } => {
                    let mouse_new_pos = vec2!(pos.x as f32, pos.y as f32);
                    if mouse_down {
                        mouse_drag = mouse_drag
                            .or(Some(vec2!()))
                            .map(|movement| movement + (mouse_new_pos - mouse_pos));
                    }
                    mouse_pos = mouse_new_pos;
                },
                WindowEvent::MouseWheel {
                    delta: MouseScrollDelta::LineDelta(_delta_x, delta_y), modifiers, ..
                } => {
                    let new_scale = option_if_then(modifiers.ctrl, || {
                        let new_scale = current_lemon!().scale + delta_y * 0.05;
                        some_if(new_scale >= 0.35 && new_scale <= 2.0, new_scale)
                    });
                    let new_s     = option_if_then(modifiers.alt, || {
                        let normalized = current_lemon!().get_normalized();
                        let new_s      = normalized.s + delta_y * 0.01;
                        if new_s >= 0.15 && new_s <= 0.95 {
                            unsafe {
                                gl::BindBuffer(gl::ARRAY_BUFFER, vbo_lemon_s);
                                gl::buffer_sub_data(gl::ARRAY_BUFFER,
                                    current_lemon_index!(), slice::from_ref(&normalized.s)
                                );
                            }
                            debug_frame_store_reset!();
                            Some(new_s)
                        } else { None }
                    });
                    if new_scale.is_some() || new_s.is_some() {
                        let mut lemon = current_lemon!();
                        let new_scale = new_scale.unwrap_or(lemon.scale);
                        let new_s     = new_s.unwrap_or_else(|| lemon.get_normalized().s);
                        lemon.mutate_shape(new_s, new_scale);
                    } else if !(modifiers.ctrl || modifiers.alt) { // regular camera zoom
                        camera_distance -= camera_distance * delta_y * 0.065;
                        if camera_distance < 0.0 { camera_distance = 0.0; }
                        mouse_drag = mouse_drag.or(Some(vec2!())); // HACK: redraw on zoom
                    }
                },
                _ => (),
            },
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::Key(KeyboardInput {
                    state, modifiers, virtual_keycode: Some(key), ..
                }) => match key {
                    VirtualKeyCode::P => if let ElementState::Pressed = state {
                        capture_image_to_file();
                    },
                    VirtualKeyCode::A => if let ElementState::Pressed = state {
                        debug_draw_axes = !debug_draw_axes;
                    },
                    VirtualKeyCode::S => if let ElementState::Pressed = state {
                        debug_draw_torus_section = !debug_draw_torus_section;
                    },
                    VirtualKeyCode::Z => if let ElementState::Pressed = state {
                        debug_draw_colliders_lemon = !debug_draw_colliders_lemon;
                    },
                    VirtualKeyCode::X => if let ElementState::Pressed = state {
                        debug_draw_colliders_floor = !debug_draw_colliders_floor;
                    },
                    VirtualKeyCode::C => if let ElementState::Pressed = state {
                        debug_draw_collision_response = !debug_draw_collision_response;
                    },
                    VirtualKeyCode::V => if let ElementState::Pressed = state {
                        debug_draw_motion_vectors = !debug_draw_motion_vectors;
                    },
                    VirtualKeyCode::B => if let ElementState::Pressed = state {
                        debug_draw_bounding_volumes = !debug_draw_bounding_volumes;
                    },
                    VirtualKeyCode::L => if let ElementState::Pressed = state {
                        debug_lemon_party = !debug_lemon_party;
                    },
                    VirtualKeyCode::Escape => if let ElementState::Pressed = state {
                        debug_pause_next_frame = !debug_pause;
                    },
                    VirtualKeyCode::Left => if let ElementState::Pressed = state {
                        if debug_pause && debug_frame_current > 0 {
                            debug_frame_current -= 1;
                            lemons.clear();
                            lemons.extend_from_slice(&debug_frame_store[debug_frame_current]);
                        }
                    },
                    VirtualKeyCode::Right => if let ElementState::Pressed = state {
                        if debug_pause && debug_frame_current + 1 < debug_frame_store.len() {
                            debug_frame_current += 1;
                            lemons.clear();
                            lemons.extend_from_slice(&debug_frame_store[debug_frame_current]);
                        }
                    },
                    VirtualKeyCode::Space => if let ElementState::Pressed = state {
                        if modifiers.ctrl { spawn_lemon(&mut lemons, vbo_lemon_s); }
                        else              { reset_lemon(&mut current_lemon!()); }
                    },
                    _ => (),
                },
                _ => (),
            },
            _ => (),
        });
        // UPDATE PAUSE STATE
        if debug_pause_next_frame != debug_pause {
            if debug_pause {
                debug_frame_store.truncate(debug_frame_current + 1);
            } else {
                debug_frame_current = debug_frame_store.len() - 1;
            }
            debug_pause = debug_pause_next_frame;
        }

        // UPDATE PHYSICS
        for (index, lemon) in lemons.iter_mut().enumerate() {
            if debug_lemon_party && !debug_pause { // this is hacky as shit but important!
                use mem::transmute as tm;

                const LSB_MASK:   u32 = 0x1;
                const PARTY_RATE: f32 = 0.5 / SECOND;

                let lsb = unsafe { tm::<_,u32>(lemon.sagitta) & LSB_MASK > 0 };
                let mut normalized = lemon.get_normalized();
                normalized = if lsb {
                    NormalizedLemon::new(normalized.s + PARTY_RATE * lemon.scale)
                } else {
                    NormalizedLemon::new(normalized.s - PARTY_RATE * lemon.scale)
                };
                let next_lsb = {
                    if      normalized.s >= LEMON_PARTY_S_MAX { false }
                    else if normalized.s <= LEMON_PARTY_S_MIN { true  }
                    else                                      { lsb   }
                };
                lemon.mutate_shape(normalized.s, lemon.scale);
                lemon.sagitta = unsafe { // set the LSB for next frame
                    if next_lsb { tm::<_,f32>(tm::<_,u32>(lemon.sagitta) | LSB_MASK) }
                    else        { tm::<_,f32>(tm::<_,u32>(lemon.sagitta) &!LSB_MASK) }
                };
                unsafe {
                    gl::BindBuffer(gl::ARRAY_BUFFER, vbo_lemon_s);
                    gl::buffer_sub_data(gl::ARRAY_BUFFER,
                        index, slice::from_ref(&normalized.s)
                    );
                }
            }

            if !debug_pause {   // INTEGRATE RIGIDBODIES
                lemon.phys.position += lemon.phys.velocity;
                lemon.phys.velocity += PHYS_GRAVITY;

                let inertia_inverse  = lemon.phys.get_inertia_inverse();
                let angular_velocity = Quat {
                    v: inertia_inverse * lemon.phys.angular_momentum,
                    s: 0.0,
                };
                lemon.phys.orientation += angular_velocity / 2.0 * lemon.phys.orientation;
                lemon.phys.orientation  = lemon.phys.orientation.normalize();

                let angular_drag = {
                    let mag  = lemon.phys.angular_momentum.magnitude();
                    if LEMON_ANGULAR_DRAG > mag {
                        lemon.phys.angular_momentum
                    } else {
                        lemon.phys.angular_momentum / mag * LEMON_ANGULAR_DRAG
                    }
                };
                lemon.phys.angular_momentum -= angular_drag;
            }

            if debug_draw_motion_vectors {
                debug.draw_ray(
                    &color!(0xFF50FFFF).truncate(), 1, &lemon.phys.position,
                    &(lemon.phys.velocity / FRAME_DELTA_TIME),
                );
                debug.draw_ray(
                    &color!(0xFFFF50FF).truncate(), 1, &lemon.phys.position,
                    & ( lemon.phys.get_inertia_inverse()
                      * lemon.phys.angular_momentum
                      / FRAME_DELTA_TIME ),
                );
            }

            let floor_collision = lemon::get_collision_halfspace(
                &lemon, (VEC3_Z, 0.0),
                some_if(debug_draw_colliders_floor, &mut debug),
            );

            // COLLISION RESPONSE
            if let Some(mut collision) = floor_collision {
                // push object out of plane // TODO: solve for velocity, orientation
                if !debug_pause {
                    lemon.phys.position += collision.normal * collision.depth;
                    collision.point     += collision.normal * collision.depth / 2.0;
                }

                let inertia_inverse  = lemon.phys.get_inertia_inverse();
                let angular_velocity = inertia_inverse * lemon.phys.angular_momentum;

                let offset   = collision.point - lemon.phys.position;
                let velocity = lemon.phys.velocity + angular_velocity.cross(offset);

                let collision_tangent = proj_onto_plane(velocity.normalize(), collision.normal);

                let reaction = {
                    let proj    = -(1.0 + LEMON_COLLISION_ELASTICITY)
                                * velocity.dot(collision.normal);
                    let linear  = 1.0 / lemon.phys.mass;
                             // + 1.0 / infinity  => 0.0
                    let angular = ( inertia_inverse
                                  * offset.cross(collision.normal)
                                  ).cross(offset);
                             // + [infinity]^-1 => 0.0
                    proj / (linear + angular.dot(collision.normal))
                };
                let reaction_vector = reaction * collision.normal;

                let friction = {
                    let proj    = velocity.dot(collision_tangent);
                    let linear  = 1.0 / lemon.phys.mass;
                    let angular = ( inertia_inverse
                                  * offset.cross(collision_tangent)
                                  ).cross(offset);

                    let cap     = proj / (linear + angular.dot(collision_tangent));
                    cap.min(reaction * LEMON_FRICTION).neg()
                };
                let friction_vector = friction * collision_tangent;

                if !debug_pause {
                    let impulse = reaction_vector + friction_vector;

                    lemon.phys.velocity         += impulse / lemon.phys.mass;
                    lemon.phys.angular_momentum += offset.cross(impulse);
                }

                if debug_draw_collision_response {
                    debug.draw_ray(
                        &color!(0x0000AFFF).truncate(), 1, &collision.point,
                        &(friction_vector / lemon.phys.mass / FRAME_DELTA_TIME)
                    );
                    debug.draw_ray(
                        &color!(0xAF0000FF).truncate(), 1, &collision.point,
                        &(reaction_vector / lemon.phys.mass / FRAME_DELTA_TIME)
                    );
                    debug.draw_ray(
                        &color!(0x00AF00FF).truncate(), 1, &collision.point,
                        &(velocity / FRAME_DELTA_TIME)
                    );
                }
            }

            if debug_draw_bounding_volumes {
                debug_depth_test.draw_line(&color!(0x708090FF).truncate(), 1,
                    &make_line_strip_capsule_billboard(
                        &lemon.get_bounding_capsule(), camera.position-lemon.phys.position, 5,
                    )
                );
            }
            if debug_draw_axes {
                debug.draw_axes(0.5, 1, &lemon.phys.get_transform());
            }
            if debug_draw_torus_section {
                let color = color!(0x708090FF).truncate();

                let lemon_vertical = lemon.get_vertical();
                let to_camera = camera.position - lemon.phys.position;
                let cross     = to_camera.cross(lemon_vertical);
                if cross != vec3!(0.0, 0.0, 0.0) {
                    let radius = cross.normalize_to(lemon.focal_radius());
                    let normal = lemon_vertical.cross(radius).normalize();
                    debug.draw_line(&color, 1, &make_line_strip_circle(
                        lemon.phys.position + radius, normal, lemon.radius, 31,
                    ));
                    debug.draw_line(&color, 1, &make_line_strip_circle(
                        lemon.phys.position - radius, normal, lemon.radius, 31,
                    ));
                }
                debug.draw_line(&color, 1, &make_line_strip_circle(
                    lemon.phys.position, lemon_vertical, lemon.focal_radius(), 31,
                ));
            }

            unsafe {
                gl::BindBuffer(gl::ARRAY_BUFFER, vbo_transform);
                gl::buffer_sub_data(
                    gl::ARRAY_BUFFER,
                    index,
                    slice::from_ref(&lemon.get_transform_with_scale()),
                );
            }
        }

        if debug_lemon_party {
            debug_frame_store_reset!();
        } else if !debug_pause {
            debug_frame_store.push_copy(&lemons);
        }

        // TEST LEMON COLLISIONS
        let lemons_len = lemons.len();
        for lemon_index in 0..lemons_len {
            let (lemon, other_lemons) = {
                let (heads, tails) = lemons.split_at_mut(lemon_index + 1);
                (&mut heads[lemon_index], tails)
            };
            for (other_relative_index, other) in other_lemons.iter_mut().enumerate() {
                let other_index = lemon_index + other_relative_index + 1;

                let bounding_volume_overlap = overlap_capsules(
                    &lemon.get_bounding_capsule(),
                    &other.get_bounding_capsule(),
                );
                let force_collision_test = debug_draw_colliders_lemon
                                        &&!debug_draw_bounding_volumes
                                        && ( lemons_len <= 4
                                          || ( lemon.phys.position-other.phys.position
                                             ).magnitude2() <= 9.0 );
                let collision = {
                    if bounding_volume_overlap || force_collision_test {
                        lemon::get_collision_lemon(
                            &lemon, &other,
                            some_if(debug_draw_colliders_lemon, &mut debug),
                        )
                    } else {
                        None
                    }
                };

                if let Some(collision) = collision {
                    assert!(bounding_volume_overlap, "lemon collision without BV overlap");
                    assert!(collision.depth >= 0.0, "collision reported negative depth");

                    if !debug_pause { // naively correct interpenetration
                        lemon.phys.position += collision.normal * collision.depth / 2.0;
                        other.phys.position -= collision.normal * collision.depth / 2.0;
                    }

                    let lemon_inertia_inverse = lemon.phys.get_inertia_inverse();
                    let lemon_point_offset    = collision.point - lemon.phys.position;
                    let lemon_point_velocity  = ( lemon_inertia_inverse
                                                * lemon.phys.angular_momentum
                                                ).cross(lemon_point_offset)
                                              + lemon.phys.velocity;

                    let other_inertia_inverse = other.phys.get_inertia_inverse();
                    let other_point_offset    = collision.point - other.phys.position;
                    let other_point_velocity  = ( other_inertia_inverse
                                                * other.phys.angular_momentum
                                                ).cross(other_point_offset)
                                              + other.phys.velocity;

                    let relative_velocity = lemon_point_velocity - other_point_velocity;
                    let collision_tangent = proj_onto_plane(relative_velocity, collision.normal)
                                           .normalize();
                    let reaction = {
                        let proj    = -(1.0 + LEMON_COLLISION_ELASTICITY) * collision.normal
                                     .dot(relative_velocity);
                        let linear  = 1.0 / lemon.phys.mass
                                    + 1.0 / other.phys.mass;
                        let angular = ( lemon_inertia_inverse
                                      * lemon_point_offset.cross(collision.normal)
                                      ).cross(lemon_point_offset)
                                    + ( other_inertia_inverse
                                      * other_point_offset.cross(collision.normal)
                                      ).cross(other_point_offset);

                        proj / (linear + angular.dot(collision.normal))
                    };
                    let reaction_vector = reaction * collision.normal;

                    let friction = {
                        let proj    = relative_velocity.dot(collision_tangent);
                        let linear  = 1.0 / lemon.phys.mass
                                    + 1.0 / other.phys.mass;
                        let angular = ( lemon_inertia_inverse
                                      * lemon_point_offset.cross(collision_tangent)
                                      ).cross(lemon_point_offset)
                                    + ( other_inertia_inverse
                                      * other_point_offset.cross(collision_tangent)
                                      ).cross(other_point_offset);

                        let cap     = proj / (linear + angular.dot(collision_tangent));
                        cap.min(reaction * LEMON_FRICTION).neg()
                    };
                    let friction_vector = friction * collision_tangent;

                    if debug_draw_collision_response {
                        debug.draw_ray(&color!(0x0000AFFF).truncate(), 1,
                            &collision.point,
                            &(friction_vector / lemon.phys.mass / FRAME_DELTA_TIME)
                        );
                        debug.draw_ray(&color!(0xAF0000FF).truncate(), 1,
                            &collision.point,
                            &(reaction_vector / lemon.phys.mass / FRAME_DELTA_TIME),
                        );
                        debug.draw_ray(&color!(0x00AF00FF).truncate(), 1,
                            &collision.point,
                            &(relative_velocity / FRAME_DELTA_TIME),
                        );
                    }

                    if !debug_pause {
                        let impulse = reaction_vector + friction_vector;

                        lemon.phys.velocity         += impulse / lemon.phys.mass;
                        lemon.phys.angular_momentum += lemon_point_offset.cross(impulse);
                        other.phys.velocity         -= impulse / other.phys.mass;
                        other.phys.angular_momentum -= other_point_offset.cross(impulse);

                        unsafe {
                            gl::BindBuffer(gl::ARRAY_BUFFER, vbo_transform);
                            gl::buffer_sub_data(
                                gl::ARRAY_BUFFER,
                                lemon_index,
                                slice::from_ref(&lemon.get_transform_with_scale()),
                            );
                            gl::buffer_sub_data(
                                gl::ARRAY_BUFFER,
                                other_index,
                                slice::from_ref(&other.get_transform_with_scale()),
                            );
                        }
                    }
                }
            }
        }

        // UPDATE CAMERA
        if let Some(movement) = mouse_drag.or(Some(vec2!())) {
            let lemon = current_lemon!();
            camera_azimuth -= (0.5 * movement.x).to_radians();

            camera_elevation += (0.5 * movement.y).to_radians();
            camera_elevation = camera_elevation.min(85.0_f32.to_radians());
            camera_elevation = camera_elevation.max(-85.0_f32.to_radians());

            let camera_dir =
                ( Quat::from_angle_z(Rad(camera_azimuth))
                * Quat::from_angle_x(Rad(-camera_elevation))
                ).rotate_vector(vec3!(0.0, 1.0, 0.0));

            let camera_focus  = point3!(
                vec3!(lemon.phys.position).mul_element_wise(vec3!(1.0, 1.0, 0.5)) + 0.5*VEC3_Z
            );
            camera.position   = camera_focus - camera_dir * camera_distance;
            camera.projection = perspective(Rad(camera_fovy), WIDTH as f32 / HEIGHT as f32, 0.1, 1000.0);
            camera.view = Mat4::look_at(
                camera.position,
                camera_focus + camera_dir,
                VEC3_Z,
            );

            debug.update_camera(&(camera.projection * camera.view));

            unsafe {
                gl::BindBuffer(gl::UNIFORM_BUFFER, camera_ubo);
                gl::buffer_data(gl::UNIFORM_BUFFER, slice::from_ref(&camera), gl::DYNAMIC_DRAW);
            }
            mouse_drag = None;
        }

        // FULL RENDER
        unsafe {
            gl::ClearColor(BACK_COLOR.x, BACK_COLOR.y, BACK_COLOR.z, BACK_COLOR.w);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            gl::Enable(gl::CULL_FACE);
            gl::Enable(gl::DEPTH_TEST);
            //gl::PolygonMode(gl::FRONT_AND_BACK, gl::LINE);
            gl::DepthFunc(gl::LESS);
            gl::DrawElementsInstanced(
                gl::TRIANGLES,
                base_mesh.indices.len() as GLsizei,
                ELEMENT_INDEX_TYPE,
                ptr::null(),
                lemons.len() as GLsizei,
            );
            debug_depth_test.render_frame();
            gl::DepthFunc(gl::ALWAYS);
            debug.render_frame();
        }
        gl_window.swap_buffers().expect("buffer swap failed");
        if let Some(sleep_duration) = FRAME_DURATION.checked_sub(frame_start_time.elapsed()) {
            thread::sleep(sleep_duration);
        }
    }
}

fn furthest_on_circle_from_point(circle: (Point3, Vec3, f32), point: Point3) -> Point3 {
    let rel  = point - circle.0;
    let proj = proj_onto_plane(rel, circle.1);
    circle.0 - proj.normalize_to(circle.2)
}

#[derive(Copy, Clone, Debug)]
pub struct Capsule {
    pub line: (Point3, Point3),
    pub radius: f32,
}

fn clamp01(t: f32) -> f32 {
    if      t <= 0.0 { 0.0 }
    else if t >= 1.0 { 1.0 }
    else             {  t  }
}

fn overlap_capsules(a: &Capsule, b: &Capsule) -> bool {
    let (point_a, point_b) = closest_on_segments(a.line, b.line);
    (point_b - point_a).magnitude2() <= (a.radius + b.radius).powi(2)
}

fn closest_on_segments(
    (p1, q1): (Point3, Point3),
    (p2, q2): (Point3, Point3),
) -> (Point3, Point3) {
    // adapted from Real-Time Collision Detection, Christer Ericson
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r  = p1 - p2;

    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    if a == 0.0 && e == 0.0 {
        return (p1, p2)
    }
    let (s, t) = {
        if a == 0.0 {
            (0.0, clamp01(f / e))
        } else {
            let c = d1.dot(r);
            if e == 0.0 {
                (clamp01(-c / a), 0.0)
            } else {
                let b = d1.dot(d2);
                let denom = a*e - b*b;

                let mut s = if denom != 0.0 {
                    clamp01((b*f - c*e) / denom)
                } else { 0.0 };

                let mut t = b*s + f;
                if t < 0.0 {
                    t = 0.0;
                    s = clamp01(-c / a);
                } else if t > e {
                    t = 1.0;
                    s = clamp01((b - c) / a);
                } else {
                    t /= e;
                }
                (s, t)
            }
        }
    };
    (p1 + d1* s, p2 + d2*t)
}

fn intersect_ray_plane(ray: (Point3, Vec3), plane: (Vec3, f32)) -> Option<Point3> {
    intersect_ray_plane_coefficient(ray, plane).map(|t| ray.0 + t * ray.1)
}

fn intersect_ray_plane_coefficient(ray: (Point3, Vec3), plane: (Vec3, f32)) -> Option<f32> {
    Some((plane.0 * plane.1 - vec3!(ray.0)).dot(plane.0) / ray.1.dot(plane.0))
        .filter(|n| !n.is_nan() && *n >= 0.0)
}

fn proj_onto_plane(v: Vec3, n_normalized: Vec3) -> Vec3 {
    v - v.dot(n_normalized) * n_normalized
}

fn proj_point_onto_plane(point: Point3, plane: (Vec3, f32)) -> Point3 {
    let plane_origin = point3!() + plane.0 * plane.1;
    let displacement = point - plane_origin;
    plane_origin + proj_onto_plane(displacement, plane.0)
}

fn get_perpendicular(v: Vec3) -> Vec3 {
    if v.y != 0.0 || v.z != 0.0 { v.cross(VEC3_X) } else { v.cross(VEC3_Y) }
}

fn make_arc_points(
    centre: Point3, arc: f32, x: Vec3, y: Vec3, segments: usize
) -> impl Iterator<Item=Point3> {
    (0..=segments).map(move |i| {
        let theta = i as f32 * arc / segments as f32;
        centre + theta.cos()*x + theta.sin()*y
    })
}

fn make_line_strip_circle(
    centre: Point3, normal: Vec3, radius: f32, segments: usize,
) -> Vec<Point3> {
    let tangent   = get_perpendicular(normal).normalize_to(radius);
    let bitangent = normal.cross(tangent);

    make_arc_points(centre, TAU, tangent, bitangent, segments).collect()
}

fn make_line_strip_capsule(
    capsule: &Capsule, axes: Option<(Vec3, Vec3)>, quater_segments: usize,
) -> Vec<Point3> {
    let required_capacity = 16 * quater_segments + 8;
    let mut points = Vec::with_capacity(required_capacity);

    let z = (capsule.line.1 - capsule.line.0).normalize();
    let (x, y) = axes.unwrap_or_else(|| {
        let x = get_perpendicular(z).normalize();
        let y = z.cross(x);
        (x, y)
    });

    let (x, y, z) = (x*capsule.radius, y*capsule.radius, z*capsule.radius);
    points.extend(make_arc_points(capsule.line.1, TAU/4.0,  z, x,   quater_segments));
    points.extend(make_arc_points(capsule.line.1, TAU,      x, y, 4*quater_segments));
    points.extend(make_arc_points(capsule.line.0, TAU,      x, y, 4*quater_segments));
    points.extend(make_arc_points(capsule.line.0, TAU/2.0,  x,-z, 2*quater_segments));
    points.extend(make_arc_points(capsule.line.1, TAU/4.0, -x, z,   quater_segments));
    points.extend(make_arc_points(capsule.line.1, TAU/4.0,  z, y,   quater_segments));
    points.extend(make_arc_points(capsule.line.0, TAU/2.0,  y,-z, 2*quater_segments));
    points.extend(make_arc_points(capsule.line.1, TAU/4.0, -y, z,   quater_segments));

    assert_eq!(
        points.len(), required_capacity,
        "make_line_strip_capsule calculated incorrect capacity"
    );
    points
}

fn make_line_strip_capsule_billboard(
    capsule: &Capsule, normal: Vec3, quater_segments: usize,
) -> Vec<Point3> {
    let required_capacity = 12 * quater_segments + 5;
    let mut points = Vec::with_capacity(required_capacity);

    let z = (capsule.line.1 - capsule.line.0).normalize();
    let x = {
        let x = z.cross(normal);
        if x == vec3!(0.0, 0.0, 0.0) {
            // just make the circles if facing parallel with capsule
            let x = get_perpendicular(z).normalize_to(capsule.radius);
            let y = z.cross(x);

            points.extend(make_arc_points(capsule.line.0, TAU, x, y, 4*quater_segments));
            points.extend(make_arc_points(capsule.line.1, TAU, x, y, 4*quater_segments));
            return points;
        } else { x.normalize() }
    };
    let y = z.cross(x);

    let (x, y, z) = (x*capsule.radius, y*capsule.radius, z*capsule.radius);
    points.extend(make_arc_points(capsule.line.1, TAU/2.0,  x, z, 2*quater_segments));
    points.extend(make_arc_points(capsule.line.1, TAU,     -x, y, 4*quater_segments));
    points.extend(make_arc_points(capsule.line.0, TAU,     -x, y, 4*quater_segments));
    points.extend(make_arc_points(capsule.line.0, TAU/2.0, -x,-z, 2*quater_segments));
    points.push(capsule.line.1 + x);

    assert_eq!(
        points.len(), required_capacity,
        "make_line_strip_capsule calculated incorrect capacity"
    );
    points
}

fn make_line_strip_grid(
    origin: Point3, axes: (Vec3, Vec3), count: usize,
) -> Vec<Point3> {
    let num_points = count * 8 + 4;
    let half       = num_points / 2;
    let mut points = unsafe {
        let mut points = Vec::with_capacity(num_points);
        points.set_len(num_points);
        points
    };

    let mut jump = (
        2.0 * count as f32 * axes.1,
       -2.0 * count as f32 * axes.0,
    );

    let mut point = (
        origin - jump.0/2.0 + jump.1/2.0 - axes.0,
        origin - jump.0/2.0 - jump.1/2.0 - axes.1,
    );

    for i in 0..=(count*2) {
        macro_rules! step { ($id:tt, $offset:expr) => {
            point.$id            += axes.$id;
            points[$offset+2*i]   = point.$id;

            point.$id            += jump.$id;
            points[$offset+2*i+1] = point.$id;

            jump.$id             *= -1.0;
        } }
        step!(0, 0); step!(1, half);
    }
    points
}

fn capture_image_to_file() -> thread::JoinHandle<()> {
    use std::fs::File;
    use std::io::BufWriter;

    let mut pixels = unsafe {
        let buf_len    = 4 * WIDTH * HEIGHT;
        let mut pixels = Vec::<u8>::with_capacity(buf_len);
        pixels.set_len(buf_len);

        gl::ReadPixels(
            0, 0, WIDTH as GLsizei, HEIGHT as GLsizei,
            gl::RGBA, gl::UNSIGNED_BYTE,
            pixels.as_mut_ptr() as *mut GLvoid,
        );
        pixels
    };

    thread::spawn(move || {
        unsafe {
            slice::from_raw_parts_mut(
                pixels.as_mut_ptr() as *mut [[u8; 4]; WIDTH], HEIGHT,
            ).reverse();
        }

        let mut writer = {
            use png::HasParameters;
            let mut encoder = png::Encoder::new(
                BufWriter::new(File::create("capture.png").expect("file creation failed")),
                WIDTH as u32, HEIGHT as u32,
            );
            encoder.set(png::ColorType::RGBA).set(png::BitDepth::Eight);
            encoder.write_header().expect("failed to write png header")
        };
        writer.write_image_data(pixels.as_slice()).expect("failed to write png data");
    })
}

#[inline]
fn some_if<T>(predicate: bool, value: T) -> Option<T> {
    if predicate { Some(value) } else { None }
}

#[inline]
fn some_if_then<T, F>(predicate: bool, f: F) -> Option<T>
where F: FnOnce() -> T
{
    if predicate { Some(f()) } else { None }
}

#[inline]
fn option_if<T>(predicate: bool, value: Option<T>) -> Option<T> {
    if predicate { value } else { None }
}

#[inline]
fn option_if_then<T, F>(predicate: bool, f: F) -> Option<T>
where F: FnOnce() -> Option<T>
{
    if predicate { f() } else { None }
}

fn as_ptr<R, P>(reference: &R) -> *const P {
    reference as *const R as *const P
}

fn as_mut_ptr<R, P>(reference: &mut R) -> *mut P {
    reference as *mut R as *mut P
}
