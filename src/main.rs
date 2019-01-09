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

mod gl {
    include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));
}
use crate::gl::types::*;

mod lemon;
use crate::lemon::Lemon;

mod debug_render;
use crate::debug_render::DebugRender;

use glutin::*;

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
    f32::consts::PI,
    mem,
    ops::*,
    os::raw::c_char,
    ptr,
    slice,
    thread,
};

type ElementIndex = u16;
const ELEMENT_INDEX_TYPE: GLenum = gl::UNSIGNED_SHORT;

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
        Mat4::from_translation(vec3!(self.position)) * Mat4::from(Mat3::from(self.orientation))
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

const SLEEP_TIME: u32 = 30;
const PHYS_DELTA_TIME: f32 = 1.0 / SLEEP_TIME as f32;

const PHYS_GRAVITY: Vec3 = vec3!(0.0, 0.0, -0.015);

const LEMON_COLLISION_ELASTICITY: f32 = 0.15;

const LEMON_FRICTION: f32 = 1.55;

const LEMON_ANGULAR_DRAG: f32 = 0.002;

const WIDTH: usize = 720;
const HEIGHT: usize = 720;

fn main() {
    let mut events_loop = EventsLoop::new();
    let window = WindowBuilder::new()
        .with_title("lemon")
        .with_dimensions(dpi::LogicalSize::new(WIDTH as _, HEIGHT as _))
        .with_resizable(false);
    let context = ContextBuilder::new()
        .with_multisampling(1);
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

        let camera_ubo = gl_gen_object(gl::GenBuffers);
        gl::BindBuffer(gl::UNIFORM_BUFFER, camera_ubo);
        gl_buffer_init::<Camera>(gl::UNIFORM_BUFFER, 1, gl::DYNAMIC_DRAW);

        let camera_binding_index: GLuint = 1;
        gl::BindBufferBase(gl::UNIFORM_BUFFER, camera_binding_index, camera_ubo);

        (camera, camera_ubo, camera_binding_index)
    };

    let mut lemon = Lemon::new(0.58);
    lemon.phys.position = point3!(0.0, 0.0, 5.0);

    let (vao, verts, normals, indices, vbo_transform) = unsafe {
        let program = gl_link_shaders(&[
            gl_compile_shader(include_str!("shader/lemon.vert.glsl"), gl::VERTEX_SHADER),
            gl_compile_shader(include_str!("shader/lemon.frag.glsl"), gl::FRAGMENT_SHADER),
        ]);
        gl::UseProgram(program);

        let camera_index = gl::GetUniformBlockIndex(program, cstr!("Camera"));
        gl::UniformBlockBinding(program, camera_index, camera_binding_index);

        let u_lemon_color = gl::GetUniformLocation(program, cstr!("u_lemon_color"));
        gl::Uniform4fv(u_lemon_color, 1, as_ptr(&LEMON_COLOR));

        let u_ambient_color = gl::GetUniformLocation(program, cstr!("u_ambient_color"));
        gl::Uniform4fv(u_ambient_color, 1, as_ptr(&BACK_COLOR));

        //let u_point_light = gl::GetUniformLocation(program, cstr!("u_point_light"));
        //gl::Uniform4fv(u_point_light, 1, as_ptr(&vec3!(2.0, 2.0, 2.0)));

        let txo_normal_map = gl_gen_object(gl::GenTextures);
        let normal_map = lemon.make_normal_map();
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

        let vao = gl_gen_object(gl::GenVertexArrays);
        gl::BindVertexArray(vao);

        let (verts, uvs, normals, indices) = lemon.make_mesh();

        let a_transform   = gl::GetAttribLocation(program, cstr!("a_transform")) as GLuint;
        let vbo_transform = gl_gen_object(gl::GenBuffers);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo_transform);
        for i in 0..4 { // all 4 column vectors
            let a_transform = a_transform + i as GLuint;
            gl::EnableVertexAttribArray(a_transform);
            gl::VertexAttribPointer(
                a_transform , 4, gl::FLOAT, gl::FALSE,
                0, ptr::null::<f32>().offset(4 * i) as *const GLvoid,
            );
            gl::VertexAttribDivisor(a_transform, 1);
        }

        let a_position   = gl::GetAttribLocation(program, cstr!("a_position")) as GLuint;
        let vbo_position = gl_gen_object(gl::GenBuffers);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo_position);
        gl_buffer_data(gl::ARRAY_BUFFER, &verts, gl::STATIC_DRAW);
        gl::EnableVertexAttribArray(a_position);
        gl::VertexAttribPointer(a_position, 3, gl::FLOAT, gl::FALSE, 0, 0 as *const GLvoid);

        let a_normal   = gl::GetAttribLocation(program, cstr!("a_normal")) as GLuint;
        let vbo_normal = gl_gen_object(gl::GenBuffers);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo_normal);
        gl_buffer_data(gl::ARRAY_BUFFER, &normals, gl::STATIC_DRAW);
        gl::EnableVertexAttribArray(a_normal);
        gl::VertexAttribPointer(a_normal, 3, gl::FLOAT, gl::FALSE, 0, 0 as *const GLvoid);

        let a_uv   = gl::GetAttribLocation(program, cstr!("a_uv")) as GLuint;
        let vbo_uv = gl_gen_object(gl::GenBuffers);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo_uv);
        gl_buffer_data(gl::ARRAY_BUFFER, &uvs, gl::STATIC_DRAW);
        gl::EnableVertexAttribArray(a_uv);
        gl::VertexAttribPointer(a_uv, 2, gl::FLOAT, gl::FALSE, 0, 0 as *const GLvoid);

        let ebo = gl_gen_object(gl::GenBuffers);
        gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
        gl_buffer_data(gl::ELEMENT_ARRAY_BUFFER, &indices, gl::STATIC_DRAW);

        (vao, verts, normals, indices, vbo_transform)
    };

    let mut debug            = DebugRender::new();
    let mut debug_depth_test = DebugRender::with_shared_context(&debug);
    {
        let grid = make_line_strip_grid(point3!(), (VEC3_X*1.5, VEC3_Y*1.5), 20);
        debug_depth_test.draw_line(&color!(0xFFFFFFFF).truncate(), !0, &grid);

        debug.draw_axes(1.5, !0, &Mat4::identity());
    }
    let mut debug_draw_colliders          = false;
    let mut debug_draw_motion_vectors     = false;
    let mut debug_draw_collision_response = false;
    let mut debug_frame_store             = vec![lemon.phys];
    let mut debug_frame_current           = 0;
    let mut debug_pause = false;

    let mut camera_fovy: f32 = 30.0_f32.to_radians();
    let mut camera_distance: f32 = 9.0_f32;
    let mut camera_elevation: f32 = 0.0_f32.to_radians();
    let mut camera_azimuth: f32 = 0.0_f32.to_radians();

    let mut mouse_pos: Vec2 = vec2!();
    let mut mouse_drag: Option<Vec2> = Some(vec2!());
    let mut mouse_down = false;

    let mut exit = false;
    while !exit {
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
                    delta: MouseScrollDelta::LineDelta(_delta_x, delta_y), ..
                } => {
                    camera_distance -= delta_y * 0.2;
                    if camera_distance < 0.0 { camera_distance = 0.0; }
                    mouse_drag = mouse_drag.or(Some(vec2!())); // HACK: redraw on zoom
                },
                _ => (),
            },
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::Key(KeyboardInput {
                    state, virtual_keycode: Some(key), ..
                }) => match key {
                    VirtualKeyCode::P => if let ElementState::Pressed = state {
                        capture_image_to_file();
                    },
                    VirtualKeyCode::X => if let ElementState::Pressed = state {
                        debug_draw_colliders = !debug_draw_colliders;
                    },
                    VirtualKeyCode::C => if let ElementState::Pressed = state {
                        debug_draw_collision_response = !debug_draw_collision_response;
                    },
                    VirtualKeyCode::V => if let ElementState::Pressed = state {
                        debug_draw_motion_vectors = !debug_draw_motion_vectors;
                    },
                    VirtualKeyCode::Escape => if let ElementState::Pressed = state {
                        if debug_pause {
                            debug_pause = false;
                            debug_frame_store.truncate(debug_frame_current + 1);
                        } else {
                            debug_pause = true;
                            debug_frame_current = debug_frame_store.len() - 1;
                        }
                    },
                    VirtualKeyCode::Left => if let ElementState::Pressed = state {
                        if debug_pause && debug_frame_current > 0 { 
                            debug_frame_current -= 1;
                            lemon.phys = debug_frame_store[debug_frame_current];
                        }
                    },
                    VirtualKeyCode::Right => if let ElementState::Pressed = state {
                        if debug_pause && debug_frame_current + 1 < debug_frame_store.len() {
                            debug_frame_current += 1;
                            lemon.phys = debug_frame_store[debug_frame_current];
                        }
                    },
                    VirtualKeyCode::Space => if let ElementState::Pressed = state {
                        lemon.phys.position         = point3!(0.0, 0.0, 5.0);
                        lemon.phys.angular_momentum = lemon.phys.get_inertia()
                                                    * (rand::random::<Vec3>() * 0.04);
                    },
                    _ => (),
                },
                _ => (),
            },
            _ => (),
        });

        // UPDATE PHYSICS
        {
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
                
                debug_frame_store.push(lemon.phys);
            }
            if debug_draw_motion_vectors {
                debug.draw_ray(
                    &color!(0xFF50FFFF).truncate(), 1, &lemon.phys.position, 
                    &(lemon.phys.velocity / PHYS_DELTA_TIME),
                );
                debug.draw_ray(
                    &color!(0xFFFF50FF).truncate(), 1, &lemon.phys.position, 
                    & ( lemon.phys.get_inertia_inverse() 
                      * lemon.phys.angular_momentum 
                      / PHYS_DELTA_TIME ),
                );
            }

            // TEST COLLISIONS
            // find point on torus focal radius furthest from plane
            let displacement = vec3!(0.0, 0.0, lemon.phys.position.z);
            let torus_focus_normal = lemon.phys.orientation * VEC3_Z;
            // TODO: check focus normal parallel to plane
            let displacement_proj = proj_onto_plane(displacement, torus_focus_normal);

            let sphere_centre = lemon.phys.position
                              + lemon.t * displacement_proj.normalize();
            let sphere_normal = torus_focus_normal.cross(displacement_proj);

            let sphere_closest = sphere_centre - VEC3_Z*lemon.r;

            let sign = torus_focus_normal.z.signum();
            let base = lemon.phys.position - sign * torus_focus_normal;
            let test_point = if sign * VEC3_Z.cross(base-sphere_centre).dot(sphere_normal)
                              > 0.0 {
                sphere_closest
            } else {
                base
            };

            if test_point.z <= 0.0 { // COLLISION RESPONSE
                // push object out of plane // TODO: solve for velocity, orientation
                let new_position     = lemon.phys.position - test_point.z * VEC3_Z;
                
                if !debug_pause {
                    let angular_drag = {
                        let mag  = lemon.phys.angular_momentum.magnitude();
                        if LEMON_ANGULAR_DRAG > mag {
                            lemon.phys.angular_momentum
                        } else {
                            lemon.phys.angular_momentum / mag * LEMON_ANGULAR_DRAG
                        }
                    };
                    lemon.phys.angular_momentum -= angular_drag;
                    lemon.phys.position          = new_position;
                }
                
                let inertia_inverse  = lemon.phys.get_inertia_inverse();
                let angular_velocity = inertia_inverse * lemon.phys.angular_momentum;

                let point    = point3!(test_point.x, test_point.y, 0.0);
                let offset   = point - new_position;
                let velocity = lemon.phys.velocity + angular_velocity.cross(offset);
                let normal   = VEC3_Z;
                let tangent  = proj_onto_plane(velocity.normalize(), normal);

                let reaction_impulse = {
                    let proj    = -(1.0 + LEMON_COLLISION_ELASTICITY) * velocity.dot(normal);
                    let linear  = 1.0 / lemon.phys.mass;
                              // + 1.0 / infinity  => 0.0
                    let angular = (inertia_inverse * offset.cross(normal)).cross(offset);
                              // + [infinity]^-1 => 0.0
                    proj / (linear + angular.dot(normal))
                };
                let reaction = reaction_impulse * normal;

                let friction = -tangent * {
                    let proj    = velocity.dot(tangent);
                    let linear  = 1.0 / lemon.phys.mass;
                    let angular = (inertia_inverse * offset.cross(tangent)).cross(offset);

                    let cap     = proj / (linear + angular.dot(tangent));
                    cap.min(reaction_impulse * LEMON_FRICTION)
                };

                if !debug_pause {
                    lemon.phys.velocity         += (reaction + friction) / lemon.phys.mass;
                    lemon.phys.angular_momentum += offset.cross(reaction + friction);
                }
                if debug_draw_collision_response {
                    debug.draw_ray(
                        &color!(0x0000AFFF).truncate(), 1, &point, 
                        &(friction / lemon.phys.mass / PHYS_DELTA_TIME)
                    );
                    debug.draw_ray(
                        &color!(0xAF0000FF).truncate(), 1, &point, 
                        &(reaction / lemon.phys.mass / PHYS_DELTA_TIME)
                    );
                    debug.draw_ray(
                        &color!(0x00AF00FF).truncate(), 1, &point, &(velocity / PHYS_DELTA_TIME)
                    );
                }
            }

            // DEBUG VISUALISATIONS
            let floor_point   = point3!(test_point.x, test_point.y, 0.0);
            let floor_tangent = sphere_normal.cross(VEC3_Z) * lemon.r * 3.0;

            debug.draw_axes(0.5, 1, &lemon.phys.get_transform());
            if false {
                debug.draw_line(&color!(0x000000FF).truncate(), 1, &[
                    test_point, floor_point
                ]);

                debug.draw_line(&color!(0x000000FF).truncate(), 1, &[
                    floor_point - floor_tangent * 2.0 * lemon.r * 3.0,
                    floor_point + floor_tangent * 2.0 * lemon.r * 3.0,
                ]);
            }
            if true && debug_draw_colliders {
                let position  = lemon.phys.position;
                let transform = lemon.phys.get_transform();
                let normal    = (transform * VEC4_Z).truncate();
                debug.draw_axes(0.25, 1, &transform);

                let mut focal_radius = make_line_strip_circle(
                    &lemon.phys.position, &normal, lemon.t, 31
                );
                debug.draw_line(&color!(0x000000FF).truncate(), 1, &focal_radius);

                let sphere_normal = normal.cross(displacement_proj).normalize();
                let mut sphere_radius = make_line_strip_circle(
                    &sphere_centre, &sphere_normal,  lemon.r, 31
                );
                debug.draw_line(&color!(0x000000FF).truncate(), 1, &sphere_radius);

                debug.draw_line(&color!(0x000000FF).truncate(), 1, &[
                    position + normal, sphere_centre, position - normal,
                    sphere_centre, sphere_closest,
                ]);

                let world_axes_transform = Mat4::from_translation(vec3!(sphere_centre));
                debug.draw_axes(0.4, 1, &world_axes_transform);
            }

            unsafe {
                gl::BindBuffer(gl::ARRAY_BUFFER, vbo_transform);
                gl_buffer_data(
                    gl::ARRAY_BUFFER,
                    slice::from_ref(&lemon.phys.get_transform()),
                    gl::DYNAMIC_DRAW
                );
            }
        }

        // UPDATE CAMERA
        if let Some(movement) = mouse_drag.or(Some(vec2!())) {
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
            camera.projection = perspective(Rad(camera_fovy), WIDTH as f32 / HEIGHT as f32, 0.1, 100.0);
            camera.view = Mat4::look_at(
                camera.position,
                camera_focus + camera_dir,
                VEC3_Z,
            );

            debug.update_camera(&(camera.projection * camera.view));

            unsafe {
                gl::BindBuffer(gl::UNIFORM_BUFFER, camera_ubo);
                gl_buffer_data(gl::UNIFORM_BUFFER, slice::from_ref(&camera), gl::DYNAMIC_DRAW);
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
            gl::DrawElementsInstanced(
                gl::TRIANGLE_STRIP,
                indices.len() as GLsizei,
                ELEMENT_INDEX_TYPE,
                ptr::null(),
                1,
            );
            gl::DepthFunc(gl::LESS);
            debug_depth_test.render_frame();
            gl::DepthFunc(gl::ALWAYS);
            debug.render_frame();
        }
        gl_window.swap_buffers().expect("buffer swap failed");
        std::thread::sleep_ms(SLEEP_TIME);
    }
}

fn proj_onto_plane(v: Vec3, n_normalized: Vec3) -> Vec3 {
    v - v.dot(n_normalized) * n_normalized
}

fn make_line_strip_circle(
    centre: &Point3, normal: &Vec3, radius: f32, segments: usize,
) -> Vec<Point3> {
    let other = if normal.y != 0.0 && normal.z != 0.0 { VEC3_X } else { VEC3_Y };
    let tangent   = normal.cross(other).normalize_to(radius);
    let bitangent = normal.cross(tangent);

    let mut points = Vec::with_capacity(segments + 1);
    for i_theta in 0..segments {
        let theta = i_theta as f32 * 2.0 * PI / segments as f32;
        points.push(centre + tangent * theta.cos() + bitangent * theta.sin());
    }
    points.push(points[0]);
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

fn as_ptr<R, P>(reference: &R) -> *const P {
    reference as *const R as *const P
}

fn as_mut_ptr<R, P>(reference: &mut R) -> *mut P {
    reference as *mut R as *mut P
}

#[inline]
unsafe fn gl_buffer_data<T>(target: GLenum, data: &[T], usage: GLenum) {
    gl::BufferData(
        target,
        mem::size_of_val(data) as GLsizeiptr,
        data.as_ptr() as *const GLvoid,
        usage,
    );
}

#[inline]
unsafe fn gl_buffer_sub_data<T>(target: GLenum, offset: usize, data: &[T]) {
    gl::BufferSubData(
        target,
        (offset  * mem::size_of::<T>()) as GLintptr,
        mem::size_of_val(data) as GLsizeiptr,
        data.as_ptr() as *const GLvoid,
    );
}

#[inline]
unsafe fn gl_buffer_init<T>(target: GLenum, count: usize, usage: GLenum) {
    gl::BufferData(
        target,
        (mem::size_of::<T>() * count) as GLsizeiptr,
        0 as *const GLvoid,
        usage,
    );
}

#[inline]
unsafe fn gl_gen_object(gl_gen_callback: unsafe fn (GLsizei, *mut GLuint)) -> GLuint {
    let mut name = GLuint::default();
    gl_gen_callback(1, &mut name);
    name
}

unsafe trait GlGet {
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
        unsafe fn gl_get(p: GLenum, r: *mut GLuint) { gl::GetIntegerv(p, r as *mut GLint); }
        gl_get
    };
}
unsafe impl GlGet for GLint64 {
    const GL_GET: unsafe fn (GLenum, *mut Self) = gl::GetInteger64v;
}
unsafe impl GlGet for GLuint64 {
    const GL_GET: unsafe fn (GLenum, *mut Self) = {
        unsafe fn gl_get(p: GLenum, r: *mut GLuint64) { gl::GetInteger64v(p, r as *mut GLint64); }
        gl_get
    };
}
unsafe impl GlGet for GLfloat {
    const GL_GET: unsafe fn (GLenum, *mut Self) = gl::GetFloatv;
}
unsafe impl GlGet for GLdouble {
    const GL_GET: unsafe fn (GLenum, *mut Self) = gl::GetDoublev;
}

#[inline]
unsafe fn gl_get<T: GlGet>(parameter: GLenum) -> T {
    let mut result: T = mem::uninitialized(); T::GL_GET(parameter, &mut result);
    result
}

fn gl_compile_shader(src: &str, ty: GLenum) -> GLuint {
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

fn gl_link_shaders(shaders: &[GLuint]) -> GLuint {
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
