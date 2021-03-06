#[macro_use]
extern crate bitflags;
extern crate glutin;
extern crate rand;

#[cfg(target_os = "windows")]
extern crate winapi;

macro_rules! cstr {
    ($s:expr) => (
        CStr::from_bytes_with_nul(concat!($s, "\0").as_bytes()).unwrap();
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

pub const fn short_color(r: u16, g: u16, b: u16, a: bool) -> u16 {
    0 | r << 11 | g << 6 | b << 1 | a as u16
}

#[macro_use]
mod math;
use crate::math::*;

#[macro_use]
mod gl;
use crate::gl::types::*;

mod render;
use crate::render::{Camera, Render};

mod render_sdf;
use crate::render_sdf::RenderSdf;

mod jagged;
use crate::jagged::Jagged;

mod lemon;
use crate::lemon::{Lemon, NormalizedLemon};

mod phys;
use crate::phys::{Collision, Rigidbody};

mod geometry;
use crate::geometry::*;

mod debug_render;
use crate::debug_render::DebugRender;

mod debug_ui;
use crate::debug_ui::{DebugUi, Histogram};

use glutin::*;

#[cfg(target_os = "windows")]
use winapi::um::{
    mmsystem::{TIMERR_NOERROR, TIMERR_NOCANDO},
    timeapi::{timeBeginPeriod, timeEndPeriod},
};

use rand::random;

use std::{
    default::Default,
    error::Error,
    ffi::{CStr, CString},
    fmt,
    fs,
    mem,
    ops::*,
    os::raw::c_char,
    path::Path,
    ptr,
    slice,
    time::{Duration, Instant},
    thread,
};

pub const TAU: f32 = PI * 2.0;

type ElementIndex = u16;
const ELEMENT_INDEX_TYPE: GLenum = gl::UNSIGNED_SHORT;

const LEMON_COLORS: &[Vec4] = &[
    color!(0xFFF44F_FF), // bright yellow
    color!(0xFFFE6B_FF), // pale yellow
    color!(0xFFE74D_FF), // warm yellow
];
const BACK_COLOR:  Vec4 = color!(0xA2EFEF_00);

const LEMON_TEX_SIZE: usize = 1;

const ARENA_WIDTH:     f32   = ARENA_GRID_STEP * ARENA_GRID_DIVS as f32;
const ARENA_GRID_DIVS: usize = 20;
const ARENA_GRID_STEP: f32   = 1.5 * METER;

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

const LEMON_SCALE_MAX:    f32 = 1.25;
const LEMON_SCALE_MIN:    f32 = 0.75;
const LEMON_S_MIN:        f32 = 0.50;
const LEMON_S_MAX:        f32 = 0.75;
const LEMON_PARTY_S_MIN:  f32 = 0.30;
const LEMON_PARTY_S_MAX:  f32 = 0.95;

const LEMON_SPAWN_HEIGHT_BASE:       Real  = 3.0;
const LEMON_SPAWN_HEIGHT_VARIANCE:   Real  = 0.5;
const LEMON_SPAWN_HORIZONTAL_RADIUS: Real  = 0.5;
const LEMON_SPAWN_INIT:              usize = 7;
const LEMON_SPAWN_MULTI_OFFSET:      Real  = 0.25*METER + 2.0*LEMON_SCALE_MAX;
const LEMON_SPAWN_MULTI_PERIOD:      usize = (1.0 * SECOND) as usize;

const CAMERA_HEIGHT_BASE:    f32 = 0.5;
const CAMERA_HEIGHT_FACTOR:  f32 = 0.6;
const CAMERA_HEIGHT_DEFAULT: f32 = CAMERA_HEIGHT_BASE + 1.0 * CAMERA_HEIGHT_FACTOR;
const CAMERA_LERP_TIME:      f32 = 0.3;

const DEFAULT_VSYNC:      bool  = false;
const DEFAULT_MSAA:       u16   = 2;
const DEFAULT_MAX_BODIES: usize = 256;
const DEFAULT_WIDTH:      usize = 1280;
const DEFAULT_HEIGHT:     usize = 720;

// these match specific values in the shader
pub const MAX_BODIES:          usize = 64;
pub const MAX_CLIPS_PER_LEMON: usize = 1;

fn main() {
    let (vsync, msaa, max_bodies, width, height) = {
        let mut vsync      = DEFAULT_VSYNC;
        let mut msaa       = DEFAULT_MSAA;
        let mut max_bodies = DEFAULT_MAX_BODIES;
        let mut width      = DEFAULT_WIDTH;
        let mut height     = DEFAULT_HEIGHT;

        let mut args = std::env::args();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-v" => args.next().map(|b| vsync      = b.parse().unwrap()),
                "-m" => args.next().map(|b| msaa       = b.parse().unwrap()),
                "-n" => args.next().map(|n| max_bodies = n.parse().unwrap()),
                "-w" => args.next().map(|n| width      = n.parse().unwrap()),
                "-h" => args.next().map(|n| height     = n.parse().unwrap()),
                "-r" => args.next().map(|s| match s.as_str() {
                    "360"  => { height = 360;  width = 640;  },
                    "480"  => { height = 480;  width = 854;  },
                    "720"  => { height = 720;  width = 1280; },
                    "1080" => { height = 1080; width = 1920; },
                    "1440" => { height = 1440; width = 2560; },
                    "2160" => { height = 2160; width = 3840; },
                    _      => panic!("unknown resolution: {}", s),
                }),
                _    => continue,
            };
        }
        (vsync, msaa, MAX_BODIES, width, height)
    };
    let aspect     = width as f32 / height as f32;
    if max_bodies <= 15  { panic!("max bodies must be greater than 15"); }
    if width      <= 127 { panic!("window width must be greater than 127"); }
    if height     <= 127 { panic!("window height must be greater than 127"); }

    let mut events_loop = EventsLoop::new();
    let windowed_context = {
        let window = WindowBuilder::new()
            .with_dimensions(dpi::LogicalSize::new(width as _, height as _))
            .with_title("lemon")
            .with_resizable(false);

        let mut windowed_context = ContextBuilder::new()
            .with_multisampling(msaa)
            .with_vsync(vsync)
            .build_windowed(window, &events_loop)
            .unwrap();
        unsafe { windowed_context.make_current().unwrap() }
    };
    gl::load_with(|symbol| windowed_context.get_proc_address(symbol) as *const _);

    let sleep_resolution_ms = if cfg!(target_os = "windows") {
        let mut sleep_resolution_ms = 1;
        #[cfg(target_os = "windows")]
        while unsafe { timeBeginPeriod(sleep_resolution_ms) } == TIMERR_NOCANDO {
            sleep_resolution_ms += 1;
            if sleep_resolution_ms > 16 { break; }
        }
        sleep_resolution_ms
    } else {
        1 // TODO: implement actual logic for non-windows platforms
    };

    let mut render = Render::init(max_bodies);
    let mut render_sdf = RenderSdf::init(max_bodies).unwrap();
    let mut render_sdf_enabled = false;
    render_sdf.update_clipping_planes(&(0..max_bodies).map(|_| 
        [Plane::new(random::<Vec3>().normalize(), 0.25); MAX_CLIPS_PER_LEMON]
    ).collect::<Vec<_>>());

    let mut debug_frame_store             = Jagged::new();
    let mut debug_frame_current           = 0;
    let mut debug_frame_step              = 0;
    let mut debug_frame_step_delay        = 0;

    let mut lemons           = Vec::<Lemon>::with_capacity(max_bodies);
    let mut lemon_transforms = vec![mat4!(0.0); max_bodies].into_boxed_slice();
    let mut lemon_spawn_frame = 0;

    macro_rules! spawn_lemon { () => {
        if lemons.len() >= max_bodies { return; }
        let id = lemons.len();

        let scale = LEMON_SCALE_MIN + (LEMON_SCALE_MAX-LEMON_SCALE_MIN) * random::<f32>();
        let s     = LEMON_S_MIN + (LEMON_S_MAX-LEMON_S_MIN) * random::<f32>();

        let color_index = random::<f32>().powi(2) * LEMON_COLORS.len() as f32;
        let color       = LEMON_COLORS[color_index.floor() as usize].truncate();

        let mut new_lemon = Lemon::new(s, scale);
        render.update_lemon(lemons.len(), Some(&s), Some(&color), None);

        reset_lemon(&mut new_lemon);
        if id > 0 && debug_frame_current < lemon_spawn_frame + LEMON_SPAWN_MULTI_PERIOD {
            new_lemon.phys.position.z = lemons[id-1].phys.position.z 
                                      + LEMON_SPAWN_MULTI_OFFSET;
        }
        lemon_spawn_frame = debug_frame_current;
        lemons.push(new_lemon);
    } };

    fn reset_lemon(lemon: &mut Lemon) {
        let theta = random::<f32>() * TAU;
        lemon.phys.position         = Point3 {
            x: theta.sin() * LEMON_SPAWN_HORIZONTAL_RADIUS,
            y: theta.cos() * LEMON_SPAWN_HORIZONTAL_RADIUS,
            z: (LEMON_SPAWN_HEIGHT_BASE + (random::<f32>()*TAU).sin() * LEMON_SPAWN_HEIGHT_VARIANCE) * METER,
        };
        lemon.phys.orientation      = Quat::from_axis_angle(
                                      random::<Vec3>().normalize(),
                                      Deg(30.0 * (random::<f32>() - 0.5)));
        lemon.phys.velocity         = vec3!() * METER / SECOND;
        lemon.phys.angular_momentum = lemon.phys.get_inertia()
                                    * (random::<Vec3>() - vec3!(0.5, 0.5, 0.5)) * 2.0
                                    * TAU / 5.0 / SECOND;
    }

    macro_rules! debug_frame_store_clear { () => {
        debug_frame_store.clear();
        debug_frame_current = 0;
        debug_frame_store.push_copy(&lemons);
    }; }

    for _ in 0..LEMON_SPAWN_INIT {
        spawn_lemon!();
    }
    debug_frame_store.push_copy(&lemons);

    let mut lemon_selection_index:  usize = !0;
    let mut lemon_selection_anim:   Real  = 0.0;
    let mut lemon_hover_index:      usize = !0;
    let mut lemon_hover_prev_index: usize = !0;
    let mut lemon_hover_depth:      Real  = INFINITY;
    let mut lemon_hover_needs_update      = true;
    let mut lemon_interact_index:   usize = 0;
    let mut lemon_drag_offset:      Vec3  = VEC3_0;
    let mut lemon_drag_plane:       Plane = Plane::new(VEC3_0, 0.0);
    render.update_selection(Some(lemon_selection_index), None);
    render_sdf.update_selection(Some(lemon_selection_index), None);

    let mut debug            = DebugRender::new();
    let mut debug_depth_test = DebugRender::with_shared_context(&debug);
    {
        debug_depth_test.draw_line(&color!(0xFFFFFFFF).truncate(), !0, &make_line_strip_grid(
            point3!(), (VEC3_X*ARENA_GRID_STEP, VEC3_Y*ARENA_GRID_STEP), ARENA_GRID_DIVS,
        ));

        debug.draw_axes(1.5, !0, &Mat4::identity());
    }

    let mut debug_ui = DebugUi::new(width, height, gl::TEXTURE1);

    let mut debug_draw_axes               = false;
    let mut debug_draw_torus_section      = false;
    let mut debug_draw_colliders_floor    = false;
    let mut debug_draw_colliders_lemon    = false;
    let mut debug_draw_bounding_volumes   = false;
    let mut debug_draw_motion_vectors     = false;
    let mut debug_draw_collision_response = false;
    let mut debug_draw_wireframe          = false;
    let mut debug_lemon_party             = false;
    let mut debug_spin_between_frames     = false;

    let debug_histogram_width:       usize = height.min(width);
    let debug_histogram_height:      usize = height / 3;
    let debug_histogram_bar_width:   usize = 2;
    let debug_histogram_bar_spacing: usize = 0;
    let debug_histogram_nano_per_px: u32   = 5 * FRAME_DURATION.subsec_nanos() / 4
                                           / debug_histogram_height as u32;
    let mut debug_histogram                = Histogram::new(
        (width - debug_histogram_width) as isize, (height / 16) as isize,
        debug_histogram_width, debug_histogram_height,
    );
    let mut debug_histogram_logging        = true;
    let mut debug_histogram_display        = false;
    let get_histogram_bar_height = |elapsed: Duration| -> usize {
        (elapsed.subsec_nanos() / debug_histogram_nano_per_px) as usize
    };

    let mut debug_pause                   = false;
    let mut debug_pause_next_frame        = false;

    let mut camera              = Camera::default();
    let mut camera_fovy         = 30.0_f32.to_radians();
    let mut camera_distance     = 15.0_f32;
    let mut camera_elevation    = 9.0_f32.to_radians();
    let mut camera_azimuth      = 0.0_f32.to_radians();
    let mut camera_direction    = VEC3_0;
    let mut camera_target       = point3!(0.0, 0.0, CAMERA_HEIGHT_DEFAULT);
    let mut camera_lerp         = NAN;
    let mut camera_lerp_origin  = point3!(VEC3_0);
    let mut camera_lerp_point   = point3!(VEC3_0);
    let mut camera_needs_update = true;

    let mut mouse_pixel    = point2!(VEC2_0);
    let mut mouse_pos      = point2!(-1.0, 1.0);
    let mut mouse_prev_pos = point2!(-1.0, 1.0);
    let mut mouse_movement = VEC2_0;

    let mut mouse_down     = false;
    let mut mouse_pressed  = false;
    let mut mouse_released = false;
    let mut mouse_clicked  = false;
    let mut mouse_dragging = false;

    let mut mouse_ray              = Ray::new(point3!(), vec3!());
    let mut mouse_ray_needs_update = true;

    macro_rules! dragging_lemon { ($lemon_index: expr) => {
        mouse_dragging && $lemon_index == lemon_hover_index
                       && $lemon_index == lemon_interact_index
    }; }

    let mut frame_sync_time = Instant::now();

    let mut window_in_focus = false;
    'main: loop {
        // RESET PER-FRAME VARIABLES
        // mouse input
        mouse_pressed  = false;
        mouse_released = false;
        mouse_clicked  = false;
        mouse_movement = VEC2_0;
        mouse_prev_pos = mouse_pos;

        // POLL INPUT
        let mut exit = false;
        events_loop.poll_events(|event| match event {
            Event::WindowEvent{ event: WindowEvent::CloseRequested, .. } => {
                exit = true;
            },
            Event::WindowEvent{ event: WindowEvent::Focused(focus), .. } => {
                window_in_focus = focus;
            },
            _ if !window_in_focus || exit => { /* ignore input while out of focus */ },

            Event::WindowEvent { event, .. } => match event {
                WindowEvent::MouseInput { button: MouseButton::Left, state, .. } => {
                    match state {
                        ElementState::Pressed  => {
                            mouse_down     = true;
                            mouse_pressed  = true;
                        },
                        ElementState::Released => {
                            mouse_down     = false;
                            mouse_released = true;
                            mouse_clicked  = !mouse_dragging;
                            mouse_dragging = false;
                        },
                    }
                },
                WindowEvent::CursorMoved { position: pos, .. } => {
                    let mouse_pixel_new      = point2!(pos.x as f32, pos.y as f32);
                    let mouse_pixel_movement = mouse_pixel_new - mouse_pixel;
                    mouse_pixel              = mouse_pixel_new;

                    if mouse_pixel_movement.x != 0.0 || mouse_pixel_movement.y != 0.0 {
                        let mouse_pos_new = Point2::new(
                            (mouse_pixel.x * 2.0) as f32 / width  as f32 - 1.0,
                            (mouse_pixel.y *-2.0) as f32 / height as f32 + 1.0,
                        );

                        mouse_pos              = mouse_pos_new;
                        mouse_dragging        |= mouse_down && !mouse_pressed;
                        camera_needs_update   |= mouse_dragging;
                        mouse_ray_needs_update = true;
                    }
                },
                WindowEvent::MouseWheel {
                    delta: MouseScrollDelta::LineDelta(_delta_x, delta_y), modifiers, ..
                } => {
                    if !(modifiers.ctrl || modifiers.alt) { // regular camera zoom
                        camera_distance -= camera_distance * delta_y * 0.065;
                        if camera_distance < 0.0 { camera_distance = 0.0; }
                        camera_needs_update = true;
                    } else if lemon_selection_index < lemons.len() {
                        let new_scale = option_if_then(modifiers.ctrl, || {
                            let new_scale = lemons[lemon_selection_index].scale + delta_y * 0.05;
                            some_if(new_scale >= 0.35 && new_scale <= 2.0, new_scale)
                        });
                        let new_s     = option_if_then(modifiers.alt, || {
                            let normalized = lemons[lemon_selection_index].get_normalized();
                            let new_s      = normalized.s + delta_y * 0.01;
                            if new_s >= 0.15 && new_s <= 0.95 {
                                render.update_lemon(lemon_selection_index, Some(&normalized.s), None, None);
                                debug_frame_store_clear!();
                                Some(new_s)
                            } else { None }
                        });
                        if new_scale.is_some() || new_s.is_some() {
                            let lemon     = &mut lemons[lemon_selection_index];
                            let new_scale = new_scale.unwrap_or(lemon.scale);
                            let new_s     = new_s.unwrap_or_else(|| lemon.get_normalized().s);
                            lemon.mutate_shape(new_s, new_scale);
                            render_sdf.update_lemon(lemon_selection_index, Some(&lemon));
                        }
                    }
                },
                WindowEvent::KeyboardInput{ input: KeyboardInput {
                    state, modifiers, virtual_keycode: Some(key), ..
                }, .. } => match key {
                    VirtualKeyCode::Q => if let ElementState::Pressed = state {
                        render_sdf_enabled = !render_sdf_enabled;
                    },
                    VirtualKeyCode::Grave => if let ElementState::Pressed = state {
                        // reload sdf renderer
                        match RenderSdf::init(max_bodies) {
                            Ok(new_render_sdf) => render_sdf = new_render_sdf,
                            Err(glsl_error)    => eprintln!("{}", glsl_error),
                        }
                        // HACK: resets clipping planes
                        render_sdf.update_clipping_planes(&(0..max_bodies).map(|_| 
                            [Plane::new(random::<Vec3>().normalize(), 0.25); MAX_CLIPS_PER_LEMON]
                        ).collect::<Vec<_>>());
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
                    VirtualKeyCode::W => if let ElementState::Pressed = state {
                        debug_draw_wireframe = !debug_draw_wireframe;
                    },
                    VirtualKeyCode::L => if let ElementState::Pressed = state {
                        debug_lemon_party = !debug_lemon_party;
                    },
                    VirtualKeyCode::H => if let ElementState::Pressed = state {
                        if debug_histogram_display {
                            debug_ui.clear_pixels();
                            debug_histogram_display = false;
                        } else {
                            debug_histogram_display = true;
                        }
                    },
                    VirtualKeyCode::J => if let ElementState::Pressed = state {
                        if !modifiers.ctrl {
                            if !debug_histogram_logging {
                                debug_histogram.clear_pixels();
                            }
                            debug_histogram_logging = true;
                        } else {
                            debug_histogram_logging = false;
                        }
                    },
                    VirtualKeyCode::T => if let ElementState::Pressed = state {
                        debug_spin_between_frames = modifiers.ctrl;
                    },
                    VirtualKeyCode::R => if let ElementState::Pressed = state {
                        if modifiers.ctrl && modifiers.shift {
                            lemons.clear();
                            debug_frame_store_clear!();
                            if lemon_selection_index != !0 {
                                lemon_selection_index = !0;
                                camera_target.z       = CAMERA_HEIGHT_DEFAULT;
                                camera_lerp           = 0.0;
                            }
                        }
                    },
                    VirtualKeyCode::Space => if let ElementState::Pressed = state {
                        if modifiers.ctrl {
                            spawn_lemon!();
                        } else {
                            if lemon_selection_index < lemons.len() {
                                reset_lemon(&mut lemons[lemon_selection_index]);
                                lemon_selection_anim = 0.0;
                                camera_lerp_origin = camera_lerp_point;
                                camera_lerp        = 0.25;
                            } else {
                                camera_target = point3!(0.0, 0.0, CAMERA_HEIGHT_DEFAULT);
                                camera_lerp   = 0.0;
                            }
                        }
                    },
                    VirtualKeyCode::Escape => if let ElementState::Pressed = state {
                        debug_pause_next_frame = !debug_pause;
                        if lemon_selection_index >= lemons.len() {
                            lemon_selection_index = !0;
                            camera_target.z       = CAMERA_HEIGHT_DEFAULT;
                            camera_lerp           = 0.0;
                        }
                    },
                    VirtualKeyCode::Right => {
                        if let ElementState::Pressed = state {
                            debug_frame_step = if modifiers.ctrl { 6 } else { 1 };
                        } else if debug_frame_step > 0 {
                            debug_frame_step = 0;
                        }
                    },
                    VirtualKeyCode::Left => {
                        if let ElementState::Pressed = state {
                            debug_frame_step = if modifiers.ctrl { -6 } else { -1 };
                        } else if debug_frame_step < 0 {
                            debug_frame_step = 0;
                        }
                    },
                    _ => (),
                },
                _ => (),
            },
            _ => (),
        });
        if exit { break 'main; }
        mouse_movement = mouse_pos - mouse_prev_pos;

        // STEP THROUGH FRAMES WHILE PAUSED
        if debug_pause && debug_frame_step != 0 && debug_frame_store.len() > 0 {
            if debug_frame_step_delay == 0 || debug_frame_step_delay > FRAME_RATE / 5 {
                let target_frame = debug_frame_current as isize + debug_frame_step;
                if target_frame < 0 {
                    debug_frame_current = 0;
                } else if target_frame >= debug_frame_store.len() as isize {
                    debug_frame_current = debug_frame_store.len() - 1;
                } else {
                    debug_frame_current = target_frame as usize;
                }
                lemons.clear();
                lemons.extend_from_slice(&debug_frame_store[debug_frame_current]);
                camera_needs_update = true;
            }
            debug_frame_step_delay += 1;
        } else {
            debug_frame_step_delay = 0;
        }

        // UPDATE PAUSE STATE
        if debug_pause_next_frame != debug_pause {
            if debug_pause {
                debug_frame_store.truncate(debug_frame_current + 1);
            } else {
                assert_eq!(debug_frame_current, debug_frame_store.len() - 1);
            }
            debug_pause = debug_pause_next_frame;
        }

        // UPDATE CAMERA
        camera_needs_update |= !debug_pause;
        camera_needs_update |= !camera_lerp.is_nan();
        if camera_needs_update {
            if dragging_lemon!(!0) {
                camera_azimuth   -= TAU/5.0 * mouse_movement.x * aspect;

                camera_elevation -= TAU/5.0 * mouse_movement.y;
                camera_elevation  = camera_elevation.min(85.0_f32.to_radians());
                camera_elevation  = camera_elevation.max(-85.0_f32.to_radians());
            }

            camera_direction =
                ( Quat::from_angle_z(Rad(camera_azimuth))
                * Quat::from_angle_x(Rad(-camera_elevation))
                ).rotate_vector(vec3!(0.0, 1.0, 0.0));

            if lemon_selection_index < lemons.len() && !dragging_lemon!(lemon_selection_index) {
                // don't chase selected lemon while being dragged
                let lemon = &lemons[lemon_selection_index];
                camera_target = Point3 {
                    x: lemon.phys.position.x,
                    y: lemon.phys.position.y,
                    z: lemon.phys.position.z * CAMERA_HEIGHT_FACTOR + CAMERA_HEIGHT_BASE,
                };
            }

            if !camera_lerp.is_nan() {
                if camera_lerp < 1.0 {
                    if camera_lerp == 0.0 {
                        // set origin on first frame of transition
                        camera_lerp_origin = camera_lerp_point;
                    } else {
                        // evaluate elliptical easing function
                        camera_lerp_point  = camera_lerp_origin
                                           + (1.0 - (camera_lerp - 1.0).powi(2)).sqrt()
                                           * (camera_target - camera_lerp_origin);
                    }
                    camera_lerp += FRAME_DELTA_TIME / 0.3;
                } else {
                    camera_lerp       = NAN;
                    camera_lerp_point = camera_target;
                }
            } else {
                camera_lerp_point = camera_target;
            }
            camera.position   = camera_lerp_point - camera_direction * camera_distance;
            camera.projection = perspective(Rad(camera_fovy), aspect, 0.1, 1e+6);
            camera.view       = Mat4::look_at(
                camera.position,
                camera_lerp_point + camera_direction,
                VEC3_Z,
            );

            render.update_camera(&camera);
            render_sdf.update_camera(&camera);
            debug.update_camera(&camera);

            mouse_ray_needs_update = true;
            camera_needs_update    = false;
        }

        // UPDATE MOUSE RAY
        if mouse_ray_needs_update {
            let camera_inverse = (camera.projection * camera.view).invert().unwrap();
            let origin = camera_inverse.transform_point(point3!(mouse_pos.x, mouse_pos.y, 0.0));
            let target = camera_inverse.transform_point(point3!(mouse_pos.x, mouse_pos.y, 1.0));
            mouse_ray  = Ray::new(origin, (target - origin).normalize()); // normalize?
        }

        debug_histogram.add_bar_segment("input processing, camera update",
            short_color(31, 31, 31, true), get_histogram_bar_height(frame_sync_time.elapsed()),
        );

        // MOUSE DRAG LEMONS (LEMON INTERACTION part 1 / 2)
        if lemon_interact_index < lemons.len() && dragging_lemon!(lemon_interact_index) {
            // set position and velocity
            let lemon    = &mut lemons[lemon_interact_index];
            let drag_pos = raycast_plane_double_sided(mouse_ray, lemon_drag_plane).point
                         + lemon_drag_offset;

            lemon.phys.velocity         = drag_pos - lemon.phys.position;
            lemon.phys.position         = drag_pos;
            lemon.phys.angular_momentum = VEC3_0;

            if lemon_interact_index == lemon_selection_index {
                // induce camera transition after lemon released
                camera_lerp = 0.0;
            }
        }
        // reset lemon_hover state
        lemon_hover_prev_index    = lemon_hover_index;
        // ensure lemon_hover is buffered on the frame the mouse is released
        lemon_hover_needs_update |= mouse_released;
        if lemon_hover_needs_update {
            lemon_hover_index     = !0;
            lemon_hover_depth     = INFINITY;
        }

        // TIMESTEP INTEGRATION AND FIXED-OBJECT COLLISIONS
        for (lemon_index, lemon) in lemons.iter_mut().enumerate() {
            // LEMON PARTY CODE
            if debug_lemon_party && !debug_pause {
                const LSB_MASK:   u32 = 0x1;
                const PARTY_RATE: f32 = 0.5 / SECOND;

                let lsb = lemon.sagitta.to_bits() & LSB_MASK != 0;
                let mut normalized = lemon.get_normalized();
                normalized = if lsb { // increase or decrease sagitta based on its LSB
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
                lemon.sagitta = f32::from_bits( // set the LSB for next frame
                    if next_lsb { lemon.sagitta.to_bits() | LSB_MASK }
                    else        { lemon.sagitta.to_bits() &!LSB_MASK }
                );
                render.update_lemon(lemon_index, Some(&normalized.s), None, None);
                render_sdf.update_lemon(lemon_index, Some(&lemon));
            }

            // INTEGRATE RIGIDBODIES
            if !debug_pause && !dragging_lemon!(lemon_index) {
                phys::integrate_rigidbody_fixed_timestep(&mut lemon.phys);
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

            // TEST STATIC WALL-PLANE COLLISIONS
            if !debug_pause || dragging_lemon!(lemon_index) {
                for &dir in [
                    vec2!(1.0, 0.0), vec2!(-1.0, 0.0), vec2!(0.0, 1.0), vec2!(0.0, -1.0)
                ].into_iter() {
                    if vec2!(lemon.phys.position).dot(dir) >= ARENA_WIDTH - lemon.scale {
                        let wall_normal = -dir.extend(0.0);
                        if let Some(collision) = lemon::get_collision_halfspace(
                            &lemon, (wall_normal, -ARENA_WIDTH), None,
                        ) {
                            if dragging_lemon!(lemon_index) {
                                phys::resolve_basic_collision_static(
                                    collision, &mut lemon.phys,
                                );
                            } else {
                                phys::resolve_collision_static(
                                    collision, &mut lemon.phys, None,
                                );
                            }
                        }
                    }
                }
            }

            // TEST STATIC FLOOR-PLANE COLLISION
            let floor_collision = lemon::get_collision_halfspace(
                &lemon, (VEC3_Z, 0.0),
                some_if(debug_draw_colliders_floor, &mut debug),
            );
            if let Some(collision) = floor_collision {
                if dragging_lemon!(lemon_index) {
                    phys::resolve_basic_collision_static(
                        collision, &mut lemon.phys,
                    );
                } else if !debug_pause {
                    phys::resolve_collision_static(
                        collision, &mut lemon.phys,
                        some_if(debug_draw_collision_response, &mut debug),
                    );
                } else if debug_draw_collision_response {
                    let mut temp_lemon_phys = lemon.phys.clone();
                    phys::resolve_collision_static(
                        collision, &mut temp_lemon_phys, Some(&mut debug),
                    );
                }
            }

            // UPDATE HOVER SELECTION
            if lemon_hover_needs_update && overlap_capsule_ray(
                lemon.get_bounding_capsule(), mouse_ray
            ) {
                let raycast = lemon::raycast(mouse_ray, lemon);
                if raycast.rear.is_non_negative() && raycast.fore.is_intersection() {
                    let hit = raycast.fore;
                    if hit.depth < lemon_hover_depth {
                        lemon_hover_index = lemon_index;
                        lemon_hover_depth = hit.depth;
                    }
                }
            }

            // DEBUG VISUALISATIONS
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

            // UPDATE TRANSFORM DATA
            lemon_transforms[lemon_index] = lemon.get_transform_with_scale();
        }

        if debug_lemon_party {
            debug_frame_store_clear!();
        } else if !debug_pause {
            debug_frame_store.push_copy(&lemons);
        }

        // UPDATE LEMON SELECTION AND PULSE EFFECTS (LEMON INTERACTION part 2 / 2)
        if mouse_pressed {
            // set candidate for dragging or selection
            lemon_interact_index     = lemon_hover_index;
            lemon_hover_needs_update = false;
            render.update_hover(Some(lemon_hover_index), Some(0.6));
            render_sdf.update_hover(Some(lemon_hover_index), Some(0.6));
        }
        if mouse_released {
            if mouse_clicked && lemon_interact_index == lemon_hover_index {
                if lemon_interact_index == !0 {
                    if lemon_selection_index == !0 {
                        // move camera focus if already in free look mode
                        let raycast_floor = raycast_plane_double_sided(
                            mouse_ray, Plane::new(VEC3_Z, 0.0),
                        );
                        if raycast_floor.is_intersection() && raycast_floor.is_non_negative()
                        && raycast_floor.point.x.abs() <= ARENA_WIDTH
                        && raycast_floor.point.y.abs() <= ARENA_WIDTH
                        {
                            // LAG: camera unable to update until next frame
                            camera_target.x = raycast_floor.point.x;
                            camera_target.y = raycast_floor.point.y;
                        }
                    }
                    camera_target.z         = CAMERA_HEIGHT_DEFAULT;
                    camera_lerp             = 0.0;
                } else if lemon_selection_index != lemon_interact_index {
                    camera_lerp             = 0.0;
                }
                lemon_selection_index = lemon_interact_index;
                lemon_selection_anim  = 0.0;
                render.update_selection(Some(lemon_selection_index), None);
                render_sdf.update_selection(Some(lemon_selection_index), None);
            }
            lemon_hover_needs_update = true;
        }
        if mouse_down {
             if !mouse_dragging && lemon_hover_index < lemons.len() {
                // test lemon still under cursor (ignoring occlusion)
                let lemon   = &lemons[lemon_hover_index];
                let raycast = lemon::raycast(mouse_ray, lemon);
                if raycast.rear.is_non_negative() && raycast.fore.is_intersection() {
                    // set axes lemon is dragged along
                    // TODO: do this once (before loop) on first frame of drag only
                    lemon_drag_offset = lemon.phys.position - raycast.fore.point;
                    lemon_drag_plane  = Plane::from_point_and_normal(
                        raycast.fore.point, -camera_direction,
                    );
                } else {
                    // cancel hover
                    lemon_hover_index = !0;
                    render.update_hover(Some(!0), None);
                    render_sdf.update_hover(Some(!0), None);
                }
            }
        } else {
            // no mouse button input: set highlight to lemon under cursor
            if lemon_hover_index != lemon_hover_prev_index || mouse_released {
                render.update_hover(Some(lemon_hover_index), Some(0.3));
                render_sdf.update_hover(Some(lemon_hover_index), Some(0.3));
            }
        }

        // lemon selection animation
        if lemon_selection_index < lemons.len() && !lemon_selection_anim.is_nan() {
            const MAX: Real = 1.4;
            const MIN: Real = 0.0;
            const LEN: Real = 0.4;

            let glow = if lemon_selection_anim < 1.0 {
                MAX - (MAX-MIN) * (1.0 - (lemon_selection_anim - 1.0).powi(2)).sqrt()
            } else {
                lemon_selection_anim = NAN;
                MIN
            };
            render.update_selection(None, Some(glow));
            render_sdf.update_selection(None, Some(glow));
            lemon_selection_anim += FRAME_DELTA_TIME / LEN;
        }

        debug_histogram.add_bar_segment("rigidbody integration, static collisions",
            short_color(31, 31, 0, true), get_histogram_bar_height(frame_sync_time.elapsed()),
        );

        // DYNAMIC OBJECT COLLISIONS
        let lemons_len = lemons.len();
        for lemon_index in 0..lemons_len { // iterate over (un-ordered) pairs of objects
            let (lemon, other_lemons) = {
                let (heads, tails) = lemons.split_at_mut(lemon_index + 1);
                (&mut heads[lemon_index], tails)
            };
            for (other_relative_index, other) in other_lemons.iter_mut().enumerate() {
                let other_index = lemon_index + other_relative_index + 1;

                let bounding_volume_overlap = overlap_capsules(
                    lemon.get_bounding_capsule(),
                    other.get_bounding_capsule(),
                );
                let force_collision_test = debug_draw_colliders_lemon
                                        &&!debug_draw_bounding_volumes
                                        && ( lemons_len <= 4
                                           || ( lemon.phys.position-other.phys.position
                                              ).magnitude2() <= 9.0 );

                let collision = option_if_then(
                    bounding_volume_overlap || force_collision_test,
                    || lemon::get_collision_lemon(
                        &lemon, &other, some_if(debug_draw_colliders_lemon, &mut debug),
                    ),
                );

                if let Some(collision) = collision {
                    assert!(bounding_volume_overlap, "lemon collision without BV overlap");
                    assert!(collision.depth >= 0.0, "collision reported negative depth");

                    if dragging_lemon!(lemon_index) {
                        if !debug_pause { phys::resolve_collision_kinematic(
                            collision.neg(), &mut other.phys, &lemon.phys,
                            some_if(debug_draw_collision_response, &mut debug),
                        ); }
                    } else if dragging_lemon!(other_index) {
                        if !debug_pause { phys::resolve_collision_kinematic(
                            collision, &mut lemon.phys, &other.phys,
                            some_if(debug_draw_collision_response, &mut debug),
                        ); }
                    } else {
                        if !debug_pause {
                            phys::resolve_collision_dynamic(
                                collision, &mut lemon.phys, &mut other.phys,
                                some_if(debug_draw_collision_response, &mut debug),
                            );
                        } else if debug_draw_collision_response { // draw debug w/o side effects
                            let mut temp_lemon_phys = lemon.phys.clone();
                            let mut temp_other_phys = other.phys.clone();
                            phys::resolve_collision_dynamic(
                                collision, &mut temp_lemon_phys, &mut temp_other_phys,
                                Some(&mut debug),
                            );
                        }
                    }

                    // UPDATE LEMONS TRANSFORM DATA
                    if !debug_pause {
                        lemon_transforms[lemon_index] = lemon.get_transform_with_scale();
                        lemon_transforms[other_index] = other.get_transform_with_scale();
                    }
                }
            }
        }
        debug_histogram.add_bar_segment("dynamic collisions",
            short_color(31, 15, 0, true), get_histogram_bar_height(frame_sync_time.elapsed()),
        );

        // RENDER SCENE
        unsafe {
            gl::ClearColor(BACK_COLOR.x, BACK_COLOR.y, BACK_COLOR.z, BACK_COLOR.w);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            gl::Enable(gl::DEPTH_TEST);
            gl::Enable(gl::CULL_FACE);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::ONE, gl::ONE_MINUS_SRC_ALPHA);
            gl::DepthFunc(gl::LESS);

            if debug_draw_wireframe { gl::PolygonMode(gl::FRONT_AND_BACK, gl::LINE); }
            render.update_lemon_transforms(&lemon_transforms[0..lemons.len()]);
            render_sdf.update_lemons(&lemons);

            if render_sdf_enabled {
                render_sdf.render_lemons(lemons.len());
            } else {
                render.render_lemons(lemons.len());
            }
            gl::PolygonMode(gl::FRONT_AND_BACK, gl::FILL);

            debug_depth_test.render_frame();

            gl::DepthFunc(gl::ALWAYS);
            debug.render_frame();

            if debug_histogram_display {
                debug_histogram.copy_pixels_to_ui(&mut debug_ui);
                debug_ui.render();
            }
        }
        debug_histogram.add_bar_segment("draw calls",
            short_color(0, 31, 0, true), get_histogram_bar_height(frame_sync_time.elapsed()),
        );

        // SYNCHRONISE BUFFER SWAP CALL
        let frame_expected_sync_time = frame_sync_time + FRAME_DURATION;
        let frame_sleep_start_time   = Instant::now();
        let frame_sleep_end_time     = if !debug_spin_between_frames
                                       && frame_sleep_start_time < frame_expected_sync_time
        {
            let remaining_ms  = (frame_expected_sync_time - frame_sleep_start_time)
                                .subsec_millis();
            let sleep_time_ms = remaining_ms as i32 - sleep_resolution_ms as i32
                              - (remaining_ms % sleep_resolution_ms) as i32;
            if sleep_time_ms > 0 {
                thread::sleep(Duration::from_millis(sleep_time_ms as u64));
            }
            Instant::now()
        } else {
            frame_sleep_start_time
        };
        if frame_sleep_end_time < frame_expected_sync_time {
            while Instant::now() < frame_expected_sync_time { /* spin remaining time */ }
        }

        // SWAP BUFFERS AND RECORD NEXT FRAME SYNC TIME
        let frame_next_sync_time   = Instant::now();
        windowed_context.swap_buffers().expect("buffer swap failed");
        let frame_buffer_swap_time = Instant::now();

        // LOG SWAP TIMING AND ADVANCE HISTOGRAM FOR DISPLAY NEXT FRAME
        if debug_histogram_logging {
            if frame_sleep_start_time != frame_sleep_end_time {
                if frame_sleep_end_time < frame_expected_sync_time {
                    debug_histogram.add_bar_segment("thread sleep",
                        short_color(0, 0, 0, false),
                        get_histogram_bar_height(frame_sleep_end_time - frame_sync_time),
                    );
                    debug_histogram.add_bar_segment("thread sleep defecit",
                        short_color(0, 0, 31, true),
                        get_histogram_bar_height(frame_next_sync_time - frame_sync_time),
                    );
                } else {
                    debug_histogram.add_bar_segment("thread sleep",
                        short_color(0, 0, 0, false),
                        get_histogram_bar_height(FRAME_DURATION),
                    );
                    debug_histogram.add_bar_segment("thread sleep excess",
                        short_color(31, 0, 0, true),
                        get_histogram_bar_height(frame_sleep_end_time - frame_sync_time),
                    );
                }
            }

            debug_histogram.add_line("frame sync point",
                short_color(0, 0, 0, true),
                get_histogram_bar_height(FRAME_DURATION),
            );
            debug_histogram.render_and_flush_buffer(
                debug_histogram_bar_width, debug_histogram_bar_spacing
            );
        } else {
            debug_histogram.clear_buffer();
        }

        debug_histogram.add_bar_segment("buffer swap",
            short_color(0, 31, 31, true),
            get_histogram_bar_height(frame_buffer_swap_time - frame_next_sync_time),
        );

        // SET NEXT FRAME SYNC TIME
        frame_sync_time  = frame_next_sync_time;

        // ADVANCE FRAME COUNTER
        if !debug_pause && !debug_lemon_party { debug_frame_current += 1;}
    }

    #[cfg(target_os = "windows")] unsafe { timeEndPeriod(sleep_resolution_ms); }
}

fn clamp01(t: f32) -> f32 {
    if      t <= 0.0 { 0.0 }
    else if t >= 1.0 { 1.0 }
    else             {  t  }
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

    let z = (capsule.segment.tail - capsule.segment.head).normalize();
    let (x, y) = axes.unwrap_or_else(|| {
        let x = get_perpendicular(z).normalize();
        let y = z.cross(x);
        (x, y)
    });

    let (x, y, z) = (x*capsule.radius, y*capsule.radius, z*capsule.radius);
    points.extend(make_arc_points(capsule.segment.tail, TAU/4.0,  z, x,   quater_segments));
    points.extend(make_arc_points(capsule.segment.tail, TAU,      x, y, 4*quater_segments));
    points.extend(make_arc_points(capsule.segment.head, TAU,      x, y, 4*quater_segments));
    points.extend(make_arc_points(capsule.segment.head, TAU/2.0,  x,-z, 2*quater_segments));
    points.extend(make_arc_points(capsule.segment.tail, TAU/4.0, -x, z,   quater_segments));
    points.extend(make_arc_points(capsule.segment.tail, TAU/4.0,  z, y,   quater_segments));
    points.extend(make_arc_points(capsule.segment.head, TAU/2.0,  y,-z, 2*quater_segments));
    points.extend(make_arc_points(capsule.segment.tail, TAU/4.0, -y, z,   quater_segments));

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

    let z = (capsule.segment.tail - capsule.segment.head).normalize();
    let x = {
        let x = z.cross(normal);
        if x == vec3!(0.0, 0.0, 0.0) {
            // just make the circles if facing parallel with capsule
            let x = get_perpendicular(z).normalize_to(capsule.radius);
            let y = z.cross(x);

            points.extend(make_arc_points(capsule.segment.head, TAU, x, y, 4*quater_segments));
            points.extend(make_arc_points(capsule.segment.tail, TAU, x, y, 4*quater_segments));
            return points;
        } else { x.normalize() }
    };
    let y = z.cross(x);

    let (x, y, z) = (x*capsule.radius, y*capsule.radius, z*capsule.radius);
    points.extend(make_arc_points(capsule.segment.tail, TAU/2.0,  x, z, 2*quater_segments));
    points.extend(make_arc_points(capsule.segment.tail, TAU,     -x, y, 4*quater_segments));
    points.extend(make_arc_points(capsule.segment.head, TAU,     -x, y, 4*quater_segments));
    points.extend(make_arc_points(capsule.segment.head, TAU/2.0, -x,-z, 2*quater_segments));
    points.push(capsule.segment.tail + x);

    assert_eq!(
        points.len(), required_capacity,
        "make_line_strip_capsule calculated incorrect capacity"
    );
    points
}

fn make_line_strip_plane(
    plane: Plane, step: Real, count: usize,
) -> Vec<Point3> {
    let origin = Point3::from_vec(plane.normal*plane.offset);
    let x = get_perpendicular(plane.normal).normalize();
    let y = plane.normal.cross(x);
    make_line_strip_grid(origin, (x*step, y*step), count)
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
