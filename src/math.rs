#![allow(unused_macros)]
#![allow(dead_code)]

extern crate cgmath;
pub use cgmath::{*, num_traits::{one, zero}};

pub use std::f32::{self, NAN, INFINITY, consts::PI};
pub const TAU: f32 = PI * 2.0;

#[macro_export]
macro_rules! vec4 {
    () => {
        Vec4 { x:zero(), y: zero(), z: zero(), w: zero() }
    };
    ($xyzw: expr) => {
        vec4!($xyzw.x, $xyzw.y, $xyzw.z, $xyzw.w)
    };
    ($x: expr, $y: expr, $z: expr, $w: expr) => (
        Vec4 { x: $x, y: $y, z: $z, w: $w }
    );
}

#[macro_export]
macro_rules! vec3 {
    () => {
        Vec3 { x: zero(), y: zero(), z: zero() }
    };
    ($xyz: expr) => {
        vec3!($xyz.x, $xyz.y, $xyz.z)
    };
    ($x: expr, $y: expr, $z: expr) => (
        Vec3 { x: $x, y: $y, z: $z }
    );
}

#[macro_export]
macro_rules! vec2 {
    () => {
        Vec2 { x: zero(), y: zero() }
    };
    ($xy: expr) => {
        vec2!($xy.x, $xy.y)
    };
    ($x: expr, $y: expr) => (
        Vec2 { x: $x, y: $y }
    );
}

#[macro_export]
macro_rules! point3 {
    () => {
        Point3 { x: zero(), y: zero(), z: zero() }
    };
    ($xyz: expr) => {
        point3!($xyz.x, $xyz.y, $xyz.z)
    };
    ($x: expr, $y: expr, $z: expr) => (
        Point3 { x: $x, y: $y, z: $z }
    );
}

#[macro_export]
macro_rules! point2 {
    () => {
        Point2 { x: zero(), y: zero() }
    };
    ($xy: expr) => {
        point2!($xy.x, $xy.y)
    };
    ($x: expr, $y: expr) => (
        Point2 { x: $x, y: $y }
    );
}

#[macro_export]
macro_rules! mat4 {
    () => (
        mat4!(1.0)
    );
    ($n: expr) => (
        Mat4 {
            x: vec4!($n, 0.0, 0.0, 0.0),
            y: vec4!(0.0, $n, 0.0, 0.0),
            z: vec4!(0.0, 0.0, $n, 0.0),
            w: vec4!(0.0, 0.0, 0.0, $n),
        }
    );
}

#[macro_export]
macro_rules! mat3 {
    () => (
        mat3!(1.0)
    );
    ($n: expr) => (
        Mat3 {
            x: vec3!($n, 0.0, 0.0),
            y: vec3!(0.0, $n, 0.0),
            z: vec3!(0.0, 0.0, $n),
        }
    );
}

#[macro_export]
macro_rules! mat2 {
    () => (
        mat2!(1.0)
    );
    ($n: expr) => (
        Mat3 {
            x: vec3!($n, 0.0)
            y: vec3!(0.0, $n),
        }
    );
}

pub type Real   = f32; // TODO: propagate this

pub type Vec4          = cgmath::Vector4<Real>;
pub const VEC4_X: Vec4 = vec4!(1.0, 0.0, 0.0, 0.0);
pub const VEC4_Y: Vec4 = vec4!(0.0, 1.0, 0.0, 0.0);
pub const VEC4_Z: Vec4 = vec4!(0.0, 0.0, 1.0, 0.0);
pub const VEC4_W: Vec4 = vec4!(0.0, 0.0, 0.0, 1.0);
pub const VEC4_0: Vec4 = vec4!(0.0, 0.0, 0.0, 0.0);
pub const VEC4_1: Vec4 = vec4!(1.0, 1.0, 1.0, 1.0);

pub type Vec3   = cgmath::Vector3<Real>;
pub const VEC3_X: Vec3 = vec3!(1.0, 0.0, 0.0);
pub const VEC3_Y: Vec3 = vec3!(0.0, 1.0, 0.0);
pub const VEC3_Z: Vec3 = vec3!(0.0, 0.0, 1.0);
pub const VEC3_0: Vec3 = vec3!(0.0, 0.0, 0.0);
pub const VEC3_1: Vec3 = vec3!(1.0, 1.0, 1.0);

pub type Vec2   = cgmath::Vector2<Real>;
pub const VEC2_X: Vec2 = vec2!(1.0, 0.0);
pub const VEC2_Y: Vec2 = vec2!(0.0, 1.0);
pub const VEC2_0: Vec2 = vec2!(0.0, 0.0);
pub const VEC2_1: Vec2 = vec2!(1.0, 1.0);

pub type Point3 = cgmath::Point3<Real>;
pub type Point2 = cgmath::Point2<Real>;
pub type Mat4   = cgmath::Matrix4<Real>;
pub type Mat3   = cgmath::Matrix3<Real>;
pub type Mat2   = cgmath::Matrix2<Real>;
pub type Quat   = cgmath::Quaternion<Real>;
