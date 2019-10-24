#![allow(dead_code)]
use super::*;

// GEOMETRIC TYPES

#[derive(Copy, Clone, Debug)]
#[repr(C)]
/// `vector` must have positive length
pub struct Ray {
    pub origin: Point3,
    pub vector: Vec3,
}

impl Ray {
    pub fn new(origin: Point3, vector: Vec3) -> Self {
        Ray { origin, vector }
    }

    pub fn from_points(origin: Point3, second: Point3) -> Self {
        Ray { origin, vector: second - origin }
    }

    pub fn normalized_from_points(origin: Point3, second: Point3) -> Self {
        Ray { origin, vector: (second - origin).normalize() }
    }

    pub fn eval(&self, t: Real) -> Point3 {
        self.origin + t*self.vector
    }

    pub fn eval_inverse(&self, point: Point3) -> Real {
        self.vector.dot(point - self.origin) / self.vector.magnitude2()
    }

    pub fn eval_inverse_if_unit_length(&self, point: Point3) -> Real {
        self.vector.dot(point - self.origin)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
/// `head` and `tail` need not be different points
pub struct Segment {
    pub head: Point3,
    pub tail: Point3,
}

impl Segment {
    pub fn new(head: Point3, tail: Point3) -> Self {
        Segment { head, tail }
    }

    pub fn eval(&self, t: Real) -> Point3 {
        if t == 0.5 {
            Point3::from_vec((vec3!(self.head) + vec3!(self.tail)) / 2.0)
        } else {
            self.head + t*(self.tail - self.head)
        }
    }

    pub fn eval_inverse(&self, point: Point3) -> Real {
        (self.tail - self.head).dot(point - self.head)
    }

    pub fn length(&self) -> Real {
        (self.head - self.tail).magnitude()
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
/// `normal` must be unit length
pub struct Plane {
    pub normal: Vec3,
    pub offset: Real,
}

impl Plane {
    pub fn new(normal: Vec3, offset: Real) -> Self {
        Plane { normal, offset }
    }

    pub fn from_point_and_normal(point: Point3, normal: Vec3) -> Self {
        Plane { normal, offset: normal.dot(vec3!(point)) }
    }

    pub fn from_triangle(triangle: &Triangle) -> Self {
        Self::from_point_and_normal(triangle.points[0], triangle.get_nonunit_normal().normalize())
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
/// `radius` must be non-negative
pub struct Sphere {
    pub centre: Point3,
    pub radius: Real,
}

impl Sphere {
    pub fn new(centre: Point3, radius: Real) -> Self {
        Sphere { centre, radius }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
/// `normal` must be unit length
/// `radius` must be non-negative
pub struct Disc {
    pub centre: Point3,
    pub normal: Vec3,
    pub radius: Real,
}

impl Disc {
    pub fn new(centre: Point3, normal: Vec3, radius: Real) -> Self {
        Disc { centre, normal, radius }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
/// `radius` must be non-negative
pub struct Capsule {
    pub segment: Segment,
    pub radius:  Real,
}

impl Capsule {
    pub fn new(segment: Segment, radius: Real) -> Self {
        Capsule { segment, radius }
    }

    pub fn from_points_and_radius(head: Point3, tail: Point3, radius: Real) -> Self {
        Capsule { segment: Segment { head, tail }, radius }
    }
}

/// Triangles are stored in counter-clockwise winding order
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Triangle {
    pub points: [Point3; 3],
}

impl Triangle {
    pub fn new(points: [Point3; 3]) -> Self {
        Triangle { points }
    }

    pub fn get_nonunit_normal(&self) -> Vec3 {
        (self.points[1] - self.points[0]).cross(self.points[2] - self.points[0])
    }
}

/// The four counter-clockwise triangular faces are defined as the following ordered triplets:
/// `points`[0, 1, 2]
/// `points`[0, 3, 1]
/// `points`[1, 3, 2]
/// `points`[2, 3, 0]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Tetrahedron {
    pub points: [Point3; 4],
}

impl Tetrahedron {
    pub fn new(points: [Point3; 4]) -> Self {
        Tetrahedron { points }
    }

    pub fn get_faces(&self) -> [Triangle; 4] { [ 
        Triangle::new([self.points[0], self.points[1], self.points[2]]),
        Triangle::new([self.points[0], self.points[3], self.points[1]]),
        Triangle::new([self.points[1], self.points[3], self.points[2]]),
        Triangle::new([self.points[2], self.points[3], self.points[0]]),
    ] }
}

// CLOSEST POINT FUNCTIONS

// Lines, Rays, Segments

pub fn closest_on_line_to_point(line: Ray, point: Point3) -> Point3 {
    line.eval(line.eval_inverse(point))
}

pub fn closest_on_ray_to_point(ray: Ray, point: Point3) -> Point3 {
    let t = ray.eval_inverse(point);
    if t <= 0.0 {
        ray.origin
    } else {
        ray.eval(t)
    }
}

pub fn closest_on_segment_to_point(segment: Segment, point: Point3) -> Point3 {
    let v = segment.tail - segment.head;
    let t = v.dot(point - segment.head) / v.magnitude2();
    if t <= 0.0 {
        segment.head
    } else if t >= 1.0 {
        segment.tail
    } else {
        segment.head + t*v
    }
}

pub fn closest_on_lines(a: Ray, b: Ray) -> (Point3, Point3) {
    let (a_t, b_t) = closest_on_lines_t(a, b);
    (a.eval(a_t), b.eval(b_t))
}

pub fn closest_on_lines_t(
    Ray { origin: p1, vector: d1 }: Ray,
    Ray { origin: p2, vector: d2 }: Ray,
) -> (Real, Real) {
    // adapted from Real-Time Collision Detection, Christer Ericson
    let r  = p1 - p2;

    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);
    let c = d1.dot(r);

    let b = d1.dot(d2);
    let d = a*e - b*b;

    let s = if d != 0.0 {
        (b*f - c*e) / d
    } else { 0.0 };
    let t = (b*s + f) / e;

    (s, t)
}

pub fn closest_on_segments(a: Segment, b: Segment) -> (Point3, Point3) {
    let (a_t, b_t) = closest_on_segments_t(a, b);
    (a.eval(a_t), b.eval(b_t))
}

pub fn closest_on_segments_t(
    Segment { head: p1, tail: q1 }: Segment,
    Segment { head: p2, tail: q2 }: Segment,
) -> (Real, Real) {
    // adapted from Real-Time Collision Detection, Christer Ericson
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r  = p1 - p2;

    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    if a == 0.0 && e == 0.0 {
        return (0.0, 0.0)
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
                let d = a*e - b*b;

                let mut s = if d != 0.0 {
                    clamp01((b*f - c*e) / d)
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
    (s, t)
}

pub fn closest_on_segment_and_ray(s: Segment, r: Ray) -> (Point3, Point3) {
    let (s_t, r_t) = closest_on_segment_and_ray_t(s, r);
    (s.eval(s_t), r.eval(r_t))
}

pub fn closest_on_segment_and_ray_t(
    Segment { head:   p1, tail:   q1 }: Segment,
    Ray     { origin: p2, vector: d2 }: Ray,
) -> (Real, Real) {
    // adapted from Real-Time Collision Detection, Christer Ericson
    let d1 = q1 - p1;
    let r  = p1 - p2;

    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    let (s, t) = {
        if a == 0.0 {
            (0.0, clamp01(f / e))
        } else {
            let c = d1.dot(r);
            let b = d1.dot(d2);
            let d = a*e - b*b;

            let s = if d != 0.0 {
                clamp01((b*f - c*e) / d)
            } else { 0.0 };
            let t = (b*s + f) / e;

            (s, t)
        }
    };
    (s, t)
}

// Spheres

pub fn closest_on_sphere_to_point(sphere: Sphere, point: Point3) -> Point3 {
    sphere.centre + (point - sphere.centre).normalize_to(sphere.radius)
}

pub fn closest_on_sphere_and_line(sphere: Sphere, line: Ray) -> (Point3, Point3) {
    let closest_line   = closest_on_line_to_point(line, sphere.centre);
    let closest_sphere = closest_on_sphere_to_point(sphere, closest_line);
    (closest_sphere, closest_line)
}

pub fn closest_on_sphere_and_ray(sphere: Sphere, ray: Ray) -> (Point3, Point3) {
    let closest_ray    = closest_on_ray_to_point(ray, sphere.centre);
    let closest_sphere = closest_on_sphere_to_point(sphere, closest_ray);
    (closest_sphere, closest_ray)
}

pub fn closest_on_sphere_and_segment(sphere: Sphere, segment: Segment) -> (Point3, Point3) {
    let closest_segment = closest_on_segment_to_point(segment, sphere.centre);
    let closest_sphere  = closest_on_sphere_to_point(sphere, closest_segment);
    (closest_sphere, closest_segment)
}

// Discs and Circles

pub fn closest_on_circle_to_point(circle: Disc, point: Point3) -> Point3 {
    let proj = proj_onto_plane(point - circle.centre, circle.normal);
    circle.centre + proj.normalize_to(circle.radius)
}

pub fn furthest_on_circle_from_point(circle: Disc, point: Point3) -> Point3 {
    let proj = proj_onto_plane(point - circle.centre, circle.normal);
    circle.centre - proj.normalize_to(circle.radius)
}

// ITERATIVE PROCEDURES

pub fn find_closest_and_distance2_on_circles(
    a: Disc, b: Disc,
    seed: Option<Point3>, epsilon2: Real, min_distance2: Real, max_iters: usize,
) -> (Point3, Point3, Real, (Real, usize)) {
    let seed = seed.unwrap_or_else(|| Segment::new(a.centre, b.centre).eval(0.5));
    find_local_closest_and_distance2_on_generic_pair(
        a, closest_on_circle_to_point,
        b, closest_on_circle_to_point,
        seed, epsilon2, min_distance2, max_iters,
    )
}

pub fn find_furthest_and_distance2_on_circles(
    a: Disc, b: Disc,
    seed: Option<Point3>, epsilon2: Real, max_distance2: Real, max_iters: usize,
) -> (Point3, Point3, Real, (Real, usize)) {
    let seed = seed.unwrap_or_else(|| Segment::new(a.centre, b.centre).eval(0.5));
    find_local_furthest_and_distance2_on_generic_pair(
        a, furthest_on_circle_from_point,
        b, furthest_on_circle_from_point,
        seed, epsilon2, max_distance2, max_iters,
    )
}

pub fn find_closest_and_distance2_on_circle_and_ray(
    c: Disc, r: Ray,
    seed: Option<Point3>, epsilon2: Real, min_distance2: Real, max_iters: usize,
) -> (Point3, Point3, Real, (Real, usize)) {
    find_local_closest_and_distance2_on_generic_pair(
        c, closest_on_circle_to_point,
        r, closest_on_ray_to_point,
        seed.unwrap_or(c.centre), epsilon2, min_distance2, max_iters,
    )
}

#[inline]
pub fn find_local_closest_and_distance2_on_generic_pair<T: Copy, U: Copy>(
    mut t: T, mut closest_on_t_to_point: impl FnMut(T, Point3) -> Point3,
    mut u: U, mut closest_on_u_to_point: impl FnMut(U, Point3) -> Point3,
    seed: Point3, epsilon2: Real, min_distance2: Real, max_iters: usize,
) -> (Point3, Point3, Real, (Real, usize)) {
    let mut closest_t = closest_on_t_to_point(t, seed);
    let mut closest_u = closest_on_u_to_point(u, seed);
    let mut distance2 = (closest_u - closest_t).magnitude2();

    let mut delta2    = INFINITY;
    let mut iters     = 0;
    while delta2    > epsilon2
       && distance2 > min_distance2
       && iters     < max_iters
    {
        let distance2_prev = distance2;

        closest_t = closest_on_t_to_point(t, closest_u);
        closest_u = closest_on_u_to_point(u, closest_t);

        distance2 = (closest_u - closest_t).magnitude2();
        delta2    = distance2_prev - distance2;
        iters    += 1;
    }
    (closest_t, closest_u, distance2, (delta2, iters))
}

#[inline]
pub fn find_local_furthest_and_distance2_on_generic_pair<T: Copy, U: Copy>(
    t: T, mut furthest_on_t_from_point: impl FnMut(T, Point3) -> Point3,
    u: U, mut furthest_on_u_from_point: impl FnMut(U, Point3) -> Point3,
    seed: Point3, epsilon2: Real, max_distance2: Real, max_iters: usize,
) -> (Point3, Point3, Real, (Real, usize)) {
    let mut furthest_t = furthest_on_t_from_point(t, seed);
    let mut furthest_u = furthest_on_u_from_point(u, seed);
    let mut distance2 = (furthest_u - furthest_t).magnitude2();

    let mut delta2    = INFINITY;
    let mut iters     = 0;
    while delta2    > epsilon2
       && distance2 < max_distance2
       && iters     < max_iters
    {
        let distance2_prev = distance2;

        furthest_t = furthest_on_t_from_point(t, furthest_u);
        furthest_u = furthest_on_u_from_point(u, furthest_t);

        distance2 = (furthest_u - furthest_t).magnitude2();
        delta2    = distance2 - distance2_prev;
        iters    += 1;
    }
    (furthest_t, furthest_u, distance2, (delta2, iters))
}

// OVERLAP TESTS

pub fn overlap_spheres(a: Sphere, b: Sphere) -> bool {
    (b.centre - a.centre).magnitude2() <= (a.radius + b.radius).powi(2)
}

pub fn overlap_sphere_ray(s: Sphere, r: Ray) -> bool {
    let closest = closest_on_ray_to_point(r, s.centre);
    (closest - s.centre).magnitude2() <= s.radius.powi(2)
}

pub fn overlap_capsules(a: Capsule, b: Capsule) -> bool {
    let (closest_a, closest_b) = closest_on_segments(a.segment, b.segment);
    (closest_b - closest_a).magnitude2() <= (a.radius + b.radius).powi(2)
}

pub fn overlap_capsule_ray(c: Capsule, r: Ray) -> bool {
    let (closest_c, closest_r) = closest_on_segment_and_ray(c.segment, r);
    (closest_r - closest_c).magnitude2() <= c.radius.powi(2)
}

pub fn overlap_capsule_sphere(c: Capsule, s: Sphere) -> bool {
    let closest = closest_on_segment_to_point(c.segment, s.centre);
    (closest - s.centre).magnitude2() <= (c.radius + s.radius).powi(2)
}

// GILBERT-JOHNSON-KEERTHI ALGORITHM (GJK)

// Based on Casey Muratori's implementation: caseymuratori.com/blog_0003
pub fn overlap_convex_sets_gilbert_johnson_keerthi<T1: Copy, T2: Copy>(
    max_iterations: usize, normalize_support_argument: bool,
    s1: T1, init1: Point3, support1: impl Fn(T1, Vec3) -> Point3,
    s2: T2, init2: Point3, support2: impl Fn(T2, Vec3) -> Point3,
    mut debug: Option<&mut DebugRender>,
) -> bool {
    // buffer for only 3 simplex points: 4th always stored in local variable
    let mut simplex     = [vec3!(NAN, NAN, NAN); 3];
    let mut simplex_len = 0;

    // macro for writing values to simplex
    macro_rules! set_simplex {
        ($a:expr, $b:expr, $c:expr) => {
            let new_simplex = [$a, $b, $c];
            for i in 0..3 { simplex[i] = new_simplex[i]; }
            simplex_len = 3;
        };
        ($a:expr, $b:expr) => {
            let new_simplex = [$a, $b];
            for i in 0..2 { simplex[i] = new_simplex[i]; }
            simplex_len = 2;
        };
        ($a:expr) => {
            simplex[0] = $a;
            simplex_len = 1;
        };
    };

    // debug visuals setup; causes no allocations if not enabled
    let mut debug_support1_points = Vec::new();
    let mut debug_support2_points = Vec::new();
    if let Some(ref mut debug) = debug {
        debug.reverse_draw_order_begin();

        debug_support1_points.push(init1);
        debug_support2_points.push(init2);

        // draw an approximate outline of the full minkowski difference
        let (x, y) = debug.get_camera().x_y_axes();
        let billboarded_samples = make_arc_points(point3!(VEC3_0), TAU, x, y, 32).map(|point| {
            let direction = vec3!(point);
            let sample    = support1(s1, direction) - support2(s2, -direction);
            point3!(sample)
        }).collect::<Vec<Point3>>();
        debug.draw_line(&color!(0x708090FF).truncate(), 1, &billboarded_samples);
    }

    // macro for generating the minkowski difference samples
    // direction is updated every iteration and depends on the simplex relative to the origin
    let mut direction = init2 - init1;
    macro_rules! sample_minkowski_difference { () => { {
        if normalize_support_argument {
            direction = direction.normalize();
        }
        let support1 = support1(s1, direction);
        let support2 = support2(s2,-direction);

        if debug.is_some() {
            // draw an L shaped line to the next support points on the volumes

            let prev1 = debug_support1_points.last().unwrap();
            let proj1 = prev1 + direction * (direction.dot(support1-prev1)) / direction.magnitude2();
            debug_support1_points.extend_from_slice(&[proj1, support1]);

            let prev2 = debug_support2_points.last().unwrap();
            let proj2 = prev2 + direction * (direction.dot(support2-prev2)) / direction.magnitude2();
            debug_support2_points.extend_from_slice(&[proj2, support2]);
        }
        support1 - support2
    } } }

    // initialise simplex
    set_simplex![sample_minkowski_difference!()];
    direction = -simplex[0];

    let mut i = 0;
    let intersection = loop {
        if i == max_iterations {
            eprintln!("GJK max iterations reached: {}", max_iterations);
            break false;
        }
        i += 1;

        // generate new sample in direction given by last iteration
        let new_point = sample_minkowski_difference!();
        if new_point.dot(direction) < 0.0 {
            // minkowski difference does not contain the origin: no intersection
            break false;
        }
        simplex_len += 1; // add new_point to simplex (written to array after simplex update)

        // macro which all tests in the do_simplex procedures are based off
        // checks whether or not moving from new_point in $dir goes towards the origin
        // this is used to determine which of the simplex's voronoi region the origin is in
        // the voronoi region defines the next iteration's simplex and search direction
        macro_rules! towards_origin { ($dir:expr) => {
            // HACK: works because new_point == $a for all macro invocations in this code
            ($dir).dot(-new_point) > 0.0
        } }

        macro_rules! do_simplex_2 { ($a:expr, $b:expr) => { {
            if let Some(ref mut debug) = debug {
                let line: [Point3; 2] = unsafe { mem::transmute([
                    $a, $b
                ]) };
                debug.draw_line(&color!(0xFF00FFFF).truncate(), 1, &line);
            }

            // check if the step from $b to $a passed the orgin
            let ab = $b - $a;
            if towards_origin!(ab) {
                // origin is between $a and $b: use the line segment next iteration
                // search in direction perpendicular to the line segment, towards the origin
                set_simplex![$a, $b];
                direction = (ab).cross(-$a).cross(ab);
            } else {
                // did not pass the orign: discard $b from next iteration
                // search directly towards origin from $a
                set_simplex![$a];
                direction = -$a;
            }
        } } }

        macro_rules! do_simplex_3 { ($a:expr, $b:expr, $c:expr) => { {
            if let Some(ref mut debug) = debug {
                let triangle: [Point3; 4] = unsafe { mem::transmute([
                    $a, $b, $c, $a,
                ]) };
                debug.draw_line(&color!(0xFFFF00FF).truncate(), 1, &triangle);
            }

            // edge vectors
            let ab  = $b - $a;
            let ac  = $c - $a;

            // face normal (CCW winding order)
            let abc = (ab).cross(ac);

            // first test if origin is in the direction of the ac edge's normal
            // note that ab and ac are not mutually exclusive, so both always need to be tested
            if towards_origin!((abc).cross(ac)) {
                // test two 2-simplices simultaneously, as ab edge is not ruled out yet
                if towards_origin!(ac) {
                    // origin lies between $a and $c
                    set_simplex![$a, $c];
                    direction = (ac).cross(-$a).cross(ac);
                } else {
                    // drop down to 2-simplex case to decide between ab edge, or just $a
                    do_simplex_2![$a, $b];
                }
            } else {
                // ac edge is ruled out, but still need to check ab
                if towards_origin!((ab).cross(abc)) {
                    do_simplex_2![$a, $b];
                } else {
                    // neither edge's normal in direction of origin, so use a 3-simplex next iteration
                    // test which side of the triangle origin is on
                    if towards_origin!(abc) {
                        // top: search in direction of the face normal and use the same simplex
                        set_simplex![$a, $b, $c];
                        direction = abc;
                    } else {
                        // bottom: reverse normal and swap $b and $c to preserve winding order
                        set_simplex![$a, $c, $b];
                        direction = -abc;
                    }
                }
            }
        } } }

        macro_rules! do_simplex_4 { ($a:expr, $b:expr, $c:expr, $d:expr) => { {
            if let Some(ref mut debug) = debug {
                let tetrahedron: [Point3; 8] = unsafe { mem::transmute([
                    $a, $b, $c, $a, $d, $b, $c, $d,
                ]) };
                debug.draw_line(&color!(0x00FF00FF).truncate(), 1, &tetrahedron);
            }

            // edge vectors
            let ab  = $b - $a;
            let ac  = $c - $a;
            let ad  = $d - $a;

            // face normals (CCW winding order)
            let abc = (ab).cross(ac);
            let acd = (ac).cross(ad);
            let adb = (ad).cross(ab);

            // find any face that is looking towards origin, then drop down to 3-simplex procedure
            if      towards_origin!(abc) { do_simplex_3![$a, $b, $c]; }
            else if towards_origin!(acd) { do_simplex_3![$a, $c, $d]; }
            else if towards_origin!(adb) { do_simplex_3![$a, $d, $b]; }
            else {
                // origin is contained within tetrahedron: volumes intersect
                if let Some(ref mut debug) = debug {
                    let to_origin: [Point3; 7] = unsafe { mem::transmute([
                        $a, VEC3_0, $b, VEC3_0, $c, VEC3_0, $d,
                    ]) };
                    debug.draw_line(&color!(0x0000FFFF).truncate(), 1, &to_origin);
                }

                break true;
            }
        } } }

        match simplex_len {
            1 => unreachable!(),
            2 => do_simplex_2![new_point, simplex[0]],
            3 => do_simplex_3![new_point, simplex[0], simplex[1]],
            4 => do_simplex_4![new_point, simplex[0], simplex[1], simplex[2]],
            _ => unreachable!(),
        }
    };

    if let Some(ref mut debug) = debug {
        // finalise debug visuals
        debug.draw_line(&color!(0xFF0000FF).truncate(), 1, &debug_support1_points);
        debug.draw_line(&color!(0x0000FFFF).truncate(), 1, &debug_support2_points);

        debug.reverse_draw_order_end();
    }
    intersection
}

// CONVEX POLYTOPES

/// `Polytope` is defined as the intersection of halfspaces
pub type Polytope = [Plane];

pub fn polytope_contains_point(polytope: &Polytope, point: Point3) -> bool {
    polytope.iter().all(|plane| (plane.normal).dot(vec3!(point)) <= plane.offset)
}

pub fn polytope_from_tetrahedron(tetrahedron: &Tetrahedron) -> [Plane; 4] {
    let mut planes = [Plane::new(VEC3_0, 0.0); 4];
    let triangles = &tetrahedron.get_faces();
    for i in 0..4 {
        planes[i] = Plane::from_triangle(&triangles[i]);
    }
    planes
}

pub fn get_planes_intersection(planes: [Plane; 3]) -> Option<Point3> {
    Mat3::from_cols(planes[0].normal, planes[1].normal, planes[2].normal)
        .transpose()
        .invert()
        .map(|inverse| inverse.transform_point(point3!(
            planes[0].offset, planes[1].offset, planes[2].offset
        )))
}

pub fn polytope_get_supporting_point(polytope: &Polytope, direction: Vec3) -> Option<Point3> {
    // always return a point that is the intersection of 3 planes
    if polytope.len() < 3 {
        return None;
    }

    // get the first combination of planes which intersect on the correct side of the polytope
    let mut intersection  = None;
    let mut planes        = [Plane::new(VEC3_NAN, NAN); 3];
    let mut plane_indices = [!0; 3];

    for i in 0..(polytope.len()-2) {
        let i_facing = polytope[i].normal.dot(direction) >= 0.0;

        for j in i..(polytope.len()-1) {
            let j_facing = polytope[j].normal.dot(direction) >= 0.0;
            if 1 > i_facing as u8 + j_facing as u8 { continue; }

            for k in i..(polytope.len()-0) {
                let k_facing = polytope[k].normal.dot(direction) >= 0.0;
                if 2 > i_facing as u8 + j_facing as u8 + k_facing as u8 { continue; }

                if let Some(point) = get_planes_intersection(planes) {
                    // verify `point` is contained within the other halfspaces of the polytope
                    if polytope_contains_point(&polytope[( 0 )..i], point)
                    && polytope_contains_point(&polytope[(i+1)..j], point)
                    && polytope_contains_point(&polytope[(j+1)..k], point)
                    && polytope_contains_point(&polytope[(k+1).. ], point)
                    {
                        planes        = [polytope[i], polytope[j], polytope[k]];
                        plane_indices = [i, j, k];
                        intersection  = Some(point);
                    }
                }
            }
        }
    }

    // hill-climb along polytope edges
    if let Some(mut intersection) = intersection { loop {
        // determine edge with highest dot product with `direction`
        let mut worst_plane_index = !0;
        let mut best_edge         = VEC3_NAN;
        let mut best_edge_dot     = 0.0; // don't consider edges with negative dot product

        let vertex_normal = planes[0].normal + planes[1].normal + planes[2].normal;
        for i in 0..3 {
            let edge = (planes[(i+1)%3].normal).cross(planes[(i+2)%3].normal);
            // sign of vertex normal dot product corrects `edge` to point away from `intersection`
            let sign = (edge).dot(vertex_normal).signum();
            let dot  = sign * ((edge).dot(direction));
            if dot > best_edge_dot {
                best_edge         = sign*edge;
                best_edge_dot     = dot;
                worst_plane_index = i;
            }
        }

        let edge_ray = if worst_plane_index != !0 {
            Ray { origin: intersection, vector: best_edge }
        } else {
            // no edges had positive dot product: current point is a local (thus global) maximum
            break Some(intersection);
        };

        // find the vertex that the best edge connects to
        let mut min_t = INFINITY;
        for i in 0..polytope.len() {
            if i == plane_indices[0] || i == plane_indices[1] || i == plane_indices[2] {
                // don't try intersecting a plane with itself or the worst plane from before
                continue;
            }
            let plane_dot = polytope[i].normal.dot(edge_ray.vector);
            if plane_dot <= 0.0 {
                // plane must be facing away from the edge to be on the outside of polytope
                continue;
            }
            let t = polytope[i].offset - polytope[i].normal.dot(vec3!(edge_ray.origin)) / plane_dot;
            if t > 0.0 && t < min_t {
                // because `edge_ray` is verified to be contained in/on the polytope, its closest
                // intersection also must be contained
                min_t                            = t;
                intersection                     = edge_ray.eval(t);
                plane_indices[worst_plane_index] = i;
                planes       [worst_plane_index] = polytope[i];
            }
        }
        if min_t == INFINITY {
            // no plane intersected the edge, so the polytope is unbounded in `direction`
            break None;
        }
    } } else {
        // every possibility exhausted: no three intersecting planes
        None
    }
}

// RAYCASTS

#[derive(Copy, Clone, Debug)]
#[repr(C)]
/// `point` is the point at which the ray intersects
/// `point` is produced by evaluating the ray for `depth`
/// `depth` is the distance along the ray in terms of the ray vector's length
/// `depth` will be NaN if the ray does not intersect
/// `depth` will be negative if the intersection is behind the ray's origin
pub struct Raycast {
    pub point: Point3,
    pub depth: Real,
}

impl Raycast {
    pub const NON_INTERSECTION: Self = Raycast { point: point3!(NAN, NAN, NAN), depth: NAN };

    pub fn from_solution(ray: Ray, t: Real) -> Self {
        Raycast { point: ray.eval(t), depth: t }
    }

    pub fn is_intersection(&self) -> bool {
       !self.depth.is_nan()
    }

    pub fn filter_non_intersection(self) -> Option<Self> {
        some_if(self.is_intersection(), self)
    }

    pub fn is_non_negative(&self) -> bool {
        self.depth >= 0.0
    }

    pub fn filter_negative(self) -> Option<Self> {
        some_if(self.is_non_negative(), self)
    }
}

pub fn raycast_plane_single_sided(ray: Ray, plane: Plane) -> Raycast {
    let d = plane.normal.dot(ray.vector);
    if d >= 0.0 {
        Raycast::NON_INTERSECTION
    } else {
        let t = (plane.offset - plane.normal.dot(vec3!(ray.origin))) / d;
        Raycast::from_solution(ray, t)
    }
}

pub fn raycast_plane_double_sided(ray: Ray, plane: Plane) -> Raycast {
    let t = (plane.offset - plane.normal.dot(vec3!(ray.origin))) / plane.normal.dot(ray.vector);
    Raycast::from_solution(ray, t)
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
/// If the ray starts outside the solid, it enters at `fore` and exits at `rear`
/// If the ray starts inside the solid, `fore` and `rear` are the exits in front and behind the ray, respectively
/// If the ray is tangent to the solid's surface, `fore` and `rear` will be equal
/// If the ray does not intersect the solid, `fore` and `rear` will be non-intersection
/// `fore` and `rear` may independently be non-intersection
pub struct RaycastSolid {
    pub fore: Raycast,
    pub rear: Raycast,
}

impl RaycastSolid {
    pub const NON_INTERSECTION: Self = RaycastSolid {
        fore: Raycast::NON_INTERSECTION,
        rear: Raycast::NON_INTERSECTION,
    };

    pub fn from_fore_and_rear_solutions(ray: Ray, fore_t: Real, rear_t: Real) -> Self {
        RaycastSolid {
            fore: Raycast::from_solution(ray, fore_t),
            rear: Raycast::from_solution(ray, rear_t),
        }
    }
}

pub fn raycast_sphere(ray: Ray, sphere: Sphere) -> RaycastSolid {
    // solve quadratic: (v . v)t^2 + 2(m . v)t + (m . m) + r^2 = 0
    let m = ray.origin - sphere.centre;

    let a = ray.vector.magnitude2();
    let b = m.dot(ray.vector);
    let c = m.magnitude2() - sphere.radius.powi(2);

    let d = b.powi(2) - a*c;
    if d < 0.0 {
        RaycastSolid::NON_INTERSECTION
    } else {
        let e = d.sqrt();

        let s = (-b - e) / a;
        let t = (-b + e) / a;
        if s >= 0.0 {
            RaycastSolid::from_fore_and_rear_solutions(ray, s, t)
        } else {
            RaycastSolid::from_fore_and_rear_solutions(ray, t, s)
        }
    }
}

pub fn raycast_sphere_normalized(ray_normalized: Ray, sphere: Sphere) -> RaycastSolid {
    let ray = ray_normalized;
    // solve quadratic: t^2 + 2(m . v)t + (m . m) + r^2 = 0
    let m = ray.origin - sphere.centre;

    //  a = 1.0
    let b = m.dot(ray.vector);
    let c = m.magnitude2() - sphere.radius.powi(2);

    let d = b.powi(2) - c;
    if d < 0.0 {
        RaycastSolid::NON_INTERSECTION
    } else {
        let e = d.sqrt();

        let s = -b - e;
        let t = -b + e;
        if s >= 0.0 {
            RaycastSolid::from_fore_and_rear_solutions(ray, s, t)
        } else {
            RaycastSolid::from_fore_and_rear_solutions(ray, t, s)
        }
    }
}
