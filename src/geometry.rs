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
