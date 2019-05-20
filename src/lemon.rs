use super::*;

#[derive(Copy, Clone, Debug)]
pub struct Lemon {
    /// Radius of the sphere swept around the focus
    pub radius:  f32,

    /// Sagitta of the arc; radius of the widest horizontal of the lemon
    pub sagitta: f32,

    /// Scale factor of lemon. Corresponds to half the height.
    pub scale:   f32,

    pub phys: Rigidbody,
}

impl Lemon {
// https://www.desmos.com/calculator/z7ijifw8pc
// https://www.researchgate.net/publication/2173504
    pub const DENSITY: f32 = 10.0 * KILOGRAM / METER / METER / METER;

    pub fn new(s: f32, scale: f32) -> Self {
        if s <= 0.0 || scale <= 0.0 {
            panic!("non-positive lemon size parameter(s)");
        } else if s > 1.0 {
            panic!("lemon sagitta greater than 1.0");
        }

        let normalized = NormalizedLemon::new(s);
        let (r, _, _)  = normalized.rst();

        let (mass, i_z, i_y) = normalized.eval_definite_integral_mass_intertia_z_y_n1_p1();
        let mut phys         = Rigidbody::default();
        phys.mass            = scale.powi(3) * Lemon::DENSITY * mass;
        phys.inertia_local   = scale.powi(3) * Lemon::DENSITY * vec3!(i_y, i_y, i_z);

        Lemon { radius: r*scale, sagitta: s*scale, scale, phys }
    }

    pub fn with_sagitta_and_half_height(sagitta: f32, half_height: f32) -> Self {
        Lemon::new(sagitta / half_height, half_height)
    }

    #[inline]
    pub fn focal_radius(&self) -> f32 {
        self.radius - self.sagitta
    }

    #[inline]
    pub fn get_vertical(&self) -> Vec3 {
        self.phys.orientation * VEC3_Z
    }

    #[inline]
    pub fn get_transform_with_scale(&self) -> Mat4 {
        self.phys.get_transform() * Mat4::from_scale(self.scale)
    }

    #[inline]
    pub fn get_bounding_capsule(&self) -> Capsule {
        let half = (self.scale - self.sagitta) * self.get_vertical();
        Capsule {
            line: (self.phys.position - half, self.phys.position + half),
            radius: self.sagitta,
        }
    }

    pub fn get_normalized(&self) -> NormalizedLemon {
        NormalizedLemon {
            r: self.radius         / self.scale,
            s: self.sagitta        / self.scale,
            t: self.focal_radius() / self.scale,
        }
    }

    pub fn mutate_shape(&mut self, s: f32, scale: f32) {
        let old_phys = self.phys;
        let old_angular_velocity = self.phys.get_inertia_inverse()
                                 * self.phys.angular_momentum;
        *self = Lemon::new(s, scale);
        self.phys.velocity         = old_phys.velocity;
        self.phys.position         = old_phys.position;
        self.phys.orientation      = old_phys.orientation;
        self.phys.angular_momentum = self.phys.get_inertia() * old_angular_velocity;
    }
}

pub const MESH_RESOLUTION_T:      usize = 32;
pub const MESH_RESOLUTION_Z:      usize = MESH_RESOLUTION_Z_HALF * 2 - 1;
pub const MESH_RESOLUTION_Z_HALF: usize = 16;

pub struct LemonBaseMesh {
    /// Vertex base position attribute.
    pub points:    Vec<Point3>,

    /// Mesh polygon index list.
    pub indices: Vec<ElementIndex>,
}

// essentially just a cone-capped cylinder
pub fn make_base_mesh() -> LemonBaseMesh {
    // choose vertex points for a uniform arc-length
    let mut vertex_points = [0.0; MESH_RESOLUTION_Z_HALF];
    for (i, z) in vertex_points.iter_mut().enumerate() {
        *z = (i as f32 / (MESH_RESOLUTION_Z_HALF-1) as f32 * TAU/4.0).sin();
    }

    // compute vertices
    let total_vertices = 2 + (MESH_RESOLUTION_Z - 2) * MESH_RESOLUTION_T;
    let mut points    = Vec::with_capacity(total_vertices);

    points.push(point3!(0.0, 0.0, -1.0));
    for i_z in 1..(MESH_RESOLUTION_Z-1) {
        let (z_index, z_sign) = {
            if i_z >= MESH_RESOLUTION_Z_HALF {
                (i_z - (MESH_RESOLUTION_Z_HALF-1),  1.0)
            } else {
                ((MESH_RESOLUTION_Z_HALF-1) - i_z, -1.0)
            }
        };
        assert!(z_index <= MESH_RESOLUTION_Z_HALF - 2, "unexpected z_index: {}", z_index);

        for i_theta in 0..MESH_RESOLUTION_T {
            // i_z term staggers polygon strips so adjacent triangles do not lie on a plane
            let theta = i_theta   as f32 * TAU / MESH_RESOLUTION_T as f32
                      + 0.5 * i_z as f32 * TAU / MESH_RESOLUTION_T as f32;
            points.push(point3!(
                theta.cos(),
                theta.sin(),
                z_sign * vertex_points[z_index]
            ));
        }
    }
    points.push(point3!(0.0, 0.0, 1.0));
    assert_eq!(points.len(), total_vertices, "incorrect total_vertices");

    // compute vertex list
    let total_indices = 3 * MESH_RESOLUTION_T * (2 + 2 * (MESH_RESOLUTION_Z-3));
    let mut indices = Vec::with_capacity(total_indices);
    // base cone
    for i_theta in 0..(MESH_RESOLUTION_T as ElementIndex) {
        const BASE:   ElementIndex = 1;
        const VERTEX: ElementIndex = 0;

        indices.push(BASE + i_theta);
        indices.push(VERTEX);
        indices.push(BASE + (i_theta+1) % MESH_RESOLUTION_T as ElementIndex);
    }
    // main cylinder body
    for i_z in 0..(MESH_RESOLUTION_Z as ElementIndex - 3) {
        let i_z_base      = i_z      * MESH_RESOLUTION_T as ElementIndex + 1;
        let i_z_base_next = i_z_base + MESH_RESOLUTION_T as ElementIndex;
        for i_theta in 0..MESH_RESOLUTION_T as ElementIndex {
            let i_theta_next = (i_theta+1) % MESH_RESOLUTION_T as ElementIndex;

            indices.push(i_z_base      + i_theta);
            indices.push(i_z_base      + i_theta_next);
            indices.push(i_z_base_next + i_theta);

            indices.push(i_z_base_next + i_theta);
            indices.push(i_z_base      + i_theta_next);
            indices.push(i_z_base_next + i_theta_next);
        }
    }
    // base cone
    for i_theta in 0..(MESH_RESOLUTION_T as ElementIndex) {
        const BASE:   ElementIndex = ((MESH_RESOLUTION_Z - 3) * MESH_RESOLUTION_T + 1) as _;
        const VERTEX: ElementIndex = BASE + MESH_RESOLUTION_T as ElementIndex;

        indices.push(BASE + (i_theta+1) % MESH_RESOLUTION_T as ElementIndex);
        indices.push(VERTEX);
        indices.push(BASE + i_theta);
    }
    assert_eq!(indices.len(), total_indices, "incorrect total_indices");

    LemonBaseMesh { points, indices, }
}

pub const MAP_RESOLUTION: usize = 512;

pub fn make_radius_normal_z_map() -> Vec<[Vec2; MAP_RESOLUTION]> {
    let mut tex = Vec::<[Vec2; MAP_RESOLUTION]>::with_capacity(MAP_RESOLUTION);
    unsafe { tex.set_len(MAP_RESOLUTION); }

    for i_s in 0..MAP_RESOLUTION {
        let s     = (i_s+1) as f32 / MAP_RESOLUTION as f32;
        let lemon = NormalizedLemon::new(s);

        for i_z in 0..(MAP_RESOLUTION-1) {
            // sqrt here to effectively increase sample density as z -> 1.
            // has a corresponding square in lemon vertex shader.
            let z = (i_z as f32 / (MAP_RESOLUTION - 1) as f32).sqrt();
            let (radius, gradient) = lemon.eval_radius_gradient(z);
            tex[i_s][i_z] = vec2!(radius, -gradient);
        }
        tex[i_s][MAP_RESOLUTION-1] = vec2!(0.0, f32::MIN);
    }
    tex
}

pub struct NormalizedLemon {
    pub r: f32,
    pub s: f32,
    pub t: f32,
}

impl NormalizedLemon {
    pub fn new(s: f32) -> Self {
        let r = (s*s + 1.0) / 2.0 / s;
        let t = r - s;
        NormalizedLemon { r, s, t }
    }

    pub fn rst(&self) -> (f32, f32, f32) {
        (self.r, self.s, self.t)
    }

    pub fn eval_radius_gradient(&self, z: f32) -> (f32, f32) {
        let (r, s, t) = self.rst();
        let (z2, r2)  = (z*z, r*r);
        let u         = (r2 - z2).sqrt();

        (u - t, -z / u)
    }

    pub fn eval_integral_mass_intertia_z_y(&self, z: f32) -> (f32, f32, f32) {
        let (r,  s,  t)  = self.rst();
        let (r2, s2, t2) = (r*r, s*s, t*t);
        let (z2, z3, z5) = (z*z, z*z*z, z*z*z*z*z);

        let u  = (r2 - z2).sqrt();
        let su = u + r2 * (1.0/r).asin();

        let mass       = PI * ( z3/3.0
                              + z*(r2 + t2)
                              - t*su );

        let inertia_z  = PI / 2.0 * ( z5/5.0
                                    - 2.0*z3*(r2 + t2/3.0)
                                    + z*(r2*r2 + 6.0*r2*t2 + t2*t2 - (r2 - z2)*u)
                                    - t*(2.0*t2 + 3.0/2.0*r2)*su );

        let inertia_y  = inertia_z / 2.0
                       - PI * (-z5/5.0
                              + z3/3.0*(r2 + t2)
                              - 2.0*t*(r2/8.0*su - z/4.0*u*u*u) );
        (mass, inertia_z, inertia_y)
    }

    pub fn eval_definite_integral_mass_intertia_z_y_n1_p1(&self) -> (f32, f32, f32) {
        let (r,  s,  t)  = self.rst();
        let (r2, s2, t2) = (r*r, s*s, t*t);

        let u  = (r2 - 1.0).sqrt();       // = u(1) =  u(-1) // <= u(z) is even
        let su = u + r2 * (1.0/r).asin(); // = U(1) = -U(-1) // <= U(z) is odd

        let mass       = PI * 2.0 * ( -1.0/3.0
                                    + (r2 + t2)
                                    - t*su );

        let inertia_z  = PI * ( 0.2
                              - 2.0*(r2/3.0 + t2)
                              + (r2*r2 + 6.0*r2*t2 + t2*t2)
                              - t * ( u  * (r2-1.0)
                                    + su * (2.0*t2 + 3.0/2.0*r2) ) );

        let inertia_y  = inertia_z / 2.0
                       + PI * 2.0 * ( -0.2
                                    + (r2 + t2)/3.0
                                    - 2.0 * t * (r2/8.0*su - u*u*u/4.0) );
        (mass, inertia_z, inertia_y)
    }
}

#[inline]
fn is_edge_case(
    lemon: &Lemon, lemon_vertical: Vec3, lemon_sphere: Point3, displacement: Vec3,
) -> bool {
    let lemon_sphere_relative = lemon_sphere - lemon.phys.position;
    // test that the sign of the angle from upper_bound to lemon_sphere_relative
    // is the same as from upper_bound to displacement
    let upper_bound     = lemon_vertical * lemon.scale - lemon_sphere_relative;
    let reference_cross = upper_bound.cross(-lemon_sphere_relative);
    let test_cross      = upper_bound.cross(displacement);
    test_cross.dot(reference_cross) < 0.0
}

pub fn get_collision_halfspace(
    lemon: &Lemon,
    plane: (Vec3, f32),
    mut debug: Option<&mut DebugRender>,
) -> Option<Collision> {
    let lemon_vertical = {
        let lemon_vertical = lemon.get_vertical();
        lemon_vertical * lemon_vertical.dot(plane.0).signum().neg()
    };
    let lemon_sphere = lemon.phys.position + {
        let proj = proj_onto_plane(plane.0, lemon_vertical);
        if proj == vec3!(0.0, 0.0, 0.0) {
            // focus parallel to plane: early return halfspace-sphere test (radius = scale)
            let centre_distance = vec3!(lemon.phys.position).dot(plane.0) - plane.1;

            if let Some(ref mut debug) = debug {
                let lemon_vertex = lemon.phys.position + lemon_vertical * lemon.scale;
                debug.draw_ray(&color!(0x0000FFFF).truncate(), 1,
                    &lemon_vertex, &(plane.0 * (lemon.scale - centre_distance)),
                );
            }
            return some_if_then(centre_distance <= lemon.scale, || {
                Collision {
                    point:  lemon.phys.position - plane.0 * (centre_distance + lemon.scale)/2.0,
                    normal: plane.0,
                    depth:  lemon.scale - centre_distance
                }
            })
        } else { proj.normalize_to(lemon.focal_radius()) }
    };

    if let Some(ref mut debug) = debug {
        let color = color!(0x708090FF).truncate();
        debug.draw_line(&color, 1, &[
            lemon.phys.position + lemon_vertical * lemon.scale, lemon_sphere,
            lemon.phys.position - lemon_vertical * lemon.scale, lemon_sphere,
            proj_point_onto_plane(lemon_sphere, plane),
        ]);
        debug.draw_line(&color, 1, &make_line_strip_circle(
            lemon_sphere, lemon_vertical.cross(plane.0).normalize(), lemon.radius, 31,
        ));
    };

    if is_edge_case(&lemon, lemon_vertical, lemon_sphere, plane.0.neg()) {
        // halfspace-point test
        let lemon_vertex = lemon.phys.position + lemon_vertical * lemon.scale;
        let dot          = vec3!(lemon_vertex).dot(plane.0);

        if let Some(ref mut debug) = debug {
            debug.draw_ray(&color!(0xBF00BFFF).truncate(), 1,
                &lemon_vertex, &(plane.0 * (plane.1 - dot)),
            );
        }
        some_if_then(dot <= plane.1, || {
            let depth = plane.1 - dot;
            Collision {
                point:  lemon_vertex + plane.0 * depth / 2.0,
                normal: plane.0,
                depth:  depth,
            }
        })
    } else {
        // halfspace-sphere test
        let sphere_distance = vec3!(lemon_sphere).dot(plane.0) - plane.1;
        some_if_then(sphere_distance <= lemon.radius, || {
            Collision {
                point:  lemon_sphere - plane.0 * (sphere_distance + lemon.radius)/2.0,
                normal: plane.0,
                depth:  lemon.radius - sphere_distance,
            }
        })
    }
}

pub fn get_collision_lemon(
    lemon: &Lemon,
    other: &Lemon,
    mut debug: Option<&mut DebugRender>,
) -> Option<Collision> {
    let lemon_vertical = lemon.get_vertical();
    let lemon_focus    = (lemon.phys.position, lemon_vertical, lemon.focal_radius());

    let other_vertical = other.get_vertical();
    let other_focus    = (other.phys.position, other_vertical, other.focal_radius());

    // find furthest points on focii for general case collision test
    let (lemon_sphere, other_sphere, displacement, distance2) = {
        // TODO: special case for when displacement parallel to single vertical
        let parallel = lemon_vertical.cross(other_vertical) == vec3!(0.0, 0.0, 0.0);
        if parallel {
            // algebraic solution available when focus normals are parallel
            let centroid_displacement = other.phys.position - lemon.phys.position;
            let syzygy = lemon_vertical.cross(centroid_displacement) == vec3!(0.0, 0.0, 0.0);
            if syzygy {
                // early return trivial case where displacement parallel to focus normals
                // iterative algorithm cannot converge with this input
                if let Some(ref mut debug) = debug {
                    let color = color!(0x0000FFFF).truncate();
                    let lemon_vertex = lemon.phys.position + lemon_vertical * lemon.scale
                                     * lemon_vertical.dot(centroid_displacement).signum();
                    let other_vertex = other.phys.position - other_vertical * other.scale
                                     * other_vertical.dot(centroid_displacement).signum();
                    debug.draw_line(&color, 1, &[lemon_vertex, other_vertex]);
                }

                let distance = centroid_displacement.dot(lemon_vertical).abs();
                return some_if_then(distance <= lemon.scale + other.scale, || Collision {
                    point:  lemon.phys.position + centroid_displacement / 2.0,
                    normal: centroid_displacement.neg() / distance,
                    depth:  lemon.scale + other.scale - distance,
                });
            } else {
                let lemon_sphere = lemon.phys.position
                                 - proj_onto_plane(centroid_displacement, lemon_vertical)
                                   .normalize_to(lemon.focal_radius());
                let other_sphere = other.phys.position
                                 + proj_onto_plane(centroid_displacement, other_vertical)
                                   .normalize_to(other.focal_radius());
                let displacement = other_sphere - lemon_sphere;
                let distance2    = displacement.magnitude2();
                (lemon_sphere, other_sphere, displacement, distance2)
            }
        } else {
            // iteratively find furthest points on focus radii
            // based on algorithm described here:
            // https://www.sciencedirect.com/science/article/pii/S0307904X0200080X
            // it also cites a result stating the following has no closed-form solution
            const MAX_ITERATIONS: usize = 50;
            const EPSILON: f32 = 0.0000;

            let mut lemon_sphere = furthest_on_circle_from_point(
                lemon_focus, other.phys.position,
            );
            let mut other_sphere = furthest_on_circle_from_point(
                other_focus, lemon.phys.position,
            );
            let mut displacement = other_sphere - lemon_sphere;
            let mut distance2    = displacement.magnitude2();

            let mut iterations = 0;
            loop {
                let distance2_prev = distance2;
                iterations  += 1;
                lemon_sphere = furthest_on_circle_from_point(lemon_focus, other_sphere);
                other_sphere = furthest_on_circle_from_point(other_focus, lemon_sphere);
                displacement = other_sphere - lemon_sphere;
                distance2    = displacement.magnitude2();

                let epsilon = distance2 - distance2_prev;
                if epsilon <= EPSILON {
                    break;
                }
                if iterations >= MAX_ITERATIONS {
                    // indicate error to better profile this edge case
                    break;
                }
            }
            (lemon_sphere, other_sphere, displacement, distance2)
        }
    };

    #[derive(Copy, Clone, Debug)]
    enum CollisionTest {
        Sphere(Point3),
        Vertex(Point3),
    }
    use CollisionTest::{Sphere, Vertex};

    // TODO: replace multiple per-lemon arguments with a struct?
    type Circle = (Point3, Vec3, f32); // for furthest_on_circle_from_point function
    #[inline]
    fn resolve_collision_test(
        lemon: &Lemon, lemon_focus: Circle, lemon_vertical: Vec3, lemon_test: CollisionTest,
        other: &Lemon, other_focus: Circle, other_vertical: Vec3, other_test: CollisionTest,
        displacement: Vec3, distance2: f32,
        mut debug: Option<&mut DebugRender>,
    ) -> Option<Collision> { match (lemon_test, other_test) {

        (Sphere(lemon_sphere), Sphere(other_sphere)) => {
            // curved surface to curved surface
            // also handles vertex-vertex collision
            if let Some(ref mut debug) = debug {
                let color = color!(0x708090FF).truncate();
                debug.draw_line(&color, 1, &[
                    lemon.phys.position + lemon_vertical * lemon.scale, lemon_sphere,
                    lemon.phys.position - lemon_vertical * lemon.scale, lemon_sphere,
                    other_sphere, other.phys.position + other_vertical * other.scale,
                    other_sphere, other.phys.position - other_vertical * other.scale,
                ]);
                debug.draw_line(&color, 1, &make_line_strip_circle(
                    lemon_sphere, lemon_vertical.cross(displacement).normalize(),
                    lemon.radius, 63,
                ));
                debug.draw_line(&color, 1, &make_line_strip_circle(
                    other_sphere, other_vertical.cross(displacement).normalize(),
                    other.radius, 63,
                ));
            }

            // test edge cases
            let lemon_edge = is_edge_case(lemon, lemon_vertical, lemon_sphere, displacement);
            let other_edge = is_edge_case(other, other_vertical, other_sphere,-displacement);

            if lemon_edge && other_edge {
                let lemon_vertex = lemon.phys.position + lemon_vertical * lemon.scale;
                let other_vertex = other.phys.position + other_vertical * other.scale;
                let displacement = other_vertex - lemon_vertex;
                let distance2    = displacement.magnitude2();

                resolve_collision_test(
                    lemon, lemon_focus, lemon_vertical, Vertex(lemon_vertex),
                    other, other_focus, other_vertical, Vertex(other_vertex),
                    displacement, distance2,
                    debug,
                )
            } else if lemon_edge {
                let lemon_vertex = lemon.phys.position + lemon_vertical * lemon.scale;
                let other_sphere = furthest_on_circle_from_point(other_focus, lemon_vertex);
                let displacement = other_sphere - lemon_vertex;
                let distance2    = displacement.magnitude2();

                resolve_collision_test(
                    lemon, lemon_focus, lemon_vertical, Vertex(lemon_vertex),
                    other, other_focus, other_vertical, Sphere(other_sphere),
                    displacement, distance2,
                    debug,
                )
            } else if other_edge {
                let other_vertex = other.phys.position + other_vertical * other.scale;
                let lemon_sphere = furthest_on_circle_from_point(lemon_focus, other_vertex);
                let displacement = other_vertex - lemon_sphere;
                let distance2    = displacement.magnitude2();

                resolve_collision_test(
                    lemon, lemon_focus, lemon_vertical, Sphere(lemon_sphere),
                    other, other_focus, other_vertical, Vertex(other_vertex),
                    displacement, distance2,
                    debug,
                )
            } else {
                some_if_then(distance2 <= (lemon.radius + other.radius).powi(2), || {
                    let radii = lemon.radius + other.radius;
                    Collision {
                        point:  lemon_sphere + displacement * lemon.radius / radii,
                        normal: displacement.normalize().neg(),
                        depth:  radii - distance2.sqrt(),
                    }
                })
            }
        },

        (Sphere(lemon_sphere), Vertex(other_vertex)) => {
            // vertex touching curved surface
            if let Some(ref mut debug) = debug {
                let color = color!(0xBF00BFFF).truncate();
                debug.draw_line(&color, 1, &[
                    lemon.phys.position + lemon_vertical * lemon.scale, lemon_sphere,
                    lemon.phys.position - lemon_vertical * lemon.scale, lemon_sphere,
                    other_vertex,
                ]);
                debug.draw_line(&color, 1, &make_line_strip_circle(
                    lemon_sphere, lemon_vertical.cross(displacement).normalize(),
                    lemon.radius, 63,
                ));
            }
            if is_edge_case(lemon, lemon_vertical, lemon_sphere, displacement) {
                let lemon_vertex = lemon.phys.position + lemon_vertical * lemon.scale;
                let displacement = other_vertex - lemon_vertex;
                let distance2    = displacement.magnitude2();

                resolve_collision_test(
                    lemon, lemon_focus, lemon_vertical, Vertex(lemon_vertex),
                    other, other_focus, other_vertical, Vertex(other_vertex),
                    displacement.neg(), distance2,
                    debug,
                )
            } else {
                some_if_then(distance2 <= lemon.radius * lemon.radius, || {
                    let distance = distance2.sqrt();
                    Collision {
                        point:  lemon_sphere + (lemon.radius + distance)/2.0
                                             * displacement/distance,
                        normal: displacement.normalize().neg(),
                        depth:  lemon.radius - distance,
                    }
                })
            }
        },

        (Vertex(lemon_vertex), Sphere(other_sphere)) => {
            // same as previous case, but reverse order and negate result
            resolve_collision_test(
                other, other_focus, other_vertical, Sphere(other_sphere),
                lemon, lemon_focus, lemon_vertical, Vertex(lemon_vertex),
                displacement.neg(), distance2,
                debug,
            ).map(Collision::neg)
        },

        (Vertex(lemon_vertex), Vertex(other_vertex)) => {
            // this case is required for closest-point, but never returns collision.
            // vertex-vertex collisions are actually handled by the general case or the
            // parallel normal and displacement case before sphere points are found.
            if let Some(ref mut debug) = debug {
                let color = color!(0x0000FFFF).truncate();
                debug.draw_line(&color, 1, &[lemon_vertex, other_vertex]);
            }
            None
        },
    } }

    let lemon_vertical_signed = lemon_vertical * lemon_vertical.dot(displacement).signum();
    let other_vertical_signed = other_vertical *-other_vertical.dot(displacement).signum();
    resolve_collision_test(
        &lemon, lemon_focus, lemon_vertical_signed, Sphere(lemon_sphere),
        &other, other_focus, other_vertical_signed, Sphere(other_sphere),
        displacement, distance2,
        debug,
    )
}
