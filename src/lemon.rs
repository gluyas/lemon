use super::*;

#[derive(Copy, Clone, Debug)]
pub struct Lemon {
    /// Radius of the sphere swept around the focus
    pub radius:  Real,

    /// Sagitta of the arc; radius of the widest horizontal of the lemon
    pub sagitta: Real,

    /// Scale factor of lemon. Corresponds to half the height.
    pub scale:   Real,

    pub color: Vec3,
    pub phys:  Rigidbody,
}

impl Lemon {
// https://www.desmos.com/calculator/z7ijifw8pc
// https://www.researchgate.net/publication/2173504
    pub const DENSITY: Real = 10.0 * KILOGRAM / METER / METER / METER;

    pub fn new(s: Real, scale: Real, color: Vec3) -> Self {
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

        Lemon { radius: r*scale, sagitta: s*scale, scale, color, phys }
    }

    #[inline]
    pub fn focal_radius(&self) -> Real {
        self.radius - self.sagitta
    }

    #[inline]
    pub fn get_vertical(&self) -> Vec3 {
        self.phys.orientation * VEC3_Z
    }

    #[inline]
    pub fn get_focus(&self) -> Disc {
        Disc {
            centre: self.phys.position,
            normal: self.get_vertical(),
            radius: self.focal_radius()
        }
    }

    #[inline]
    pub fn get_transform_with_scale(&self) -> Mat4 {
        self.phys.get_transform() * Mat4::from_scale(self.scale)
    }

    #[inline]
    pub fn get_bounding_capsule(&self) -> Capsule {
        let half = (self.scale - self.sagitta) * self.get_vertical();
        Capsule {
            segment: Segment::new(self.phys.position - half, self.phys.position + half),
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

    pub fn mutate_shape(&mut self, s: Real, scale: Real) {
        let old_phys = self.phys;
        let old_angular_velocity = self.phys.get_inertia_inverse()
                                 * self.phys.angular_momentum;
        *self = Lemon::new(s, scale, self.color);
        self.phys.velocity         = old_phys.velocity;
        self.phys.position         = old_phys.position;
        self.phys.orientation      = old_phys.orientation;
        self.phys.angular_momentum = self.phys.get_inertia() * old_angular_velocity;
    }

    pub fn eval_radius_gradient(&self, z: Real) -> (Real, Real) {
        let root     = (self.radius.powi(2) - z.powi(2)).sqrt();
        let radius   = root - self.focal_radius();
        let gradient = -z / root;
        (radius, gradient)
    }

    pub fn eval_radius_inverse(&self, radius: Real) -> Real {
        let c2 = (self.focal_radius() + radius).powi(2);
        let r2 = self.radius.powi(2);
        (r2 - c2).sqrt()
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
        *z = (i as Real / (MESH_RESOLUTION_Z_HALF-1) as Real * TAU/4.0).sin();
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
            let theta = i_theta   as Real * TAU / MESH_RESOLUTION_T as Real
                      + 0.5 * i_z as Real * TAU / MESH_RESOLUTION_T as Real;
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
        let s     = (i_s+1) as Real / MAP_RESOLUTION as Real;
        let lemon = NormalizedLemon::new(s);

        for i_z in 0..(MAP_RESOLUTION-1) {
            // sqrt here to effectively increase sample density as z -> 1.
            // has a corresponding square in lemon vertex shader.
            let z = (i_z as Real / (MAP_RESOLUTION - 1) as Real).sqrt();
            let (radius, gradient) = lemon.eval_radius_gradient(z);
            tex[i_s][i_z] = vec2!(radius, -gradient);
        }
        tex[i_s][MAP_RESOLUTION-1] = vec2!(0.0, f32::MIN);
    }
    tex
}

pub struct NormalizedLemon {
    pub r: Real,
    pub s: Real,
    pub t: Real,
}

impl NormalizedLemon {
    pub fn new(s: Real) -> Self {
        let r = (s*s + 1.0) / 2.0 / s;
        let t = r - s;
        NormalizedLemon { r, s, t }
    }

    pub fn rst(&self) -> (Real, Real, Real) {
        (self.r, self.s, self.t)
    }

    pub fn eval_radius_gradient(&self, z: Real) -> (Real, Real) {
        let (r, s, t) = self.rst();
        let (z2, r2)  = (z*z, r*r);
        let u         = (r2 - z2).sqrt();

        (u - t, -z / u)
    }

    pub fn eval_integral_mass_intertia_z_y(&self, z: Real) -> (Real, Real, Real) {
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

    pub fn eval_definite_integral_mass_intertia_z_y_n1_p1(&self) -> (Real, Real, Real) {
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

    pub fn eval_radius_inverse(&self, radius: Real) -> Real {
        let c2 = (self.t + radius).powi(2);
        let r2 = self.r.powi(2);
        (r2 - c2).sqrt()
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
    plane: (Vec3, Real),
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
    let lemon_focus    = lemon.get_focus();
    let lemon_vertical = lemon_focus.normal;

    let other_focus    = other.get_focus();
    let other_vertical = other_focus.normal;

    let combined_radii2 = (lemon.radius + other.radius).powi(2);
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
            let (lemon_sphere, other_sphere, distance2, (epsilon, _)) = {
                find_furthest_and_distance2_on_circles(
                    lemon_focus, other_focus,
                    // cap at combined_radii distance: see early-out in resolve_collision
                    // TODO: seed algorithm with previous frame's result
                    None, 0.0, (lemon.radius + other.radius).powi(2), 25,
                )
            };
            let displacement = other_sphere - lemon_sphere;
            (lemon_sphere, other_sphere, displacement, distance2)
        }
    };

    #[derive(Copy, Clone, Debug)]
    enum CollisionTest {
        Sphere(Point3),
        Vertex(Point3),
    }
    use CollisionTest::{Sphere, Vertex};

    #[inline(always)]
    fn resolve_collision(
        lemon: &Lemon, lemon_focus: Disc, lemon_vertical: Vec3, lemon_test: CollisionTest,
        other: &Lemon, other_focus: Disc, other_vertical: Vec3, other_test: CollisionTest,
        displacement: Vec3, distance2: Real,
        mut debug: Option<&mut DebugRender>,
    ) -> Option<Collision> { match (lemon_test, other_test) {

        (Sphere(lemon_sphere), Sphere(other_sphere)) => {
            // curved surface to curved surface
            // also handles vertex-vertex collision
            if let Some(ref mut debug) = debug {
                let color = color!(0x708090FF).truncate();
                debug.draw_line(&color, 1, &make_line_strip_circle(
                    lemon_sphere, lemon_vertical.cross(displacement).normalize(),
                    lemon.radius, 63,
                ));
                debug.draw_line(&color, 1, &make_line_strip_circle(
                    other_sphere, other_vertical.cross(displacement).normalize(),
                    other.radius, 63,
                ));
                debug.draw_line(&color, 1, &[
                    lemon.phys.position + lemon_vertical * lemon.scale, lemon_sphere,
                    lemon.phys.position - lemon_vertical * lemon.scale, lemon_sphere,
                    other_sphere, other.phys.position + other_vertical * other.scale,
                    other_sphere, other.phys.position - other_vertical * other.scale,
                ]);
            }
            // early out: lemons too far to possibly collide
            if distance2 > (lemon.radius + other.radius).powi(2) {
                return None;
            }

            // test edge cases
            let lemon_edge = is_edge_case(lemon, lemon_vertical, lemon_sphere, displacement);
            let other_edge = is_edge_case(other, other_vertical, other_sphere,-displacement);

            if lemon_edge && other_edge {
                let lemon_vertex = lemon.phys.position + lemon_vertical * lemon.scale;
                let other_vertex = other.phys.position + other_vertical * other.scale;
                let displacement = other_vertex - lemon_vertex;
                let distance2    = displacement.magnitude2();

                resolve_collision(
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

                resolve_collision(
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

                resolve_collision(
                    lemon, lemon_focus, lemon_vertical, Sphere(lemon_sphere),
                    other, other_focus, other_vertical, Vertex(other_vertex),
                    displacement, distance2,
                    debug,
                )
            } else {
                let combined_radii = lemon.radius + other.radius;
                Some(Collision {
                    point:  lemon_sphere + displacement * lemon.radius / combined_radii,
                    normal: displacement.normalize().neg(),
                    depth:  combined_radii - distance2.sqrt(),
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

                resolve_collision(
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
            resolve_collision(
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
    resolve_collision(
        &lemon, lemon_focus, lemon_vertical_signed, Sphere(lemon_sphere),
        &other, other_focus, other_vertical_signed, Sphere(other_sphere),
        displacement, distance2,
        debug,
    )
}

pub fn raycast(mut ray: Ray, lemon: &Lemon) -> RaycastSolid {
    const EPSILON:        f32   = 1.0e-5;
    const MAX_ITERATIONS: usize = 25;

    let lemon_vertical = lemon.get_vertical();

    let ray_vector_dot_lemon_vertical = ray.vector.dot(lemon_vertical);
    if (ray_vector_dot_lemon_vertical.powi(2) / ray.vector.magnitude2()).abs() <= EPSILON {
        // perpendicular vectors: project ray onto axis and defer to sphere raycast
        let t = (ray.origin - lemon.phys.position).dot(lemon_vertical);
        return if t.abs() > lemon.scale {
            // ray above or below lemon
            RaycastSolid::NON_INTERSECTION
        } else {
            raycast_sphere(ray, Sphere {
                centre: lemon.phys.position + t * lemon_vertical,
                radius: lemon.eval_radius_gradient(t).0,
            })
        };
    }

    let offset = -vec3!(lemon.phys.position);
    ray.origin += offset;

    let j = ( 1.0 // = lemon_vertical.dot(lemon_vertical)
            / ray_vector_dot_lemon_vertical
            * ray.vector )
          - lemon_vertical;

    let k = vec3!(ray.origin)
          - ( vec3!(ray.origin).dot(lemon_vertical)
            / ray_vector_dot_lemon_vertical
            * ray.vector );

    // h(x) = sqrt(h_a*x^2 + 2*h_b*x + h_c)
    let h_a = j.dot(j);
    let h_b = j.dot(k);
    let h_c = k.dot(k);

    let (x1, x2) = if h_a.abs() <= EPSILON {
        // parallel vectors: solve f(x) = sqrt(c)
        let x = lemon.eval_radius_inverse(h_c.sqrt());
        if x.is_nan() {
            return RaycastSolid::NON_INTERSECTION
        } else {
            (-x, x)
        }
    } else {
        // general case: use newton's method to solve f(x) - h(x) = 0

        // h(h_min_x) = sqrt(h_min_y2)
        let h_min_y2 = h_c - h_b.powi(2) / h_a;
        let h_min_x  =-h_b / h_a;

        if h_min_y2 > lemon.sagitta.powi(2) {
            // ray never gets closer to axis than lemon's widest point
            return RaycastSolid::NON_INTERSECTION;
        }
let mut roots = (0.0, 0.0, 0.0, 0.0);
        // choose iteration starting points based on the best combination of:
        //     f(a_x) = +/- sqrt(h_c)*(x + h_b/h_c) = asymptotes of h
        //     f(b_x) = h_min_y                     = horizontal at minimum value of h
        let (mut x1, mut x2) = {
            // asymptote intersection
            let (a_x1, a_x2) = {
                // a(x)  = +/- a_m*(x - a_c)
                let a_m2 = h_a;
                let a_m  = a_m2.sqrt();
                let a_c  = h_min_x;

                // choose asymptotes based on relative position of hyperbola vertex
                if h_min_x.abs() > lemon.scale {
                    // take both solutions of a single asymptote
                    let sign = h_min_x.signum();

                    let a = a_m2 + 1.0;
                    let b = a_m*lemon.focal_radius() + sign*(a_m2*a_c);

                    let det = a*lemon.radius.powi(2)
                            - (lemon.focal_radius() + sign*(a_m*a_c)).powi(2);
                    if det <= 0.0 {
                        // asymptote does not intersect lemon
                        // h(x) > a(x) & a(x) >= f(x) => h(x) != f(x)
                        return RaycastSolid::NON_INTERSECTION;
                    }
                    let root = det.sqrt();

                    // solutions in ascending order
                    let x1 = (sign*b - root) / a;
                    let x2 = (sign*b + root) / a;
                    (x1, x2)
                } else {
                    // take the single true solution from both asymptotes
                    let a = a_m2 + 1.0;

                    // negative asymptote
                    let b_n    = a_m*lemon.focal_radius() + a_m2*a_c;
                    let root_n = ( a*lemon.radius.powi(2)
                                 - (lemon.focal_radius() + a_m*a_c).powi(2)
                                 ).sqrt();
                    assert!(!root_n.is_nan(), "asymptote should intersect lemon");

                    // positive asymptote
                    let b_p    = a_m*lemon.focal_radius() - a_m2*a_c;
                    let root_p = ( a*lemon.radius.powi(2)
                                 - (lemon.focal_radius() - a_m*a_c).powi(2)
                                 ).sqrt();
                    assert!(!root_p.is_nan(), "asymptote should intersect lemon");

                    // least solution from negative asymptote
                    let x1 = (b_n  - root_n) / a;
                    // greatest solution from positive asymptote
                    let x2 = (-b_p + root_p) / a;
                    (x1, x2)
                }
            };

            // horizontal intersection (+/-)
            let b_x = lemon.eval_radius_inverse(h_min_y2.sqrt());
            // because f and h are convex with respect to eachother, the best high and low
            // initial values are the least and greated valued ones, respectively
            (a_x1.max(-b_x), a_x2.min(b_x))
        };

        // define newton's method
        let find_root = |x_initial: Real| -> Real {
            let mut x          = x_initial;
            let mut trend      = 0.0;
            for _ in 0..MAX_ITERATIONS {
                let (f, f_) = lemon.eval_radius_gradient(x);
                let h       = (h_a*x.powi(2) + 2.0*h_b*x + h_c).sqrt();
                let h_      = (h_a*x + h_b) / h;

                let value = f - h;
                if value.abs() <= EPSILON {
                    break;
                }

                let gradient = f_ - h_;
                if gradient * trend < 0.0 {
                    return NAN;
                }
                trend = gradient;
                
                x -= value / gradient;
                if x.abs() > lemon.scale {
                    return NAN;
                }
            }
            x
        };

        x1 = find_root(x1);
        if x1.is_nan() { return RaycastSolid::NON_INTERSECTION; }
        x2 = find_root(x2);
        if x2.is_nan() { return RaycastSolid::NON_INTERSECTION; }
        (x1, x2)
    };

    // convert x solutions into ray parameters
    let map_x_to_t = |x: Real| -> Real {
        (x - point3!(ray.origin).dot(lemon_vertical)) / ray_vector_dot_lemon_vertical
    };
    let (t1, t2) = if ray_vector_dot_lemon_vertical > 0.0 {
        (map_x_to_t(x1), map_x_to_t(x2))
    } else {
        (map_x_to_t(x2), map_x_to_t(x1))
    };

    // correctly order result
    ray.origin -= offset;
    if t1 >= 0.0 {
        RaycastSolid::from_fore_and_rear_solutions(ray, t1, t2)
    } else {
        RaycastSolid::from_fore_and_rear_solutions(ray, t2, t1)
    }
}