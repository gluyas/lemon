use super::*;

#[derive(Copy, Clone, Debug)]
pub struct Lemon {
    /// Radius of the sphere the arc lies on
    pub r: f32,

    /// Sagitta of the arc; radius of the widest horizontal of the lemon
    pub s: f32,

    /// Offset of the sphere from the centre; focal radius
    /// Equal to `r - s`
    pub t: f32,

    pub phys: Rigidbody,
}

impl Lemon {
// https://www.desmos.com/calculator/z7ijifw8pc
// https://www.researchgate.net/publication/2173504
    pub const SUBDIV_T: usize = 32;
    pub const SUBDIV_Z: usize = 32;

    pub const DENSITY: f32 = 10.0;

    pub fn new(s: f32) -> Self {
        let r = (s*s + 1.0) / 2.0 / s;
        let t = r - s;
        let (s2, r2, t2) = (s*s, r*r, t*t);

        let u  = (r2 - 1.0).sqrt();       // = u(1) =  u(-1) // <= u(z) is even
        let su = u + r2 * (1.0/r).asin(); // = U(1) = -U(-1) // <= U(z) is odd

        let mass = Lemon::DENSITY * PI * 2.0
                 * ( -1.0/3.0 + (r2 + t2) - t*su );

        let i_z  = Lemon::DENSITY * PI
                 * ( 0.2 - 2.0*(r2/3.0 + t2) + (r2*r2 + 6.0*r2*t2 + t2*t2)
                   - t*((r2-1.0)*u + (2.0*t2 + 3.0/2.0*r2)*su) );

        let i_y  = i_z/2.0  +  Lemon::DENSITY * PI * 2.0
                 * ( -0.2 + (r2 + t2)/3.0 - 2.0*t*(r2/8.0*su - u*u*u/4.0) );

        let mut phys = Rigidbody::default();
        phys.mass    = mass;
        phys.inertia_local = vec3!(i_y, i_y, i_z);

        Lemon { r, s, t, phys }
    }

    pub fn get_vertical(&self) -> Vec3 {
        self.phys.orientation * VEC3_Z
    }

    pub fn eval_radius_gradient(&self, z: f32) -> (f32, f32) {
        let (z2, r2) = (z*z, self.r*self.r);
        let u        = (r2 - z2).sqrt();

        (u - self.t, -z / u)
    }

    pub fn eval_integral_mass_intertia_z_y(&self, z: f32) -> (f32, f32, f32) {
        let (z2, z3, z5) = (z*z, z*z*z, z*z*z*z*z);
        let (s,  r,  t)  = (self.s, self.r, self.t);
        let (s2, r2, t2) = (s*s,    r*r,    t*t);

        let u  = (r2 - z2).sqrt();
        let su = u + r2 * (1.0/r).asin();

        let mass = Lemon::DENSITY * PI
                 * ( z3/3.0
                   + z*(r2 + t2)
                   - t*su );

        let i_z  = Lemon::DENSITY * PI / 2.0
                 * ( z5/5.0
                   - 2.0*z3*(r2 + t2/3.0)
                   + z*(r2*r2 + 6.0*r2*t2 + t2*t2 - (r2 - z2)*u)
                   - t*(2.0*t2 + 3.0/2.0*r2)*su );

        let i_y  = i_z / 2.0  -  Lemon::DENSITY * PI
                 * (-z5/5.0
                   + z3/3.0*(r2 + t2)
                   - 2.0*t*(r2/8.0*su - z/4.0*u*u*u) );
        (mass, i_z, i_y)
    }

    pub fn make_mesh(&self) -> (Vec<Point3>, Vec<Vec2>, Vec<Vec3>, Vec<ElementIndex>) {
        let mut zs = [0.0; Lemon::SUBDIV_Z];
        zs.iter_mut().zip(0..Lemon::SUBDIV_Z)
            .for_each(|(z, i)| {
                let z_inverse = 1.0 - (2*i) as f32 / (Lemon::SUBDIV_Z-1) as f32;
                *z = (PI * z_inverse / 2.0).sin();
            });

        let mut surface = [(0.0, 0.0); Lemon::SUBDIV_Z];
        surface.iter_mut().zip(zs.iter())
            .for_each(|(r_m, &z)| *r_m = self.eval_radius_gradient(z));

        let mut verts   = Vec::with_capacity(Lemon::SUBDIV_T * Lemon::SUBDIV_Z);
        let mut uvs     = Vec::with_capacity(verts.capacity());
        let mut normals = Vec::with_capacity(verts.capacity());
        let mut indices = Vec::with_capacity(Lemon::SUBDIV_T * (2 * Lemon::SUBDIV_Z + 4));

        for i_theta in 0..Lemon::SUBDIV_T {
            let q_theta = i_theta as f32 / (Lemon::SUBDIV_T-1) as f32;
            let i_theta_next = (i_theta + 1) % Lemon::SUBDIV_T;
            let theta = q_theta * 2.0 * PI;
            let unit = vec3!(theta.sin(), theta.cos(), 0.0);

            indices.push((Lemon::SUBDIV_Z * i_theta) as ElementIndex);
            indices.push((Lemon::SUBDIV_Z * i_theta) as ElementIndex);
            for i_z in 0..Lemon::SUBDIV_Z {
                let q_z = (Lemon::SUBDIV_Z-i_z-1) as f32 / (Lemon::SUBDIV_Z-1) as f32;
                let (radius, gradient) = surface[i_z];

                verts.push(point3!(0.0, 0.0, zs[i_z]) + radius * unit);
                uvs.push(vec2!(q_theta, q_z));
                normals.push(vec3!(unit.x, unit.y, -gradient).normalize());

                indices.push((1 + i_z + Lemon::SUBDIV_Z * i_theta)      as ElementIndex);
                indices.push((1 + i_z + Lemon::SUBDIV_Z * i_theta_next) as ElementIndex);
            }
            indices.push((2 * Lemon::SUBDIV_Z * i_theta_next) as ElementIndex);
            indices.push((2 * Lemon::SUBDIV_Z * i_theta_next) as ElementIndex);
        }
        (verts, uvs, normals, indices)
    }

    pub fn make_normal_map(&self) -> Vec<Vec3> {
        let mut tex = vec![vec3!(0.0, 0.0, 1.0); LEMON_TEX_SIZE * LEMON_TEX_SIZE];
        /*for i in 0..LEMON_TEX_SIZE {
            for j in 0..LEMON_TEX_SIZE {
                tex[j + i * LEMON_TEX_SIZE] = vec3!(
                    0.2 * (j as f32 * 35.0 * PI / LEMON_TEX_SIZE as f32).sin(),
                    0.0,
                    1.0).normalize();
                //tex[j + i * LEMON_TEX_SIZE] = -vec3!(i as f32 / LEMON_TEX_SIZE as f32, j as f32 / LEMON_TEX_SIZE as f32, 0.0);
            }
        }*/
        /*for (i, pixel) in
            image::load_from_memory(include_bytes!("../test_normal_map2.png")).unwrap()
                .flipv()
                .to_rgb()
                .pixels()
                .enumerate()
        {
            fn norm(val: u8) -> f32 {
                2.0 * val as f32 / 255.0 - 1.0
            }
            tex[i] = vec3!(norm(pixel[0]), norm(pixel[1]), norm(pixel[2]));
        }*/
    tex
    }
}

pub fn get_collision_lemon(
    lemon: &Lemon,
    other: &Lemon,
    mut debug: Option<&mut DebugRender>
) -> Option<Collision> {
    let lemon_vertical = lemon.get_vertical();
    let lemon_focus    = (lemon.phys.position, lemon_vertical, lemon.t);

    let other_vertical = other.get_vertical();
    let other_focus    = (other.phys.position, other_vertical, other.t);

    // find furthest points on focii for general case collision test
    let (lemon_sphere, other_sphere, displacement, distance2) = {
        // TODO: special case for when displacement parallel to single vertical
        let parallel = lemon_vertical.cross(other_vertical) == vec3!(0.0, 0.0, 0.0);
        if parallel {
            // algebraic solution available when focus normals are parallel
            let centroid_displacement = other.phys.position - lemon.phys.position;
            let syzygy = lemon_vertical.cross(centroid_displacement) == vec3!(0.0, 0.0, 0.0);
            if syzygy {
                // NOTE: following assumes that all lemons have a height of 2.0
                // early return trivial case where displacement parallel to focus normals
                // iterative algorithm cannot converge with this input
                if let Some(ref mut debug) = debug {
                    let color = color!(0x0000FFFF).truncate();
                    let lemon_vertex = lemon.phys.position + lemon_vertical
                                     * lemon_vertical.dot(centroid_displacement).signum();
                    let other_vertex = other.phys.position - other_vertical
                                     * other_vertical.dot(centroid_displacement).signum();
                    debug.draw_line(&color, 1, &[lemon_vertex, other_vertex]);
                }

                let distance = centroid_displacement.dot(lemon_vertical).abs();
                return some_if_then(distance <= 2.0, || Collision {
                    point:  lemon.phys.position + centroid_displacement / 2.0,
                    normal: centroid_displacement.neg() / distance,
                    depth:  2.0 - distance,
                });
            } else {
                let lemon_sphere = lemon.phys.position
                                 - proj_onto_plane(centroid_displacement, lemon_vertical)
                                   .normalize_to(lemon.t);
                let other_sphere = other.phys.position
                                 + proj_onto_plane(centroid_displacement, other_vertical)
                                   .normalize_to(other.t);
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
                    eprintln!("lemon-lemon collision test failed to converge");
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

    #[inline]
    fn is_edge_case(
        lemon: &Lemon, lemon_vertical: Vec3, lemon_sphere: Point3,
        displacement: Vec3,
    ) -> bool {
        let lemon_sphere_relative = lemon_sphere - lemon.phys.position;
        // test that
        let upper_bound     = lemon_vertical - lemon_sphere_relative;
        let reference_cross = upper_bound.cross(-lemon_sphere_relative);
        let test_cross      = upper_bound.cross(displacement);
        test_cross.dot(reference_cross) < 0.0
    }

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
                let color = color!(0xA0A0A0FF).truncate();
                debug.draw_line(&color, 1, &[
                    lemon.phys.position + lemon_vertical, lemon_sphere,
                    lemon.phys.position - lemon_vertical, lemon_sphere,
                    other_sphere, other.phys.position + other_vertical,
                    other_sphere, other.phys.position - other_vertical,
                ]);
                debug.draw_line(&color, 1, &make_line_strip_circle(
                    &lemon_sphere, &lemon_vertical.cross(displacement).normalize(), lemon.r, 63,
                ));
                debug.draw_line(&color, 1, &make_line_strip_circle(
                    &other_sphere, &other_vertical.cross(displacement).normalize(), other.r, 63,
                ));
            }

            // test edge cases
            let lemon_edge = is_edge_case(lemon, lemon_vertical, lemon_sphere, displacement);
            let other_edge = is_edge_case(other, other_vertical, other_sphere,-displacement);

            if lemon_edge && other_edge {
                let lemon_vertex = lemon.phys.position + lemon_vertical;
                let other_vertex = other.phys.position + other_vertical;
                let displacement = other_vertex - lemon_vertex;
                let distance2    = displacement.magnitude2();

                resolve_collision_test(
                    lemon, lemon_focus, lemon_vertical, Vertex(lemon_vertex),
                    other, other_focus, other_vertical, Vertex(other_vertex),
                    displacement, distance2,
                    debug,
                )
            } else if lemon_edge {
                let lemon_vertex = lemon.phys.position + lemon_vertical;
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
                let other_vertex = other.phys.position + other_vertical;
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
                some_if_then(distance2 <= (lemon.r + other.r).powi(2), || Collision {
                    point:  lemon_sphere + displacement / 2.0,
                    normal: displacement.normalize().neg(),
                    depth:  lemon.r + other.r - distance2.sqrt(),
                })
            }
        },

        (Sphere(lemon_sphere), Vertex(other_vertex)) => {
            // vertex touching curved surface
            if let Some(ref mut debug) = debug {
                let color = color!(0xBF00BFFF).truncate();
                debug.draw_line(&color, 1, &[
                    lemon.phys.position + lemon_vertical, lemon_sphere,
                    lemon.phys.position - lemon_vertical, lemon_sphere,
                    other_vertex,
                ]);
                debug.draw_line(&color, 1, &make_line_strip_circle(
                    &lemon_sphere, &lemon_vertical.cross(displacement).normalize(), lemon.r, 63,
                ));
            }
            if is_edge_case(lemon, lemon_vertical, lemon_sphere, displacement) {
                let lemon_vertex = lemon.phys.position + lemon_vertical;
                let displacement = other_vertex - lemon_vertex;
                let distance2    = displacement.magnitude2();

                resolve_collision_test(
                    lemon, lemon_focus, lemon_vertical, Vertex(lemon_vertex),
                    other, other_focus, other_vertical, Vertex(other_vertex),
                    displacement.neg(), distance2,
                    debug,
                )
            } else {
                some_if_then(distance2 <= lemon.r * lemon.r, || {
                    let distance = distance2.sqrt();
                    Collision {
                        point:  lemon_sphere + (lemon.r + distance)/2.0 * displacement/distance,
                        normal: displacement.normalize().neg(),
                        depth:  lemon.r - distance,
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
