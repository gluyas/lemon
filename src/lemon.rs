use super::*;

#[derive(Copy, Clone, Debug)]
pub struct Lemon {
    pub r: f32,
    pub s: f32,
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
