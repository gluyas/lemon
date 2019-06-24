use super::*;

// TODO: 9.81 gravity and scale lemons to a resonable size?
const GRAVITY: Vec3 = vec3!(0.0, 0.0, -27.0 * METER / SECOND / SECOND);

const COLLISION_ELASTICITY: f32 = 0.15;

const FRICTION:     f32 = 1.55 * NEWTON / NEWTON;
const ANGULAR_DRAG: f32 = 0.1 * TAU / SECOND / SECOND;

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

pub fn integrate_rigidbody_fixed_timestep(rb: &mut Rigidbody) {
    rb.position += rb.velocity + GRAVITY/2.0;
    rb.velocity += GRAVITY;

    if rb.angular_momentum != VEC3_0 {
        let angular_velocity = rb.get_inertia_inverse()  * rb.angular_momentum;
        rb.orientation += Quat { v: angular_velocity, s: 0.0 } / 2.0 * rb.orientation;
        rb.orientation  = rb.orientation.normalize();

        // this is a hacked in solution to rigidbodies tending to spin for too long.
        // the impulse-based friction model does not affect spinning points.
        let angular_speed = angular_velocity.magnitude();
        rb.angular_momentum -= rb.get_inertia() * angular_velocity
                             * ANGULAR_DRAG / angular_speed;
    }
}

pub fn resolve_collision_static(
    mut collision: Collision,
    rb1: &mut Rigidbody,
    mut debug: Option<&mut DebugRender>,
) {
    rb1.position    += collision.normal * collision.depth;
    collision.point += collision.normal * collision.depth / 2.0;
    collision.depth  = 0.0;

    let inertia_inverse  = rb1.get_inertia_inverse();
    let angular_velocity = inertia_inverse * rb1.angular_momentum;

    let offset   = collision.point - rb1.position;
    let velocity = rb1.velocity + angular_velocity.cross(offset);

    let collision_tangent = proj_onto_plane(velocity.normalize(), collision.normal);

    let reaction = {
        let proj    = -(1.0 + COLLISION_ELASTICITY)
                    * velocity.dot(collision.normal);
        let linear  = 1.0 / rb1.mass;
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
        let linear  = 1.0 / rb1.mass;
        let angular = ( inertia_inverse
                      * offset.cross(collision_tangent)
                      ).cross(offset);

        let cap     = proj / (linear + angular.dot(collision_tangent));
        cap.min(reaction * FRICTION).neg()
    };
    let friction_vector = friction * collision_tangent;

    if let Some(ref mut debug) = debug {
        debug.draw_ray(
            &color!(0x0000AFFF).truncate(), 1, &collision.point,
            &(friction_vector / FRAME_DELTA_TIME)
        );
        debug.draw_ray(
            &color!(0xAF0000FF).truncate(), 1, &collision.point,
            &(reaction_vector / FRAME_DELTA_TIME)
        );
        debug.draw_ray(
            &color!(0x00AF00FF).truncate(), 1, &collision.point,
            &(velocity / FRAME_DELTA_TIME)
        );
    }

    let impulse = reaction_vector + friction_vector;
    rb1.velocity         += impulse / rb1.mass;
    rb1.angular_momentum += offset.cross(impulse);
}

// TODO: don't store mass, inertia, on kinematic bodies?
pub fn resolve_collision_kinematic(
    mut collision: Collision,
    rb1: &mut Rigidbody,
    rb2: &Rigidbody,
    mut debug: Option<&mut DebugRender>,
) {
    rb1.position    += collision.normal * collision.depth;
    collision.point += collision.normal * collision.depth / 2.0;
    collision.depth  = 0.0;

    let rb1_inertia_inverse = rb1.get_inertia_inverse();
    let rb1_point_offset    = collision.point - rb1.position;
    let rb1_point_velocity  = ( rb1_inertia_inverse
                                * rb1.angular_momentum
                              ).cross(rb1_point_offset)
                              + rb1.velocity;

    let rb2_inertia_inverse = rb2.get_inertia_inverse();
    let rb2_point_offset    = collision.point - rb2.position;
    let rb2_point_velocity  = ( rb2_inertia_inverse
                                * rb2.angular_momentum
                              ).cross(rb2_point_offset)
                              + rb2.velocity;

    let relative_velocity = rb1_point_velocity - rb2_point_velocity;
    let collision_tangent = proj_onto_plane(relative_velocity, collision.normal)
                           .normalize();

    let reaction = {
        let proj    = -(1.0 + COLLISION_ELASTICITY)
                    * relative_velocity.dot(collision.normal);
        let linear  = 1.0 / rb1.mass;
                 // + 1.0 / infinity  => 0.0
        let angular = ( rb1_inertia_inverse
                      * rb1_point_offset.cross(collision.normal)
                      ).cross(rb1_point_offset);
                 // + [infinity]^-1 => 0.0
        proj / (linear + angular.dot(collision.normal))
    };
    let reaction_vector = reaction * collision.normal;

    let friction = {
        let proj    = relative_velocity.dot(collision_tangent);
        let linear  = 1.0 / rb1.mass;
        let angular = ( rb1_inertia_inverse
                      * rb1_point_offset.cross(collision_tangent)
                      ).cross(rb1_point_offset);

        let cap     = proj / (linear + angular.dot(collision_tangent));
        cap.min(reaction * FRICTION).neg()
    };
    let friction_vector = friction * collision_tangent;

    if let Some(ref mut debug) = debug {
        debug.draw_ray(
            &color!(0x0000AFFF).truncate(), 1, &collision.point,
            &(friction_vector / FRAME_DELTA_TIME)
        );
        debug.draw_ray(
            &color!(0xAF0000FF).truncate(), 1, &collision.point,
            &(reaction_vector / FRAME_DELTA_TIME)
        );
        debug.draw_ray(
            &color!(0x00AF00FF).truncate(), 1, &collision.point,
            &(relative_velocity / FRAME_DELTA_TIME)
        );
    }

    let impulse = reaction_vector + friction_vector;
    rb1.velocity         += impulse / rb1.mass;
    rb1.angular_momentum += rb1_point_offset.cross(impulse);
}

pub fn resolve_collision_dynamic(
    mut collision: Collision,
    rb1: &mut Rigidbody,
    rb2: &mut Rigidbody,
    mut debug: Option<&mut DebugRender>,
) {
    // push objects out based on their relative velocity
    let velocity_ratio = {
        let rb1_dot = rb1.velocity.dot(collision.normal).min(0.0).neg();
        let rb2_dot = rb2.velocity.dot(collision.normal).max(0.0);
        if rb1_dot + rb2_dot != 0.0 {
            rb1_dot / (rb1_dot + rb2_dot)
        } else {
            0.5
        }
    };
    assert!(velocity_ratio >= 0.0 && velocity_ratio <= 1.0,
        "velocity_ratio out of range: {}", velocity_ratio
    );
    rb1.position    += collision.normal * collision.depth * velocity_ratio;
    rb2.position    -= collision.normal * collision.depth * (1.0 - velocity_ratio);
    collision.point += collision.normal * collision.depth * (velocity_ratio - 0.5);
    collision.depth  = 0.0;

    let rb1_inertia_inverse = rb1.get_inertia_inverse();
    let rb1_point_offset    = collision.point - rb1.position;
    let rb1_point_velocity  = ( rb1_inertia_inverse
                                * rb1.angular_momentum
                              ).cross(rb1_point_offset)
                              + rb1.velocity;

    let rb2_inertia_inverse = rb2.get_inertia_inverse();
    let rb2_point_offset    = collision.point - rb2.position;
    let rb2_point_velocity  = ( rb2_inertia_inverse
                                * rb2.angular_momentum
                              ).cross(rb2_point_offset)
                              + rb2.velocity;

    let relative_velocity = rb1_point_velocity - rb2_point_velocity;
    let collision_tangent = proj_onto_plane(relative_velocity, collision.normal)
                           .normalize();
    let reaction = {
        let proj    = -(1.0 + COLLISION_ELASTICITY) * collision.normal
                     .dot(relative_velocity);
        let linear  = 1.0 / rb1.mass
                    + 1.0 / rb2.mass;
        let angular = ( rb1_inertia_inverse
                      * rb1_point_offset.cross(collision.normal)
                      ).cross(rb1_point_offset)
                    + ( rb2_inertia_inverse
                      * rb2_point_offset.cross(collision.normal)
                      ).cross(rb2_point_offset);

        proj / (linear + angular.dot(collision.normal))
    };
    let reaction_vector = reaction * collision.normal;

    let friction = {
        let proj    = relative_velocity.dot(collision_tangent);
        let linear  = 1.0 / rb1.mass
                    + 1.0 / rb2.mass;
        let angular = ( rb1_inertia_inverse
                      * rb1_point_offset.cross(collision_tangent)
                      ).cross(rb1_point_offset)
                    + ( rb2_inertia_inverse
                      * rb2_point_offset.cross(collision_tangent)
                      ).cross(rb2_point_offset);

        let cap     = proj / (linear + angular.dot(collision_tangent));
        cap.min(reaction * FRICTION).neg()
    };
    let friction_vector = friction * collision_tangent;

    if let Some(ref mut debug) = debug {
        debug.draw_ray(&color!(0x0000AFFF).truncate(), 1,
            &collision.point,
            &(friction_vector / FRAME_DELTA_TIME)
        );
        debug.draw_ray(&color!(0xAF0000FF).truncate(), 1,
            &collision.point,
            &(reaction_vector / FRAME_DELTA_TIME),
        );
        debug.draw_ray(&color!(0x00AF00FF).truncate(), 1,
            &collision.point,
            &(relative_velocity / FRAME_DELTA_TIME),
        );
    }

    let impulse = reaction_vector + friction_vector;

    rb1.velocity         += impulse / rb1.mass;
    rb1.angular_momentum += rb1_point_offset.cross(impulse);
    rb2.velocity         -= impulse / rb2.mass;
    rb2.angular_momentum -= rb2_point_offset.cross(impulse);
}
