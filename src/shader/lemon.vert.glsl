#version 330

layout(std140) uniform Camera {
    mat4 view;
    mat4 projection;

    vec3 position;
} u_camera;

uniform sampler2D u_radius_normal_z_map;

// per-mesh
in mat4  a_transform;
in float a_lemon_s;

// per-vertex
in vec3 a_position;

out vec3 v_position;
out vec3 v_normal;

//flat out vec3 up;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    // square here corresponds to sqrt in make_radius_normal_z_map. this gives better accuracy
    // for z values closer to +/-1, where the derivative is highest.
    vec2  sample   = texture(u_radius_normal_z_map, vec2(pow(a_position.z, 2), a_lemon_s)).xy;
    float radius   = sample.x;
    float normal_z = sample.y;

    vec4 world_position = a_transform * vec4(a_position.xy * radius, a_position.z, 1.0);
    v_position = vec3(world_position);
    v_normal   = mat3(a_transform) * vec3(a_position.xy, sign(a_position.z) * normal_z);

    gl_Position = u_camera.projection * u_camera.view * world_position;

    //up = mat3(a_transform) * vec3(0.0, 0.0, 1.0);
}
