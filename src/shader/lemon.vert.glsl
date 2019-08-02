precision mediump float;

uniform mat4 u_camera_view;
uniform mat4 u_camera_projection;
uniform vec3 u_camera_position;

uniform sampler2D u_radius_normal_z_map;

// per-mesh
attribute mat4  a_transform;
attribute float a_lemon_s;
attribute vec3  a_lemon_color;
attribute float a_lemon_glow;

// per-vertex
attribute vec3 a_position;

varying vec3 v_position;
varying vec3 v_normal;

varying vec3 v_color;
varying float v_ui_glow;

//flat varying vec3 up;

void main() {
    // square here corresponds to sqrt in make_radius_normal_z_map. this gives better accuracy
    // for z values closer to +/-1, where the derivative is highest.
    vec2  sample   = texture2D(u_radius_normal_z_map, vec2(a_position.z*a_position.z, a_lemon_s)).xy;
    float radius   = sample.x;
    float normal_z = tan(sample.y);

    vec4 world_position = a_transform * vec4(a_position.xy * radius, a_position.z, 1.0);

    v_position  = vec3(world_position);
    v_normal    = mat3(a_transform) * vec3(a_position.xy, sign(a_position.z) * normal_z);

    v_color     = a_lemon_color;
    v_ui_glow   = a_lemon_glow;

    gl_Position = u_camera_projection * u_camera_view * world_position;

    //up = mat3(a_transform) * vec3(0.0, 0.0, 1.0);
}
