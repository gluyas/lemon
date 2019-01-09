#version 330

layout(std140) uniform Camera {
    mat4 view;
    mat4 projection;

    vec3 position;
} u_camera;

in mat4 a_transform;

in vec3 a_position;
in vec3 a_normal;
in vec2 a_uv;

out vec3 v_position;
out vec3 v_normal;
out vec2 v_uv;

flat out vec3 up;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    v_position = vec3(a_transform * vec4(a_position, 1.0));
    v_normal   = mat3(a_transform) * a_normal;
    v_uv       = a_uv;

    gl_Position = u_camera.projection * u_camera.view * a_transform * vec4(a_position, 1.0);

    up = mat3(a_transform) * vec3(0.0, 0.0, 1.0);
}
