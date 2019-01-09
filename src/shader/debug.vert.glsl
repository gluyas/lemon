#version 330

uniform mat4 u_camera;

in vec3 a_position;
in vec3 a_color;

flat out vec3 v_color;

out vec4 gl_Position;

void main() {
    v_color = a_color;
    gl_Position = u_camera * vec4(a_position, 1.0);
}
