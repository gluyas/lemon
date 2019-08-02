precision mediump float;

uniform mat4 u_camera;

attribute vec3 a_position;
attribute vec3 a_color;

varying vec3 v_color;

void main() {
    v_color = a_color;
    gl_Position = u_camera * vec4(a_position, 1.0);
}
