precision mediump float;

uniform float u_depth;

attribute vec2 a_position;

varying vec2 v_uv;

void main() {
    v_uv = 0.5 * vec2(1.0 + a_position.x, 1.0 + a_position.y);
    gl_Position = vec4(a_position, 0.0, 1.0);
}
