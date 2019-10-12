#version 330

in vec2 a_ndc;

out vec2 v_ndc;
out vec4 gl_Position;

void main() {
    gl_Position = vec4(a_ndc, 0.0, 1.0);
    v_ndc = a_ndc;
}