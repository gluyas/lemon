#version 330

flat in vec3 v_color;

out vec4 gl_FragColor;

void main() {
    gl_FragColor = vec4(v_color, 1.0);
}
