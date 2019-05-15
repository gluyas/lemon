#version 330

uniform sampler2DRect u_pixels;

layout(pixel_center_integer) in vec4 gl_FragCoord;

out vec4 gl_FragColor;

void main() {
    vec4 color = texture(u_pixels, vec2(gl_FragCoord));
    gl_FragColor = color;
}
