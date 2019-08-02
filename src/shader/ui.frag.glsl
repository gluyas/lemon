precision mediump float;

uniform sampler2D u_pixels;

varying vec2 v_uv;

void main() {
    vec4 color = texture2D(u_pixels, v_uv);
    if (color.a == 0.0) {
        discard;
    } else {
        gl_FragColor = color;
    }
}
