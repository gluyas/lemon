#version 330

layout(std140) uniform Camera {
    mat4 view;
    mat4 projection;

    vec3 position;
} u_camera;

uniform sampler2D u_normal_map;

uniform vec4 u_ambient_color;

uniform vec3 u_direction_light = vec3(2.5, 0.5, 1.5);

in vec3 v_position;
in vec3 v_normal;

in vec3 v_color;
flat in float v_ui_glow;

//flat in vec3 up;

out vec4 gl_FragColor;

//mat3 tbn(vec3 normal) {
//    vec3 u = normalize(cross(normal, up));
//    vec3 v = cross(u, normal);
//    return mat3(u, v, normal);
//}

void main() {
    //mat3 tbn = tbn(normalize(v_normal));
    //vec3 normal = tbn * texture(u_normal_map, v_uv).xyz;
    vec3 normal = normalize(v_normal);
    vec3 eye    = normalize(u_camera.position - v_position);
    vec3 light  = normalize(u_direction_light);
    //vec3 halfway = (eye + light) / 2.0;
    vec3 reflection = reflect(light, normal);

    float ambient  = 0.15;
    const float LAMBERT_BLEED = 1.7;
    float lambert  = pow(clamp((dot(normal, light) + LAMBERT_BLEED) / (LAMBERT_BLEED + 1.0), 0, 1), 1);
    float specular = pow(clamp(-dot(eye, reflection), 0, 1), 8);

    float ui_glow  = v_ui_glow * pow(1.0-dot(eye, normal), 2);

    //gl_FragColor = vec4(0.5 * (texture(u_normal_map, v_uv).xyz + vec3(1.0, 1.0, 1.0)), 1.0);
    //gl_FragColor = (vec4(1.0) + texture(u_normal_map, v_uv)) / 2.0;
    //gl_FragColor = vec4(v_uv.x, 0.0, v_uv.y, 1.0);
    //gl_FragColor = (vec4(1.0) + normal) / 2.0;

    gl_FragColor   = vec4((ambient + lambert) * v_color + vec3(max(specular, ui_glow)), 1.0);
}
