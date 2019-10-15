#version 330

// TODO: integrate with consts defined in main.rs
#define MAX_LEMONS 256

in vec2 v_ndc;

uniform mat4 u_camera_inverse;
uniform vec3 u_camera_pos;

vec3 getCameraRay() {
    vec4 origin = u_camera_inverse * vec4(v_ndc, 0.0, 1.0);
    origin /= origin.w;
    vec4 target = u_camera_inverse * vec4(v_ndc, 1.0, 1.0);
    target /= target.w;
    return normalize(target.xyz - origin.xyz);
}

// position [0][0..2], radius [0][3]
// vertical [1][0..2], focus  [1][3]
uniform mat2x4[MAX_LEMONS] u_lemons;
uniform uint               u_lemons_len;

uniform int   u_hover_instance_id;
uniform float u_hover_glow;

uniform int   u_selection_instance_id;
uniform float u_selection_glow;

// `Lemon` is defined as the self-intersecting region of a spindle torus
//
// `position` defines the centre of the lemon (point of symmetry)
//
// `vertical` defines the axis of rotation
//
// `radius` defines the minor radius of the torus
// `focus` defines the major radius of the torus
//
// `vertical`, `radius`, and `focus` are multiplied by the scale of the lemon
//
struct Lemon
{
    vec3 position;
    vec3 vertical;
    float radius;
    float focus;
};

float sdLemonPerfect(vec3 p, Lemon lemon)
{
    vec3 s = p - lemon.position;
    float t = dot(s, lemon.vertical) / dot(lemon.vertical, lemon.vertical);

    vec3 x = t*lemon.vertical;
    vec3 y = s - x;
    float y2 = dot(y, y);
    if (abs(t) >= 1.0) {
        float l = lemon.focus*(abs(t) - 1.0);
        if (y2 <= l*l) {
            // corner case
            return length(sign(t) * lemon.vertical - s);
        }
    }
    // general case
    return length(lemon.focus*-(y/sqrt(y2)) - s) - lemon.radius;
}

float sdLemon(vec3 p, Lemon lemon)
{
    vec3 s = p - lemon.position;
    float t = dot(s, lemon.vertical) / dot(lemon.vertical, lemon.vertical);

    vec3 y = s - t*lemon.vertical;

    return length(lemon.focus*-normalize(y) - s) - lemon.radius;
}

float sdWorld(vec3 p, out int id) {
    float min_sd = 2000.0;
    for (int i = 0; i < int(u_lemons_len); i++) {
        mat2x4 mat = u_lemons[i];
        Lemon lemon = Lemon(mat[0].xyz, mat[1].xyz, mat[0].w, mat[1].w);
        float sd = sdLemon(p, lemon);
        if (sd < min_sd) {
            min_sd = sd;
            id = i;
        }
    }
    return min_sd;
}

void main() {
    vec3 ray_point  = u_camera_pos;
    vec3 ray_vector = getCameraRay();
    float ray_depth = 0.0;

    int i = 0;
    int id = -1;
    while (ray_depth <= 1000.0) {
        if (++i >= 128) { break; }
        float sd = sdWorld(ray_point, id);
        ray_depth += sd;
        if (sd <= 0.0) {
            break;
        }
        ray_point += sd * ray_vector;
    }

    if (ray_depth <= 1000.0) {
        gl_FragColor = vec4(1.0, 0.8, 0.2, 1.0);

        float glow = 0.0;
        if (id == u_hover_instance_id)     glow += u_hover_glow;
        if (id == u_selection_instance_id) glow += u_selection_glow;
        gl_FragColor = mix(gl_FragColor, vec4(1.0), glow);
    } else {
        discard;
    }
}

// `sagitta` defines the shape of the lemon and is between 0 (zero volume) and 1 (sphere)
void makeLemon(in float sagitta, out float radius, out float focus)
{
    radius = (sagitta*sagitta + 1.0) / 2.0 / sagitta;
    focus  = radius - sagitta;
}