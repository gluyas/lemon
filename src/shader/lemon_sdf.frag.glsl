#version 330

vec3 quat_rotate(vec4 q, vec3 v) {
    return 2.0*(dot(q.xyz, v)*q.xyz + q.w*cross(q.xyz, v)) + v*(q.w*q.w - dot(q.xyz, q.xyz));
}

vec4 quat_inverse(vec4 q) {
    return vec4(-q.xyz, q.w);
}

// TODO: integrate with consts defined in main.rs
#define MAX_LEMONS 64
#define MAX_CLIPS 6

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

// [0][0,1,2] = position
// [1][0,1,2] = inverse rotation (vector part)
// [2][0]     = inverse rotation (scalar part)
// [2][1]     = lemon radius
// [2][2]     = lemon focal radius
uniform mat3[MAX_LEMONS]           u_lemons;
uniform vec4[MAX_LEMONS*MAX_CLIPS] u_lemon_clips;
uniform uint                       u_lemons_len;

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
// UNUSED: using quaternion rotations instead to simplify clipping planes
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
    return length(s + lemon.focus*(y/sqrt(y2))) - lemon.radius;
}

float sdLemon(vec3 p, Lemon lemon)
{
    vec3 s = p - lemon.position;
    float t = dot(s, lemon.vertical) / dot(lemon.vertical, lemon.vertical);

    vec3 y = s - t*lemon.vertical;

    return length(s + lemon.focus*normalize(y)) - lemon.radius;
}

vec3 normalLemon(vec3 p, Lemon lemon) {
    vec3 s = p - lemon.position;
    float t = dot(s, lemon.vertical) / dot(lemon.vertical, lemon.vertical);

    vec3 y = s - t*lemon.vertical;

    return normalize(s + lemon.focus*normalize(y));
}

float sdLemonAxisAlignedRelative(vec3 r, float radius, float focus) {
    return length(vec3(r.xy + focus*normalize(r.xy), r.z)) - radius;
}

vec3 normalLemonAxisAlignedRelative(vec3 r, float radius, float focus) {
    return normalize(vec3(r.xy + focus*normalize(r.xy), r.z));
}

float sdLemonClipped(vec3 p, int id) {
    mat3 mat = u_lemons[id];

    float radius = mat[2][1];
    float focus  = mat[2][2];

    vec3 s = p - mat[0];
    float s2 = dot(s, s);
    if (s2 >= radius*radius + 0.1) {
        return sqrt(s2) - radius;
    }

    // translate then rotate p into the lemon's relative, oriented space
    vec4 q = vec4(mat[1], mat[2][0]);
    vec3 r = quat_rotate(q, s);

    float sd = sdLemonAxisAlignedRelative(r, radius, focus);
    for (int j = 0; j < MAX_CLIPS; j++) {
        vec4 clip  = u_lemon_clips[id+j];
        float clipped = dot(r, clip.xyz) - clip.w;
        if (clipped > sd) {
            sd = clipped;
        }
    }
    return sd;
}

vec3 normalLemonClipped(vec3 p, int id) {
    mat3 mat = u_lemons[id];

    float radius = mat[2][1];
    float focus  = mat[2][2];
    // translate then rotate p into the lemon's relative, oriented space
    vec4 q = vec4(mat[1], mat[2][0]);
    vec3 r = quat_rotate(q, p - mat[0]);

    float sd = sdLemonAxisAlignedRelative(r, radius, focus);
    vec3 normal = normalLemonAxisAlignedRelative(r, radius, focus);
    for (int j = 0; j < MAX_CLIPS; j++) {
        vec4 clip  = u_lemon_clips[id+j];
        float clipped = dot(r, clip.xyz) - clip.w;
        if (clipped > sd) {
            sd = clipped;
            normal = clip.xyz;
        }
    }
    return quat_rotate(quat_inverse(q), normal);
}

float sdWorld(vec3 p, out int id) {
    float min_sd = 2000.0;
    for (int i = 0; i < int(u_lemons_len); i++) {
        // TODO: raycast clipping planes
        float sd = sdLemonClipped(p, i);
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
        vec3 normal = normalLemonClipped(ray_point, id);
        gl_FragColor = vec4(normal, 1.0);
        //gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);

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