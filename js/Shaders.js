
import {gl, ext, compileShader} from "./WebGL.js";
// import {config} from "./config.js"
// import * as LGL from "./WebGL.js";


export const baseVertexShader = compileShader(gl.VERTEX_SHADER, `
precision highp float;

attribute vec2 aPosition;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform vec2 texelSize;

void main () {
    vUv = aPosition * 0.5 + 0.5;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`);


export const noiseVertexShader = compileShader(gl.VERTEX_SHADER, `#version 300 es
precision highp float;

in vec2 aPosition;
out vec2 vUv;
out vec2 vL;
out vec2 vR;
out vec2 vT;
out vec2 vB;
uniform vec2 texelSize;

void main () {
    vUv = aPosition * 0.5 + 0.5;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`);


//create a vertex shader with blurred coordiates for neighbors 
//just horizontal blur
export const blurVertexShader = compileShader(gl.VERTEX_SHADER, `
precision highp float;

attribute vec2 aPosition;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
uniform vec2 texelSize;

void main () {
vUv = aPosition * 0.5 + 0.5;
float offset = 1.33333333;
vL = vUv - texelSize * offset;
vR = vUv + texelSize * offset;
gl_Position = vec4(aPosition, 0.0, 1.0);
}
`);


//TODO - seems like this should be updated to a gaussian blur or something 
export const blurShader = compileShader(gl.FRAGMENT_SHADER, `
precision mediump float;
precision mediump sampler2D;

varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
uniform sampler2D uTexture;

void main () {
    vec4 sum = texture2D(uTexture, vUv) * 0.29411764;
    sum += texture2D(uTexture, vL) * 0.35294117;
    sum += texture2D(uTexture, vR) * 0.35294117;
    gl_FragColor = sum;
}
`);

export const copyShader = compileShader(gl.FRAGMENT_SHADER, `
precision mediump float;
precision mediump sampler2D;

varying highp vec2 vUv;
uniform sampler2D uTexture;

void main () {
    gl_FragColor = texture2D(uTexture, vUv);
}
`);

export const clearShader = compileShader(gl.FRAGMENT_SHADER, `
precision mediump float;
precision mediump sampler2D;

varying highp vec2 vUv;
uniform sampler2D uTexture;
uniform float value;

void main () {
    gl_FragColor = value * texture2D(uTexture, vUv);
}
`);

export const colorShader = compileShader(gl.FRAGMENT_SHADER, `
precision mediump float;

uniform vec4 color;

void main () {
    gl_FragColor = color;
}
`);

export const checkerboardShader = compileShader(gl.FRAGMENT_SHADER, `
precision highp float;
precision highp sampler2D;

varying vec2 vUv;
uniform sampler2D uTexture;
uniform float aspectRatio;

#define SCALE 25.0

void main () {
    vec2 uv = floor(vUv * SCALE * vec2(aspectRatio, 1.0));
    float v = mod(uv.x + uv.y, 2.0);
    v = v * 0.1 + 0.8;
    gl_FragColor = vec4(vec3(v), 1.0);
}
`);

export const displayShaderSource = `
precision highp float;
precision highp sampler2D;

varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform sampler2D uTexture;
uniform sampler2D uBloom;
uniform sampler2D uSunrays;
uniform sampler2D uDithering;
uniform vec2 ditherScale;
uniform vec2 texelSize;

vec3 linearToGamma (vec3 color) {
    color = max(color, vec3(0));
    return max(1.055 * pow(color, vec3(0.416666667)) - 0.055, vec3(0));
}

void main () {
    vec3 c = texture2D(uTexture, vUv).rgb;

#ifdef SHADING
    vec3 lc = texture2D(uTexture, vL).rgb;
    vec3 rc = texture2D(uTexture, vR).rgb;
    vec3 tc = texture2D(uTexture, vT).rgb;
    vec3 bc = texture2D(uTexture, vB).rgb;

    float dx = length(rc) - length(lc);
    float dy = length(tc) - length(bc);

    vec3 n = normalize(vec3(dx, dy, length(texelSize)));
    vec3 l = vec3(0.0, 0.0, 1.0);

    float diffuse = clamp(dot(n, l) + 0.7, 0.7, 1.0);
    c *= diffuse;
#endif

#ifdef BLOOM
    vec3 bloom = texture2D(uBloom, vUv).rgb;
#endif

#ifdef SUNRAYS
    float sunrays = texture2D(uSunrays, vUv).r;
    c *= sunrays;
#ifdef BLOOM
    bloom *= sunrays;
#endif
#endif

#ifdef BLOOM
    float noise = texture2D(uDithering, vUv * ditherScale).r;
    noise = noise * 2.0 - 1.0;
    bloom += noise / 255.0;
    bloom = linearToGamma(bloom);
    c += bloom;
#endif

    float a = max(c.r, max(c.g, c.b));
    gl_FragColor = vec4(c, a);
}
`;

export const bloomPrefilterShader = compileShader(gl.FRAGMENT_SHADER, `
precision mediump float;
precision mediump sampler2D;

varying vec2 vUv;
uniform sampler2D uTexture;
uniform vec3 curve;
uniform float threshold;

void main () {
    vec3 c = texture2D(uTexture, vUv).rgb;
    float br = max(c.r, max(c.g, c.b));
    float rq = clamp(br - curve.x, 0.0, curve.y);
    rq = curve.z * rq * rq;
    c *= max(rq, br - threshold) / max(br, 0.0001);
    gl_FragColor = vec4(c, 0.0);
}
`);

export const bloomBlurShader = compileShader(gl.FRAGMENT_SHADER, `
precision mediump float;
precision mediump sampler2D;

varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform sampler2D uTexture;

void main () {
    vec4 sum = vec4(0.0);
    sum += texture2D(uTexture, vL);
    sum += texture2D(uTexture, vR);
    sum += texture2D(uTexture, vT);
    sum += texture2D(uTexture, vB);
    sum *= 0.25;
    gl_FragColor = sum;
}
`);

export const bloomFinalShader = compileShader(gl.FRAGMENT_SHADER, `
precision mediump float;
precision mediump sampler2D;

varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform sampler2D uTexture;
uniform float intensity;

void main () {
    vec4 sum = vec4(0.0);
    sum += texture2D(uTexture, vL);
    sum += texture2D(uTexture, vR);
    sum += texture2D(uTexture, vT);
    sum += texture2D(uTexture, vB);
    sum *= 0.25;
    gl_FragColor = sum * intensity;
}
`);

export const sunraysMaskShader = compileShader(gl.FRAGMENT_SHADER, `
precision highp float;
precision highp sampler2D;

varying vec2 vUv;
uniform sampler2D uTexture;

void main () {
    vec4 c = texture2D(uTexture, vUv);
    float br = max(c.r, max(c.g, c.b));
    c.a = 1.0 - min(max(br * 20.0, 0.0), 0.8);
    gl_FragColor = c;
}
`);

export const sunraysShader = compileShader(gl.FRAGMENT_SHADER, `
precision highp float;
precision highp sampler2D;

varying vec2 vUv;
uniform sampler2D uTexture;
uniform float weight;

#define ITERATIONS 16

void main () {
    float Density = 0.3;
    float Decay = 0.95;
    float Exposure = 0.7;

    vec2 coord = vUv;
    vec2 dir = vUv - 0.5;

    dir *= 1.0 / float(ITERATIONS) * Density;
    float illuminationDecay = 1.0;

    float color = texture2D(uTexture, vUv).a;

    for (int i = 0; i < ITERATIONS; i++)
    {
        coord -= dir;
        float col = texture2D(uTexture, coord).a;
        color += col * illuminationDecay * weight;
        illuminationDecay *= Decay;
    }

    gl_FragColor = vec4(color * Exposure, 0.0, 0.0, 1.0);
}
`);


//lets get some noise! 
//noise shader saved in project dir
export const noiseShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision highp float;

uniform float uPeriod;
uniform vec3 uTranslate;
uniform float uAmplitude;
uniform float uSeed;
uniform float uExponent;
uniform float uRidgeThreshold;
uniform vec3 uScale;
uniform float uAspect;
uniform float uLacunarity;
uniform float uGain;
uniform int uOctaves;

#define Index 1
#define PI 3.141592653589793
#define TWOPI 6.28318530718

in vec2 vUv;

out vec4 fragColor;


vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

// vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }
// float snoise(vec2 v, sampler2D tex){
// 	return texture(tex, vUV.st).r;
// }
//	Simplex 3D Noise 
//	by Ian McEwan, Ashima Arts
//
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

float snoise(vec3 v){ 
const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
vec3 i  = floor(v + dot(v, C.yyy) );
vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
vec3 g = step(x0.yzx, x0.xyz);
vec3 l = 1.0 - g;
vec3 i1 = min( g.xyz, l.zxy );
vec3 i2 = max( g.xyz, l.zxy );

//  x0 = x0 - 0. + 0.0 * C 
vec3 x1 = x0 - i1 + 1.0 * C.xxx;
vec3 x2 = x0 - i2 + 2.0 * C.xxx;
vec3 x3 = x0 - 1. + 3.0 * C.xxx;

// Permutations
i = mod(i, 289.0 ); 
vec4 p = permute( permute( permute( 
    i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
    + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
    + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
    
    // Gradients
    // ( N*N points uniformly over a square, mapped onto an octahedron.)
    float n_ = 1.0/7.0; // N=7
    vec3  ns = n_ * D.wyz - D.xzx;
    
    vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)
    
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)
    
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    
    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );
    
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
    
    vec3 p0 = vec3(a0.xy,h.x);
vec3 p1 = vec3(a0.zw,h.y);
vec3 p2 = vec3(a1.xy,h.z);
vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
p0 *= norm.x;
p1 *= norm.y;
p2 *= norm.z;
p3 *= norm.w;

// Mix final noise value
vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
m = m * m;
return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                            dot(p2,x2), dot(p3,x3) ) );
}
// float ridge(vec2 st, float t){
//     float n = snoise(st);
//     n = abs(t + (n-t));
//     return n/t;
// }

float ridge(float n, float threshold){
return (abs(threshold + (n - threshold))/threshold);
}

float power_noise(float n, float power){
return pow(n, power);
}

float monoNoise(vec3 st){
st.x /= uAspect;
st *= uScale;
st *= uPeriod;
st.z += uSeed;
st += uTranslate;
float noise = snoise(st);
noise *= uAmplitude;
noise = ridge(noise, uRidgeThreshold);
noise = power_noise(noise, uExponent);
return noise;
}


#define FBM(NOISE, SEED) float G=uGain; float freq = 1.0; float a = 1.0; float t = 0.0;for(int i=0; i<8; i++){t+= a*NOISE(freq*st, SEED);freq*=uLacunarity;a*=G;}


float monoSimplex(vec3 st, float seed){ 
st.x /= uAspect;
st *= uScale;
st *= uPeriod;
st.z += uSeed + seed;
st += uTranslate;
float noise = snoise(st);
noise *= uAmplitude;
noise = ridge(noise, uRidgeThreshold);
noise = power_noise(noise, uExponent);
return noise;
}

float monoSimplex(vec3 st){
st.x /= uAspect;
st *= uScale;
st *= uPeriod;
st.z += uSeed;
st += uTranslate;
float noise = snoise(st);
noise *= uAmplitude;
noise = ridge(noise, uRidgeThreshold);
noise = power_noise(noise, uExponent);
return noise;
}
vec2 rotate2D(vec2 _st, float _angle){
_st -= 0.5;
_st =  mat2(cos(_angle),-sin(_angle),
            sin(_angle),cos(_angle)) * _st;
_st += 0.5;
return _st;
}

vec2 tile(vec2 _st, float _zoom){
_st *= _zoom;
return fract(_st);
}

float box(vec2 _st, vec2 _size, float _smoothEdges){
_size = vec2(0.5)-_size*0.5;
vec2 aa = vec2(_smoothEdges*0.5);
vec2 uv = smoothstep(_size,_size+aa,_st);
uv *= smoothstep(_size,_size+aa,vec2(1.0)-_st);
return uv.x*uv.y;
}

float grid(vec2 st, float res, float smoothEdges, float rotate){
// Divide the space in 4
vec2 uv = st;
uv.x /= uAspect; 
uv = tile(uv,res);
// Use a matrix to rotate the space 45 degrees
uv = rotate2D(uv,rotate);

// Draw a square
return box(uv, vec2(.9), smoothEdges);
}

vec4 rgbSimplex(vec3 st){
float n1 = monoSimplex(st, 0.0); //take orig seed 
float n2 = monoSimplex(st, uSeed +500.0);
float n3 = monoSimplex(st, uSeed -500.0);
return vec4(n1, n2, n3, 1.0);
}

vec4 rgbSimplex(vec3 st, float seed){
float n1 = monoSimplex(st, 0.0 + seed); //take orig seed 
float n2 = monoSimplex(st, uSeed + 500.0 + seed);
float n3 = monoSimplex(st, uSeed - 500.0 + seed);
return vec4(n1, n2, n3, 1.0);
}


vec2 displace(vec2 st, vec2 vector, float scale){
vec2 offset = vec2(0.5);
vec2 midpoint = vec2(0.5);
vec2 uvi = st + scale * (vector.xy - midpoint.xy); 
return uvi;
}

vec3 displace(vec3 st , vec2 vector, float scale){ //overload to pass vec3s
vec2 offset = vec2(0.5);
vec2 midpoint = vec2(0.5);
vec2 uvi = st.xy * scale * (vector.xy);
return vec3(uvi, st.z);
}

#define NOISE_RGB(NOISE, SEED) vec3 noiseRGB = vec3(NOISE(st, uSeed + SEED), NOISE(st, uSeed + 500.0 + SEED), NOISE(st, uSeed - 500.0 + SEED));

vec3 displace(vec3 st , vec2 vector, vec2 offset, vec2 midpoint, float scale){ //overload to pass vec3s
vec2 uvi = st.xy + scale * (vector.xy); 
return vec3(uvi, st.z);
}

float recursiveWarpNoise(vec3 st, float seed){
float color = monoSimplex( st*st.x, 2.0) * monoSimplex(displace(st, st.xy, vec2(0.5), vec2(0.5), 0.1));
for(int i = 0; i<5; i++){
NOISE_RGB(monoSimplex, 2.4);

color = monoSimplex(displace(st, noiseRGB.rg*float(i)/5.0, vec2(.5), vec2(0.5), 0.05*float(i)), seed*float(i));
}
return color;
}

float ang(vec3 st){
return sin(st.y*st.x);
}


float dis(vec3 st){
float d = grid(vUv, 7.0, 0.45, PI/4.);
// d = texture(sTD2DInputs[0],vUv).r;
FBM(monoSimplex, -743.4838)
return d *t;
}

float dis2(vec3 st){
NOISE_RGB(monoSimplex, 2.4);
FBM(recursiveWarpNoise, 2.4);
return t;
}


#define EPSILON 0.0001

#define GRAD(NOISE, SEED) float st1 = NOISE(vec3(st.x + EPSILON, st.y, st.z), SEED).r;  float st2 = NOISE(vec3(st.x - EPSILON, st.y, st.z), SEED).r; float st3 = NOISE(vec3(st.x, st.y + EPSILON, st.z), SEED).r; float st4 = NOISE(vec3(st.x, st.y - EPSILON, st.z), SEED).r; vec2 grad = normalize(vec2(st1-st2, st3-st4));

#define DISP(ANG, DIST, MAX) st.xy = st.xy + vec2(cos(ANG(st)*TWOPI), sin(ANG(st)*TWOPI)) * DIST(st) * MAX;


vec4 palette[5] = vec4[5](vec4(.875, .0859375, 0.16796875, 1.0), vec4(1.), vec4(0,.3203125, 0.64453125, 1.0), vec4(0.0, 0.0, 0.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0));

vec4 fbm(vec3 st, float seed){
float G=uGain; 
float freq = 1.0; 
float a = 1.0; 
vec4 t = vec4(0.0);
for(int i=0; i<8; i++){
t += a*rgbSimplex(freq*st, seed);
freq*= uLacunarity;
//freq = pow(2.0, float(i));
a*=G;
}
return t;
}

void main()
{
//create vec3 with z value for translate
vec3 st = vec3(vUv, 0.0);
//create vec3 with z value for translate
NOISE_RGB(monoSimplex, 2.4);
// FBM(recursiveWarpNoise, 2.4);
vec4 color = fbm(st, uSeed); 
//output


fragColor = (color);

}
`);


export const errataNoiseShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision highp float;

uniform float uPeriod;
uniform vec3 uTranslate;
uniform float uAmplitude;
uniform float uSeed;
uniform float uExponent;
uniform float uRidgeThreshold;
uniform vec3 uScale;
uniform float uAspect;
uniform float uLacunarity;
uniform float uGain;
uniform int uOctaves;

#define Index 1
#define PI 3.141592653589793
#define TWOPI 6.28318530718

in vec2 vUv;

out vec4 fragColor;


vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

// vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }
// float snoise(vec2 v, sampler2D tex){
// 	return texture(tex, vUV.st).r;
// }
//	Simplex 3D Noise 
//	by Ian McEwan, Ashima Arts
//
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

float snoise(vec3 v){ 
const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
vec3 i  = floor(v + dot(v, C.yyy) );
vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
vec3 g = step(x0.yzx, x0.xyz);
vec3 l = 1.0 - g;
vec3 i1 = min( g.xyz, l.zxy );
vec3 i2 = max( g.xyz, l.zxy );

//  x0 = x0 - 0. + 0.0 * C 
vec3 x1 = x0 - i1 + 1.0 * C.xxx;
vec3 x2 = x0 - i2 + 2.0 * C.xxx;
vec3 x3 = x0 - 1. + 3.0 * C.xxx;

// Permutations
i = mod(i, 289.0 ); 
vec4 p = permute( permute( permute( 
    i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
    + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
    + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
    
    // Gradients
    // ( N*N points uniformly over a square, mapped onto an octahedron.)
    float n_ = 1.0/7.0; // N=7
    vec3  ns = n_ * D.wyz - D.xzx;
    
    vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)
    
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)
    
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    
    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );
    
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
    
    vec3 p0 = vec3(a0.xy,h.x);
vec3 p1 = vec3(a0.zw,h.y);
vec3 p2 = vec3(a1.xy,h.z);
vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
p0 *= norm.x;
p1 *= norm.y;
p2 *= norm.z;
p3 *= norm.w;

// Mix final noise value
vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
m = m * m;
return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                            dot(p2,x2), dot(p3,x3) ) );
}
// float ridge(vec2 st, float t){
//     float n = snoise(st);
//     n = abs(t + (n-t));
//     return n/t;
// }

float ridge(float n, float threshold){
return (abs(threshold + (n - threshold))/threshold);
}

float power_noise(float n, float power){
return pow(n, power);
}

float monoNoise(vec3 st){
st.x /= uAspect;
st *= uScale;
st *= uPeriod;
st.z += uSeed;
st += uTranslate;
float noise = snoise(st);
noise *= uAmplitude;
noise = ridge(noise, uRidgeThreshold);
noise = power_noise(noise, uExponent);
return noise;
}


#define FBM(NOISE, SEED) float G=uGain; float freq = 1.0; float a = 1.0; float t = 0.0;for(int i=0; i<8; i++){t+= a*NOISE(freq*st, SEED);freq*=uLacunarity;a*=G;}


float monoSimplex(vec3 st, float seed){ 
st.x /= uAspect;
st *= uScale;
st *= uPeriod;
st.z += uSeed + seed;
st += uTranslate;
float noise = snoise(st);
noise *= uAmplitude;
noise = ridge(noise, uRidgeThreshold);
noise = power_noise(noise, uExponent);
return noise;
}

float monoSimplex(vec3 st){
st.x /= uAspect;
st *= uScale;
st *= uPeriod;
st.z += uSeed;
st += uTranslate;
float noise = snoise(st);
noise *= uAmplitude;
noise = ridge(noise, uRidgeThreshold);
noise = power_noise(noise, uExponent);
return noise;
}
vec2 rotate2D(vec2 _st, float _angle){
_st -= 0.5;
_st =  mat2(cos(_angle),-sin(_angle),
            sin(_angle),cos(_angle)) * _st;
_st += 0.5;
return _st;
}

vec2 tile(vec2 _st, float _zoom){
_st *= _zoom;
return fract(_st);
}

float box(vec2 _st, vec2 _size, float _smoothEdges){
_size = vec2(0.5)-_size*0.5;
vec2 aa = vec2(_smoothEdges*0.5);
vec2 uv = smoothstep(_size,_size+aa,_st);
uv *= smoothstep(_size,_size+aa,vec2(1.0)-_st);
return uv.x*uv.y;
}

float grid(vec2 st, float res, float smoothEdges, float rotate){
// Divide the space in 4
vec2 uv = st;
uv.x /= uAspect; 
uv = tile(uv,res);
// Use a matrix to rotate the space 45 degrees
uv = rotate2D(uv,rotate);

// Draw a square
return box(uv, vec2(.9), smoothEdges);
}

vec4 rgbSimplex(vec3 st){
float n1 = monoSimplex(st, 0.0); //take orig seed 
float n2 = monoSimplex(st, uSeed +500.0);
float n3 = monoSimplex(st, uSeed -500.0);
return vec4(n1, n2, n3, 1.0);
}

vec4 rgbSimplex(vec3 st, float seed){
float n1 = monoSimplex(st, 0.0 + seed); //take orig seed 
float n2 = monoSimplex(st, uSeed + 500.0 + seed);
float n3 = monoSimplex(st, uSeed - 500.0 + seed);
return vec4(n1, n2, n3, 1.0);
}


vec2 displace(vec2 st, vec2 vector, float scale){
vec2 offset = vec2(0.5);
vec2 midpoint = vec2(0.5);
vec2 uvi = st + scale * (vector.xy - midpoint.xy); 
return uvi;
}

vec3 displace(vec3 st , vec2 vector, float scale){ //overload to pass vec3s
vec2 offset = vec2(0.5);
vec2 midpoint = vec2(0.5);
vec2 uvi = st.xy * scale * (vector.xy);
return vec3(uvi, st.z);
}

#define NOISE_RGB(NOISE, SEED) vec3 noiseRGB = vec3(NOISE(st, uSeed + SEED), NOISE(st, uSeed + 500.0 + SEED), NOISE(st, uSeed - 500.0 + SEED));

vec3 displace(vec3 st , vec2 vector, vec2 offset, vec2 midpoint, float scale){ //overload to pass vec3s
vec2 uvi = st.xy + scale * (vector.xy); 
return vec3(uvi, st.z);
}

float recursiveWarpNoise(vec3 st, float seed){
float color = monoSimplex( st*st.x, 2.0) * monoSimplex(displace(st, st.xy, vec2(0.5), vec2(0.5), 0.1));
for(int i = 0; i<5; i++){
NOISE_RGB(monoSimplex, 2.4);

color = monoSimplex(displace(st, noiseRGB.rg*float(i)/5.0, vec2(.5), vec2(0.5), 0.05*float(i)), seed*float(i));
}
return color;
}

float ang(vec3 st){
return sin(st.y*st.x);
}


float dis(vec3 st){
float d = grid(vUv, 7.0, 0.45, PI/4.);
// d = texture(sTD2DInputs[0],vUv).r;
FBM(monoSimplex, -743.4838)
return d *t;
}

float dis2(vec3 st){
NOISE_RGB(monoSimplex, 2.4);
FBM(recursiveWarpNoise, 2.4);
return t;
}


#define EPSILON 0.0001

#define GRAD(NOISE, SEED) float st1 = NOISE(vec3(st.x + EPSILON, st.y, st.z), SEED).r;  float st2 = NOISE(vec3(st.x - EPSILON, st.y, st.z), SEED).r; float st3 = NOISE(vec3(st.x, st.y + EPSILON, st.z), SEED).r; float st4 = NOISE(vec3(st.x, st.y - EPSILON, st.z), SEED).r; vec2 grad = normalize(vec2(st1-st2, st3-st4));

#define DISP(ANG, DIST, MAX) st.xy = st.xy + vec2(cos(ANG(st)*TWOPI), sin(ANG(st)*TWOPI)) * DIST(st) * MAX;


vec4 palette[5] = vec4[5](vec4(.875, .0859375, 0.16796875, 1.0), vec4(1.), vec4(0,.3203125, 0.64453125, 1.0), vec4(0.0, 0.0, 0.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0));

vec4 fbm(vec3 st, float seed){
float G=uGain; 
float freq = 1.0; 
float a = 1.0; 
vec4 t = vec4(0.0);
for(int i=0; i<8; i++){
t += a*rgbSimplex(freq*st, seed);
freq*= uLacunarity;
//freq = pow(2.0, float(i));
a*=G;
}
return t;
}

void main()
{
//create vec3 with z value for translate
vec3 st = vec3(vUv, 0.0);
DISP(ang, dis, monoSimplex(st, 4444.0)*0.25);
DISP(ang, dis2, 0.5);
float seed = 1.8;
float noise = monoSimplex(st, seed);
int idx = int(floor(noise*5.0));
float pct = fract(noise *5.0);
vec4 color = mix(palette[int(mod(float(idx),5.))], palette[int(mod(float(idx)+1.0,5.0))], pct);
// FBM(recursiveWarpNoise, 2.4);
// vec4 color = vec4(noise);
//output

fragColor = (color);

}
`);

export const malformedNoiseShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision highp float;

uniform float uPeriod;
uniform vec3 uTranslate;
uniform float uAmplitude;
uniform float uSeed;
uniform float uExponent;
uniform float uRidgeThreshold;
uniform vec3 uScale;
uniform float uAspect;
uniform float uLacunarity;
uniform float uGain;
uniform int uOctaves;
uniform sampler2D palette;


uniform float uNoiseMix;
uniform float uFrequency;
uniform float uMaxDist;

uniform vec4 uColor1;
uniform vec4 uColor2;
uniform vec4 uColor3;
uniform vec4 uColor4;
uniform vec4 uColor5;


#define Index 1
#define PI 3.141592653589793
#define TWOPI 6.28318530718

in vec2 vUv;

out vec4 fragColor;


vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

// vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }
// float snoise(vec2 v, sampler2D tex){
// 	return texture(tex, vUV.st).r;
// }
//	Simplex 3D Noise 
//	by Ian McEwan, Ashima Arts
//
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

float snoise(vec3 v){ 
const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
vec3 i  = floor(v + dot(v, C.yyy) );
vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
vec3 g = step(x0.yzx, x0.xyz);
vec3 l = 1.0 - g;
vec3 i1 = min( g.xyz, l.zxy );
vec3 i2 = max( g.xyz, l.zxy );

//  x0 = x0 - 0. + 0.0 * C 
vec3 x1 = x0 - i1 + 1.0 * C.xxx;
vec3 x2 = x0 - i2 + 2.0 * C.xxx;
vec3 x3 = x0 - 1. + 3.0 * C.xxx;

// Permutations
i = mod(i, 289.0 ); 
vec4 p = permute( permute( permute( 
    i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
    + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
    + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
    
    // Gradients
    // ( N*N points uniformly over a square, mapped onto an octahedron.)
    float n_ = 1.0/7.0; // N=7
    vec3  ns = n_ * D.wyz - D.xzx;
    
    vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)
    
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)
    
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    
    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );
    
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
    
    vec3 p0 = vec3(a0.xy,h.x);
vec3 p1 = vec3(a0.zw,h.y);
vec3 p2 = vec3(a1.xy,h.z);
vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
p0 *= norm.x;
p1 *= norm.y;
p2 *= norm.z;
p3 *= norm.w;

// Mix final noise value
vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
m = m * m;
return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                            dot(p2,x2), dot(p3,x3) ) );
}
// float ridge(vec2 st, float t){
//     float n = snoise(st);
//     n = abs(t + (n-t));
//     return n/t;
// }

float ridge(float n, float threshold){
return (abs(threshold + (n - threshold))/threshold);
}

float power_noise(float n, float power){
return pow(n, power);
}

float monoNoise(vec3 st){
st.x /= uAspect;
st *= uScale;
st *= uPeriod;
st.z += uSeed;
st += uTranslate;
float noise = snoise(st);
noise *= uAmplitude;
noise = ridge(noise, uRidgeThreshold);
noise = power_noise(noise, uExponent);
return noise;
}


#define FBM(NOISE, SEED) float G=uGain; float freq = 1.0; float a = 1.0; float t = 0.0;for(int i=0; i<uOctaves; i++){t+= a*NOISE(freq*st, SEED);freq*=uLacunarity;a*=G;}


float monoSimplex(vec3 st, float seed){ 
st.x /= uAspect;
st *= uScale;
st *= uPeriod;
st.z += uSeed + seed;
st += uTranslate;
float noise = snoise(st);
noise *= uAmplitude;
noise = ridge(noise, uRidgeThreshold);
noise = power_noise(noise, uExponent);
return noise;
}

float monoSimplex(vec3 st){
st.x /= uAspect;
st *= uScale;
st *= uPeriod;
st.z += uSeed;
st += uTranslate;
float noise = snoise(st);
noise *= uAmplitude;
noise = ridge(noise, uRidgeThreshold);
noise = power_noise(noise, uExponent);
return noise;
}
vec2 rotate2D(vec2 _st, float _angle){
_st -= 0.5;
_st =  mat2(cos(_angle),-sin(_angle),
            sin(_angle),cos(_angle)) * _st;
_st += 0.5;
return _st;
}

vec2 tile(vec2 _st, float _zoom){
_st *= _zoom;
return fract(_st);
}

float box(vec2 _st, vec2 _size, float _smoothEdges){
_size = vec2(0.5)-_size*0.5;
vec2 aa = vec2(_smoothEdges*0.5);
vec2 uv = smoothstep(_size,_size+aa,_st);
uv *= smoothstep(_size,_size+aa,vec2(1.0)-_st);
return uv.x*uv.y;
}

float grid(vec2 st, float res, float smoothEdges, float rotate){
// Divide the space in 4
vec2 uv = st;
uv.x /= uAspect; 
uv = tile(uv,res);
// Use a matrix to rotate the space 45 degrees
uv = rotate2D(uv,rotate);

// Draw a square
return box(uv, vec2(.9), smoothEdges);
}

vec4 rgbSimplex(vec3 st){
float n1 = monoSimplex(st, 0.0); //take orig seed 
float n2 = monoSimplex(st, uSeed +500.0);
float n3 = monoSimplex(st, uSeed -500.0);
return vec4(n1, n2, n3, 1.0);
}

vec4 rgbSimplex(vec3 st, float seed){
float n1 = monoSimplex(st, 0.0 + seed); //take orig seed 
float n2 = monoSimplex(st, uSeed + 500.0 + seed);
float n3 = monoSimplex(st, uSeed - 500.0 + seed);
return vec4(n1, n2, n3, 1.0);
}


vec2 displace(vec2 st, vec2 vector, float scale){
vec2 offset = vec2(0.5);
vec2 midpoint = vec2(0.5);
vec2 uvi = st + scale * (vector.xy - midpoint.xy); 
return uvi;
}

vec3 displace(vec3 st , vec2 vector, float scale){ //overload to pass vec3s
vec2 offset = vec2(0.5);
vec2 midpoint = vec2(0.5);
vec2 uvi = st.xy * scale * (vector.xy);
return vec3(uvi, st.z);
}

#define NOISE_RGB(NOISE, SEED) vec3 noiseRGB = vec3(NOISE(st, uSeed + SEED), NOISE(st, uSeed + 500.0 + SEED), NOISE(st, uSeed - 500.0 + SEED));

vec3 displace(vec3 st , vec2 vector, vec2 offset, vec2 midpoint, float scale){ //overload to pass vec3s
vec2 uvi = st.xy + scale * (vector.xy); 
return vec3(uvi, st.z);
}

float recursiveWarpNoise(vec3 st, float seed){
float color = monoSimplex( st*st.x, 2.0) * monoSimplex(displace(st, st.xy, vec2(0.5), vec2(0.5), 0.1));
for(int i = 0; i<5; i++){
NOISE_RGB(monoSimplex, 2.4);

color = monoSimplex(displace(st, noiseRGB.rg*float(i)/5.0, vec2(.5), vec2(0.5), 0.05*float(i)), seed*float(i));
}
return color;
}



float toBeNamedNoise(vec3 st, float seed){
    float color = 0.;// monoSimplex( st*st.x, 2.0) * monoSimplex(displace(st, st.xy, vec2(0.5), vec2(0.5), 0.1));
    float m = monoSimplex(st, seed * 234.54);
    color = monoSimplex(displace(st, vec2(m), vec2(0.5), vec2(0.5), m), seed);
    return color;
  }


  float dis(vec3 st){
    //st.xy *= vec2(2.3, 1.5);
    FBM(toBeNamedNoise, 2.4);
    t += 0.;
    return t;
  }

float dis2(vec3 st){
NOISE_RGB(monoSimplex, 2.4);
FBM(recursiveWarpNoise, 2.4);
return t;
}


#define EPSILON 0.0001

#define GRAD(NOISE, SEED) float st1 = NOISE(vec3(st.x + EPSILON, st.y, st.z), SEED).r;  float st2 = NOISE(vec3(st.x - EPSILON, st.y, st.z), SEED).r; float st3 = NOISE(vec3(st.x, st.y + EPSILON, st.z), SEED).r; float st4 = NOISE(vec3(st.x, st.y - EPSILON, st.z), SEED).r; vec2 grad = normalize(vec2(st1-st2, st3-st4));

#define DISP(ANG, DIST, MAX) st.xy = st.xy + vec2(cos(ANG(st)*TWOPI), sin(ANG(st)*TWOPI)) * DIST(st) * MAX(st);


// vec4 test[5] = vec4[5](uColor1, uColor2, uColor3, uColor4, uColor5);


vec4 palettes[5] = vec4[5](vec4(.875, .0859375, 0.16796875, 1.0), vec4(1.), vec4(0,.3203125, 0.64453125, 1.0), vec4(0.0, 0.0, 0.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0));

vec4 fbm(vec3 st, float seed){
    float G=uGain; 
    float freq = 1.0; 
    float a = 1.0; 
    vec4 t = vec4(0.0);
    for(int i=0; i<uOctaves; i++){
    t += a*rgbSimplex(freq*st, seed);
    freq*= uLacunarity;
    //freq = pow(2.0, float(i));
    a*=G;
    }
    return t;
}

vec4 fbm(vec3 st, float g, float lac, int oct){
    float freq = 1.0; 
    float a = 1.0; 
    vec4 t = vec4(0.0);
    for(int i=0; i<oct; i++){
    t += a*rgbSimplex(freq*st);
    freq*= lac;
    //freq = pow(2.0, float(i));
    a*=g;
    }
    return t;
}

float ang(vec3 st){
	vec3 tst = st;
	st.y *= 5.5;
	st.x *= 3.9;
  return monoSimplex(tst, 44.333) * fbm(st, 0.6, 2.0, 8).r;
}

float ridge(vec3 st, float t){
    float n = monoSimplex(st);
    n = abs(t + (n-t));
    return n/t;
}

float falloff(vec3 st){
	st += ridge(st, 0.5) * st.y;
  return min(monoSimplex(st), uMaxDist);
} 


float noise(vec3 st){
	float r = fbm(st, 0.5, 2.0, 8).r;
	r = mix(r, fbm(st, uGain, uLacunarity, 8).r, uNoiseMix);
  return r;
}

void main()
{
//create vec3 with z value for translate
vec3 st = vec3(vUv*2.0-1.0, 0.0);
DISP(ang, dis, falloff);
// DISP(ang, dis2, 0.5);
float seed = 1.8;
float noise = noise(st);
float idx = (floor(noise*5.0));
float pct = fract(noise *5.0);
// vec4 pal2 = texture(palette, vec2((idx+1.0)/5.0,0));
// vec4 pal1 = texture(palette, vec2(idx/5.0,0));
// vec4 color = mix(pal1, pal2, smoothstep(0., 1., pct));
vec4 color = mix(palettes[int(mod(float(idx),5.0))], palettes[int(mod(float(idx)+1.0,5.0))], smoothstep(0.,1.,pct));
// FBM(recursiveWarpNoise, 2.4);
// vec4 color = vec4(noise);
//output

fragColor = ((color));

}
`);
