
'use strict';


// Simulation section

//getElementsByTagName fxn will return a list of elements that meet your search criteria (familiar if used any browser traverse lib like selenium)
const canvas = document.getElementsByTagName('canvas')[0];
//function that will adjust canvas bounds in case screen size changes 
resizeCanvas();

//inital config for sim params 
let config = {
    SIM_RESOLUTION: 256, //simres
    DYE_RESOLUTION: 1024, //output res 
    ASPECT: 1.0,
    EXPONENT: 1.0,
    PERIOD: 3.0,
    RIDGE: 1.0,
    AMP: 1.0,
    LACUNARITY: 1.0,
    GAIN: 0.5,
    OCTAVES: 4,
    MONO: false,
    BACK_COLOR: { r: 0, g: 0, b: 0 },
    TRANSPARENT: false,
    BLOOM: false,
    BLOOM_ITERATIONS: 8,
    BLOOM_RESOLUTION: 256,
    BLOOM_INTENSITY: 0.8,
    BLOOM_THRESHOLD: 0.6,
    BLOOM_SOFT_KNEE: 0.7,
    PAUSED: false,
    SUNRAYS: true,
    SUNRAYS_RESOLUTION: 196,
    SHADING: true,
    SUNRAYS_WEIGHT: 0.5,
    NOISE_TRANSLATE_SPEED: 0.25
}


//create a prototype data structure for our pointers (ie a click or touch)
//we want to be a able to have more than one in the case of a multi - touch input 
function pointerPrototype () {
    this.id = -1;
    this.texcoordX = 0;
    this.texcoordY = 0;
    this.prevTexcoordX = 0;
    this.prevTexcoordY = 0;
    this.deltaX = 0;
    this.deltaY = 0;
    this.down = false;
    this.moved = false;
    this.color = [30, 0, 300];
}

//initialize arrays 
let pointers = [];
// let splatStack = [];


//add first pointer the array of pointers 
pointers.push(new pointerPrototype());

//create webgl context 
const { gl, ext } = getWebGLContext(canvas);

//set output res on mobile 
if (isMobile()) {
    config.DYE_RESOLUTION = 512;
}
//if the supported version of webgl does not support these features, turn off 
if (!ext.supportLinearFiltering) {
    config.DYE_RESOLUTION = 512;
    config.SHADING = false;
    config.BLOOM = false;
    config.SUNRAYS = false;
}

//start gui
startGUI();

function getWebGLContext (canvas) {
    const params = { alpha: true, depth: false, stencil: false, antialias: false, preserveDrawingBuffer: false };

    //get webgl context. note webgl2
    let gl = canvas.getContext('webgl2', params);
    const isWebGL2 = !!gl;
    if (!isWebGL2)
        gl = canvas.getContext('webgl', params) || canvas.getContext('experimental-webgl', params);

    //find out if our current webgl context supports certain features 
    let halfFloat;
    let supportLinearFiltering;
    if (isWebGL2) {
        gl.getExtension('EXT_color_buffer_float');
        supportLinearFiltering = gl.getExtension('OES_texture_float_linear');
    } else {
        halfFloat = gl.getExtension('OES_texture_half_float');
        supportLinearFiltering = gl.getExtension('OES_texture_half_float_linear');
    }

    gl.clearColor(0.0, 0.0, 0.0, 1.0);

    const halfFloatTexType = isWebGL2 ? gl.HALF_FLOAT : halfFloat.HALF_FLOAT_OES;
    let formatRGBA;
    let formatRG;
    let formatR;

    if (isWebGL2)//believe this is standardizing texture pixel format (aliases) based on webgl version
    {
        formatRGBA = getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, halfFloatTexType);
        formatRG = getSupportedFormat(gl, gl.RG16F, gl.RG, halfFloatTexType);
        formatR = getSupportedFormat(gl, gl.R16F, gl.RED, halfFloatTexType);
    }
    else
    {
        formatRGBA = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
        formatRG = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
        formatR = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
    }

    //ga() is for sending data to Google Analytics 
    ga('send', 'event', isWebGL2 ? 'webgl2' : 'webgl', formatRGBA == null ? 'not supported' : 'supported');

    return {
        gl,
        ext: {
            formatRGBA,
            formatRG,
            formatR,
            halfFloatTexType,
            supportLinearFiltering
        }
    };
}

function getSupportedFormat (gl, internalFormat, format, type)
{
    if (!supportRenderTextureFormat(gl, internalFormat, format, type))
    {
        switch (internalFormat)
        {
            case gl.R16F:
                return getSupportedFormat(gl, gl.RG16F, gl.RG, type);
            case gl.RG16F:
                return getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, type);
            default:
                return null;
        }
    }

    return {
        internalFormat,
        format
    }
}

//test case to check that the correct pixel types are supported 
//setup a gl texture
//set the texture params 
//create a 2d image tex
//create a frame buffer and bind the texture to it 
//check tosee if the buffer object correctly accepcted texture 
//TODO - enhance understanding of texture setup 
function supportRenderTextureFormat (gl, internalFormat, format, type) {
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 4, 4, 0, format, type, null);

    let fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

    let status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    return status == gl.FRAMEBUFFER_COMPLETE;
}


function bindForceWithDensityMap () {
    if (config.FORCE_MAP_ENABLE) config.DENSITY_MAP_ENABLE = true;
}

function startGUI () {
    const parName = 'quality_2';
    //dat is a library developed by Googles Data Team for building JS interfaces. Needs to be included in project directory 
    var gui = new dat.GUI({ width: 300 });
    gui.add(config, 'DYE_RESOLUTION', { 'high': 1024, 'medium': 512, 'low': 256, 'very low': 128 }).name(parName).onFinishChange(initFramebuffers);
    gui.add(config, 'SIM_RESOLUTION', { '32': 32, '64': 64, '128': 128, '256': 256 }).name('sim resolution').onFinishChange(initFramebuffers);
    gui.add(config, 'PERIOD', 0, 5.0).name('Period');
    gui.add(config, 'EXPONENT', 0, 4.0).name('Exponent');
    gui.add(config, 'RIDGE', 0, 1.5).name('Ridge');
    gui.add(config, 'AMP', 0, 4.0).name('Amplitude');
    gui.add(config, 'LACUNARITY', 0, 2).name('Lacunarity');
    gui.add(config, 'NOISE_TRANSLATE_SPEED', 0, 2).name('Noise Translate Speed');
    gui.add(config, 'GAIN', 0.0, 1.0).name('Gain');
    gui.add(config, 'OCTAVES', 0, 8).name('Octaves').step(1);
    gui.add(config, 'MONO').name('Mono');
    gui.add(config, 'SHADING').name('shading').onFinishChange(updateKeywords);
    gui.add(config, 'PAUSED').name('paused').listen();

    //create a function to assign to a button, here linking my github
    let github = gui.add({ fun : () => {
        window.open('https://github.com/lakeheck/Fluid-Simulation-WebGL');
        ga('send', 'event', 'link button', 'github');
    } }, 'fun').name('Github');
    github.__li.className = 'cr function bigFont';
    github.__li.style.borderLeft = '3px solid #8C8C8C';
    let githubIcon = document.createElement('span');
    github.domElement.parentElement.appendChild(githubIcon);
    githubIcon.className = 'icon github';

    if (isMobile())
        gui.close();
}

//TODO - dont understand the alchemy here but it works
function isMobile () {
    return /Mobi|Android/i.test(navigator.userAgent);
}


function captureScreenshot () {
    let res = getResolution(config.CAPTURE_RESOLUTION);
    //use helper fxn to create frame buffer to render for screenshot 
    let target = createFBO(res.width, res.height, ext.formatRGBA.internalFormat, ext.formatRGBA.format, ext.halfFloatTexType, gl.NEAREST);
    render(target);

    //create a texture from the frame buffer 
    let texture = framebufferToTexture(target);
    texture = normalizeTexture(texture, target.width, target.height);

    let captureCanvas = textureToCanvas(texture, target.width, target.height);
    let datauri = captureCanvas.toDataURL();
    //use helper fxn to download data 
    downloadURI('fluid.png', datauri);
    //tell browser we can forget about this url
    URL.revokeObjectURL(datauri);
}

function framebufferToTexture (target) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    let length = target.width * target.height * 4; //take length time width, and multiply by 4 since we have 4 channels (rgba)
    let texture = new Float32Array(length);
    //webgl fxn that will read pixels into a textue (texture type needs to match passed pixel data type, eg gl.FLOAT and Float32Array)
    gl.readPixels(0, 0, target.width, target.height, gl.RGBA, gl.FLOAT, texture);
    return texture;
}


//helper to rerange to integer values on [0,255] and return array of unsigned ints 
function normalizeTexture (texture, width, height) {
    let result = new Uint8Array(texture.length);
    let id = 0;
    for (let i = height - 1; i >= 0; i--) {
        for (let j = 0; j < width; j++) {
            let nid = i * width * 4 + j * 4;
            result[nid + 0] = clamp01(texture[id + 0]) * 255;
            result[nid + 1] = clamp01(texture[id + 1]) * 255;
            result[nid + 2] = clamp01(texture[id + 2]) * 255;
            result[nid + 3] = clamp01(texture[id + 3]) * 255;
            id += 4;
        }
    }
    return result;
}

function clamp01 (input) {
    return Math.min(Math.max(input, 0), 1);
}

function textureToCanvas (texture, width, height) {
    let captureCanvas = document.createElement('canvas');
    let ctx = captureCanvas.getContext('2d');
    captureCanvas.width = width;
    captureCanvas.height = height;
    //createImageData comes from the canvas 2d api
    let imageData = ctx.createImageData(width, height);
    //set data with our texture 
    imageData.data.set(texture);
    //render texture to canvas
    ctx.putImageData(imageData, 0, 0);

    return captureCanvas;
}

//helper function that creates a temp element on our site 
//this element is populated with a download link from HTMLCanvasElement.toDataURL()
//and then is virtually clicked, initiating download 
//then the link is removed
function downloadURI (filename, uri) {
    let link = document.createElement('a');
    link.download = filename;
    link.href = uri;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

class Material {
    constructor (vertexShader, fragmentShaderSource) {
        this.vertexShader = vertexShader;
        this.fragmentShaderSource = fragmentShaderSource;
        this.programs = [];
        this.activeProgram = null;
        this.uniforms = [];
    }

    setKeywords (keywords) {
        let hash = 0;
        for (let i = 0; i < keywords.length; i++)
            hash += hashCode(keywords[i]);

        let program = this.programs[hash];
        if (program == null)
        {
            let fragmentShader = compileShader(gl.FRAGMENT_SHADER, this.fragmentShaderSource, keywords);
            program = createProgram(this.vertexShader, fragmentShader);
            this.programs[hash] = program;
        }

        if (program == this.activeProgram) return;

        this.uniforms = getUniforms(program);
        this.activeProgram = program;
    }

    bind () {
        gl.useProgram(this.activeProgram);
    }
}

class Program {
    constructor (vertexShader, fragmentShader) {
        this.uniforms = {};
        this.program = createProgram(vertexShader, fragmentShader);
        this.uniforms = getUniforms(this.program);
    }

    bind () {
        gl.useProgram(this.program);
    }
}

function createProgram (vertexShader, fragmentShader) {
    let program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        console.trace(gl.getProgramInfoLog(program));

    return program;
}

function getUniforms (program) {
    let uniforms = [];
    //get the number of active uniforms in our shader 
    //this helps in optimization because most compilers will "know" to ignore uniforms that are not used to generate output
    //for example, 
    //uniform vec4 uBase; 
    //vec4 color = vec4(0.) + uBase;
    //color = vec4(1.0); //this line means the prior line (using uBase) is relevant, so compiler would ignore uBase assuming it is not used elsewhere
    //getProgramParameter will return a lot of info depending on what you ask for in second arg 
    let uniformCount = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    //then we loop through and fill up our array 
    for (let i = 0; i < uniformCount; i++) {
        //getActiveUniform will return size, type, name 
        let uniformName = gl.getActiveUniform(program, i).name;
        //getUniformLocation returns location in GPU memory 
        uniforms[uniformName] = gl.getUniformLocation(program, uniformName);
    }
    return uniforms;
}

function compileShader (type, source, keywords) {
    source = addKeywords(source, keywords);

    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    //test to ensure compile worked 
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
        console.trace(gl.getShaderInfoLog(shader));

    return shader;
};

//used in advection shader to assign a keyword in the case that webgl does not natively support linear filtering 
function addKeywords (source, keywords) {
    if (keywords == null) return source;
    let keywordsString = '';
    keywords.forEach(keyword => {
        keywordsString += '#define ' + keyword + '\n';
    });
    return keywordsString + source;
}


//shader defintion 
//shaders are defined inline using the compile shader fxn and passing the shader code as a string

//vertex shader 
//compute the current pixels uv by using vertex position 
//also compute our neighbors for easy calculations later 
//texelsize = 1/resolution 
const baseVertexShader = compileShader(gl.VERTEX_SHADER, `
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




//create a vertex shader with blurred coordiates for neighbors 
//just horizontal blur
const blurVertexShader = compileShader(gl.VERTEX_SHADER, `
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

//lets get some noise! 
//noise shader saved in project dir
const noiseShader = compileShader(gl.FRAGMENT_SHADER, `
precision highp float;

uniform float uPeriod;
uniform vec3 uTranslate;
uniform float uAmplitude;
uniform float uSeed;
uniform float uExponent;
uniform float uRidgeThreshold;
uniform vec3 uScale;
uniform float uAspect;

#define Index 1
#define PI 3.141592653589793
#define TWOPI 6.28318530718

varying vec2 vUv;


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


#define FBM(NOISE, SEED) float G=0.5; float freq = 1.0; float a = 1.0; float t = 0.0;for(int i=0; i<4; i++){t+= a*NOISE(freq*st, SEED);freq*=2;a*=G;}


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


// vec2 displace(vec2 st, vec2 vector, float scale){
//     vec2 offset = vec2(0.5);
//     vec2 midpoint = vec2(0.5);
//     vec2 uvi = st + scale * (vector.xy - midpoint.xy); 
//   return uvi;
// }

// vec3 displace(vec3 st , vec2 vector, float scale){ //overload to pass vec3s
//     vec2 offset = vec2(0.5);
//     vec2 midpoint = vec2(0.5);
//     vec2 uvi = st.xy * scale * (vector.xy);
//   return vec3(uvi, st.z);
// }

// #define NOISE_RGB(NOISE, SEED) vec3 noiseRGB = vec3(NOISE(st, uSeed + SEED), NOISE(st, uSeed + 500.0 + SEED), NOISE(st, uSeed - 500.0 + SEED));

// vec3 displace(vec3 st , vec2 vector, vec2 offset, vec2 midpoint, float scale){ //overload to pass vec3s
//     vec2 uvi = st.xy + scale * (vector.xy); 
//   return vec3(uvi, st.z);
// }

// float recursiveWarpNoise(vec3 st, float seed){
//   float color = monoSimplex( st*st.x, 2.0) * monoSimplex(displace(st, st.xy, vec2(0.5), vec2(0.5), 0.1));
//   for(int i = 0; i<5; i++){
//     NOISE_RGB(monoSimplex, 2.4);

//     color = monoSimplex(displace(st, noiseRGB.rg*float(i)/5.0, vec2(.5), vec2(0.5), 0.05*float(i)), seed*float(i));
//   }
//   return color;
// }

// float ang(vec3 st){
//   return sin(st.y*st.x);
// }


// float dis(vec3 st){
//   float d = grid(vUV, 7.0, 0.45, PI/4.);
//   // d = texture(sTD2DInputs[0],vUV.st).r;
//   FBM(monoSimplex, -743.4838)
//   return d *t;
// }

// float dis2(vec3 st){
//   NOISE_RGB(monoSimplex, 2.4);
//   FBM(recursiveWarpNoise, 2.4);
//   return t;
// }


#define EPSILON 0.0001

#define GRAD(NOISE, SEED) float st1 = NOISE(vec3(st.x + EPSILON, st.y, st.z), SEED).r;  float st2 = NOISE(vec3(st.x - EPSILON, st.y, st.z), SEED).r; float st3 = NOISE(vec3(st.x, st.y + EPSILON, st.z), SEED).r; float st4 = NOISE(vec3(st.x, st.y - EPSILON, st.z), SEED).r; vec2 grad = normalize(vec2(st1-st2, st3-st4));

#define DISP(ANG, DIST, MAX) st.xy = st.xy + vec2(cos(ANG(st)*TWOPI), sin(ANG(st)*TWOPI)) * DIST(st) * MAX;


//vec4[] palette = {vec4(.875, .0859375, 0.16796875, 1.0), vec4(1.), vec4(0,.3203125, 0.64453125, 1.0), vec4(0.0, 0.0, 0.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0)};

vec4 fbm(vec3 st, float seed){
    float G=0.5; 
    float freq = 1.0; 
    float a = 1.0; 
    vec4 t = vec4(0.0);
    for(int i=0; i<4; i++){
    // #ifdef MONO
    t += a*rgbSimplex(freq*st, seed);
    // #else
    // t += a*vec4(vec3(monoSimplex(freq*st, seed)),1.0);
    // #endif
    freq*= 2.0;
    //freq = pow(2.0, float(i));
    a*=G;
    }
    return t;
}

void main()
{
    //create vec3 with z value for translate
    vec3 st = vec3(vUv, 0.0);
    vec4 color = fbm(st, uSeed);
    //output
    gl_FragColor = (color);

}
`);

// const noiseShader = compileShader(gl.FRAGMENT_SHADER, noiseFrag());

//TODO - seems like this should be updated to a gaussian blur or something 
const blurShader = compileShader(gl.FRAGMENT_SHADER, `
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

const copyShader = compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    precision mediump sampler2D;

    varying highp vec2 vUv;
    uniform sampler2D uTexture;

    void main () {
        gl_FragColor = texture2D(uTexture, vUv);
    }
`);

const displayShaderSource = `
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

const bloomPrefilterShader = compileShader(gl.FRAGMENT_SHADER, `
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

const bloomBlurShader = compileShader(gl.FRAGMENT_SHADER, `
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

const bloomFinalShader = compileShader(gl.FRAGMENT_SHADER, `
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

const sunraysMaskShader = compileShader(gl.FRAGMENT_SHADER, `
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

const sunraysShader = compileShader(gl.FRAGMENT_SHADER, `
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

//Creating a simple mesh for rendering, using two triangles (ie subdivide canvas with diagonal line from (-1,-1) to (1,1))
//we need 4 vertices 0,1,2,3
//and each vertex gets 2 attributes which is their position in x,y space 
//assign these positions to the corners of our canvas
const blit = (() => {
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    //create an array of attributes for our vertices
    //two numbers per vertex, for x and y coordinates 
    //(-1,1) -> bottom left
    //(-1,1) -> top left 
    //(1,1,) -> top right
    //(1,-1) -> bottom right
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    //think this is numbering our vertices and grouping them
    //one triangle with vertices numbered 0,1,2
    //one trianlge with vertices numbered 0,2,3
    //ie triangles one and two both share vertices 0, 2 and each have a unique vertex (vtx 1 and 3 respectivey)
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);

    //intended to be used with output from createFBO
    //if we dont pass a target, then we want to create a viewport with the overall dimensions 
    //otherwise we can take our target dimensions (means we dont have to worry about sim res vs output res here)
    //clear = false is a keyword arguement set to "false" by default 
    return (target, clear = false) => {
        if (target == null)
        {
            gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        }
        else
        {
            gl.viewport(0, 0, target.width, target.height);
            gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
        }
        if (clear)
        {
            gl.clearColor(0.0, 0.0, 0.0, 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT);
        }        
        
        //do the actual drawing 
        //here we will use a triangle mesh 
        //draw 6 triangles 
        //unsigned short is the type of our vertex data 
        //offest is 0
        gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    }
})();

function CHECK_FRAMEBUFFER_STATUS () {
    let status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status != gl.FRAMEBUFFER_COMPLETE)
        console.trace("Framebuffer error: " + status);
}

//actual simulation construction
let bloom;
let bloomFramebuffers = [];
let sunrays;
let sunraysTemp;
let noise;

//load texture for dithering
let ditheringTexture = createTextureAsync('img/LDR_LLL1_0.png');

//create all our shader programs 

//POST SHADERS
const blurProgram               = new Program(blurVertexShader, blurShader);
const copyProgram               = new Program(baseVertexShader, copyShader);
const bloomPrefilterProgram     = new Program(baseVertexShader, bloomPrefilterShader);
const bloomBlurProgram          = new Program(baseVertexShader, bloomBlurShader);
const bloomFinalProgram         = new Program(baseVertexShader, bloomFinalShader);
const sunraysMaskProgram        = new Program(baseVertexShader, sunraysMaskShader);
const sunraysProgram            = new Program(baseVertexShader, sunraysShader);

//CONTENT SHADERS
const noiseProgram              = new Program(baseVertexShader, noiseShader); //noise generator 


//create a material from our display shader source to capitalize on the #defines for optimization 
//TODO - do we have to compile this source differently since there are the defines? 
//this also allows us to only use the active uniforms 
const displayMaterial = new Material(baseVertexShader, displayShaderSource);

function initFramebuffers () {
    let dyeRes = getResolution(config.DYE_RESOLUTION);//getResolution basically just applies view aspect ratio to the passed resolution 


    //we want to rescale the texture based on the canvas
    dyeRes.width = scaleByPixelRatio(canvas.clientWidth);
    dyeRes.height = scaleByPixelRatio(canvas.clientHeight);

    const texType = ext.halfFloatTexType; //TODO - should be 32 bit floats? 
    const rgba    = ext.formatRGBA;
    const rg      = ext.formatRG;
    const r       = ext.formatR;
    const filtering = ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST;

    gl.disable(gl.BLEND);


    //use helper function to create pairs of buffer objects that will be ping pong'd for our sim 
    //this lets us define the buffer objects that we wil want to use for feedback 
    if (noise == null)
        noise = createDoubleFBO(dyeRes.width, dyeRes.height, rgba.internalFormat, rgba.format, texType, filtering);
    else //resize if needed 
        noise = resizeDoubleFBO(noise, dyeRes.width, dyeRes.height, rgba.internalFormat, rgba.format, texType, filtering);
    //setup buffers for post process 
    initBloomFramebuffers();
    initSunraysFramebuffers();
}

function initBloomFramebuffers () {
    let res = getResolution(config.BLOOM_RESOLUTION);

    const texType = ext.halfFloatTexType;
    const rgba = ext.formatRGBA;
    const filtering = ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST;

    bloom = createFBO(res.width, res.height, rgba.internalFormat, rgba.format, texType, filtering);

    bloomFramebuffers.length = 0;
    for (let i = 0; i < config.BLOOM_ITERATIONS; i++)
    {
        //right shift resolution by iteration amount 
        // ie we reduce the resolution by a factor of 2^i, or rightshift(x,y) -> x/pow(2,y)
        // (1024 >> 1 = 512)
        // so basically creating mipmaps
        let width = res.width >> (i + 1);
        let height = res.height >> (i + 1);

        if (width < 2 || height < 2) break;

        let fbo = createFBO(width, height, rgba.internalFormat, rgba.format, texType, filtering);
        bloomFramebuffers.push(fbo);
    }
}

function initSunraysFramebuffers () {
    let res = getResolution(config.SUNRAYS_RESOLUTION);

    const texType = ext.halfFloatTexType;
    const r = ext.formatR;
    const filtering = ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST;

    sunrays     = createFBO(res.width, res.height, r.internalFormat, r.format, texType, filtering);
    sunraysTemp = createFBO(res.width, res.height, r.internalFormat, r.format, texType, filtering);
}

function createFBO (w, h, internalFormat, format, type, param) {
    gl.activeTexture(gl.TEXTURE0);
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

    let fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT); //can also clear depth or stencil buffers 

    let texelSizeX = 1.0 / w;
    let texelSizeY = 1.0 / h;

    return {
        texture,
        fbo,
        width: w,
        height: h,
        texelSizeX,
        texelSizeY,
        attach (id) { //assigns textures to entries in the txture buffer array (like sTD2DInputs[...])
            gl.activeTexture(gl.TEXTURE0 + id);
            gl.bindTexture(gl.TEXTURE_2D, texture);
            return id;
        }
    };
}

function createDoubleFBO (w, h, internalFormat, format, type, param) {
    let fbo1 = createFBO(w, h, internalFormat, format, type, param);
    let fbo2 = createFBO(w, h, internalFormat, format, type, param);

    return {
        width: w,
        height: h,
        texelSizeX: fbo1.texelSizeX,
        texelSizeY: fbo1.texelSizeY,
        get read () {
            return fbo1;
        },
        set read (value) {
            fbo1 = value;
        },
        get write () {
            return fbo2;
        },
        set write (value) {
            fbo2 = value;
        },
        swap () {
            let temp = fbo1;
            fbo1 = fbo2;
            fbo2 = temp;
        }
    }
}

function resizeFBO (target, w, h, internalFormat, format, type, param) {
    let newFBO = createFBO(w, h, internalFormat, format, type, param);
    copyProgram.bind();
    gl.uniform1i(copyProgram.uniforms.uTexture, target.attach(0));
    blit(newFBO);
    return newFBO;
}

function resizeDoubleFBO (target, w, h, internalFormat, format, type, param) {
    if (target.width == w && target.height == h)
        return target;
    target.read = resizeFBO(target.read, w, h, internalFormat, format, type, param);
    target.write = createFBO(w, h, internalFormat, format, type, param);
    target.width = w;
    target.height = h;
    target.texelSizeX = 1.0 / w;
    target.texelSizeY = 1.0 / h;
    return target;
}

function createTextureAsync (url) {
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, 1, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, new Uint8Array([255, 255, 255]));

    let obj = {
        texture,
        width: 1,
        height: 1,
        attach (id) {
            gl.activeTexture(gl.TEXTURE0 + id);
            gl.bindTexture(gl.TEXTURE_2D, texture);
            return id;
        }
    };

    let image = new Image();
    image.onload = () => {
        obj.width = image.width;
        obj.height = image.height;
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, image);
    };
    image.src = url;

    return obj;
}

function updateKeywords () {
    let displayKeywords = [];
    if (config.SHADING) displayKeywords.push("SHADING");
    if (config.BLOOM) displayKeywords.push("BLOOM");
    if (config.SUNRAYS) displayKeywords.push("SUNRAYS");
    displayMaterial.setKeywords(displayKeywords);
}


//actually calling our functions to make program work 
updateKeywords();
initFramebuffers();
let noiseSeed = 0.0; 
let lastUpdateTime = Date.now();
let colorUpdateTimer = 0.0;
update();


//simulation step 
function update () {
    //time step 
    const dt = calcDeltaTime();
    noiseSeed += dt * config.NOISE_TRANSLATE_SPEED;
    if (resizeCanvas()) //resize if needed 
        initFramebuffers();
    if (!config.PAUSED)
        step(dt); //do a calculation step 
    render(null);
    requestAnimationFrame(update);
}

function calcDeltaTime () {
    let now = Date.now();
    let dt = (now - lastUpdateTime) / 1000;
    dt = Math.min(dt, 0.016666); //never want to update slower than 60fps
    lastUpdateTime = now;
    return dt;
}

function resizeCanvas () {
    let width = scaleByPixelRatio(canvas.clientWidth);
    let height = scaleByPixelRatio(canvas.clientHeight);
    if (canvas.width != width || canvas.height != height) {
        canvas.width = width;
        canvas.height = height;
        return true;
    }
    return false;
}


//the simulation, finally! 
function step (dt) {
    gl.disable(gl.BLEND);
    noiseProgram.bind();
    gl.uniform1f(noiseProgram.uniforms.uPeriod, config.PERIOD); 
    gl.uniform3f(noiseProgram.uniforms.uTranslate, 0.0, 0.0, 0.0);
    gl.uniform1f(noiseProgram.uniforms.uAmplitude, config.AMP); 
    gl.uniform1f(noiseProgram.uniforms.uSeed, noiseSeed); 
    gl.uniform1f(noiseProgram.uniforms.uExponent, config.EXPONENT); 
    gl.uniform1f(noiseProgram.uniforms.uRidgeThreshold, config.RIDGE); 
    gl.uniform3f(noiseProgram.uniforms.uScale, 1., 1., 1.); 
    gl.uniform1f(noiseProgram.uniforms.uAspect, config.ASPECT); 
    blit(noise.write);
    noise.swap();
}

function render (target) {
    if (config.BLOOM)
        applyBloom(noise.read, bloom);
        if (config.SUNRAYS) {
            applySunrays(noise.read, noise.write, sunrays);
            blur(sunrays, sunraysTemp, 1);
        }
        
        if (target == null || !config.TRANSPARENT) {
            gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
            gl.enable(gl.BLEND);
        }
        else {
            gl.disable(gl.BLEND);
        }
        drawDisplay(noise);
}


function drawDisplay (target) {
    let width = target == null ? gl.drawingBufferWidth : target.width;
    let height = target == null ? gl.drawingBufferHeight : target.height;
    
    displayMaterial.bind();
    if (config.SHADING)
        gl.uniform2f(displayMaterial.uniforms.texelSize, 1.0 / width, 1.0 / height);
    gl.uniform1i(displayMaterial.uniforms.uTexture, noise.read.attach(0));
    if (config.BLOOM) {
        gl.uniform1i(displayMaterial.uniforms.uBloom, bloom.attach(1));
        gl.uniform1i(displayMaterial.uniforms.uDithering, ditheringTexture.attach(2));
        let scale = getTextureScale(ditheringTexture, width, height);
        gl.uniform2f(displayMaterial.uniforms.ditherScale, scale.x, scale.y);
    }
    if (config.SUNRAYS)
        gl.uniform1i(displayMaterial.uniforms.uSunrays, sunrays.attach(3));
    blit(target);
}

function applyBloom (source, destination) {
    if (bloomFramebuffers.length < 2)
        return;

    let last = destination;

    gl.disable(gl.BLEND);
    bloomPrefilterProgram.bind();
    let knee = config.BLOOM_THRESHOLD * config.BLOOM_SOFT_KNEE + 0.0001;
    let curve0 = config.BLOOM_THRESHOLD - knee;
    let curve1 = knee * 2;
    let curve2 = 0.25 / knee;
    gl.uniform3f(bloomPrefilterProgram.uniforms.curve, curve0, curve1, curve2);
    gl.uniform1f(bloomPrefilterProgram.uniforms.threshold, config.BLOOM_THRESHOLD);
    gl.uniform1i(bloomPrefilterProgram.uniforms.uTexture, source.attach(0));
    blit(last);

    bloomBlurProgram.bind();
    for (let i = 0; i < bloomFramebuffers.length; i++) {
        let dest = bloomFramebuffers[i];
        gl.uniform2f(bloomBlurProgram.uniforms.texelSize, last.texelSizeX, last.texelSizeY);
        gl.uniform1i(bloomBlurProgram.uniforms.uTexture, last.attach(0));
        blit(dest);
        last = dest;
    }

    gl.blendFunc(gl.ONE, gl.ONE);
    gl.enable(gl.BLEND);

    for (let i = bloomFramebuffers.length - 2; i >= 0; i--) {
        let baseTex = bloomFramebuffers[i];
        gl.uniform2f(bloomBlurProgram.uniforms.texelSize, last.texelSizeX, last.texelSizeY);
        gl.uniform1i(bloomBlurProgram.uniforms.uTexture, last.attach(0));
        gl.viewport(0, 0, baseTex.width, baseTex.height);
        blit(baseTex);
        last = baseTex;
    }

    gl.disable(gl.BLEND);
    bloomFinalProgram.bind();
    gl.uniform2f(bloomFinalProgram.uniforms.texelSize, last.texelSizeX, last.texelSizeY);
    gl.uniform1i(bloomFinalProgram.uniforms.uTexture, last.attach(0));
    gl.uniform1f(bloomFinalProgram.uniforms.intensity, config.BLOOM_INTENSITY);
    blit(destination);
}

function applySunrays (source, mask, destination) {
    gl.disable(gl.BLEND);
    sunraysMaskProgram.bind();
    gl.uniform1i(sunraysMaskProgram.uniforms.uTexture, source.attach(0));
    blit(mask);

    sunraysProgram.bind();
    gl.uniform1f(sunraysProgram.uniforms.weight, config.SUNRAYS_WEIGHT);
    gl.uniform1i(sunraysProgram.uniforms.uTexture, mask.attach(0));
    blit(destination);
}

function blur (target, temp, iterations) {
    blurProgram.bind();
    for (let i = 0; i < iterations; i++) {
        gl.uniform2f(blurProgram.uniforms.texelSize, target.texelSizeX, 0.0);
        gl.uniform1i(blurProgram.uniforms.uTexture, target.attach(0));
        blit(temp);

        gl.uniform2f(blurProgram.uniforms.texelSize, 0.0, target.texelSizeY);
        gl.uniform1i(blurProgram.uniforms.uTexture, temp.attach(0));
        blit(target);
    }
}


canvas.addEventListener('mousedown', e => {
    let posX = scaleByPixelRatio(e.offsetX);
    let posY = scaleByPixelRatio(e.offsetY);
    let pointer = pointers.find(p => p.id == -1);
    if (pointer == null)
        pointer = new pointerPrototype();
    updatePointerDownData(pointer, -1, posX, posY);
});

canvas.addEventListener('mousemove', e => {
    let pointer = pointers[0];
    if (!pointer.down) return;
    let posX = scaleByPixelRatio(e.offsetX);
    let posY = scaleByPixelRatio(e.offsetY);
    updatePointerMoveData(pointer, posX, posY);
});

window.addEventListener('mouseup', () => {
    updatePointerUpData(pointers[0]);
});

canvas.addEventListener('touchstart', e => {
    e.preventDefault();
    const touches = e.targetTouches;
    while (touches.length >= pointers.length)
        pointers.push(new pointerPrototype());
    for (let i = 0; i < touches.length; i++) {
        let posX = scaleByPixelRatio(touches[i].pageX);
        let posY = scaleByPixelRatio(touches[i].pageY);
        updatePointerDownData(pointers[i + 1], touches[i].identifier, posX, posY);
    }
});

canvas.addEventListener('touchmove', e => {
    e.preventDefault();
    const touches = e.targetTouches;
    for (let i = 0; i < touches.length; i++) {
        let pointer = pointers[i + 1];
        if (!pointer.down) continue;
        let posX = scaleByPixelRatio(touches[i].pageX);
        let posY = scaleByPixelRatio(touches[i].pageY);
        updatePointerMoveData(pointer, posX, posY);
    }
}, false);

window.addEventListener('touchend', e => {
    const touches = e.changedTouches;
    for (let i = 0; i < touches.length; i++)
    {
        let pointer = pointers.find(p => p.id == touches[i].identifier);
        if (pointer == null) continue;
        updatePointerUpData(pointer);
    }
});

window.addEventListener('keydown', e => {
    if (e.code === 'KeyP')
        config.PAUSED = !config.PAUSED;
    if (e.key === ' '){}
        // splatStack.push(parseInt(Math.random() * 20) + 5);
});

function updatePointerDownData (pointer, id, posX, posY) {
    pointer.id = id;
    pointer.down = true;
    pointer.moved = false;
    pointer.texcoordX = posX / canvas.width;
    pointer.texcoordY = 1.0 - posY / canvas.height;
    pointer.prevTexcoordX = pointer.texcoordX;
    pointer.prevTexcoordY = pointer.texcoordY;
    pointer.deltaX = 0;
    pointer.deltaY = 0;
    pointer.color = generateColor();
}

function updatePointerMoveData (pointer, posX, posY) {
    pointer.prevTexcoordX = pointer.texcoordX;
    pointer.prevTexcoordY = pointer.texcoordY;
    pointer.texcoordX = posX / canvas.width;
    pointer.texcoordY = 1.0 - posY / canvas.height;
    pointer.deltaX = correctDeltaX(pointer.texcoordX - pointer.prevTexcoordX);
    pointer.deltaY = correctDeltaY(pointer.texcoordY - pointer.prevTexcoordY);
    pointer.moved = Math.abs(pointer.deltaX) > 0 || Math.abs(pointer.deltaY) > 0;
}

function updatePointerUpData (pointer) {
    pointer.down = false;
}

function correctDeltaX (delta) {
    let aspectRatio = canvas.width / canvas.height;
    if (aspectRatio < 1) delta *= aspectRatio;
    return delta;
}

function correctDeltaY (delta) {
    let aspectRatio = canvas.width / canvas.height;
    if (aspectRatio > 1) delta /= aspectRatio;
    return delta;
}

function generateColor () {
    let c = HSVtoRGB(Math.random(), 1.0, 1.0);
    c.r *= 0.15;
    c.g *= 0.15;
    c.b *= 0.15;
    return c;
}

function HSVtoRGB (h, s, v) {
    let r, g, b, i, f, p, q, t;
    i = Math.floor(h * 6);
    f = h * 6 - i;
    p = v * (1 - s);
    q = v * (1 - f * s);
    t = v * (1 - (1 - f) * s);

    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }

    return {
        r,
        g,
        b
    };
}

function normalizeColor (input) {
    let output = {
        r: input.r / 255,
        g: input.g / 255,
        b: input.b / 255
    };
    return output;
}

function wrap (value, min, max) {
    let range = max - min;
    if (range == 0) return min;
    return (value - min) % range + min;
}

function getResolution (resolution) {
    let aspectRatio = gl.drawingBufferWidth / gl.drawingBufferHeight;
    if (aspectRatio < 1)
        aspectRatio = 1.0 / aspectRatio;

    let min = Math.round(resolution);
    let max = Math.round(resolution * aspectRatio);

    if (gl.drawingBufferWidth > gl.drawingBufferHeight)
        return { width: max, height: min };
    else
        return { width: min, height: max };
}

function getTextureScale (texture, width, height) {
    return {
        x: width / texture.width,
        y: height / texture.height
    };
}


//check for differences in pixel size based on diplay
// the px ratio gives the ratio of one physical px to one css px
function scaleByPixelRatio (input) {
    let pixelRatio = window.devicePixelRatio || 1;
    return Math.floor(input * pixelRatio);
}

function hashCode (s) {
    if (s.length == 0) return 0;
    let hash = 0;
    for (let i = 0; i < s.length; i++) {
        hash = (hash << 5) - hash + s.charCodeAt(i);
        hash |= 0; // Convert to 32bit integer
    }
    return hash;
};