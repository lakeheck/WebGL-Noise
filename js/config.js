export const config = {
    SIM_RESOLUTION: 256, //simres
    DYE_RESOLUTION: 1024, //output res 
    ASPECT: 1.0,
    CAPTURE_RESOLUTION: 1024, //screen capture res 
    DENSITY_DISSIPATION: .85, //def need to figure out this one, think perhaps bc im squaring the color in splatColor
    SHADING: false,
    PAUSED: false,
    //not used 
    BLOOM: false,
    BLOOM_ITERATIONS: 8,
    BLOOM_RESOLUTION: 256,
    BLOOM_INTENSITY: 0.8,
    BLOOM_THRESHOLD: 0.6,
    BLOOM_SOFT_KNEE: 0.7,
    //not used 
    SUNRAYS: false,
    SUNRAYS_RESOLUTION: 196,
    SUNRAYS_WEIGHT: 0.6,
    //noise settings 
    EXPONENT: 2.5,
    PERIOD: 1.0,
    RIDGE: 0.9,
    AMP: 1.0,
    LACUNARITY: 3.5,
    GAIN: 0.4,
    OCTAVES: 4,
    MONO: false,
    NOISE_TRANSLATE_SPEED: 0.15,
    ERRATA: true,
    SHOWSTATS: true,
    WARP: true,
    //TODO - need to connect palette 
    COLOR1: { r: 223, g: 22, b: 43 },
    COLOR2: { r: 255, g: 255, b: 255 }, 
    COLOR3: { r: 0, g: 81, b: 164 }, 
    COLOR4: { r: 0, g: 0, b: 0 }, 
    COLOR5: { r: 255, g: 255, b: 255 }, 
    // warp settings 
    NOISECROSS: 1., 
    MAXDIST: 1.,
    SCALEX: 1.,
    SCALEY: 1.,
    RESET: false,
    RANDOM: false
    
};
