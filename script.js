
'use strict';

import {gl , ext, canvas } from "./js/WebGL.js";
import {config} from "./js/config.js";
import {Noise} from "./js/Noise.js";
import * as LGL from "./js/WebGL.js";

if (LGL.isMobile()) {
    config.DYE_RESOLUTION = 512;
}

LGL.resizeCanvas();


// const pixels = new Uint8Array([223,
//     21,
//     42,
//     255,
//     255,
//     255,
//     255,
//     255,
//     0,
//     81,
//     164,
//     255,
//     0,
//     0,
//     0,
//     255,
//     255,
//     255,
//     255,
//     255]); // opaque blue


// var rgbTex = LGL.textureFromPixelArray(gl, pixels, gl.RGBA, 5, 1);

let n = new Noise();
n.initFramebuffers();
n.startGUI();
n.simulate();
