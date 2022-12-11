
'use strict';

import {gl , ext, canvas } from "./js/WebGL.js";
import {config} from "./js/config.js";
import {Noise} from "./js/Noise.js";
import * as LGL from "./js/WebGL.js";


import {Stats} from "./stats.module.js";

LGL.resizeCanvas();

let stats = new Stats();
let container = document.createElement('div')
document.body.appendChild(container );
container.appendChild(stats.dom);

let n = new Noise();
n.initFramebuffers();
n.startGUI();
n.simulate();
stats.update();
