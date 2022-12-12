
'use strict';

import {gl , ext, canvas } from "./js/WebGL.js";
import {config} from "./js/config.js";
import {Noise} from "./js/Noise.js";
import * as LGL from "./js/WebGL.js";

LGL.resizeCanvas();

let n = new Noise();
n.initFramebuffers();
n.startGUI();
n.simulate();
