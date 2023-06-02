// from https://radzion.com/blog/linear-algebra/vectors

class Vector {
  constructor (...components) {
    this.components = components;
  }

  add({ components }) {
    return new Vector(
      ...components.map(
        (component, index) => this.components[index] + component
      )
    );
  }

  subtract({ components }) {
    return new Vector(
      ...components.map(
        (component, index) => this.components[index] - component
      )
    );
  }

  length() {
    return Math.hypot(...this.components);
  }

  normalize() {
    return this.scaleBy(1 / this.length());
  }
}


/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-webgpu';
import * as mpPose from '@mediapipe/pose';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPaths(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

import * as posedetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';

import { setupStats } from './stats_panel';
import { Context } from './camera';
import { setupDatGui } from './option_panel';
import { STATE } from './params';
import { setBackendAndEnvFlags } from './util';
const OSC = require('osc-js');
let osc = new OSC();
osc.open(); // connect by default to ws://localhost:8080

document.getElementById('send').addEventListener('click', () => {
  var message = new OSC.Message('/test/random', 33);
  osc.send(message);
});


// websocket
const socket = new WebSocket('ws://10.209.2.60:8025'); //12345 //ws://10.209.2.60:8025 // ws://localhost:8080
let socketConnected = false;
// Connection opened
socket.addEventListener('open', (event) => {
  // socket.send('Hello Server!');
  socket.send({ 'hello': 'Hello Server!' });
  socketConnected = true;
});
socket.addEventListener('close', (event) => {
  socketConnected = false;
  console.log('socket is closed');
});

// Listen for messages
socket.addEventListener('message', (event) => {
  console.log('Message from server ', event.data);
});
// [end] websocket



let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;

const statusElement = document.getElementById('status');

let counter = 0;

async function createDetector() {
  switch (STATE.model) {
    case posedetection.SupportedModels.PoseNet:
      return posedetection.createDetector(STATE.model, {
        quantBytes: 4,
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: { width: 500, height: 500 },
        multiplier: 0.75
      });
    case posedetection.SupportedModels.BlazePose:
      const runtime = STATE.backend.split('-')[0];
      if (runtime === 'mediapipe') {
        return posedetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${mpPose.VERSION}`
        });
      } else if (runtime === 'tfjs') {
        return posedetection.createDetector(
          STATE.model, { runtime, modelType: STATE.modelConfig.type });
      }
    case posedetection.SupportedModels.MoveNet:
      const modelType = STATE.modelConfig.type == 'lightning' ?
        posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING :
        posedetection.movenet.modelType.SINGLEPOSE_THUNDER;
      return posedetection.createDetector(STATE.model, { modelType });
  }
}

async function checkGuiUpdate() {
  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    detector.dispose();

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    detector = await createDetector(STATE.model);
    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimatePosesStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimatePosesStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
      1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  // FPS only counts the time it takes to finish estimatePoses.
  beginEstimatePosesStats();

  const poses = await detector.estimatePoses(
    camera.video,
    { maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false });

  endEstimatePosesStats();

  camera.drawCtx();

  if (poses && poses.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(poses);

    let msgArray = [];

    //https://github.com/tommymitch/posenetosc/blob/master/cameraosc.js
    for (let i = 0; i < poses.length; i++) {
      const pose = poses[i];
      const nose = pose.keypoints[0]; // nose

      if (socketConnected) {
        let msgPoseObj = new Object();
        msgPoseObj.nPose = i;
        msgPoseObj.keypoint = nose.name;
        msgPoseObj.xPos = nose.x / camera.video.width;
        msgPoseObj.yPos = nose.y / camera.video.height;
        msgArray.push(msgPoseObj);
      }

      /* let message = new OSC.Message('/pose/' + i + '/nose');
      message.add(poses[i].keypoints[0].x / camera.video.width);
      message.add(poses[i].keypoints[0].y / camera.video.height);
      // console.log(poses[i].keypoints[0].x / camera.video.width, poses[i].keypoints[0].y / camera.video.height);
      osc.send(message); */
    }

    // direct osc
    let message = new OSC.Message('/num_poses');
    message.add(poses.length);
    osc.send(message);

    msgArray = msgArray.sort((a, b) => {
      if (a.xPos < b.xPos) {
        return -1;
      }
    });

    for (let i = 0; i < msgArray.length; i++) {
      let message = new OSC.Message('/pose/' + i + '/nose');
      message.add(msgArray[i].xPos);
      message.add(msgArray[i].yPos);
      osc.send(message);
    }

    // websocket send
    if (socketConnected && (counter % 12) == 0) {
      let msgObj = new Object();
      msgObj.data = msgArray;

      let msgJsonString = JSON.stringify(msgObj);
      // console.log(msgJsonString);
      // console.log("prin");
      socket.send(msgJsonString);

    }

    counter++;


  }

}

async function updateVideo(event) {
  // Clear reference to any previous uploaded video.
  URL.revokeObjectURL(camera.video.currentSrc);
  const file = event.target.files[0];
  camera.source.src = URL.createObjectURL(file);

  // Wait for video to be loaded.
  camera.video.load();
  await new Promise((resolve) => {
    camera.video.onloadeddata = () => {
      resolve(video);
    };
  });

  const videoWidth = camera.video.videoWidth;
  const videoHeight = camera.video.videoHeight;
  // Must set below two lines, otherwise video element doesn't show.
  camera.video.width = videoWidth;
  camera.video.height = videoHeight;
  camera.canvas.width = videoWidth;
  camera.canvas.height = videoHeight;

  statusElement.innerHTML = 'Video is loaded.';
}

async function runFrame() {
  await checkGuiUpdate();
  if (video.paused) {
    // video has finished.
    camera.mediaRecorder.stop();
    camera.clearCtx();
    camera.video.style.visibility = 'visible';
    return;
  }
  await renderResult();
  rafId = requestAnimationFrame(runFrame);
}

async function run() {
  statusElement.innerHTML = 'Warming up model.';
  // Warming up pipeline.
  const [runtime, $backend] = STATE.backend.split('-');

  console.log(camera.video.height, camera.video.width);

  if (runtime === 'tfjs') {
    const warmUpTensor =
      tf.fill([camera.video.height, camera.video.width, 3], 0, 'float32');
    await detector.estimatePoses(
      warmUpTensor,
      { maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false });
    warmUpTensor.dispose();
    statusElement.innerHTML = 'Model is warmed up.';
  }

  camera.video.style.visibility = 'hidden';
  video.pause();
  video.currentTime = 0;
  video.play();
  //video.loop = true;
  camera.mediaRecorder.start();

  video.onended = (event) => {
    run();
    console.log(
      "restart"
    );
  };

  console.log("run pum");

  await new Promise((resolve) => {
    camera.video.onseeked = () => {
      resolve(video);
    };
  });

  await runFrame();
}

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    alert('Cannot find model in the query string.');
    return;
  }

  await setupDatGui(urlParams);
  stats = setupStats();
  camera = new Context();

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);
  detector = await createDetector();

  const runButton = document.getElementById('submit');
  runButton.onclick = run;

  const uploadButton = document.getElementById('videofile');
  uploadButton.onchange = updateVideo;
};

app();

