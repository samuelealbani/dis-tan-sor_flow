// from https://radzion.com/blog/linear-algebra/vectors

class Vector {
  constructor(...components) {
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



let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;

const statusElement = document.getElementById('status');

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

  let pose0NosePos, pose1NosePos;
  let pose0leftHandPos, pose0rightHandPos;
  let pose1leftHandPos, pose1rightHandPos;
  let pose0leftHipPos, pose0rightHipPos;
  let pose1leftHipPos, pose1rightHipPos;

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old
  // model, which shouldn't be rendered.
  /*   if (poses.length > 0 && !STATE.isModelChanged) {
      camera.drawResults(poses);
      // console.log(camera.width);
      // console.log(poses[0]);
      // var message = new OSC.Message('/test/random');
      // message.add(poses[0].keypoints[0].x);
      // osc.send(message);

      // -------------here OSC-------------
      //https://github.com/tommymitch/posenetosc/blob/master/cameraosc.js
      // for (let i = 0; i < poses.length; i++) {
      //   const pose = poses[i];
      //   // var message = new OSC.Message('/pose/' + i);
      //   //        message.add(pose.keypoints[0].x);
      //   // osc.send(message);
      //     let message = new OSC.Message('/num_poses');
      //     message.add (poses.length);
      //     osc.send (message);

      //   for (let j = 0; j < pose.keypoints.length; j++) {
      //     const keypoint = pose.keypoints[j];
      //     let message = new OSC.Message('/pose/' + i + '/' + keypoint.name );
      //     message.add (keypoint.x/camera.video.width);
      //     message.add (keypoint.y/camera.video.height);
      //     message.add (keypoint.score);
      //     osc.send (message);
      //     console.log(message);
      //   }
      // }


    } */
  if (poses && poses.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(poses);
    //https://github.com/tommymitch/posenetosc/blob/master/cameraosc.js
    for (let i = 0; i < poses.length; i++) {
      const pose = poses[i];

      const nose = pose.keypoints[0]; // nose
      const leftHand = pose.keypoints[9]; // left_wrist
      const rightHand = pose.keypoints[10]; // right_elbow
      const leftHip = pose.keypoints[11]; // left_hip
      const rightHip = pose.keypoints[12]; // right_hip
      switch (i) {
        case 0:
          // console.log('nose', pose.keypoints[0]); // nose
          pose0NosePos = new Vector(
            nose.x / camera.video.width,
            nose.y / camera.video.height
          );

          pose0leftHandPos = new Vector(
            leftHand.x / camera.video.width,
            leftHand.y / camera.video.height
          );
          pose0rightHandPos = new Vector(
            rightHand.x / camera.video.width,
            rightHand.y / camera.video.height
          );

          pose0leftHipPos = new Vector(
            leftHip.x / camera.video.width,
            leftHip.y / camera.video.height
          );
          pose0rightHipPos = new Vector(
            rightHip.x / camera.video.width,
            rightHip.y / camera.video.height
          );

          break;
        case 1:
          pose1NosePos = new Vector(
            nose.x / camera.video.width,
            nose.y / camera.video.height
          );

          pose1leftHandPos = new Vector(
            leftHand.x / camera.video.width,
            leftHand.y / camera.video.height
          );
          pose1rightHandPos = new Vector(
            rightHand.x / camera.video.width,
            rightHand.y / camera.video.height
          );

          pose1leftHipPos = new Vector(
            leftHip.x / camera.video.width,
            leftHip.y / camera.video.height
          );
          pose1rightHipPos = new Vector(
            rightHip.x / camera.video.width,
            rightHip.y / camera.video.height
          );

          const diffNose = pose0NosePos.subtract(pose1NosePos);
          const lengthDistNose = diffNose.length();
          let message = new OSC.Message("/distances/noses");
          message.add(lengthDistNose);
          osc.send(message);

          const diffLeftHand = pose0leftHandPos.subtract(pose1leftHandPos);
          const lengthDistLeftHand = diffLeftHand.length();
          message = new OSC.Message("/distances/leftHands");
          message.add(lengthDistLeftHand);
          osc.send(message);

          const diffRightHand = pose0rightHandPos.subtract(pose1rightHandPos);
          const lengthDistRightHand = diffRightHand.length();
          message = new OSC.Message("/distances/rightHands");
          message.add(lengthDistRightHand);
          osc.send(message);

          const diffLeftHand0Nose1 = pose0leftHandPos.subtract(pose1NosePos);
          const lengthDistLeftHand0Nose1 = diffLeftHand0Nose1.length();
          message = new OSC.Message("/distances/leftHand0nose1");
          message.add(lengthDistLeftHand0Nose1);
          osc.send(message);

          const diffRightHand0Nose1 = pose0rightHandPos.subtract(pose1NosePos);
          const lengthDistRightHand0Nose1 = diffRightHand0Nose1.length();
          message = new OSC.Message("/distances/rightHand0nose1");
          message.add(lengthDistRightHand0Nose1);
          osc.send(message);

          const diffLeftHand1Nose0 = pose1leftHandPos.subtract(pose0NosePos);
          const lengthDistLeftHand1Nose0 = diffLeftHand1Nose0.length();
          message = new OSC.Message("/distances/leftHand1nose0");
          message.add(lengthDistLeftHand1Nose0);
          osc.send(message);

          const diffRightHand1Nose0 = pose1rightHandPos.subtract(pose0NosePos);
          const lengthDistRightHand1Nose0 = diffRightHand1Nose0.length();
          message = new OSC.Message("/distances/rightHand1nose0");
          message.add(lengthDistRightHand1Nose0);
          osc.send(message);


          /* ---- */
          const diffLeftHand0Hip1 = pose0leftHandPos.subtract(pose1leftHipPos);
          const lengthDistLeftHand0Hip1 = diffLeftHand0Hip1.length();
          message = new OSC.Message("/distances/leftHand0hip1");
          message.add(lengthDistLeftHand0Hip1);
          osc.send(message);

          const diffRightHand0Hip1 = pose0rightHandPos.subtract(pose1leftHipPos);
          const lengthDistRightHand0Hip1 = diffRightHand0Hip1.length();
          message = new OSC.Message("/distances/rightHand0hip1");
          message.add(lengthDistRightHand0Hip1);
          osc.send(message);

          const diffLeftHand1Hip0 = pose1leftHandPos.subtract(pose0leftHipPos);
          const lengthDistLeftHand1Hip0 = diffLeftHand1Hip0.length();
          message = new OSC.Message("/distances/leftHand1hip0");
          message.add(lengthDistLeftHand1Hip0);
          osc.send(message);

          const diffRightHand1Hip0 = pose1rightHandPos.subtract(pose0leftHipPos);
          const lengthDistRightHand1Hip0 = diffRightHand1Hip0.length();
          message = new OSC.Message("/distances/rightHand1hip0");
          message.add(lengthDistRightHand1Hip0);
          osc.send(message);

          /*  */
          const diffHip0Hip1 = pose0leftHipPos.subtract(pose1leftHipPos);
          const lengthDistHip0Hip1 = diffHip0Hip1.length();
          message = new OSC.Message("/distances/hip0hip1");
          message.add(lengthDistHip0Hip1);
          osc.send(message);


          break;

        default:
          break;
      }


      let message = new OSC.Message('/num_poses');
      message.add(poses.length);
      osc.send(message);

      for (let j = 0; j < pose.keypoints.length; j++) {
        const keypoint = pose.keypoints[j];
        let message = new OSC.Message('/pose/' + i + '/' + keypoint.name);
        message.add(keypoint.x / camera.video.width);
        message.add(keypoint.y / camera.video.height);
        message.add(keypoint.score);
        osc.send(message);
        console.log(message);
      }
    }
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

