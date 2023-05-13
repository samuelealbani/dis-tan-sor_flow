/* eslint-disable no-trailing-spaces */
/*
"bridge" between your UDP and WebSocket clients

to  be excecuted before the main server

this sends to the udp port 9129 (to be opened in MAX/MSP)
all the messages triggered by the TF app

*/

const OSC = require('osc-js');

const config = { udpClient: { port: 9130 } };
const osc = new OSC({ plugin: new OSC.BridgePlugin(config) });

console.log('buba');

osc.open(); // start a WebSocket server on port 8080
