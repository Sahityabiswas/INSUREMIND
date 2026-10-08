"use strict";

class MicrophoneCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.remaining = sampleRate * 10;
  }
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel && this.remaining > 0) {
      const count = Math.min(channel.length, this.remaining);
      this.port.postMessage(channel.slice(0, count));
      this.remaining -= count;
    }
    return this.remaining > 0;
  }
}
registerProcessor("microphone-capture", MicrophoneCapture);
