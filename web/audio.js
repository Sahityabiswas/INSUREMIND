"use strict";

// Audio stays local: the browser produces PCM WAV, and the server runs Vosk.
class LocalRecorder {
  constructor(onLevel) { this.onLevel = onLevel; this.cancelled = false; this.chunks = []; }

  async start() {
    this.cancelled = false;
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({audio: {
        channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true,
      }});
      if (this.cancelled) { await this.release(); return false; }
      this.context = new AudioContext();
      this.rate = this.context.sampleRate;
      await this.context.audioWorklet.addModule("/audio-worklet.js");
      if (this.cancelled) { await this.release(); return false; }
      this.node = new AudioWorkletNode(this.context, "microphone-capture");
      this.node.port.onmessage = ({data}) => {
        if (this.cancelled) return;
        this.chunks.push(data);
        const rms = Math.sqrt(data.reduce((sum, value) => sum + value * value, 0) / data.length);
        this.onLevel(Math.min(1, rms * 8));
      };
      this.source = this.context.createMediaStreamSource(this.stream);
      this.mute = this.context.createGain();
      this.mute.gain.value = 0;
      this.source.connect(this.node).connect(this.mute).connect(this.context.destination);
      await this.context.resume();
      if (this.cancelled) { await this.release(); return false; }
      return true;
    } catch (error) { await this.release(); throw error; }
  }

  async release() {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.source?.disconnect();
    this.node?.disconnect();
    this.mute?.disconnect();
    if (this.node) this.node.port.onmessage = null;
    if (this.context && this.context.state !== "closed") await this.context.close();
    this.onLevel(0);
  }

  async stop(discard = false) {
    this.cancelled = true;
    await this.release();
    const chunks = this.chunks;
    this.chunks = [];
    if (discard) return null;
    const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    if (!length) throw new Error("No microphone audio captured. Please retry.");
    const input = new AudioBuffer({length, numberOfChannels: 1, sampleRate: this.rate});
    let offset = 0;
    for (const chunk of chunks) {input.copyToChannel(chunk, 0, offset); offset += chunk.length;}
    const offline = new OfflineAudioContext(1, Math.ceil(length * 16000 / this.rate), 16000);
    const source = offline.createBufferSource();
    source.buffer = input;
    source.connect(offline.destination);
    source.start();
    const rendered = await offline.startRendering();
    return LocalRecorder.wav(rendered.getChannelData(0));
  }

  static wav(samples) {
    const bytes = new ArrayBuffer(44 + samples.length * 2), view = new DataView(bytes);
    const word = (offset, text) => [...text].forEach((letter, index) => view.setUint8(offset + index, letter.charCodeAt(0)));
    word(0, "RIFF"); view.setUint32(4, bytes.byteLength - 8, true); word(8, "WAVE");
    word(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, 16000, true); view.setUint32(28, 32000, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true); word(36, "data");
    view.setUint32(40, samples.length * 2, true);
    samples.forEach((sample, index) => view.setInt16(44 + index * 2, Math.round(Math.max(-1, Math.min(1, sample)) * (sample < 0 ? 32768 : 32767)), true));
    return new Blob([bytes], {type: "audio/wav"});
  }
}
window.LocalRecorder = LocalRecorder;
