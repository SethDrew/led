export class AudioEngine {
  private audioContext: AudioContext;
  private analyser: AnalyserNode;
  private source: MediaElementAudioSourceNode | null = null;
  private freqData: Float32Array;
  private timeData: Float32Array;

  constructor(fftSize = 2048) {
    this.audioContext = new AudioContext();
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = fftSize;
    this.analyser.smoothingTimeConstant = 0.8;
    this.analyser.connect(this.audioContext.destination);

    const binCount = this.analyser.frequencyBinCount;
    this.freqData = new Float32Array(binCount);
    this.timeData = new Float32Array(fftSize);
  }

  connectElement(audioEl: HTMLAudioElement): void {
    if (this.source) {
      this.source.disconnect();
    }
    this.source = this.audioContext.createMediaElementSource(audioEl);
    this.source.connect(this.analyser);

    // Resume context on user interaction (autoplay policy)
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
  }

  getFrequencyData(): Float32Array {
    this.analyser.getFloatFrequencyData(this.freqData);
    return this.freqData;
  }

  getTimeDomainData(): Float32Array {
    this.analyser.getFloatTimeDomainData(this.timeData);
    return this.timeData;
  }

  getRMS(): number {
    this.analyser.getFloatTimeDomainData(this.timeData);
    let sum = 0;
    for (let i = 0; i < this.timeData.length; i++) {
      sum += this.timeData[i] * this.timeData[i];
    }
    return Math.sqrt(sum / this.timeData.length);
  }

  get sampleRate(): number {
    return this.audioContext.sampleRate;
  }

  get fftSize(): number {
    return this.analyser.fftSize;
  }
}
