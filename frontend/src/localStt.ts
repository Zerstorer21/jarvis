/**
 * Riconoscimento vocale locale (JARVIS_STT=local).
 *
 * Al posto della Web Speech API di Chrome (che manda l'audio a Google):
 * il microfono è ascoltato qui, una frase viene isolata con un semplice
 * rilevatore di voce a soglia di energia, convertita in WAV 16 kHz mono e
 * mandata a POST /api/stt, dove Whisper la trascrive sul PC.
 * Il testo torna a `onTranscript`, esattamente come faceva Chrome.
 */
import type { VoiceInput } from "./voice";

const TARGET_RATE = 16000;
const FRAME = 2048;                 // campioni per blocco (~43 ms a 48 kHz)
const PREROLL_MS = 400;             // audio tenuto prima dell'inizio della voce
const END_SILENCE_MS = 800;         // silenzio che chiude la frase
const MIN_SPEECH_MS = 350;          // più corta di così = rumore
const MAX_UTTERANCE_MS = 20000;
const START_FRAMES = 3;             // blocchi consecutivi sopra soglia per iniziare
const MIN_LEVEL = 0.012;            // soglia minima assoluta (RMS)

export function createLocalVoiceInput(
  onTranscript: (text: string) => void,
  onInterim: (text: string) => void,
  onError: (msg: string) => void,
  onMicEvent: (event: string) => void = () => {}
): VoiceInput {
  let listening = false;
  let paused = false;
  let ctx: AudioContext | null = null;
  let stream: MediaStream | null = null;
  let node: ScriptProcessorNode | null = null;

  let noise = 0.01;                 // stima del rumore di fondo
  let loudRun = 0;
  let recording = false;
  let startedAt = 0;
  let lastLoudAt = 0;
  let chunks: Float32Array[] = [];
  const preroll: Float32Array[] = [];
  let inFlight = 0;

  const mark = (what: string) => {
    console.info(`[voice-local] ${new Date().toLocaleTimeString()} ${what}`);
    onMicEvent(what);
  };

  function rms(buf: Float32Array): number {
    let s = 0;
    for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
    return Math.sqrt(s / buf.length);
  }

  function downsample(input: Float32Array, rate: number): Int16Array {
    const ratio = rate / TARGET_RATE;
    const out = new Int16Array(Math.floor(input.length / ratio));
    for (let i = 0; i < out.length; i++) {
      const start = Math.floor(i * ratio);
      const end = Math.min(input.length, Math.floor((i + 1) * ratio));
      let sum = 0;
      for (let j = start; j < end; j++) sum += input[j];
      const v = Math.max(-1, Math.min(1, sum / Math.max(1, end - start)));
      out[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
    }
    return out;
  }

  function wav(pcm: Int16Array): Blob {
    const header = new ArrayBuffer(44);
    const v = new DataView(header);
    const str = (o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    str(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); str(8, "WAVE");
    str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, TARGET_RATE, true); v.setUint32(28, TARGET_RATE * 2, true);
    v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, "data"); v.setUint32(40, pcm.length * 2, true);
    return new Blob([header, pcm.buffer as ArrayBuffer], { type: "audio/wav" });
  }

  async function send(samples: Float32Array, rate: number) {
    const body = wav(downsample(samples, rate));
    inFlight++;
    onInterim("…");
    mark(`heard ${(samples.length / rate).toFixed(1)}s, transcribing locally`);
    try {
      const res = await fetch("/api/stt", { method: "POST", body, headers: { "Content-Type": "audio/wav" } });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        mark(`local stt error ${res.status}: ${data.error || ""}`);
        return;
      }
      const text = String(data.text || "").trim();
      if (text) onTranscript(text);
      else mark("local stt: nothing recognisable");
    } catch (e) {
      mark(`local stt request failed: ${e}`);
    } finally {
      inFlight--;
    }
  }

  function finish(rate: number) {
    recording = false;
    const dur = Date.now() - startedAt - END_SILENCE_MS;
    const total = chunks.reduce((n, c) => n + c.length, 0);
    const all = new Float32Array(total);
    let o = 0;
    for (const c of chunks) { all.set(c, o); o += c.length; }
    chunks = [];
    if (dur < MIN_SPEECH_MS) return;
    void send(all, rate);
  }

  function onAudio(buf: Float32Array, rate: number) {
    if (!listening || paused) return;
    const copy = new Float32Array(buf);
    const level = rms(copy);
    const threshold = Math.max(MIN_LEVEL, noise * 3);
    const now = Date.now();

    if (!recording) {
      preroll.push(copy);
      const keep = Math.ceil((PREROLL_MS / 1000) * rate / FRAME);
      while (preroll.length > keep) preroll.shift();
      if (level > threshold) {
        loudRun++;
        if (loudRun >= START_FRAMES) {
          recording = true;
          startedAt = now;
          lastLoudAt = now;
          chunks = preroll.splice(0);
          loudRun = 0;
        }
      } else {
        loudRun = 0;
        noise = noise * 0.97 + level * 0.03;   // si adatta al rumore della stanza
      }
      return;
    }

    chunks.push(copy);
    if (level > threshold) lastLoudAt = now;
    if (now - lastLoudAt > END_SILENCE_MS || now - startedAt > MAX_UTTERANCE_MS) finish(rate);
  }

  async function open() {
    if (ctx) return;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
      });
    } catch (e) {
      onError("Microphone not available");
      mark(`could not open the microphone: ${e}`);
      return;
    }
    ctx = new AudioContext();
    const source = ctx.createMediaStreamSource(stream);
    node = ctx.createScriptProcessor(FRAME, 1, 1);
    const rate = ctx.sampleRate;
    node.onaudioprocess = (ev) => onAudio(ev.inputBuffer.getChannelData(0), rate);
    source.connect(node);
    node.connect(ctx.destination);           // necessario perché il nodo giri; l'uscita è silenziosa
    mark(`listening (local whisper, ${rate} Hz)`);
  }

  return {
    start() {
      listening = true;
      paused = false;
      void open();
      if (ctx && ctx.state === "suspended") void ctx.resume();
    },
    stop() {
      listening = false;
      recording = false;
      chunks = [];
    },
    pause() {
      paused = true;
      recording = false;
      chunks = [];
    },
    resume() {
      paused = false;
      if (ctx && ctx.state === "suspended") void ctx.resume();
    },
    restart(reason: string) {
      // Il segnale "DEAF" del monitor riguarda Chrome; qui non c'è nulla da
      // riavviare, ma lo si annota se una trascrizione è ancora in corso.
      if (inFlight) mark(`ignored restart (${reason}): transcription in progress`);
    },
  };
}
