// pwa_manager.js
import * as webllm from "https://esm.run/@mlc-ai/web-llm";

// 1. Declare Offline PWA Custom Model Registry
const CUSTOM_MODEL_ID = "qwen2.5-0.5b-custom-q4f16_1";

const appConfig = {
  model_list: [
    {
      // Point either to local public folders or custom cache boundaries for offline PWA operation
      model: "./mlc_weights/",
      model_id: CUSTOM_MODEL_ID,
      model_lib: "./libs/qwen2_5-0_5b-q4f16_1-webgpu.wasm",
      overrides: {
        context_window_size: 2048,
      },
    },
  ],
  // Cache weights directly inside the PWA's local Origin Private File System (OPFS)
  cacheBackend: "opfs",
};

class EdgeInferenceManager {
  constructor() {
    this.engine = null;
    this.tokenHistory = [];
    this.historyWindowSize = 16;
    // Base temperature config
    this.temperature = 0.7;
  }

  /**
   * Initializes the Web Worker and loads the WebGPU custom model
   */
  async initializeModel(onProgress) {
    console.log("[PWA Manager] Launching Web Worker background thread...");

    const worker = new Worker(
      new URL("./pwa_worker.js", import.meta.url),
      { type: "module" }
    );

    this.engine = new webllm.WebWorkerMLCEngine(worker, {
      appConfig: appConfig,
      initProgressCallback: (report) => {
        if (onProgress) onProgress(report.text);
      },
      logLevel: "INFO",
    });

    console.log("[PWA Manager] Reloading custom Qwen2.5 WASM model...");
    await this.engine.reload(CUSTOM_MODEL_ID);
    console.log("[PWA Manager] Model loaded and ready on WebGPU.");
  }

  /**
   * Port of VAR active-governor concepts: calculates sliding window token repetition
   */
  detectRepetitionAnomaly() {
    if (this.tokenHistory.length < this.historyWindowSize) {
      return false; // Not enough tokens to evaluate
    }

    const uniqueTokens = new Set(this.tokenHistory);
    const uniquenessRatio = uniqueTokens.size / this.tokenHistory.length;

    // If less than 35% of recent tokens are unique, we are entering a loop degeneration
    const anomalyDetected = uniquenessRatio < 0.35;

    if (anomalyDetected) {
      console.warn(
        `[VAR Monitor] Repetition anomaly flagged! Uniqueness ratio: ${(uniquenessRatio * 100).toFixed(1)}%`
      );
    }

    return anomalyDetected;
  }

  /**
   * Generates text with self-healing feedback intervention
   */
  async generateText(prompt, onTokenCallback) {
    this.tokenHistory = [];
    this.temperature = 0.7; // Reset to safe baseline temperature

    const messages = [
      { role: "system", content: "You are an offline survival assistant. Answer clearly." },
      { role: "user", content: prompt },
    ];

    console.log("[PWA Manager] Initiating streaming inference pass...");

    const completion = await this.engine.chat.completions.create({
      messages: messages,
      stream: true,
      temperature: this.temperature,
      max_tokens: 256,
      // Pass a callback to read progress and run active intervention
      stream_options: { include_usage: false },
    });

    let generatedText = "";

    for await (const chunk of completion) {
      const token = chunk.choices[0]?.delta?.content || "";
      if (!token) continue;

      generatedText += token;
      onTokenCallback(token);

      // Track token structure
      this.tokenHistory.push(token);
      if (this.tokenHistory.length > this.historyWindowSize) {
        this.tokenHistory.shift();
      }

      // Check for quantization-induced loop decay
      if (this.detectRepetitionAnomaly()) {
        // Shift temperature up to break the low-precision local minima
        this.temperature = Math.min(this.temperature + 0.3, 1.4);
        console.log(
          `[VAR Intervention] Increasing temperature to ${this.temperature} to force token divergence.`
        );

        // Update live runtime engine options for the remaining token generations
        await this.engine.setOptions({ temperature: this.temperature });
      }
    }

    return generatedText;
  }
}

export { EdgeInferenceManager };
