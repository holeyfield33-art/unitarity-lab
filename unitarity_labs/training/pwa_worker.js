// pwa_worker.js
// Hosted locally in your PWA build or imported via bundler/CDN
import { WebWorkerMLCEngineHandler } from "https://esm.run/@mlc-ai/web-llm";

// Instantiate the prebuilt engine handler to process incoming Main-Thread instructions
const handler = new WebWorkerMLCEngineHandler();

self.onmessage = (msg) => {
  handler.onmessage(msg);
};
