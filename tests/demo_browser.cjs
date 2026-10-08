/* Run with the bundled Playwright package on NODE_PATH and DEMO_URL set. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require("playwright");

const url = process.env.DEMO_URL || "http://127.0.0.1:8765";
const artifacts = path.resolve(__dirname, "../.runtime");

async function main() {
  const voiceWav = process.env.DEMO_VOICE_WAV || path.join(artifacts, "speech/validation/clean-0.wav");
  const browser = await chromium.launch({headless: true, channel: "msedge", args: ["--mute-audio",
    "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-audio-capture=${voiceWav}`]});
  const context = await browser.newContext({viewport: {width: 1440, height: 900}, acceptDownloads: true});
  const page = await context.newPage();
  const errors = [];
  const speechRequests = [];
  page.on("request", (request) => {if (request.url().endsWith("/speech")) speechRequests.push(request);});
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {if (message.type() === "error") errors.push(message.text());});
  page.setDefaultTimeout(30000);

  async function noOverflow() {
    const sizes = await page.evaluate(() => ({width: innerWidth, content: document.documentElement.scrollWidth}));
    assert(sizes.content <= sizes.width + 1, `Horizontal overflow: ${JSON.stringify(sizes)}`);
  }
  async function screenshot(name) {
    await noOverflow();
    await page.screenshot({path: path.join(artifacts, name), fullPage: false});
  }
  async function send(text, timeout = 15000) {
    const previous = await page.evaluate(() => state.session.history.length);
    await page.locator("#message").fill(text);
    const result = page.waitForResponse((response) => response.url().endsWith("/messages") && response.request().method() === "POST", {timeout});
    await page.locator("#send").click();
    const response = await result;
    assert.strictEqual(response.status(), 200);
    await page.waitForFunction((turn) => state.session.history.length === turn && !state.busy, previous + 1);
    return page.evaluate(() => state.session.history.at(-1));
  }
  async function reset() {
    await page.locator("#new-session").click();
    if (await page.locator("#reset-dialog").isVisible()) await page.locator("#confirm-reset").click();
    await page.waitForFunction(() => document.querySelector("#turn-count").textContent === "0 turns" && !document.querySelector("#message").disabled);
  }
  async function apply() {
    await page.locator("#apply-profile").click();
    if (await page.locator("#reset-dialog").isVisible()) await page.locator("#confirm-reset").click();
    await page.waitForFunction(() => !document.querySelector("#message").disabled);
  }
  function nextSpeech(timeout) {
    return page.waitForResponse((response) => response.url().endsWith("/speech"), {timeout}).catch((error) => error);
  }
  async function checkSpeech(responsePromise) {
    const response = await responsePromise;
    if (response instanceof Error) throw response;
    assert.strictEqual(response.status(), 200);
    assert(Number(response.headers()["content-length"]) > 44);
    // Edge's DevTools transport omits some audio bodies; verify the decoded player.
    await page.waitForFunction(() => playback && playback.readyState >= 2 && Number.isFinite(playback.duration) && playback.duration > 0);
    await page.locator("#stop-playback").click();
  }

  try {
    await page.goto(url);
    await page.waitForFunction(() => !document.querySelector("#message").disabled, null, {timeout: 125000});
    assert(await page.locator("svg.lucide").count() > 15, "Icon assets must render");
    assert(!(await page.locator("#review-transcript").isChecked()), "Voice must answer directly by default");
    assert(await page.locator("#auto-speak").isChecked(), "Voice replies must be spoken by default");
    await screenshot("ui-desktop.png");
    console.log("PASS initial desktop UI and local assets");

    await page.locator("[data-generator='template']").click();
    await apply();
    const first = await send("I need health insurance for my family.");
    assert.strictEqual(first.response.understanding.classifier.backend, "transformer");
    assert((await page.locator("#runtime-trace").textContent()).includes("MiniLM"));
    assert.strictEqual(first.response.state.need, "FAMILY_HEALTH");
    assert.strictEqual(first.response.source, "template");
    assert(!first.response.closed);
    assert.strictEqual(speechRequests.length, 0, "Typed turns must not start automatic playback");
    assert((await page.locator("#understanding").textContent()).includes("Text emotion"));
    assert(!(await page.locator("#acoustic-section").isVisible()));
    assert((await page.locator("#action-name").textContent()) !== "Awaiting buyer");
    await screenshot("ui-desktop-chat.png");

    const rejection = await send("No, I am not interested. Please stop.");
    assert.strictEqual(rejection.response.action, "RESPECT_REJECTION");
    await page.waitForFunction(() => document.querySelector("#message").disabled);
    assert(await page.locator("#conversation-closed").isVisible());
    const downloadPromise = page.waitForEvent("download");
    await page.locator("#export").click();
    const download = await downloadPromise;
    const transcript = path.join(artifacts, "ui-transcript.json");
    await download.saveAs(transcript);
    assert.strictEqual(JSON.parse(fs.readFileSync(transcript, "utf8")).history.length, 2);
    const session = await page.locator("#session-label").textContent();
    await page.reload();
    await page.waitForFunction(() => document.querySelector("#turn-count").textContent === "2 turns");
    assert.strictEqual(await page.locator("#session-label").textContent(), session);
    assert(await page.locator("#conversation-closed").isVisible());
    await page.locator("#new-session").click();
    await page.locator("#cancel-reset").click();
    assert.strictEqual(await page.locator("#turn-count").textContent(), "2 turns");
    await reset();
    console.log("PASS text conversation, rejection, export, restore, reset confirmation");

    await page.locator("[data-scenario='coverage']").click();
    await page.waitForFunction(() => document.querySelector("#message").value.includes("5 lakh"));
    assert.strictEqual(await page.locator("#age").inputValue(), "42");
    assert.strictEqual(await page.locator("#budget").inputValue(), "mid");
    await page.locator("[data-scenario='family']").click();
    await page.waitForFunction(() => document.querySelector("#message").value === "I need health insurance for my family.");
    await send('I need family health insurance. <img src=x onerror="window.__xss=1">');
    assert.strictEqual(await page.evaluate(() => window.__xss), undefined);
    assert.strictEqual(await page.locator(".message-text img").count(), 0);
    await reset();
    console.log("PASS scenario presets and literal rendering of buyer input");

    await page.locator("[data-view='catalogue'].nav-button").click();
    await page.locator("#catalogue-need").selectOption("FAMILY_HEALTH");
    await page.locator("#eligible-only").check();
    await page.waitForFunction(() => document.querySelectorAll(".product").length === 1);
    assert((await page.locator("#product-grid").textContent()).includes("HLTH-5L"));
    await page.locator("#catalogue-need").selectOption("RETIREMENT");
    await page.locator(".no-products").waitFor();
    await page.locator("#eligible-only").uncheck();
    await page.waitForFunction(() => document.querySelectorAll(".product").length === 8);
    await screenshot("ui-catalogue.png");
    await page.locator("[data-view='results'].nav-button").click();
    await page.locator("#results-pipeline").selectOption("understanding");
    await page.waitForFunction(() => document.querySelectorAll("#results-table tr").length === 3);
    assert((await page.locator("#results-table").textContent()).includes("Multi-label objections"));
    assert((await page.locator("#results-source").textContent()).includes("understanding_v1"));
    await screenshot("ui-transformer-results.png");
    await page.locator("#results-pipeline").selectOption("voice");
    await page.waitForFunction(() => document.querySelectorAll("#results-table tr").length === 4);
    assert((await page.locator("#results-table").textContent()).includes("6.33%"));
    assert((await page.locator("#results-source").textContent()).includes("voice_v3_transformer"));
    assert((await page.locator("#voice-metrics").textContent()).includes("4.67%"));
    assert(!(await page.locator("#text-chart").isVisible()));
    await screenshot("ui-voice-results.png");
    await page.locator("#results-pipeline").selectOption("text");
    await page.waitForFunction(() => document.querySelector("#results-source").textContent.includes("rl_comparison.csv"));
    assert((await page.locator("#results-table").textContent()).includes("4.120"));
    assert(await page.locator(".results-chart img").evaluate((img) => img.complete && img.naturalWidth > 0));
    await screenshot("ui-results.png");
    console.log("PASS catalogue eligibility, empty match state, measured results and training image");

    await page.locator("[data-view='conversation'].nav-button").click();
    await page.locator("#nlp-backend").selectOption("nb");
    await apply();
    assert.strictEqual((await send("I need family health insurance.")).response.understanding.classifier.backend, "nb");
    await page.locator("#nlp-backend").selectOption("transformer");
    await apply();
    const compound = await send("The premium is too expensive and I don't trust this insurer.");
    assert(compound.response.raw_understanding.objections.includes("PRICE_TOO_HIGH"));
    assert(compound.response.raw_understanding.objections.includes("DO_NOT_TRUST_INSURER"));
    assert((await page.locator("#understanding").textContent()).includes("All objections"));
    await screenshot("ui-multilabel.png");
    await reset();
    await page.locator("[data-generator='hybrid']").click();
    await apply();
    const routed = await send("I need health insurance for my family.");
    assert.strictEqual(routed.response.generation_route, "deterministic_task");
    assert((await page.locator(".message-source").allTextContents()).includes("Deterministic reply"));
    assert(!(await page.locator(".message-source").allTextContents()).includes("Template fallback"));
    const amount = await send("around 250000");
    assert.strictEqual(amount.response.decision_reason, "ambiguous_amount");
    const payment = await send("what will be my payment");
    assert.strictEqual(payment.response.decision_reason, "no_verified_premium_quote");
    assert(!payment.response.llm_attempted);
    assert((await page.locator("#runtime-trace").textContent()).includes("Deterministic Task"));
    await reset();

    for (const text of ["i am looking for cavalier saudis are my family", "i am looking for health insurance in my family",
      "yes i am happy to share of my information ask 1 by 1"]) {
      const continued = await send(text);
      assert(!continued.response.closed && !continued.response.llm_attempted);
    }
    assert((await page.locator("#workflow-fields").textContent()).includes("Coverage target"));
    assert((await send("500000")).response.state.dialogue_task === "discovery_existing");
    assert((await send("no")).response.state.workflow.phase === "ready_for_review");
    const summary = await send("yes please");
    assert(summary.response.state.dialogue_task === "handoff_summary" && !summary.response.closed);
    assert((await page.locator("#workflow-phase").textContent()).includes("Summary Ready"));
    await screenshot("ui-guided-workflow.png");
    assert((await send("Please stop.")).response.closed_reason === "buyer_stop");
    await reset();
    console.log("PASS reported false-closure regression, guided questions, product review and handoff summary");

    // Browser capture uses only the generated WAV supplied by Chromium's fake device.
    if (fs.existsSync(voiceWav)) {
      assert(!(await page.locator("#review-transcript").isChecked()));
      await page.evaluate(() => {
        window.__streams = [];
        window.__getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
        navigator.mediaDevices.getUserMedia = async (...args) => {
          const stream = await window.__getUserMedia(...args);
          window.__streams.push(stream);
          return stream;
        };
      });
      await page.locator("#record").click();
      await page.waitForFunction(() => document.querySelector("#recording-status").textContent.startsWith("Recording"));
      await screenshot("ui-recording.png");
      await page.locator("#cancel-recording").click();
      await page.waitForFunction(() => document.querySelector("#recording-strip").hidden);
      assert.strictEqual(await page.locator("#turn-count").textContent(), "0 turns");
      assert(await page.evaluate(() => window.__streams.every((stream) => stream.getTracks().every((track) => track.readyState === "ended"))));

      await page.locator("#record").click();
      await page.waitForFunction(() => document.querySelector("#recording-status").textContent.startsWith("Recording"));
      await page.locator("[data-view='catalogue'].nav-button").click();
      await page.waitForFunction(() => !state.recording);
      assert(await page.evaluate(() => window.__streams.every((stream) => stream.getTracks().every((track) => track.readyState === "ended"))));
      await page.locator("[data-view='conversation'].nav-button").click();
      assert.strictEqual(await page.locator("#turn-count").textContent(), "0 turns");

      await page.locator("#record").click();
      await page.waitForFunction(() => document.querySelector("#recording-status").textContent.startsWith("Recording 4"));
      const capturedPromise = page.waitForResponse((response) => response.url().endsWith("/audio"), {timeout: 90000});
      // Automatic speech follows both audio processing (90 s) and synthesis (65 s).
      const automaticSpeech = nextSpeech(160000);
      await page.locator("#record").click();
      const captured = await capturedPromise;
      assert.strictEqual(captured.status(), 200);
      await page.waitForFunction(() => document.querySelector("#turn-count").textContent === "1 turn");
      const capturedEntry = await page.evaluate(() => state.session.history[0]);
      assert.strictEqual(capturedEntry.response.state.need, "FAMILY_HEALTH");
      assert.strictEqual(capturedEntry.voice.acoustic_emotion.source, "audio_waveform");
      assert(["estimated", "uncertain"].includes(capturedEntry.voice.acoustic_emotion.status));
      assert(!(await page.locator("#transcript-review").isVisible()), "No Send confirmation for direct voice turns");
      assert.strictEqual(await page.locator(".chat-message.agent").count(), 1);
      assert((await page.locator("#understanding").textContent()).includes("Spoken intent"));
      assert(!(await page.locator("#understanding").textContent()).includes("Text emotion"));
      assert(await page.locator("#vocal-emotion").isVisible());
      assert((await page.locator("#vocal-emotion").textContent()).includes(capturedEntry.voice.acoustic_emotion.status === "uncertain" ?
        "Uncertain" : `${capturedEntry.voice.acoustic_emotion.label[0]}${capturedEntry.voice.acoustic_emotion.label.slice(1).toLowerCase()}`));
      assert((await page.locator("#acoustic-trace").textContent()).includes("wav2vec2"));
      assert(!(await page.locator("#text-emotion-trace").isVisible()));
      assert(await page.evaluate(() => window.__streams.every((stream) => stream.getTracks().every((track) => track.readyState === "ended"))));
      await checkSpeech(automaticSpeech);
      await screenshot("ui-voice-conversation.png");
      await page.locator("#acoustic-section summary").click();
      assert(await page.locator("#acoustic-trace").isVisible());
      await page.locator("#acoustic-section summary").click();
      await page.locator("#text-emotion-section summary").click();
      assert((await page.locator("#text-emotion-trace").textContent()).includes("RoBERTa"));
      await page.locator("#text-emotion-section summary").click();

      // Presentation-only fixtures must never replace a missing waveform estimate with text emotion.
      for (const [audio, expected] of [
        [{source: "audio_waveform", status: "estimated", label: "ANGRY", confidence: .9}, "Angry (Estimate)"],
        [{source: "audio_waveform", status: "uncertain", label: "SAD", confidence: .4}, "Uncertain"],
        [{source: "audio_waveform", status: "unavailable", error: "Model unavailable"}, "Unavailable"],
        [{source: "audio_waveform", status: "insufficient_audio"}, "Insufficient Audio"],
        [{source: "audio_waveform", status: "skipped_stop"}, "Skipped For Stop Request"],
        [{source: "audio_waveform", status: "disabled"}, "Disabled"],
        [{source: "text", status: "estimated", label: "HAPPY", confidence: .99}, "Not Analyzed"],
        [null, "Not Analyzed"],
      ]) {
        await page.evaluate((acoustic) => {
          const entry = structuredClone(state.session.history[0]);
          entry.voice.acoustic_emotion = acoustic;
          renderInspector(entry);
        }, audio);
        assert.strictEqual(await page.locator("#vocal-emotion strong").textContent(), expected);
        if (expected !== "Angry (Estimate)") assert.strictEqual(await page.locator("#vocal-emotion .confidence").count(), 0);
      }
      await page.evaluate(() => renderInspector(state.session.history[0]));
      await page.setViewportSize({width: 390, height: 844});
      await page.locator(".mobile-panels [data-panel='inspector']").click();
      await page.locator(".inspector-panel").evaluate((panel) => {panel.scrollTop = 0;});
      await screenshot("ui-mobile-voice-inspector.png");
      const vocalBox = await page.locator("#vocal-emotion").boundingBox();
      assert(vocalBox && vocalBox.y >= 0 && vocalBox.y + vocalBox.height <= 844, "Vocal emotion must be above the fold");
      await page.locator(".mobile-panels [data-panel='chat']").click();
      await page.setViewportSize({width: 1440, height: 900});

      await page.locator("#auto-speak").uncheck();
      const speechPromise = nextSpeech(70000);
      await page.locator("[data-speak='1']").click();
      await checkSpeech(speechPromise);
      await reset();
      const speechCount = speechRequests.length;
      const uploadPromise = page.waitForResponse((response) => response.url().endsWith("/audio"));
      await page.locator("#audio-file").setInputFiles(voiceWav);
      assert.strictEqual((await uploadPromise).status(), 200);
      await page.waitForFunction(() => document.querySelector("#turn-count").textContent === "1 turn");
      assert(await page.evaluate(() => state.session.history[0].voice.transcript.confidence > .55));
      assert((await page.locator("#runtime-trace").textContent()).includes("ASR confidence"));
      assert.strictEqual(speechRequests.length, speechCount, "Spoken replies can be disabled");
      await reset();
      await page.locator("#review-transcript").check();
      await page.locator("#auto-speak").check();
      const reviewPromise = page.waitForResponse((response) => response.url().includes("/audio?review=1"));
      await page.locator("#audio-file").setInputFiles(voiceWav);
      assert.strictEqual((await reviewPromise).status(), 200);
      await page.locator("#transcript-review").waitFor();
      assert.strictEqual(await page.locator("#turn-count").textContent(), "0 turns");
      assert((await page.locator("#message").inputValue()).includes("insurance"));
      assert.strictEqual(speechRequests.length, speechCount, "Preview must not generate or speak an answer");
      await screenshot("ui-transcript-review.png");
      const reviewedSpeech = nextSpeech(90000);
      const reviewed = await send("I need health insurance for my family. Please ask one by one.");
      assert(reviewed.voice.corrected && reviewed.voice.reviewed);
      assert.strictEqual(reviewed.voice.acoustic_emotion.source, "audio_waveform");
      assert((await page.locator(".message-source").allTextContents()).includes("Corrected voice transcript"));
      assert(!(await page.locator("#transcript-review").isVisible()));
      await checkSpeech(reviewedSpeech);
      await reset();
      await page.evaluate(() => { navigator.mediaDevices.getUserMedia = async () => {throw new DOMException("Denied", "NotAllowedError");}; });
      await page.locator("#record").click();
      await page.waitForFunction(() => document.querySelector("#error-text").textContent.includes("permission denied"));
      assert(!(await page.locator("#record").isDisabled()));
      assert(!(await page.locator("#recording-strip").isVisible()));
      await page.locator("#dismiss-error").click();
      await page.evaluate(() => { navigator.mediaDevices.getUserMedia = window.__getUserMedia; });
      console.log("PASS direct voice answers, automatic native TTS, separate vocal/text emotion, optional review, capture cleanup and permission failure");
    }
    await page.locator("[data-generator='template']").click();
    await apply();

    await page.setViewportSize({width: 390, height: 844});
    await page.locator("[data-view='conversation'].nav-button").click();
    await screenshot("ui-mobile.png");
    await page.locator(".mobile-panels [data-panel='buyer']").click();
    assert(await page.locator(".buyer-panel").isVisible());
    await page.locator("#age").fill("99");
    await apply();
    await page.locator("[data-view='catalogue'].nav-button").click();
    await page.locator("#eligible-only").check();
    await page.locator(".no-products").waitFor();
    assert((await page.locator("#catalogue-summary").textContent()).includes("Age 99"));
    await page.locator("[data-view='conversation'].nav-button").click();
    await page.locator(".mobile-panels [data-panel='buyer']").click();
    await page.locator("#age").fill("35");
    await apply();
    await page.locator(".mobile-panels [data-panel='chat']").click();
    await send("I need health insurance for my family.");
    await screenshot("ui-mobile-chat.png");
    await page.locator(".message-inspect").click();
    assert(await page.locator(".inspector-panel").isVisible());
    await screenshot("ui-mobile-inspector.png");
    await page.locator(".mobile-panels [data-panel='chat']").click();
    await page.setViewportSize({width: 320, height: 740});
    await noOverflow();
    await screenshot("ui-small-mobile.png");
    await page.locator("[data-view='results'].nav-button").click();
    await page.locator("#results-pipeline").selectOption("voice");
    await page.waitForFunction(() => document.querySelector("#results-source").textContent.includes("voice_v3_transformer"));
    await screenshot("ui-mobile-results.png");
    await page.locator("[data-view='conversation'].nav-button").click();
    await page.setViewportSize({width: 1280, height: 720});
    await screenshot("ui-compact-desktop.png");
    console.log("PASS mobile panels, profile changes, age eligibility and 320px/1280px layouts");

    if (process.env.DEMO_LLM_TEST === "1") {
      await reset();
      await page.locator("[data-generator='ollama']").click();
      await apply();
      const generated = await send("I need health insurance for my family.", 145000);
      assert.strictEqual(generated.response.llm_error, null);
      assert(generated.response.source === "ollama" || generated.response.llm_rejected,
        "The local LLM must generate an accepted response or an explicit guard rejection");
      await screenshot("ui-live-llm.png");
      console.log(`PASS real LLM request: source=${generated.response.source}, rejected=${generated.response.llm_rejected}, duration=${generated.duration_ms}ms`);
      const amount = await send("around 250000", 145000);
      assert.strictEqual(amount.response.llm_error, null);
      assert.strictEqual(amount.response.decision_reason, "ambiguous_amount");
      assert.strictEqual(amount.response.state.amount_context.kind, "unknown");
      assert(amount.response.validation.ok && !amount.response.closed);
      console.log(`PASS amount clarification: source=${amount.response.source}, reasons=${JSON.stringify(amount.response.llm_rejection_reasons)}, duration=${amount.duration_ms}ms`);
      const payment = await send("what will be my payment", 145000);
      assert.strictEqual(payment.response.llm_error, null);
      assert.strictEqual(payment.response.decision_reason, "no_verified_premium_quote");
      assert(payment.response.validation.ok && !payment.response.validation.repeated);
      assert(!/\d/.test(payment.response.text), "A payment reply must not invent a numeric quote");
      await screenshot("ui-fixed-llm.png");
      console.log(`PASS payment limitation: source=${payment.response.source}, reasons=${JSON.stringify(payment.response.llm_rejection_reasons)}, duration=${payment.duration_ms}ms`);
    }
    assert.deepStrictEqual(errors, [], "No browser JavaScript or asset errors");
    console.log("PASS browser console is clean");
  } catch (error) {
    console.error("UI failure state", await page.evaluate(() => ({
      error: document.querySelector("#error-text").textContent,
      feedback: document.querySelector("#voice-feedback").textContent,
      turn: state.session?.history.at(-1), understanding: document.querySelector("#understanding").textContent,
    })).catch(() => "Page unavailable"));
    await page.screenshot({path: path.join(artifacts, "ui-test-failure.png")}).catch(() => {});
    throw error;
  } finally {
    await context.close();
    await browser.close();
  }
}
main().catch((error) => {console.error(error); process.exitCode = 1;});
