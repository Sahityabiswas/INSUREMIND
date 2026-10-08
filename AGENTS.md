# Project Maintenance

- Treat the command-line clients, local web UI, API, training and evaluation as one project.
- When changing user-visible agent behavior, settings, model selection, speech, or result formats,
  update the corresponding UI controls, status/trace labels, API contract, tests and run documentation
  in the same change. Do not leave the UI on an older implementation.
- Reuse ConversationSession and VoiceSession; do not duplicate policy, safety or speech decisions
  in the browser. Resolve voice artifacts from configs/voice.yaml rather than hard-coded versions.
- Keep existing text and voice experiment artifacts. Do not label results from different simulator
  versions as a direct training improvement. Distinguish deterministic routing from LLM fallback.
- Browser microphone use must require an explicit user action. Release microphone tracks on stop,
  cancel and failure; do not retain raw recordings or use them for training without consent.
- Verify changed API behavior with pytest and UI workflows with tests/demo_browser.cjs at desktop
  and mobile widths. Native speech tests use generated audio, never an unattended microphone.
- This is a localhost research prototype with synthetic insurance products, not policy issuance,
  verified pricing, payment processing or a production sales service.
