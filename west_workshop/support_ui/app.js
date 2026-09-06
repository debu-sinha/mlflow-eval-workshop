"use strict";
const byId = id => document.getElementById(id);
let configuration = null;
let variant = "repaired";
let latest = null;
let busy = false;
const labels = {full_refund: "Full refund eligible", store_credit: "Store credit eligible", support_review: "Support review needed"};

function setBusy(value) {
  busy = value;
  document.querySelectorAll("#send, #order-select, [data-question], [data-variant]").forEach(element => element.disabled = value);
  byId("question").disabled = value;
  byId("chat-form").setAttribute("aria-busy", String(value));
}

function resetAnswer() {
  latest = null;
  byId("exchange").hidden = true;
  byId("welcome").hidden = false;
  byId("question").value = "";
}

function updateOrder() {
  if (!configuration) return;
  const order = configuration.orders[byId("order-select").value];
  byId("order-id").textContent = order.id;
  byId("order-age").textContent = order.label;
  byId("order-condition").textContent = order.condition;
  byId("refund-suggestion").dataset.question = order.defective
    ? `The item I bought ${order.days} days ago is defective. Can you help me?`
    : `I bought this ${order.days} days ago. Can you give me a full refund?`;
  byId("refund-suggestion").firstChild.textContent = order.defective ? "My item is defective " : "Can I get a refund? ";
  resetAnswer();
}

document.querySelectorAll("[data-question]").forEach(button => button.addEventListener("click", () => {
  byId("question").value = button.dataset.question;
  byId("question").focus();
}));
byId("order-select").addEventListener("change", updateOrder);
document.querySelectorAll("[data-variant]").forEach(button => button.addEventListener("click", () => {
  variant = button.dataset.variant;
  document.querySelectorAll("[data-variant]").forEach(item => {
    item.classList.toggle("selected", item.dataset.variant === variant);
    item.setAttribute("aria-pressed", String(item.dataset.variant === variant));
  });
  byId("policy-notice").hidden = variant !== "candidate";
  resetAnswer();
}));
byId("question").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    byId("chat-form").requestSubmit();
  }
});

byId("chat-form").addEventListener("submit", async event => {
  event.preventDefault();
  const question = byId("question").value.trim();
  if (busy || !question) return;
  latest = null;
  byId("welcome").hidden = true;
  byId("exchange").hidden = false;
  byId("customer-question").textContent = question;
  byId("thinking").hidden = false;
  byId("answer-block").hidden = true;
  byId("request-error").hidden = true;
  setBusy(true);
  try {
    const response = await fetch("/api/chat", {
      method: "POST", headers: {"Content-Type": "application/json", "X-Northstar-Request": "chat"},
      body: JSON.stringify({question, order: byId("order-select").value, variant}),
    });
    let data;
    try { data = await response.json(); } catch { throw new Error("The app connection was interrupted. Refresh the page and try again."); }
    if (!response.ok) throw new Error(data.error || "This request could not be completed. Please try again.");
    latest = data;
    byId("answer").textContent = data.body;
    byId("eligibility").textContent = labels[data.eligibility] || "Eligibility unclear";
    byId("eligibility").classList.toggle("unparsed", !labels[data.eligibility]);
    byId("answer-context").textContent = `${data.elapsed_seconds}s · ${data.variant === "candidate" ? "Stale" : "Current"} policy · Live response`;
    byId("answer-block").hidden = false;
    byId("question").value = "";
  } catch (error) {
    byId("request-error").textContent = error.message;
    byId("request-error").hidden = false;
  } finally {
    byId("thinking").hidden = true;
    setBusy(false);
  }
});

function showEvidence(useResponse) {
  const response = useResponse ? latest : null;
  byId("trace-details").hidden = !response;
  byId("evidence-policy-label").textContent = response ? "RETRIEVED IN THIS TRACE" : "SELECTED POLICY";
  byId("evidence-policy-version").textContent = response ? response.policy_version : variant === "candidate" ? "2024-01" : "2026-09";
  byId("evidence-policy").textContent = response ? response.retrieved_policy : configuration?.policies[variant] || "Policy details are unavailable. Refresh the page to load them.";
  byId("evidence-intro").textContent = response
    ? "This document came from the saved retrieval span for the answer you just received. The application uses the same predictor as the evaluation notebooks."
    : "The assistant receives one selected policy document. This is a simple document selector. After sending a question, follow its answer to inspect the actual saved retrieval.";
  if (response) {
    byId("evidence-model").textContent = response.model;
    byId("evidence-time").textContent = `${response.elapsed_seconds} seconds`;
    byId("evidence-trace").textContent = response.trace_id;
    byId("original-answer").textContent = response.answer;
    byId("trace-link").hidden = !response.trace_url;
    byId("local-trace-hint").hidden = !!response.trace_url;
    if (response.trace_url) byId("trace-link").href = response.trace_url;
  }
  byId("evidence-dialog").showModal();
}
byId("inspect-source").addEventListener("click", () => showEvidence(false));
byId("inspect-answer").addEventListener("click", () => showEvidence(true));
byId("close-evidence").addEventListener("click", () => byId("evidence-dialog").close());
byId("evidence-dialog").addEventListener("click", event => {
  if (event.target === byId("evidence-dialog")) {
    const bounds = event.target.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) event.target.close();
  }
});
fetch("/api/config").then(response => {
  if (!response.ok) throw new Error("Configuration unavailable");
  return response.json();
}).then(data => { configuration = data; updateOrder(); }).catch(() => {
  byId("live-status").textContent = "REFRESH TO RECONNECT";
});
