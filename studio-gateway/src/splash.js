// Provisioning splash shown for the initial navigation while a user's backend
// stack is being created (~60-90s on first visit). It also hosts the per-user
// "bring your own keys" step: on first visit a user adds one or more model
// providers (each POSTed to /gateway/provider, which persists it), then clicks
// Continue to provision the stack once with the full set. Polls /gateway/ready
// and reloads into the studio SPA once ready.

const PROVIDER_LABELS = { openai: 'OpenAI', anthropic: 'Anthropic', vertex: 'Vertex (ADC)' };

function renderSplash(state, basePath = '') {
  const st = (state && state.state) || 'provisioning';
  const isError = st === 'error';
  // Poll/action endpoints must include the embed prefix so requests from inside
  // the dashboard iframe route back through the dashboard proxy to the gateway.
  const readyUrl = `${basePath}/gateway/ready`;
  const retryUrl = `${basePath}/gateway/retry`;
  const providerUrl = `${basePath}/gateway/provider`;
  const providers = Array.isArray(state && state.providers) && state.providers.length
    ? state.providers
    : ['openai', 'anthropic', 'vertex'];
  const detail = isError
    ? escapeHtml(state.error || 'Provisioning failed.')
    : 'Setting up your isolated workspace (server, database, and compute namespace). This usually takes about a minute on first launch.';
  const options = providers
    .map((p) => `<option value="${escapeHtml(p)}">${escapeHtml(PROVIDER_LABELS[p] || p)}</option>`)
    .join('');
  // Escape "<" so injected strings containing "</script>" can't break out of the inline
  // <script> below (JSON.stringify alone doesn't escape it) — markup-injection safe.
  const esc = (v) => JSON.stringify(v).replace(/</g, '\\u003c');
  const initial = esc({ state: st, error: (state && state.error) || '' });

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Amortized Studio — preparing your workspace</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font-family: "Red Hat Text", "RedHatText", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    display: flex; align-items: center; justify-content: center;
    background: #f2f2f2; color: #151515;
  }
  @media (prefers-color-scheme: dark) { body { background: #0f1214; color: #e0e0e0; } .card { background: #1b1f22 !important; box-shadow: none !important; } }
  .card {
    background: #fff; border-radius: 12px; padding: 40px 44px; max-width: 460px; width: 92%; text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
  }
  .spinner {
    width: 44px; height: 44px; margin: 0 auto 24px; border-radius: 50%;
    border: 4px solid rgba(238,0,0,0.15); border-top-color: #ee0000;
    animation: spin 0.9s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  h1 { font-size: 1.25rem; margin: 0 0 8px; font-weight: 600; }
  p { font-size: 0.9rem; line-height: 1.5; margin: 0; opacity: 0.8; }
  .err { color: #c9190b; white-space: pre-wrap; text-align: left; font-family: monospace; font-size: 0.75rem;
         margin-top: 16px; max-height: 160px; overflow: auto; }
  button {
    margin-top: 20px; padding: 8px 20px; font-size: 0.9rem; border: 0; border-radius: 6px;
    background: #ee0000; color: #fff; cursor: pointer;
  }
  button:hover { background: #be0000; }
  button:disabled { opacity: 0.6; cursor: default; }
  .btn-secondary { background: transparent; color: inherit; border: 1px solid rgba(0,0,0,0.28); }
  .btn-secondary:hover { background: rgba(0,0,0,0.05); }
  @media (prefers-color-scheme: dark) {
    .btn-secondary { border-color: rgba(255,255,255,0.28); }
    .btn-secondary:hover { background: rgba(255,255,255,0.08); }
  }
  .hidden { display: none; }
  .keyform { text-align: left; }
  .keyform label { display: block; font-size: 0.85rem; margin: 16px 0 5px; font-weight: 600; }
  .keyform select, .keyform input, .keyform textarea {
    width: 100%; padding: 9px 11px; font-size: 0.9rem; border-radius: 6px;
    border: 1px solid rgba(0,0,0,0.22); background: #fff; color: inherit;
  }
  .keyform textarea { font-family: monospace; font-size: 0.78rem; resize: vertical; min-height: 120px; }
  @media (prefers-color-scheme: dark) {
    .keyform select, .keyform input, .keyform textarea { background: #0f1214; border-color: rgba(255,255,255,0.22); color: #e0e0e0; }
  }
  .keyform button { width: 100%; margin-top: 22px; }
  .keyform .btn-add { margin-top: 18px; }
  .configured-label { font-size: 0.8rem; font-weight: 600; margin-top: 18px; }
  .configured { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
  .chip { font-size: 0.8rem; padding: 4px 11px; border-radius: 999px;
          background: #e9f5e8; color: #1e4f18; border: 1px solid #95d58e; }
  @media (prefers-color-scheme: dark) { .chip { background: rgba(13,32,9,0.4); color: #5ba352; border-color: #163b11; } }
  .formerr { color: #c9190b; font-size: 0.8rem; margin-top: 12px; }
  .hint { font-size: 0.78rem; opacity: 0.7; margin-top: 6px; }
</style>
</head>
<body>
  <div class="card">
    <div id="status">
      <div class="spinner" id="spinner"></div>
      <h1 id="title">Preparing your Amortized Studio…</h1>
      <p id="detail">${detail}</p>
    </div>
    <form class="keyform hidden" id="keyform" onsubmit="return addProvider(event)">
      <h1>Connect your models</h1>
      <p>Add one or more model providers for Morty. Each is stored for your account only and used solely for your sandbox. You can add or change these later in Settings.</p>
      <div class="configured-label hidden" id="configured-label">Added</div>
      <div class="configured hidden" id="configured"></div>
      <label for="provider">Provider</label>
      <select id="provider">${options}</select>
      <label for="apikey" id="credlabel">API key</label>
      <input id="apikey" type="password" autocomplete="off" spellcheck="false" placeholder="paste your API key" />
      <textarea id="adcjson" class="hidden" rows="8" autocomplete="off" spellcheck="false" placeholder="paste your Vertex ADC JSON (application_default_credentials.json)"></textarea>
      <div class="hint" id="hint"></div>
      <div class="formerr hidden" id="formerr"></div>
      <button type="submit" id="addbtn" class="btn-secondary btn-add">Add provider</button>
      <button type="button" id="continuebtn" class="hidden" onclick="continueProvision()">Continue</button>
    </form>
    <div class="err hidden" id="err"></div>
    <button class="hidden" id="retry" onclick="retry()">Retry</button>
  </div>
<script>
  var READY_URL = ${esc(readyUrl)};
  var RETRY_URL = ${esc(retryUrl)};
  var PROVIDER_URL = ${esc(providerUrl)};
  var LABELS = ${esc(PROVIDER_LABELS)};
  var HINTS = { openai: 'OpenAI keys start with "sk-".', anthropic: 'Anthropic keys start with "sk-ant-".', vertex: 'Paste the Vertex ADC JSON (a Google credentials file). Stored for your account only.' };
  var POLL_MS = 2500;
  var CONFIGURED = {};   // provider -> true (added this session / already stored)
  function el(id){ return document.getElementById(id); }
  function show(id, on){ var e = el(id); if (e) e.classList.toggle('hidden', !on); }
  // Vertex is ADC-only: its credential is a JSON blob (textarea), not a key string (input).
  function isAdc(p){ return p === 'vertex'; }
  function updateHint(){
    var p = el('provider'); if (!p) return;
    var adc = isAdc(p.value);
    show('apikey', !adc); show('adcjson', adc);
    var lbl = el('credlabel'); if (lbl) lbl.textContent = adc ? 'Vertex ADC JSON' : 'API key';
    el('hint').textContent = HINTS[p.value] || '';
  }
  function renderConfigured(){
    var names = Object.keys(CONFIGURED);
    var box = el('configured');
    box.innerHTML = names.map(function(p){
      return '<span class="chip">' + (LABELS[p] || p) + '</span>';
    }).join('');
    show('configured', names.length > 0);
    show('configured-label', names.length > 0);
    show('continuebtn', names.length > 0);   // >=1 provider required before provisioning
  }
  function loadConfigured(){
    // Populate already-stored providers (e.g. after a reload mid-setup).
    fetch(PROVIDER_URL, { headers: { 'Accept': 'application/json' }, cache: 'no-store' })
      .then(function(r){ return r.ok ? r.json() : null; })
      .then(function(b){
        if (b && Array.isArray(b.configured)) {
          CONFIGURED = {};
          b.configured.forEach(function(p){ CONFIGURED[p] = true; });
          renderConfigured();
        }
      })
      .catch(function(){});
  }
  function showProvisioning(){
    show('keyform', false); show('err', false); show('retry', false); show('status', true);
    el('spinner').classList.remove('hidden');
    el('title').textContent = 'Preparing your Amortized Studio…';
    el('detail').textContent = 'Setting up your isolated workspace. This usually takes about a minute.';
  }
  function showNeedsKey(){
    el('spinner').classList.add('hidden');
    show('status', false); show('err', false); show('retry', false); show('keyform', true);
    updateHint();
    loadConfigured();
  }
  function toError(msg){
    el('spinner').classList.add('hidden');
    show('keyform', false); show('status', true);
    el('title').textContent = 'Could not prepare your workspace';
    el('detail').textContent = 'Something went wrong provisioning your environment.';
    var e = el('err'); e.textContent = msg || ''; show('err', !!msg); show('retry', true);
  }
  function applyState(s){
    if (!s || !s.state) { setTimeout(poll, POLL_MS); return; }
    if (s.state === 'ready') { window.location.reload(); return; }
    if (s.state === 'needs_key') { showNeedsKey(); return; }
    if (s.state === 'error') { toError(s.error); return; }
    showProvisioning(); setTimeout(poll, POLL_MS);
  }
  function poll(){
    fetch(READY_URL, { headers: { 'Accept': 'application/json' }, cache: 'no-store' })
      .then(function(r){
        if (r.status === 401 || r.status === 403) { toError('Your session has expired. Reload the page to sign in again.'); return null; }
        return r.json();
      })
      .then(function(s){ if (s) applyState(s); })
      .catch(function(){ setTimeout(poll, POLL_MS); });
  }
  function retry(){
    show('retry', false); show('err', false); showProvisioning();
    fetch(RETRY_URL, { method: 'POST', cache: 'no-store' })
      .then(function(r){ return r.json(); })
      .then(applyState)
      .catch(function(){ setTimeout(poll, POLL_MS); });
  }
  // Add ONE provider (persist only; does not provision yet). Stays on the form so the
  // user can add more; Continue then provisions with the full set.
  function addProvider(ev){
    ev.preventDefault();
    var provider = el('provider').value;
    var adc = isAdc(provider);
    var key = adc ? el('adcjson').value : el('apikey').value;
    var fe = el('formerr'); fe.classList.add('hidden');
    if (adc) {
      var ok = false;
      try { var o = JSON.parse(key); ok = !!o && typeof o === 'object' && !!o.type; } catch (e) { ok = false; }
      if (!ok) { fe.textContent = 'Paste a valid Vertex ADC JSON (a Google credentials file).'; fe.classList.remove('hidden'); return false; }
    } else if (!key || key.trim().length < 8) {
      fe.textContent = 'Enter a valid API key.'; fe.classList.remove('hidden'); return false;
    }
    var btn = el('addbtn'); btn.disabled = true; btn.textContent = 'Adding…';
    fetch(PROVIDER_URL, {
      method: 'POST', cache: 'no-store',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify({ provider: provider, key: key })
    })
      .then(function(r){ return r.json().then(function(b){ return { ok: r.ok, body: b }; }); })
      .then(function(res){
        btn.disabled = false; btn.textContent = 'Add provider';
        if (!res.ok) { fe.textContent = (res.body && res.body.error) || 'Could not add provider.'; fe.classList.remove('hidden'); return; }
        if (res.body && Array.isArray(res.body.configured)) {
          CONFIGURED = {};
          res.body.configured.forEach(function(p){ CONFIGURED[p] = true; });
        } else {
          CONFIGURED[provider] = true;
        }
        el('apikey').value = ''; el('adcjson').value = '';
        renderConfigured();
      })
      .catch(function(){ btn.disabled = false; btn.textContent = 'Add provider'; fe.textContent = 'Network error. Try again.'; fe.classList.remove('hidden'); });
    return false;
  }
  // Provision the stack with everything added so far (retry clears the needs_key entry,
  // then ensureUserStack provisions with the full provider set).
  function continueProvision(){
    showProvisioning();
    fetch(RETRY_URL, { method: 'POST', cache: 'no-store' })
      .then(function(r){ return r.json(); })
      .then(applyState)
      .catch(function(){ setTimeout(poll, POLL_MS); });
  }
  var providerSel = el('provider'); if (providerSel) providerSel.addEventListener('change', updateHint);
  applyState(${initial});
</script>
</body>
</html>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

module.exports = { renderSplash };
