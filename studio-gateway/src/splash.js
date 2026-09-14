// Provisioning splash shown for the initial navigation while a user's backend
// stack is being created (~60-90s on first visit). It also hosts the per-user
// "bring your own key" step: on first visit a user picks a model provider and
// pastes its API key, which is POSTed to /gateway/provider before provisioning
// proceeds. Polls /gateway/ready and reloads into the studio SPA once ready.

const PROVIDER_LABELS = { openai: 'OpenAI', anthropic: 'Anthropic' };

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
    : ['openai', 'anthropic'];
  const detail = isError
    ? escapeHtml(state.error || 'Provisioning failed.')
    : 'Setting up your isolated workspace (server, database, and compute namespace). This usually takes about a minute on first launch.';
  const options = providers
    .map((p) => `<option value="${escapeHtml(p)}">${escapeHtml(PROVIDER_LABELS[p] || p)}</option>`)
    .join('');
  const initial = JSON.stringify({ state: st, error: (state && state.error) || '' });

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
    background: #fff; border-radius: 12px; padding: 40px 44px; max-width: 440px; width: 92%; text-align: center;
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
  .hidden { display: none; }
  .keyform { text-align: left; }
  .keyform label { display: block; font-size: 0.85rem; margin: 16px 0 5px; font-weight: 600; }
  .keyform select, .keyform input {
    width: 100%; padding: 9px 11px; font-size: 0.9rem; border-radius: 6px;
    border: 1px solid rgba(0,0,0,0.22); background: #fff; color: inherit;
  }
  @media (prefers-color-scheme: dark) {
    .keyform select, .keyform input { background: #0f1214; border-color: rgba(255,255,255,0.22); color: #e0e0e0; }
  }
  .keyform button { width: 100%; margin-top: 22px; }
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
    <form class="keyform hidden" id="keyform" onsubmit="return submitKey(event)">
      <h1>Connect your model</h1>
      <p>Morty needs a model API key to chat. It is stored for your account only and used solely for your sandbox.</p>
      <label for="provider">Provider</label>
      <select id="provider">${options}</select>
      <label for="apikey">API key</label>
      <input id="apikey" type="password" autocomplete="off" spellcheck="false" placeholder="paste your API key" />
      <div class="hint" id="hint"></div>
      <div class="formerr hidden" id="formerr"></div>
      <button type="submit" id="savebtn">Save &amp; continue</button>
    </form>
    <div class="err hidden" id="err"></div>
    <button class="hidden" id="retry" onclick="retry()">Retry</button>
  </div>
<script>
  var READY_URL = ${JSON.stringify(readyUrl)};
  var RETRY_URL = ${JSON.stringify(retryUrl)};
  var PROVIDER_URL = ${JSON.stringify(providerUrl)};
  var HINTS = { openai: 'OpenAI keys start with "sk-".', anthropic: 'Anthropic keys start with "sk-ant-".' };
  var POLL_MS = 2500;
  function el(id){ return document.getElementById(id); }
  function show(id, on){ var e = el(id); if (e) e.classList.toggle('hidden', !on); }
  function updateHint(){ var p = el('provider'); if (p) el('hint').textContent = HINTS[p.value] || ''; }
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
      .then(function(r){ return r.json(); })
      .then(applyState)
      .catch(function(){ setTimeout(poll, POLL_MS); });
  }
  function retry(){
    show('retry', false); show('err', false); showProvisioning();
    fetch(RETRY_URL, { method: 'POST', cache: 'no-store' })
      .then(function(r){ return r.json(); })
      .then(applyState)
      .catch(function(){ setTimeout(poll, POLL_MS); });
  }
  function submitKey(ev){
    ev.preventDefault();
    var provider = el('provider').value;
    var key = el('apikey').value;
    var fe = el('formerr'); fe.classList.add('hidden');
    if (!key || key.trim().length < 8) { fe.textContent = 'Enter a valid API key.'; fe.classList.remove('hidden'); return false; }
    var btn = el('savebtn'); btn.disabled = true; btn.textContent = 'Saving…';
    fetch(PROVIDER_URL, {
      method: 'POST', cache: 'no-store',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify({ provider: provider, key: key })
    })
      .then(function(r){ return r.json().then(function(b){ return { ok: r.ok, body: b }; }); })
      .then(function(res){
        btn.disabled = false; btn.textContent = 'Save & continue';
        if (!res.ok) { fe.textContent = (res.body && res.body.error) || 'Could not save key.'; fe.classList.remove('hidden'); return; }
        applyState(res.body);
      })
      .catch(function(){ btn.disabled = false; btn.textContent = 'Save & continue'; fe.textContent = 'Network error. Try again.'; fe.classList.remove('hidden'); });
    return false;
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
