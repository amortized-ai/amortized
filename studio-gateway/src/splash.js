// Provisioning splash shown for the initial navigation while a user's backend
// stack is being created (~60-90s on first visit). Polls /gateway/ready and
// reloads into the studio SPA once the backend is healthy.

function renderSplash(state, basePath = '') {
  const isError = state && state.state === 'error';
  // Poll/retry endpoints must include the embed prefix so requests from inside
  // the dashboard iframe route back through the dashboard proxy to the gateway.
  const readyUrl = `${basePath}/gateway/ready`;
  const retryUrl = `${basePath}/gateway/retry`;
  const detail = isError
    ? escapeHtml(state.error || 'Provisioning failed.')
    : 'Setting up your isolated workspace (server, database, and compute namespace). This usually takes about a minute on first launch.';

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
    background: #fff; border-radius: 12px; padding: 40px 44px; max-width: 440px; text-align: center;
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
  .hidden { display: none; }
</style>
</head>
<body>
  <div class="card">
    <div class="spinner" id="spinner"></div>
    <h1 id="title">Preparing your Amortized Studio…</h1>
    <p id="detail">${detail}</p>
    <div class="err hidden" id="err"></div>
    <button class="hidden" id="retry" onclick="retry()">Retry</button>
  </div>
<script>
  var READY_URL = ${JSON.stringify(readyUrl)};
  var RETRY_URL = ${JSON.stringify(retryUrl)};
  var POLL_MS = 2500;
  function show(el, on){ document.getElementById(el).classList.toggle('hidden', !on); }
  function toError(msg){
    document.getElementById('spinner').classList.add('hidden');
    document.getElementById('title').textContent = 'Could not prepare your workspace';
    document.getElementById('detail').textContent = 'Something went wrong provisioning your environment.';
    var e = document.getElementById('err'); e.textContent = msg || ''; show('err', !!msg);
    show('retry', true);
  }
  function poll(){
    fetch(READY_URL, { headers: { 'Accept': 'application/json' }, cache: 'no-store' })
      .then(function(r){ return r.json(); })
      .then(function(s){
        if (s.state === 'ready') { window.location.reload(); return; }
        if (s.state === 'error') { toError(s.error); return; }
        setTimeout(poll, POLL_MS);
      })
      .catch(function(){ setTimeout(poll, POLL_MS); });
  }
  function retry(){
    show('retry', false); show('err', false);
    document.getElementById('spinner').classList.remove('hidden');
    document.getElementById('title').textContent = 'Preparing your Amortized Studio…';
    document.getElementById('detail').textContent = 'Retrying…';
    fetch(RETRY_URL, { method: 'POST', cache: 'no-store' }).then(function(){ setTimeout(poll, POLL_MS); });
  }
  ${isError ? 'toError(' + JSON.stringify(state.error || '') + ');' : 'setTimeout(poll, POLL_MS);'}
</script>
</body>
</html>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

module.exports = { renderSplash };
