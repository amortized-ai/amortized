# Deploying the Plugin on OpenShift

This guide walks through deploying the plugin on an OpenShift cluster that already has the Red Hat OpenShift AI (RHOAI) Dashboard running.

## Prerequisites

- **Helm** — to install the plugin chart
- **`oc` CLI** — logged in to the target OpenShift cluster
- **Access to `redhat-ods-applications`** — typically requires cluster-admin, since you need to modify the dashboard's Deployment

> **ODH vs RHOAI:** This guide uses the RHOAI dashboard namespace `redhat-ods-applications` and deployment name `rhods-dashboard`. If you are running the Open Data Hub (ODH) upstream distribution instead, substitute `opendatahub` for the namespace and `odh-dashboard` for the deployment name throughout.

---

## 1. Install the Plugin

Install directly from the OCI registry — no need to clone the repo:

```bash
helm install amortized-studio oci://quay.io/rh-ai-community-plugins/amortized-studio-chart \
  --version 0.4.2 \
  --namespace cp-amortized-studio \
  --create-namespace
```

Or, from a local checkout of the repository:

```bash
helm install amortized-studio chart/ \
  --namespace cp-amortized-studio \
  --create-namespace
```

This creates:

- A **Deployment** and **Service** (`amortized-studio`) serving the plugin's static assets (including `remoteEntry.js`) via Nginx on port 8080
- A **BFF Deployment** and **Service** (`amortized-studio-bff`) running the plugin's backend service on port 3000 (enabled by default)

### Overriding Defaults

Pass `--set` flags to customize the installation:

```bash
helm install amortized-studio oci://quay.io/rh-ai-community-plugins/amortized-studio-chart \
  --version 0.4.2 \
  --namespace cp-amortized-studio \
  --create-namespace \
  --set replicaCount=2
```

To deploy the frontend only (no BFF):

```bash
helm install amortized-studio oci://quay.io/rh-ai-community-plugins/amortized-studio-chart \
  --version 0.4.2 \
  --namespace cp-amortized-studio \
  --create-namespace \
  --set bff.enabled=false
```

See [Helm Chart Reference](#helm-chart-reference) for the full list of configurable values.

---

## 2. Register with the RHOAI Dashboard

The dashboard discovers plugins through the `MODULE_FEDERATION_CONFIG` environment variable on its Deployment. You need to append this plugin's entry to that configuration.

### Frontend Only

If you deployed without the BFF (or want to register the frontend first), use this configuration:

```bash
oc get configmap federation-config \
  -n redhat-ods-applications \
  -o jsonpath='{.data.module-federation-config\.json}' \
| python3 -c "
import json, sys
config = json.load(sys.stdin)
config.append({
  'name': 'amortizedStudio',
  'backend': {
    'remoteEntry': '/remoteEntry.js',
    'authorize': False,
    'tls': False,
    'service': {
      'name': 'amortized-studio',
      'namespace': 'cp-amortized-studio',
      'port': 8080
    }
  }
})
print(json.dumps(config))
" > /tmp/mf-config-extended.json

oc set env deployment/rhods-dashboard \
  -n redhat-ods-applications \
  "MODULE_FEDERATION_CONFIG=$(cat /tmp/mf-config-extended.json)"
```

### Frontend + BFF

If you deployed with the BFF enabled, add a `proxyService` entry so the dashboard proxies API requests to the BFF service:

```bash
oc get configmap federation-config \
  -n redhat-ods-applications \
  -o jsonpath='{.data.module-federation-config\.json}' \
| python3 -c "
import json, sys
config = json.load(sys.stdin)
config.append({
  'name': 'amortizedStudio',
  'backend': {
    'remoteEntry': '/remoteEntry.js',
    'authorize': False,
    'tls': False,
    'service': {
      'name': 'amortized-studio',
      'namespace': 'cp-amortized-studio',
      'port': 8080
    }
  },
  'proxyService': [{
    'path': '/amortized-studio/api',
    'pathRewrite': '/api',
    'authorize': True,
    'tls': False,
    'service': {
      'name': 'amortized-studio-bff',
      'namespace': 'cp-amortized-studio',
      'port': 3000
    }
  }]
})
print(json.dumps(config))
" > /tmp/mf-config-extended.json

oc set env deployment/rhods-dashboard \
  -n redhat-ods-applications \
  "MODULE_FEDERATION_CONFIG=$(cat /tmp/mf-config-extended.json)"
```

The `proxyService` entry tells the dashboard to forward requests from `/amortized-studio/api/*` to the BFF service, rewriting the path to `/api/*` and forwarding the user's Bearer token (`authorize: true`).

### Why `MODULE_FEDERATION_CONFIG` Instead of the ConfigMap?

The RHOAI operator reconciles the `federation-config` ConfigMap, which means direct edits to it may be reverted. Setting the environment variable on the Deployment overrides the ConfigMap value and survives operator reconciliation.

New dashboard pods roll out automatically after the environment variable is set. After roughly two minutes, reload the RHOAI Dashboard in your browser to see the plugin's sidebar entries.

---

## 3. Verify

### Check registration

Confirm the plugin appears in the dashboard's federation config:

```bash
oc set env deployment/rhods-dashboard -n redhat-ods-applications --list \
  | grep MODULE_FEDERATION_CONFIG \
  | python3 -c "
import json, sys
data = json.loads(sys.stdin.read().split('=', 1)[1])
for entry in data:
    name = entry['name']
    has_proxy = bool(entry.get('proxyService'))
    print(f'  {name}' + (' (+ BFF proxy)' if has_proxy else ''))
"
```

### Check pods

Verify the plugin pods are running:

```bash
oc get pods -n cp-amortized-studio
```

You should see pods for `amortized-studio` (and `amortized-studio-bff` if BFF is enabled), all in `Running` status.

### Check the dashboard

Open the RHOAI Dashboard in your browser. You should see the plugin's pages in the sidebar.

---

## Uninstalling

### 1. Remove from the dashboard federation config

Retrieve the current config, remove the `amortizedStudio` entry, and re-apply:

```bash
oc get configmap federation-config \
  -n redhat-ods-applications \
  -o jsonpath='{.data.module-federation-config\.json}' \
| python3 -c "
import json, sys
config = json.load(sys.stdin)
config = [e for e in config if e.get('name') != 'amortizedStudio']
print(json.dumps(config))
" > /tmp/mf-config-reduced.json

oc set env deployment/rhods-dashboard \
  -n redhat-ods-applications \
  "MODULE_FEDERATION_CONFIG=$(cat /tmp/mf-config-reduced.json)"
```

### 2. Uninstall the Helm release

```bash
helm uninstall amortized-studio -n cp-amortized-studio
oc delete namespace cp-amortized-studio   # optional: remove the namespace entirely
```

---

## Helm Chart Reference

Key values in `chart/values.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `namespace` | `cp-amortized-studio` | Target namespace for all namespaced resources |
| `image.repository` | `quay.io/rh-ai-community-plugins/amortized-studio` | Frontend container image |
| `image.tag` | `""` (defaults to appVersion) | Frontend image tag |
| `image.pullPolicy` | `IfNotPresent` | Image pull policy |
| `replicaCount` | `1` | Frontend replicas |
| `service.type` | `ClusterIP` | Frontend Service type |
| `service.port` | `8080` | Frontend Service port |
| `resources.requests.cpu` | `50m` | Frontend CPU request |
| `resources.requests.memory` | `64Mi` | Frontend memory request |
| `resources.limits.cpu` | `100m` | Frontend CPU limit |
| `resources.limits.memory` | `128Mi` | Frontend memory limit |
| `bff.enabled` | `true` | Deploy the BFF service |
| `bff.image.repository` | `quay.io/rh-ai-community-plugins/amortized-studio-bff` | BFF container image |
| `bff.image.tag` | `""` (defaults to appVersion) | BFF image tag |
| `bff.service.port` | `3000` | BFF Service port |
| `bff.resources.requests.cpu` | `100m` | BFF CPU request |
| `bff.resources.requests.memory` | `128Mi` | BFF memory request |
| `bff.resources.limits.cpu` | `200m` | BFF CPU limit |
| `bff.resources.limits.memory` | `256Mi` | BFF memory limit |

For the complete list, see [`chart/values.yaml`](../../chart/values.yaml).
