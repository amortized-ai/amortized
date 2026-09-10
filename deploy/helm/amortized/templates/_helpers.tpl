{{/*
Common labels applied to every resource (mirrors the base kustomize manifests).
The base uses a flat `app: amortized` label plus a per-component `component:` label;
we keep that exactly so selectors stay compatible.
*/}}
{{- define "amortized.labels" -}}
app: amortized
{{- end -}}

{{- define "amortized.namespace" -}}
{{- .Values.namespace -}}
{{- end -}}

{{- define "amortized.jobsNamespace" -}}
{{- .Values.jobsNamespace -}}
{{- end -}}

{{/* In-cluster service names (<svc>.<ns>.svc), parameterized by the release
namespace. The short `.svc` form resolves via the pod's DNS search domains
regardless of the cluster domain, so it does not hard-code `cluster.local`. */}}
{{- define "amortized.serverFqdn" -}}
amortized-server.{{ .Values.namespace }}.svc
{{- end -}}

{{- define "amortized.mlflowFqdn" -}}
mlflow.{{ .Values.namespace }}.svc
{{- end -}}

{{- define "amortized.minioFqdn" -}}
minio.{{ .Values.namespace }}.svc
{{- end -}}

{{- define "amortized.postgresFqdn" -}}
postgres.{{ .Values.namespace }}.svc
{{- end -}}

{{/*
Per-store "bundled" resolution. Each store defaults to dataStores.bundled but can be
overridden individually (postgres.bundled / minio.bundled / mlflow.bundled) so you can,
e.g., bundle PostgreSQL while pointing MLflow at an external / enterprise server.
Enterprise MLflow always implies external MLflow. Helpers return the string "true"/"false"
(use with `eq ... "true"`); hasKey is used so an explicit `false` override is honored
(Helm's `default` would treat false as unset).
*/}}
{{- define "amortized.postgresBundled" -}}
{{- if hasKey .Values.postgres "bundled" -}}{{ .Values.postgres.bundled }}{{- else -}}{{ .Values.dataStores.bundled }}{{- end -}}
{{- end -}}

{{- define "amortized.minioBundled" -}}
{{- if hasKey .Values.minio "bundled" -}}{{ .Values.minio.bundled }}{{- else -}}{{ .Values.dataStores.bundled }}{{- end -}}
{{- end -}}

{{- define "amortized.mlflowBundled" -}}
{{- if .Values.mlflow.enterprise.enabled -}}false{{- else if hasKey .Values.mlflow "bundled" -}}{{ .Values.mlflow.bundled }}{{- else -}}{{ .Values.dataStores.bundled }}{{- end -}}
{{- end -}}

{{/* MLflow workspace (X-MLFLOW-WORKSPACE) for the enterprise RHOAI MLflow == the namespace. */}}
{{- define "amortized.mlflowWorkspace" -}}
{{- .Values.mlflow.enterprise.workspace | default .Values.namespace -}}
{{- end -}}

{{/*
Wiring helpers. Each store points at its in-cluster service when bundled, else the
external / operator-supplied endpoint.
*/}}
{{- define "amortized.databaseUrl" -}}
{{- if eq (include "amortized.postgresBundled" .) "true" -}}
postgresql://{{ .Values.postgres.user }}:{{ .Values.postgres.password }}@{{ include "amortized.postgresFqdn" . }}:5432/{{ .Values.postgres.database }}
{{- else -}}
{{- required "database.url is required when PostgreSQL is not bundled" .Values.database.url -}}
{{- end -}}
{{- end -}}

{{- define "amortized.mlflowTrackingUri" -}}
{{- if eq (include "amortized.mlflowBundled" .) "true" -}}
http://{{ include "amortized.mlflowFqdn" . }}:5000
{{- else -}}
{{- required "mlflow.trackingUri is required when MLflow is not bundled (external/enterprise)" .Values.mlflow.trackingUri -}}
{{- end -}}
{{- end -}}

{{- define "amortized.gatewayUrl" -}}
{{- if eq (include "amortized.mlflowBundled" .) "true" -}}
http://{{ include "amortized.mlflowFqdn" . }}:5000/gateway/mlflow/v1
{{- else -}}
{{- .Values.mlflow.gatewayUrl -}}
{{- end -}}
{{- end -}}

{{- define "amortized.s3Endpoint" -}}
{{- if eq (include "amortized.minioBundled" .) "true" -}}
http://{{ include "amortized.minioFqdn" . }}:9000
{{- else -}}
{{- required "s3.endpoint is required when MinIO is not bundled" .Values.s3.endpoint -}}
{{- end -}}
{{- end -}}

{{- define "amortized.s3AccessKey" -}}
{{- if eq (include "amortized.minioBundled" .) "true" -}}
{{- .Values.minio.rootUser -}}
{{- else -}}
{{- required "s3.accessKey is required when MinIO is not bundled" .Values.s3.accessKey -}}
{{- end -}}
{{- end -}}

{{- define "amortized.s3SecretKey" -}}
{{- if eq (include "amortized.minioBundled" .) "true" -}}
{{- .Values.minio.rootPassword -}}
{{- else -}}
{{- required "s3.secretKey is required when MinIO is not bundled" .Values.s3.secretKey -}}
{{- end -}}
{{- end -}}

{{/*
Secret holding SDG teacher-model provider keys, loaded into the server env.
Empty output when no teacher keys are configured (falsy in `if`/`with`).
*/}}
{{- define "amortized.teacherKeysSecret" -}}
{{- if .Values.teacherKeys.existingSecret -}}
{{- .Values.teacherKeys.existingSecret -}}
{{- else if .Values.teacherKeys.values -}}
{{- .Values.teacherKeys.secretName -}}
{{- end -}}
{{- end -}}

{{/* Render an image reference from an images.<component> block. */}}
{{- define "amortized.image" -}}
{{- printf "%s:%s" .repository (.tag | toString) -}}
{{- end -}}

{{/* storageClassName line for PVCs / volumeClaimTemplates (omitted when empty). */}}
{{- define "amortized.storageClass" -}}
{{- if .Values.global.storageClass }}
storageClassName: {{ .Values.global.storageClass }}
{{- end }}
{{- end -}}
