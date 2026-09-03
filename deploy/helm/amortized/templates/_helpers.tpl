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

{{/* In-cluster service FQDNs, parameterized by the release namespace. */}}
{{- define "amortized.serverFqdn" -}}
amortized-server.{{ .Values.namespace }}.svc.cluster.local
{{- end -}}

{{- define "amortized.mlflowFqdn" -}}
mlflow.{{ .Values.namespace }}.svc.cluster.local
{{- end -}}

{{- define "amortized.minioFqdn" -}}
minio.{{ .Values.namespace }}.svc.cluster.local
{{- end -}}

{{- define "amortized.postgresFqdn" -}}
postgres.{{ .Values.namespace }}.svc.cluster.local
{{- end -}}

{{/*
Wiring helpers. When dataStores.bundled is true we point at the in-cluster
services; otherwise we use the operator-supplied external endpoints.
*/}}
{{- define "amortized.databaseUrl" -}}
{{- if .Values.dataStores.bundled -}}
postgresql://{{ .Values.postgres.user }}:{{ .Values.postgres.password }}@{{ include "amortized.postgresFqdn" . }}:5432/{{ .Values.postgres.database }}
{{- else -}}
{{- required "database.url is required when dataStores.bundled=false" .Values.database.url -}}
{{- end -}}
{{- end -}}

{{- define "amortized.mlflowTrackingUri" -}}
{{- if .Values.dataStores.bundled -}}
http://{{ include "amortized.mlflowFqdn" . }}:5000
{{- else -}}
{{- required "mlflow.trackingUri is required when dataStores.bundled=false" .Values.mlflow.trackingUri -}}
{{- end -}}
{{- end -}}

{{- define "amortized.gatewayUrl" -}}
{{- if .Values.dataStores.bundled -}}
http://{{ include "amortized.mlflowFqdn" . }}:5000/gateway/mlflow/v1
{{- else -}}
{{- .Values.mlflow.gatewayUrl -}}
{{- end -}}
{{- end -}}

{{- define "amortized.s3Endpoint" -}}
{{- if .Values.dataStores.bundled -}}
http://{{ include "amortized.minioFqdn" . }}:9000
{{- else -}}
{{- required "s3.endpoint is required when dataStores.bundled=false" .Values.s3.endpoint -}}
{{- end -}}
{{- end -}}

{{- define "amortized.s3AccessKey" -}}
{{- if .Values.dataStores.bundled -}}
{{- .Values.minio.rootUser -}}
{{- else -}}
{{- required "s3.accessKey is required when dataStores.bundled=false" .Values.s3.accessKey -}}
{{- end -}}
{{- end -}}

{{- define "amortized.s3SecretKey" -}}
{{- if .Values.dataStores.bundled -}}
{{- .Values.minio.rootPassword -}}
{{- else -}}
{{- required "s3.secretKey is required when dataStores.bundled=false" .Values.s3.secretKey -}}
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
