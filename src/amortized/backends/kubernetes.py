"""Kubernetes compute backend — runs jobs on a K8s/OpenShift cluster."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

from amortized.backends import (
    BackendHandle,
    BackendStatus,
    Capability,
    JobSpec,
)

logger = logging.getLogger("amortized.backends.kubernetes")

_LABEL_APP = "amortized"


class KubernetesBackend:
    def __init__(
        self,
        name: str = "kubernetes",
        namespace: str = "amortized-jobs",
        image_registry: str = "ghcr.io/amortized-ai",
        image_pull_policy: str = "Always",
    ) -> None:
        self.name = name
        self._namespace = namespace
        self._image_registry = image_registry
        self._image_pull_policy = image_pull_policy
        self._client: Any | None = None

    def capabilities(self) -> set[Capability]:
        return {Capability.GPU, Capability.LOG_STREAM, Capability.STOP}

    def _labels(self, job_id: str, job_type: str = "", user_id: str = "") -> dict[str, str]:
        labels: dict[str, str] = {"app": _LABEL_APP, "amortized/job-id": job_id}
        if job_type:
            labels["amortized/job-type"] = job_type
        if user_id:
            labels["amortized/user"] = user_id
        return labels

    def _resource_name(self, job_id: str) -> str:
        return f"amortized-{job_id[:53]}"

    async def _get_client(self) -> Any:
        if self._client is None:
            from kubernetes_asyncio import config
            from kubernetes_asyncio.client import ApiClient

            config.load_incluster_config()  # type: ignore[no-untyped-call]
            self._client = ApiClient()
        return self._client

    async def _gcp_secret_exists(self) -> bool:
        """Check if the gcp-credentials secret exists in the jobs namespace."""
        api_client = await self._get_client()
        from kubernetes_asyncio.client import CoreV1Api

        core = CoreV1Api(api_client)
        try:
            await core.read_namespaced_secret("gcp-credentials", self._namespace)
            return True
        except Exception:
            return False

    async def _sync_gcp_secret(self, source_namespace: str) -> None:
        """Copy the GCP credentials secret from the source namespace to the jobs namespace."""
        api_client = await self._get_client()
        from kubernetes_asyncio.client import CoreV1Api, V1ObjectMeta, V1Secret

        core = CoreV1Api(api_client)

        try:
            source = await core.read_namespaced_secret("opencode-gcp", source_namespace)
        except Exception:
            logger.debug("No opencode-gcp secret in namespace %s", source_namespace)
            return

        # Also read the LLM secret for project/location
        project = ""
        location = ""
        try:
            llm_secret = await core.read_namespaced_secret("opencode-llm", source_namespace)
            import base64

            project = base64.b64decode(llm_secret.data.get("google-cloud-project", "")).decode()
            location = base64.b64decode(llm_secret.data.get("vertex-location", "")).decode()
        except Exception:
            logger.debug("No opencode-llm secret in namespace %s", source_namespace)

        data = dict(source.data or {})
        if project:
            import base64 as b64mod

            data["google-cloud-project"] = b64mod.b64encode(project.encode()).decode()
        if location:
            import base64 as b64mod

            data["vertex-location"] = b64mod.b64encode(location.encode()).decode()

        target = V1Secret(
            metadata=V1ObjectMeta(
                name="gcp-credentials",
                namespace=self._namespace,
            ),
            data=data,
        )

        try:
            await core.create_namespaced_secret(self._namespace, target)
            logger.info("Synced GCP credentials to namespace %s", self._namespace)
        except Exception:
            try:
                await core.replace_namespaced_secret("gcp-credentials", self._namespace, target)
            except Exception:
                logger.warning("Failed to sync GCP credentials to %s", self._namespace)

    def _build_config_map(self, spec: JobSpec, resource_name: str) -> Any:
        from kubernetes_asyncio.client import V1ConfigMap, V1ObjectMeta

        return V1ConfigMap(
            metadata=V1ObjectMeta(
                name=f"{resource_name}-config",
                namespace=self._namespace,
                labels=self._labels(spec.job_id, spec.job_type, spec.user_id),
            ),
            data=dict(spec.config_files),
        )

    def _build_pod_spec(self, spec: JobSpec, resource_name: str, *, mount_gcp: bool = False) -> Any:
        from kubernetes_asyncio.client import (
            V1Capabilities,
            V1Container,
            V1EmptyDirVolumeSource,
            V1EnvVar,
            V1EnvVarSource,
            V1HostPathVolumeSource,
            V1PodSecurityContext,
            V1PodSpec,
            V1ResourceRequirements,
            V1SecretKeySelector,
            V1SecurityContext,
            V1Volume,
            V1VolumeMount,
        )

        work_host_path = f"/var/local-path-provisioner/job-work/{spec.job_id}"
        volumes = [
            V1Volume(
                name="config",
                config_map={"name": f"{resource_name}-config"},  # type: ignore[arg-type]
            ),
            V1Volume(
                name="work",
                host_path=V1HostPathVolumeSource(
                    path=work_host_path, type="DirectoryOrCreate",
                ),
            ),
            V1Volume(
                name="shm",
                empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="12Gi"),
            ),
        ]

        volume_mounts = [
            V1VolumeMount(name="config", mount_path="/amortized", read_only=True),
            V1VolumeMount(name="work", mount_path="/amortized/work"),
            V1VolumeMount(name="shm", mount_path="/dev/shm"),
        ]

        env_vars = [
            V1EnvVar(name="AMORTIZED_JOB_ID", value=spec.job_id),
            V1EnvVar(name="AMORTIZED_WORK_DIR", value="/amortized/work"),
            V1EnvVar(name="AMORTIZED_CONFIG_PATH", value="/amortized/config.json"),
            V1EnvVar(name="HOME", value="/amortized/work"),
            V1EnvVar(name="HF_HOME", value="/amortized/work/.cache"),
            V1EnvVar(name="TRANSFORMERS_CACHE", value="/amortized/work/.cache"),
            V1EnvVar(name="TORCHINDUCTOR_CACHE_DIR", value="/amortized/work/.cache/torchinductor"),
            V1EnvVar(name="USER", value="amortized"),
        ]

        if mount_gcp:
            from kubernetes_asyncio.client import V1SecretVolumeSource

            volumes.append(
                V1Volume(
                    name="gcp-credentials",
                    secret=V1SecretVolumeSource(secret_name="gcp-credentials"),
                ),
            )
            volume_mounts.append(
                V1VolumeMount(
                    name="gcp-credentials",
                    mount_path="/gcp",
                    read_only=True,
                ),
            )
            env_vars.extend(
                [
                    V1EnvVar(name="GOOGLE_APPLICATION_CREDENTIALS", value="/gcp/credentials.json"),
                    V1EnvVar(
                        name="GOOGLE_CLOUD_PROJECT",
                        value_from=V1EnvVarSource(
                            secret_key_ref=V1SecretKeySelector(
                                name="gcp-credentials",
                                key="google-cloud-project",
                                optional=True,
                            )
                        ),
                    ),
                    V1EnvVar(
                        name="VERTEX_LOCATION",
                        value_from=V1EnvVarSource(
                            secret_key_ref=V1SecretKeySelector(
                                name="gcp-credentials",
                                key="vertex-location",
                                optional=True,
                            )
                        ),
                    ),
                ]
            )

        secret_name = f"{resource_name}-env"
        for k in spec.env:
            env_vars.append(
                V1EnvVar(
                    name=k,
                    value_from=V1EnvVarSource(
                        secret_key_ref=V1SecretKeySelector(
                            name=secret_name,
                            key=k,
                        )
                    ),
                )
            )

        resources: dict[str, Any] = {}
        if spec.resources.gpus > 0:
            resources["limits"] = {"nvidia.com/gpu": str(spec.resources.gpus)}
            resources["requests"] = {"nvidia.com/gpu": str(spec.resources.gpus)}
        if spec.resources.memory_gb:
            mem = f"{spec.resources.memory_gb}Gi"
            resources.setdefault("requests", {})["memory"] = mem
            resources.setdefault("limits", {})["memory"] = mem
        if spec.resources.cpus:
            resources.setdefault("requests", {})["cpu"] = str(spec.resources.cpus)

        from kubernetes_asyncio.client import V1EnvFromSource, V1SecretEnvSource

        container_security_context = V1SecurityContext(
            allow_privilege_escalation=False,
            capabilities=V1Capabilities(drop=["ALL"]),
        )

        container = V1Container(
            name="job",
            image=spec.image or f"{self._image_registry}/worker:latest",
            image_pull_policy=self._image_pull_policy,
            command=spec.command or None,
            env=env_vars,
            env_from=[V1EnvFromSource(secret_ref=V1SecretEnvSource(name="amortized-s3"))],
            volume_mounts=volume_mounts,
            resources=V1ResourceRequirements(**resources) if resources else None,
            working_dir="/amortized/work",
            security_context=container_security_context,
        )

        node_selector = None
        runtime_class_name = None
        if spec.resources.gpus > 0:
            node_selector = {"nvidia.com/gpu.present": "true"}
            runtime_class_name = "nvidia"

        init_security_context = V1SecurityContext(
            allow_privilege_escalation=False,
            run_as_non_root=False,
        )

        init_containers: list[Any] = []
        for download in spec.s3_downloads:
            s3_cmd = "aws s3 sync" if download.is_directory else "aws s3 cp"
            init_containers.append(
                V1Container(
                    name=f"s3-download-{len(init_containers)}",
                    image="docker.io/amazon/aws-cli:latest",
                    command=[
                        "sh",
                        "-c",
                        f"mkdir -p $(dirname {download.local_path}) && cd / && "
                        f"{s3_cmd} {download.s3_uri} {download.local_path} "
                        f"--endpoint-url $AWS_S3_ENDPOINT && "
                        f"ls -la {download.local_path}",
                    ],
                    env_from=[V1EnvFromSource(secret_ref=V1SecretEnvSource(name="amortized-s3"))],
                    volume_mounts=[V1VolumeMount(name="work", mount_path="/amortized/work")],
                    security_context=init_security_context,
                    resources=V1ResourceRequirements(
                        requests={"cpu": "100m", "memory": "128Mi"},
                        limits={"cpu": "500m", "memory": "512Mi"},
                    ),
                )
            )

        return V1PodSpec(
            init_containers=init_containers or None,
            containers=[container],
            volumes=volumes,
            restart_policy="Never",
            node_selector=node_selector,
            runtime_class_name=runtime_class_name,
            security_context=V1PodSecurityContext(run_as_non_root=False),
        )

    async def _create_secret(self, spec: JobSpec, resource_name: str, api_client: Any) -> None:
        from kubernetes_asyncio.client import CoreV1Api, V1ObjectMeta, V1Secret

        if not spec.env:
            return

        secret = V1Secret(
            metadata=V1ObjectMeta(
                name=f"{resource_name}-env",
                namespace=self._namespace,
                labels=self._labels(spec.job_id, spec.job_type, spec.user_id),
            ),
            string_data=dict(spec.env),
        )

        core = CoreV1Api(api_client)
        await core.create_namespaced_secret(self._namespace, secret)

    async def _submit_job(self, spec: JobSpec) -> BackendHandle:
        from kubernetes_asyncio.client import (
            BatchV1Api,
            CoreV1Api,
            V1Job,
            V1JobSpec,
            V1ObjectMeta,
        )

        resource_name = self._resource_name(spec.job_id)
        api_client = await self._get_client()

        core = CoreV1Api(api_client)
        batch = BatchV1Api(api_client)

        await self._create_secret(spec, resource_name, api_client)

        config_map = self._build_config_map(spec, resource_name)
        await core.create_namespaced_config_map(self._namespace, config_map)

        mount_gcp = await self._gcp_secret_exists()
        pod_spec = self._build_pod_spec(spec, resource_name, mount_gcp=mount_gcp)
        pod_spec.restart_policy = "Never"

        job = V1Job(
            metadata=V1ObjectMeta(
                name=resource_name,
                namespace=self._namespace,
                labels=self._labels(spec.job_id, spec.job_type, spec.user_id),
            ),
            spec=V1JobSpec(
                template={  # type: ignore[arg-type]
                    "metadata": {"labels": self._labels(spec.job_id, spec.job_type, spec.user_id)},
                    "spec": pod_spec,
                },
                backoff_limit=0,
                ttl_seconds_after_finished=3600,
            ),
        )

        created_job = await batch.create_namespaced_job(self._namespace, job)

        job_uid = created_job.metadata.uid
        try:
            await core.patch_namespaced_config_map(
                f"{resource_name}-config",
                self._namespace,
                {
                    "metadata": {
                        "ownerReferences": [
                            {
                                "apiVersion": "batch/v1",
                                "kind": "Job",
                                "name": resource_name,
                                "uid": job_uid,
                            }
                        ]
                    }
                },
            )
            await core.patch_namespaced_secret(
                f"{resource_name}-env",
                self._namespace,
                {
                    "metadata": {
                        "ownerReferences": [
                            {
                                "apiVersion": "batch/v1",
                                "kind": "Job",
                                "name": resource_name,
                                "uid": job_uid,
                            }
                        ]
                    }
                },
            )
        except Exception:
            logger.warning(
                "Failed to set ownerReferences on ConfigMap/Secret — manual cleanup may be needed"
            )

        logger.info("Created K8s Job %s for job %s", resource_name, spec.job_id)

        return BackendHandle(
            backend_name=self.name,
            job_id=spec.job_id,
            scheduler_id=resource_name,
            container_id="job",
        )

    async def submit(self, spec: JobSpec) -> BackendHandle:
        return await self._submit_job(spec)

    async def status(self, handle: BackendHandle) -> BackendStatus:
        resource_name = handle.scheduler_id
        if not resource_name:
            return BackendStatus(
                running=False,
                error="Cannot check job status — no Kubernetes resource name was recorded.",
            )

        api_client = await self._get_client()
        from kubernetes_asyncio.client import BatchV1Api

        batch = BatchV1Api(api_client)
        try:
            job = await batch.read_namespaced_job(resource_name, self._namespace)
        except Exception as e:
            err_str = str(e)
            if "404" in err_str or "NotFound" in err_str:
                msg = (
                    f"Job '{resource_name}' no longer exists."
                    " It may have been cancelled or cleaned up."
                )
            else:
                msg = "Unable to check job status. Please try again later."
            logger.warning(
                "K8s API error for job %s in %s: %s",
                resource_name,
                self._namespace,
                e,
            )
            return BackendStatus(running=False, error=msg)

        status = job.status
        if status.succeeded and status.succeeded > 0:
            return BackendStatus(running=False, exit_code=0)
        if status.failed and status.failed > 0:
            reason = await self._get_pod_failure_reason(
                resource_name, api_client
            )
            return BackendStatus(
                running=False,
                exit_code=1,
                error=reason or "Job failed. Check logs for details.",
            )
        return BackendStatus(running=True)

    async def _get_pod_failure_reason(
        self, job_name: str, api_client: Any
    ) -> str | None:
        from kubernetes_asyncio.client import CoreV1Api

        try:
            core = CoreV1Api(api_client)
            pods = await core.list_namespaced_pod(
                self._namespace, label_selector=f"job-name={job_name}"
            )
            for pod in pods.items or []:
                for cs in pod.status.container_statuses or []:
                    if cs.state and cs.state.waiting:
                        reason = cs.state.waiting.reason or ""
                        if "ImagePull" in reason or "ErrImagePull" in reason:
                            image = cs.image or "unknown"
                            return (
                                f"Failed to pull container image '{image}'."
                                " Check that the image exists and is"
                                " accessible."
                            )
                    if cs.state and cs.state.terminated:
                        exit_code = cs.state.terminated.exit_code
                        reason = cs.state.terminated.reason or ""
                        if exit_code != 0:
                            return (
                                f"Job exited with code {exit_code}"
                                f"{f': {reason}' if reason else ''}."
                                " Check logs for details."
                            )
        except Exception:
            logger.debug("Could not inspect pod for %s", job_name, exc_info=True)
        return None

    async def cancel(self, handle: BackendHandle) -> None:
        resource_name = handle.scheduler_id
        if not resource_name:
            return

        api_client = await self._get_client()
        from kubernetes_asyncio.client import BatchV1Api, CoreV1Api

        batch = BatchV1Api(api_client)
        try:
            await batch.delete_namespaced_job(
                resource_name,
                self._namespace,
                propagation_policy="Foreground",
            )
            logger.info("Deleted K8s Job %s", resource_name)
        except Exception:
            logger.exception("Failed to delete Job %s", resource_name)

        core = CoreV1Api(api_client)
        with contextlib.suppress(Exception):
            await core.delete_namespaced_secret(f"{resource_name}-env", self._namespace)

    async def logs(self, handle: BackendHandle) -> AsyncIterator[str]:
        resource_name = handle.scheduler_id
        if not resource_name:
            return

        api_client = await self._get_client()
        from kubernetes_asyncio.client import CoreV1Api

        core = CoreV1Api(api_client)
        pods = await core.list_namespaced_pod(
            self._namespace,
            label_selector=f"amortized/job-id={handle.job_id}",
        )
        if not pods.items:
            return

        pod_name = pods.items[0].metadata.name

        try:
            log_text = await core.read_namespaced_pod_log(
                name=pod_name,
                namespace=self._namespace,
                tail_lines=2000,
            )
            if log_text:
                for line in log_text.split("\n"):
                    yield line
        except Exception:
            pass
