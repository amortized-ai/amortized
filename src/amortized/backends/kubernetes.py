"""Kubernetes compute backend — runs jobs on a K8s/OpenShift cluster."""

from __future__ import annotations

import contextlib
import json
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

EVENTS_URL = "http://amortized-server.amortized.svc.cluster.local:8000/api/v1/events/ingest"
_LABEL_APP = "amortized"


class KubernetesBackend:
    def __init__(
        self,
        name: str = "kubernetes",
        namespace: str = "amortized-jobs",
        image_registry: str = "ghcr.io/amortized-ai",
    ) -> None:
        self.name = name
        self._namespace = namespace
        self._image_registry = image_registry
        self._client: Any | None = None

    def capabilities(self) -> set[Capability]:
        return {Capability.GPU, Capability.LOG_STREAM, Capability.STOP}

    def _labels(self, job_id: str) -> dict[str, str]:
        return {"app": _LABEL_APP, "amortized/job-id": job_id}

    def _resource_name(self, job_id: str) -> str:
        return f"amortized-{job_id[:53]}"

    async def _get_client(self) -> Any:
        if self._client is None:
            from kubernetes_asyncio import config
            from kubernetes_asyncio.client import ApiClient

            config.load_incluster_config()
            self._client = ApiClient()
        return self._client

    def _build_config_map(self, spec: JobSpec, resource_name: str) -> dict[str, Any]:
        from kubernetes_asyncio.client import V1ConfigMap, V1ObjectMeta

        data: dict[str, str] = {}

        run_config = spec.env.get("_run_config")
        if run_config:
            data["config.yaml"] = run_config

        synth_config = spec.env.get("_synth_config")
        if synth_config:
            data["synth_config.yaml"] = synth_config

        run_script = spec.env.get("_run_script")
        if run_script:
            data["run.py"] = run_script

        raw_config: object = spec.env.get("_config", {})
        if isinstance(raw_config, str):
            raw_config = json.loads(raw_config)
        data["config.json"] = json.dumps({"config": raw_config, "artifacts": {}})

        return V1ConfigMap(
            metadata=V1ObjectMeta(
                name=f"{resource_name}-config",
                namespace=self._namespace,
                labels=self._labels(spec.job_id),
            ),
            data=data,
        )

    def _build_pod_spec(self, spec: JobSpec, resource_name: str) -> Any:
        from kubernetes_asyncio.client import (
            V1Capabilities,
            V1Container,
            V1EmptyDirVolumeSource,
            V1EnvVar,
            V1EnvVarSource,
            V1PodSecurityContext,
            V1PodSpec,
            V1ResourceRequirements,
            V1SecretKeySelector,
            V1SecurityContext,
            V1Volume,
            V1VolumeMount,
        )

        volumes = [
            V1Volume(
                name="config",
                config_map={"name": f"{resource_name}-config"},
            ),
            V1Volume(
                name="work",
                empty_dir=V1EmptyDirVolumeSource(),
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
            V1EnvVar(name="AMORTIZED_EVENTS_URL", value=EVENTS_URL),
            V1EnvVar(name="HOME", value="/amortized/work"),
            V1EnvVar(name="HF_HOME", value="/amortized/work/.cache"),
            V1EnvVar(name="TRANSFORMERS_CACHE", value="/amortized/work/.cache"),
            V1EnvVar(name="TORCHINDUCTOR_CACHE_DIR", value="/amortized/work/.cache/torchinductor"),
            V1EnvVar(name="USER", value="amortized"),
        ]

        secret_name = f"{resource_name}-env"
        filtered_env = {
            k: v
            for k, v in spec.env.items()
            if k not in ("_config", "_run_script", "_run_config", "_synth_config", "_s3_data_path")
        }
        for k in filtered_env:
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
            run_as_non_root=True,
            capabilities=V1Capabilities(drop=["ALL"]),
        )

        is_serve = bool(spec.ports)
        container = V1Container(
            name="job",
            image=spec.image or f"{self._image_registry}/worker:latest",
            image_pull_policy="Always",
            command=None if is_serve else (spec.command or None),
            args=spec.command if is_serve else None,
            env=env_vars,
            env_from=[V1EnvFromSource(secret_ref=V1SecretEnvSource(name="amortized-s3"))],
            volume_mounts=volume_mounts,
            resources=V1ResourceRequirements(**resources) if resources else None,
            working_dir="/amortized/work",
            security_context=container_security_context,
        )

        node_selector = None
        if spec.resources.gpus > 0:
            node_selector = {"nvidia.com/gpu.present": "true"}

        init_containers = []
        s3_data_path = spec.env.get("_s3_data_path", "")
        if s3_data_path:
            local_name = s3_data_path.split("/")[-1]
            init_containers.append(
                V1Container(
                    name="s3-download",
                    image="docker.io/amazon/aws-cli:latest",
                    command=[
                        "sh",
                        "-c",
                        f"mkdir -p /amortized/work && "
                        f"cd / && "
                        f"aws s3 cp {s3_data_path} /amortized/work/{local_name} "
                        f"--endpoint-url $AWS_S3_ENDPOINT && "
                        f"ls -la /amortized/work/{local_name}",
                    ],
                    env_from=[V1EnvFromSource(secret_ref=V1SecretEnvSource(name="amortized-s3"))],
                    volume_mounts=[V1VolumeMount(name="work", mount_path="/amortized/work")],
                    security_context=container_security_context,
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
            security_context=V1PodSecurityContext(run_as_non_root=True),
        )

    async def _create_secret(self, spec: JobSpec, resource_name: str, api_client: Any) -> None:
        from kubernetes_asyncio.client import CoreV1Api, V1ObjectMeta, V1Secret

        filtered_env = {
            k: v
            for k, v in spec.env.items()
            if k not in ("_config", "_run_script", "_run_config", "_synth_config", "_s3_data_path")
        }
        if not filtered_env:
            return

        secret = V1Secret(
            metadata=V1ObjectMeta(
                name=f"{resource_name}-env",
                namespace=self._namespace,
                labels=self._labels(spec.job_id),
            ),
            string_data=filtered_env,
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

        pod_spec = self._build_pod_spec(spec, resource_name)
        pod_spec.restart_policy = "Never"

        job = V1Job(
            metadata=V1ObjectMeta(
                name=resource_name,
                namespace=self._namespace,
                labels=self._labels(spec.job_id),
            ),
            spec=V1JobSpec(
                template={"metadata": {"labels": self._labels(spec.job_id)}, "spec": pod_spec},
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
        except Exception:
            logger.warning(
                "Failed to set ownerReference on ConfigMap — manual cleanup may be needed"
            )

        logger.info("Created K8s Job %s for job %s", resource_name, spec.job_id)

        return BackendHandle(
            backend_name=self.name,
            job_id=spec.job_id,
            scheduler_id=resource_name,
            container_id="job",
        )

    async def _submit_deployment(self, spec: JobSpec) -> BackendHandle:
        from kubernetes_asyncio.client import (
            AppsV1Api,
            CoreV1Api,
            V1Deployment,
            V1DeploymentSpec,
            V1LabelSelector,
            V1ObjectMeta,
            V1PodTemplateSpec,
            V1Service,
            V1ServicePort,
            V1ServiceSpec,
        )

        resource_name = self._resource_name(spec.job_id)
        api_client = await self._get_client()

        core = CoreV1Api(api_client)
        apps = AppsV1Api(api_client)

        await self._create_secret(spec, resource_name, api_client)

        config_map = self._build_config_map(spec, resource_name)
        await core.create_namespaced_config_map(self._namespace, config_map)

        pod_spec = self._build_pod_spec(spec, resource_name)
        pod_spec.restart_policy = "Always"

        deployment = V1Deployment(
            metadata=V1ObjectMeta(
                name=resource_name,
                namespace=self._namespace,
                labels=self._labels(spec.job_id),
            ),
            spec=V1DeploymentSpec(
                replicas=1,
                selector=V1LabelSelector(match_labels=self._labels(spec.job_id)),
                template=V1PodTemplateSpec(
                    metadata=V1ObjectMeta(labels=self._labels(spec.job_id)),
                    spec=pod_spec,
                ),
            ),
        )

        await apps.create_namespaced_deployment(self._namespace, deployment)

        if spec.ports:
            service_ports = [
                V1ServicePort(
                    name=f"port-{container_port}",
                    port=container_port,
                    target_port=container_port,
                )
                for container_port in spec.ports.values()
            ]
            service = V1Service(
                metadata=V1ObjectMeta(
                    name=resource_name,
                    namespace=self._namespace,
                    labels=self._labels(spec.job_id),
                ),
                spec=V1ServiceSpec(
                    selector=self._labels(spec.job_id),
                    ports=service_ports,
                ),
            )
            await core.create_namespaced_service(self._namespace, service)

        logger.info("Created K8s Deployment %s for job %s", resource_name, spec.job_id)

        return BackendHandle(
            backend_name=self.name,
            job_id=spec.job_id,
            scheduler_id=resource_name,
            container_id="deployment",
        )

    async def submit(self, spec: JobSpec) -> BackendHandle:
        is_serve = bool(spec.ports)
        if is_serve:
            return await self._submit_deployment(spec)
        return await self._submit_job(spec)

    async def status(self, handle: BackendHandle) -> BackendStatus:
        kind = handle.container_id
        resource_name = handle.scheduler_id
        if not resource_name:
            return BackendStatus(running=False, error="No scheduler_id")

        api_client = await self._get_client()
        if kind == "deployment":
            return await self._deployment_status(resource_name, api_client)
        return await self._job_status(resource_name, api_client)

    async def _job_status(self, resource_name: str, api_client: Any) -> BackendStatus:
        from kubernetes_asyncio.client import BatchV1Api

        batch = BatchV1Api(api_client)
        try:
            job = await batch.read_namespaced_job(resource_name, self._namespace)
        except Exception as e:
            return BackendStatus(running=False, error=str(e))

        status = job.status
        if status.succeeded and status.succeeded > 0:
            return BackendStatus(running=False, exit_code=0)
        if status.failed and status.failed > 0:
            return BackendStatus(running=False, exit_code=1, error="Job failed")
        return BackendStatus(running=True)

    async def _deployment_status(self, resource_name: str, api_client: Any) -> BackendStatus:
        from kubernetes_asyncio.client import AppsV1Api, CoreV1Api

        apps = AppsV1Api(api_client)
        try:
            deployment = await apps.read_namespaced_deployment(resource_name, self._namespace)
        except Exception as e:
            return BackendStatus(running=False, error=str(e))

        conditions = deployment.status.conditions or []
        for cond in conditions:
            if cond.type == "Available" and cond.status == "True":
                return BackendStatus(running=True)

        core = CoreV1Api(api_client)
        try:
            job_id = deployment.metadata.labels.get("amortized/job-id", "")
            pods = await core.list_namespaced_pod(
                self._namespace,
                label_selector=f"amortized/job-id={job_id}",
            )
            for pod in pods.items:
                for cs in pod.status.container_statuses or []:
                    if (
                        cs.state
                        and cs.state.waiting
                        and cs.state.waiting.reason == "CrashLoopBackOff"
                    ):
                        return BackendStatus(
                            running=False,
                            exit_code=1,
                            error="CrashLoopBackOff",
                        )
        except Exception:
            pass

        # No conditions yet — deployment is still spinning up
        return BackendStatus(running=True, error="provisioning")

    async def cancel(self, handle: BackendHandle) -> None:
        kind = handle.container_id
        resource_name = handle.scheduler_id
        if not resource_name:
            return

        api_client = await self._get_client()
        if kind == "deployment":
            await self._cancel_deployment(resource_name, api_client)
        else:
            await self._cancel_job(resource_name, api_client)

        from kubernetes_asyncio.client import CoreV1Api

        core = CoreV1Api(api_client)
        with contextlib.suppress(Exception):
            await core.delete_namespaced_secret(f"{resource_name}-env", self._namespace)

    async def _cancel_job(self, resource_name: str, api_client: Any) -> None:
        from kubernetes_asyncio.client import BatchV1Api

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

    async def _cancel_deployment(self, resource_name: str, api_client: Any) -> None:
        from kubernetes_asyncio.client import AppsV1Api, CoreV1Api

        apps = AppsV1Api(api_client)
        core = CoreV1Api(api_client)
        try:
            await apps.delete_namespaced_deployment(resource_name, self._namespace)
            logger.info("Deleted K8s Deployment %s", resource_name)
        except Exception:
            logger.exception("Failed to delete Deployment %s", resource_name)
        try:
            await core.delete_namespaced_service(resource_name, self._namespace)
            logger.info("Deleted K8s Service %s", resource_name)
        except Exception:
            pass

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

        from kubernetes_asyncio import watch

        w = watch.Watch()
        try:
            async for line in w.stream(
                core.read_namespaced_pod_log,
                name=pod_name,
                namespace=self._namespace,
                follow=True,
            ):
                yield line
        except Exception:
            pass
        finally:
            w.stop()
