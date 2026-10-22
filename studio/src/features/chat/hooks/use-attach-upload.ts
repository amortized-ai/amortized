import { useCallback, useEffect, useRef, useState } from "react"
import { toast } from "sonner"
import { useQueryClient } from "@tanstack/react-query"
import { getJob, uploadDataset } from "@/lib/api-client"
import type { Job } from "@/types/api"

const POLL_INTERVAL_MS = 2000
const POLL_TIMEOUT_MS = 10 * 60 * 1000

type AttachKind = "dataset" | "document"

async function pollJobUntilTerminal(jobId: string): Promise<Job> {
  const deadline = Date.now() + POLL_TIMEOUT_MS
  for (;;) {
    const job = await getJob(jobId)
    if (job.status === "succeeded" || job.status === "failed" || job.status === "cancelled") {
      return job
    }
    if (Date.now() >= deadline) {
      throw new Error(`Timed out waiting for ${jobId} to finish`)
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS))
  }
}

function referenceText(kind: AttachKind, filename: string, runId: string): string {
  // The handle Morty's tools key on: get_dataset takes run_id, get_document_chunks
  // takes document_id (which is the MLflow run id). Name it explicitly so the agent
  // can resolve the upload without guessing.
  const idField = kind === "dataset" ? "run_id" : "document_id"
  return `Attached ${kind} "${filename}" (${idField}: ${runId}). Use this ${kind} for the next step.`
}

/**
 * Drives in-chat attachment: run (or track) the async upload job, wait for it to
 * finish, then post a reference to the resolved artifact so Morty can use it via its
 * existing dataset/document tools. Both upload endpoints are 202-async, so the
 * artifact id (the completed job's mlflow_run_id) is only known after the job lands.
 *
 * References go through a one-at-a-time queue, dispatched only while the chat is idle
 * (`canSend`). That both defers a reference that resolves while the chat is busy
 * (streaming / locked / job in flight) AND serializes references that resolve
 * together — handing them straight to `sendMessage` would let the send lock silently
 * drop all but the first. Attaches themselves track concurrently, so none is dropped.
 */
export function useAttachUpload(onReference: (text: string) => void, canSend = true) {
  const queryClient = useQueryClient()
  const [activeCount, setActiveCount] = useState(0)
  const pendingRefs = useRef<string[]>([])
  const [flushTick, setFlushTick] = useState(0)
  const dispatchingRef = useRef(false)

  // Read onReference at dispatch time (which can be minutes after a track starts)
  // without re-creating callbacks; synced after render, never during it.
  const onReferenceRef = useRef(onReference)
  useEffect(() => {
    onReferenceRef.current = onReference
  })

  const postReference = useCallback((text: string) => {
    pendingRefs.current.push(text)
    setFlushTick((t) => t + 1)
  }, [])

  // Dispatch at most one queued reference per idle window. While the chat is busy,
  // clear the in-flight flag so the next reference waits for the following idle tick —
  // keeping references ordered and clear of the send lock.
  useEffect(() => {
    if (!canSend) {
      dispatchingRef.current = false
      return
    }
    if (dispatchingRef.current || pendingRefs.current.length === 0) return
    dispatchingRef.current = true
    onReferenceRef.current(pendingRefs.current.shift()!)
  }, [canSend, flushTick])

  const track = useCallback(
    async (jobId: string, kind: AttachKind, filename: string, toastId: string | number) => {
      const job = await pollJobUntilTerminal(jobId)
      if (job.status === "succeeded" && job.mlflow_run_id) {
        toast.success(`Attached ${filename}`, { id: toastId, duration: 4000 })
        void queryClient.invalidateQueries({ queryKey: ["jobs"] })
        void queryClient.invalidateQueries({
          queryKey: kind === "dataset" ? ["mlflow", "datasets"] : ["documents"],
        })
        postReference(referenceText(kind, filename, job.mlflow_run_id))
      } else {
        toast.error(`${filename} failed: ${job.error ?? "processing did not complete"}`, {
          id: toastId,
          duration: 4000,
        })
      }
    },
    [postReference, queryClient],
  )

  // No single-slot guard: dataset and document attaches track concurrently, so a
  // document upload that completes while a dataset attach is active is still tracked
  // and referenced (and vice versa).
  const run = useCallback(
    async (loadingMessage: string, work: (toastId: string | number) => Promise<void>) => {
      setActiveCount((n) => n + 1)
      const toastId = toast.loading(loadingMessage)
      try {
        await work(toastId)
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Attachment failed", {
          id: toastId,
          duration: 4000,
        })
      } finally {
        setActiveCount((n) => n - 1)
      }
    },
    [],
  )

  // Dataset has no reusable upload component, so upload here then track the job.
  const attachDataset = useCallback(
    (file: File) => {
      void run(`Uploading ${file.name}...`, async (toastId) => {
        const job = await uploadDataset(file)
        await track(job.id, "dataset", file.name, toastId)
      })
    },
    [run, track],
  )

  // Document reuses UploadDocumentDialog, which already submitted; just track the job.
  const attachDocument = useCallback(
    (jobId: string, filename: string) => {
      void run(`Processing ${filename}...`, async (toastId) => {
        await track(jobId, "document", filename, toastId)
      })
    },
    [run, track],
  )

  return { attachDataset, attachDocument, isUploading: activeCount > 0 }
}
