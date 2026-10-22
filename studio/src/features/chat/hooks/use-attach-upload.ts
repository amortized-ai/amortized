import { useCallback, useEffect, useRef, useState } from "react"
import { toast } from "sonner"
import { useQueryClient } from "@tanstack/react-query"
import { getJob, uploadDataset } from "@/lib/api-client"
import type { Job } from "@/types/api"

const POLL_INTERVAL_MS = 2000
const POLL_TIMEOUT_MS = 10 * 60 * 1000

type AttachKind = "dataset" | "document"

export interface AttachStatus {
  id: number
  message: string
}

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
 * Progress is reported via `statuses` (rendered inline, non-blocking) rather than a
 * toast that would cover the composer; success needs no toast because the posted
 * reference turn is the confirmation; only failures toast (brief + obvious).
 *
 * References are delivered one at a time by a drain driven off each send's completion.
 * `onReference` returns whether it delivered (false = the chat couldn't accept it): an
 * undelivered reference stays queued and a later idle transition resumes the drain, so
 * none is lost or stuck. Attaches track concurrently, so a second is never dropped.
 */
export function useAttachUpload(onReference: (text: string) => unknown, canSend = true) {
  const queryClient = useQueryClient()
  const [statuses, setStatuses] = useState<AttachStatus[]>([])
  const pendingRefs = useRef<string[]>([])
  const drainingRef = useRef(false)
  const idRef = useRef(0)

  // Read latest props at dispatch time (which can be minutes after a track starts)
  // without re-creating callbacks; synced after render, never during it.
  const onReferenceRef = useRef(onReference)
  const canSendRef = useRef(canSend)
  useEffect(() => {
    onReferenceRef.current = onReference
    canSendRef.current = canSend
  })

  // Deliver queued references one at a time. Await each send and advance only once it
  // reports delivery; a send that reports false (chat busy / couldn't accept) leaves
  // the reference queued and stops the drain — the idle effect below resumes it.
  const drain = useCallback(async () => {
    if (drainingRef.current) return
    drainingRef.current = true
    try {
      while (canSendRef.current && pendingRefs.current.length > 0) {
        const next = pendingRefs.current[0]!
        const delivered = await onReferenceRef.current(next)
        if (delivered === false) break
        pendingRefs.current.shift()
      }
    } finally {
      drainingRef.current = false
    }
  }, [])

  const postReference = useCallback(
    (text: string) => {
      pendingRefs.current.push(text)
      void drain()
    },
    [drain],
  )

  // Resume draining when the chat returns to idle — a reference may have been deferred
  // because the chat was busy, or a prior send reported it couldn't be delivered.
  useEffect(() => {
    if (canSend) void drain()
  }, [canSend, drain])

  const track = useCallback(
    async (jobId: string, kind: AttachKind, filename: string) => {
      const job = await pollJobUntilTerminal(jobId)
      if (job.status === "succeeded" && job.mlflow_run_id) {
        void queryClient.invalidateQueries({ queryKey: ["jobs"] })
        void queryClient.invalidateQueries({
          queryKey: kind === "dataset" ? ["mlflow", "datasets"] : ["documents"],
        })
        postReference(referenceText(kind, filename, job.mlflow_run_id))
      } else {
        toast.error(`${filename} failed: ${job.error ?? "processing did not complete"}`, {
          duration: 4000,
        })
      }
    },
    [postReference, queryClient],
  )

  // Inline, non-blocking progress (see `statuses`); errors stay as a toast. No
  // single-slot guard — dataset and document attaches track concurrently.
  const run = useCallback(
    async (message: string, work: () => Promise<void>) => {
      const id = idRef.current++
      setStatuses((s) => [...s, { id, message }])
      try {
        await work()
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Attachment failed", { duration: 4000 })
      } finally {
        setStatuses((s) => s.filter((x) => x.id !== id))
      }
    },
    [],
  )

  // Dataset has no reusable upload component, so upload here then track the job.
  const attachDataset = useCallback(
    (file: File) => {
      void run(`Uploading ${file.name}`, async () => {
        const job = await uploadDataset(file)
        await track(job.id, "dataset", file.name)
      })
    },
    [run, track],
  )

  // Document reuses UploadDocumentDialog, which already submitted; just track the job.
  const attachDocument = useCallback(
    (jobId: string, filename: string) => {
      void run(`Processing ${filename}`, async () => {
        await track(jobId, "document", filename)
      })
    },
    [run, track],
  )

  return { attachDataset, attachDocument, statuses, isUploading: statuses.length > 0 }
}
