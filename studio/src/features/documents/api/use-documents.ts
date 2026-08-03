import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { toast } from "sonner"
import {
  getDocuments,
  getDocumentContent,
  getDocumentChunks,
  uploadDocument,
  convertDocumentUrl,
  deleteDocument,
} from "@/lib/api-client"
import type { DocumentChunksResponse, DocumentRecord, DocumentUploadResponse } from "@/types/api"

export function useDocuments() {
  return useQuery<DocumentRecord[]>({
    queryKey: ["documents"],
    queryFn: getDocuments,
  })
}

export function useDocumentContent(id: string | null) {
  return useQuery<DocumentUploadResponse>({
    queryKey: ["documents", id, "content"],
    queryFn: () => getDocumentContent(id!),
    enabled: !!id,
  })
}

export function useDocumentChunks(id: string | null) {
  return useQuery<DocumentChunksResponse>({
    queryKey: ["documents", id, "chunks"],
    queryFn: () => getDocumentChunks(id!),
    enabled: !!id,
  })
}

export function useUploadDocument() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      file,
      options,
    }: {
      file: File
      options?: { output_format?: string; chunk_max_tokens?: number }
    }) => uploadDocument(file, options),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] })
    },
  })
}

export function useConvertDocumentUrl() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      url,
      options,
    }: {
      url: string
      options?: { output_format?: string; chunk_max_tokens?: number }
    }) => convertDocumentUrl(url, options),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] })
    },
  })
}


export function useDeleteDocument() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => deleteDocument(id),
    onSuccess: () => {
      toast.success("Document deleted successfully")
    },
    onError: (err) => {
      toast.error(`Failed to delete document: ${err instanceof Error ? err.message : "Unknown error"}`)
    },
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ["documents"] })
    },
  })
}
