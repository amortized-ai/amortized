import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  getDocuments,
  getDocumentContent,
  uploadDocument,
  convertDocumentUrl,
} from "@/lib/api-client"
import type { DocumentRecord, DocumentUploadResponse } from "@/types/api"

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

export function useUploadDocument() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      file,
      options,
    }: {
      file: File
      options?: { output_format?: string }
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
      options?: { output_format?: string }
    }) => convertDocumentUrl(url, options),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] })
    },
  })
}
