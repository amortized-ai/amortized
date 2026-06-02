import { listConversations, getConversation, ConversationSummary, ChatMessage } from "./api";

const STORAGE_KEY = "amortized_conversation_id";

export function getSavedConversationId(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(STORAGE_KEY);
}

export function saveConversationId(id: string): void {
  localStorage.setItem(STORAGE_KEY, id);
}

export function clearConversationId(): void {
  localStorage.removeItem(STORAGE_KEY);
}

export async function loadConversation(id: string): Promise<ChatMessage[] | null> {
  try {
    const detail = await getConversation(id);
    return detail.messages;
  } catch {
    // Conversation no longer exists
    clearConversationId();
    return null;
  }
}

export async function loadRecentConversations(): Promise<ConversationSummary[]> {
  try {
    return await listConversations();
  } catch {
    return [];
  }
}
