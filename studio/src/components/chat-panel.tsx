"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { MessageCircle, Send, X, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  sendChatMessage,
  type ChatResponse,
  type SuggestedAction,
} from "@/lib/api";

interface DisplayMessage {
  role: "user" | "assistant";
  content: string;
  suggestedAction?: SuggestedAction | null;
}

export function ChatPanel() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (open) {
      inputRef.current?.focus();
    }
  }, [open]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);

    try {
      const res: ChatResponse = await sendChatMessage(text, conversationId);
      setConversationId(res.conversation_id);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.message,
          suggestedAction: res.suggested_action,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, something went wrong. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, conversationId]);

  const handleActionClick = useCallback(
    (action: SuggestedAction) => {
      switch (action.type) {
        case "create_training_job":
          router.push("/jobs/new?tab=training");
          break;
        case "create_sdg_job":
          router.push("/jobs/new?tab=sdg");
          break;
        case "estimate_memory":
          router.push("/jobs/new?tab=training");
          break;
        case "view_jobs":
          router.push("/jobs");
          break;
        case "view_flows":
          router.push("/flows");
          break;
        default:
          break;
      }
      setOpen(false);
    },
    [router]
  );

  const handleNewChat = useCallback(() => {
    setMessages([]);
    setConversationId(undefined);
  }, []);

  return (
    <>
      {/* Toggle button */}
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          "fixed bottom-6 right-6 z-50 flex h-12 w-12 items-center justify-center",
          "rounded-full bg-primary text-primary-foreground shadow-lg",
          "transition-transform hover:scale-105",
          open && "hidden"
        )}
        aria-label="Open chat"
      >
        <MessageCircle className="h-5 w-5" />
      </button>

      {/* Panel */}
      <div
        className={cn(
          "fixed right-0 top-0 z-40 flex h-full w-96 max-w-full flex-col",
          "border-l border-border bg-background shadow-xl",
          "transition-transform duration-200 ease-in-out",
          open ? "translate-x-0" : "translate-x-full"
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <h2 className="text-sm font-semibold text-foreground">
            Amortized Assistant
          </h2>
          <div className="flex items-center gap-2">
            <button
              onClick={handleNewChat}
              className="text-xs text-muted-foreground hover:text-foreground"
            >
              New chat
            </button>
            <button
              onClick={() => setOpen(false)}
              className="text-muted-foreground hover:text-foreground"
              aria-label="Close chat"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="text-sm text-muted-foreground text-center mt-8">
              <p className="mb-2 font-medium">How can I help?</p>
              <p>Ask me about fine-tuning, data generation, VRAM estimation, or job status.</p>
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={cn("flex", msg.role === "user" ? "justify-end" : "justify-start")}>
              <div
                className={cn(
                  "max-w-[85%] rounded-lg px-3 py-2 text-sm",
                  msg.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-foreground"
                )}
              >
                <div className="whitespace-pre-wrap">{msg.content}</div>
                {msg.suggestedAction && (
                  <button
                    onClick={() => handleActionClick(msg.suggestedAction!)}
                    className={cn(
                      "mt-2 w-full rounded-md border px-3 py-1.5 text-xs font-medium",
                      "border-border bg-background text-foreground",
                      "hover:bg-muted transition-colors"
                    )}
                  >
                    {msg.suggestedAction.label || msg.suggestedAction.type}
                  </button>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-muted rounded-lg px-3 py-2">
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-border p-4">
          <div className="flex gap-2">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Ask about training, SDG, VRAM..."
              className={cn(
                "flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm",
                "placeholder:text-muted-foreground",
                "focus:outline-none focus:ring-1 focus:ring-ring"
              )}
              disabled={loading}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || loading}
              className={cn(
                "flex h-9 w-9 items-center justify-center rounded-md",
                "bg-primary text-primary-foreground",
                "disabled:opacity-50 disabled:cursor-not-allowed",
                "hover:bg-primary/90 transition-colors"
              )}
              aria-label="Send message"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
