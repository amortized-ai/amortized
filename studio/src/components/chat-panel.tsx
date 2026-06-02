"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Send, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { streamChatMessage } from "@/lib/api";

interface DisplayMessage {
  role: "user" | "assistant";
  content: string;
}

interface ChatPanelProps {
  mode?: "center" | "panel";
}

const WELCOME_MESSAGE =
  "Hi, I'm your Amortized assistant. I can help you generate training data, " +
  "fine-tune models, and optimize your agent workflows. What would you like to do?";

export function ChatPanel({ mode = "panel" }: ChatPanelProps) {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [streamingText, setStreamingText] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);
    setStreamingText("");

    try {
      let accumulated = "";
      await streamChatMessage(
        text,
        conversationId,
        (delta) => {
          accumulated += delta;
          setStreamingText(accumulated);
        },
        (fullText) => {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: fullText },
          ]);
          setStreamingText("");
          setLoading(false);
        },
        (metadata) => {
          setConversationId(metadata.conversation_id);
        },
        () => {
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: accumulated || "Sorry, something went wrong. Please try again.",
            },
          ]);
          setStreamingText("");
          setLoading(false);
        }
      );
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, something went wrong. Please try again.",
        },
      ]);
      setStreamingText("");
      setLoading(false);
    }
  }, [input, loading, conversationId]);

  const handleNewChat = useCallback(() => {
    setMessages([]);
    setConversationId(undefined);
    setStreamingText("");
  }, []);

  const isCenter = mode === "center";

  return (
    <div
      className={cn(
        "flex flex-col",
        isCenter
          ? "h-full w-full max-w-3xl mx-auto"
          : "h-full w-full"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3">
        <h2
          className={cn(
            "font-semibold text-foreground",
            isCenter ? "text-lg" : "text-sm"
          )}
        >
          Amortized Assistant
        </h2>
        {messages.length > 0 && (
          <button
            onClick={handleNewChat}
            className="text-xs text-muted-foreground hover:text-foreground"
          >
            New chat
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 space-y-4">
        {messages.length === 0 && !streamingText && (
          <div
            className={cn(
              "text-muted-foreground",
              isCenter
                ? "text-center mt-24 max-w-lg mx-auto"
                : "text-center mt-8"
            )}
          >
            <p className={cn("mb-2 font-medium", isCenter ? "text-lg" : "text-sm")}>
              {WELCOME_MESSAGE}
            </p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={cn(
              "flex",
              msg.role === "user" ? "justify-end" : "justify-start"
            )}
          >
            <div
              className={cn(
                "rounded-lg px-3 py-2 text-sm whitespace-pre-wrap",
                isCenter ? "max-w-[75%]" : "max-w-[85%]",
                msg.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-foreground"
              )}
            >
              {msg.content}
            </div>
          </div>
        ))}
        {streamingText && (
          <div className="flex justify-start">
            <div
              className={cn(
                "rounded-lg px-3 py-2 text-sm whitespace-pre-wrap bg-muted text-foreground",
                isCenter ? "max-w-[75%]" : "max-w-[85%]"
              )}
            >
              {streamingText}
            </div>
          </div>
        )}
        {loading && !streamingText && (
          <div className="flex justify-start">
            <div className="bg-muted rounded-lg px-3 py-2">
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-4 py-4">
        <div className="flex gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Ask about training, SDG, VRAM..."
            rows={1}
            className={cn(
              "flex-1 resize-none rounded-md border border-input bg-background px-3 py-2 text-sm",
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
  );
}
