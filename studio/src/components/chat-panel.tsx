"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Send, Loader2, Wrench, Play, MessageSquarePlus, History } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  streamChatMessage,
  ActionEvent,
  createTrainingJob,
  createSDGJob,
  ConversationSummary,
  ChatMessage,
} from "@/lib/api";
import {
  getSavedConversationId,
  saveConversationId,
  clearConversationId,
  loadConversation,
  loadRecentConversations,
} from "@/lib/chat-store";

interface DisplayMessage {
  role: "user" | "assistant";
  content: string;
  action?: ActionEvent | null;
}

interface ChatPanelProps {
  mode?: "center" | "panel";
}

const WELCOME_MESSAGE =
  "Hi, I'm your Amortized assistant. I can help you generate training data, " +
  "fine-tune models, and optimize your agent workflows. What would you like to do?";

const TOOL_LABELS: Record<string, string> = {
  list_sdg_flows: "Checking SDG flows",
  submit_sdg_job: "Submitting SDG job",
  submit_training_job: "Submitting training job",
  check_job_status: "Checking job status",
  get_job_metrics: "Fetching metrics",
  list_jobs: "Listing jobs",
  estimate_vram: "Estimating VRAM",
  propose_action: "Preparing action",
};

function parseMessageContent(msg: ChatMessage): string {
  if (typeof msg.content === "string") return msg.content;
  if (msg.content && typeof msg.content === "object" && "message" in msg.content) {
    return msg.content.message;
  }
  return String(msg.content);
}

function formatTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
}

export function ChatPanel({ mode = "panel" }: ChatPanelProps) {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [streamingText, setStreamingText] = useState("");
  const [thinkingTool, setThinkingTool] = useState<string | null>(null);
  const [pendingAction, setPendingAction] = useState<ActionEvent | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [restoringChat, setRestoringChat] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const historyRef = useRef<HTMLDivElement>(null);

  // Restore conversation on mount — load ALL messages including any
  // that arrived while the user was on another page
  useEffect(() => {
    const savedId = getSavedConversationId();
    if (savedId) {
      loadConversation(savedId).then((msgs) => {
        if (msgs && msgs.length > 0) {
          setConversationId(savedId);
          const restored: DisplayMessage[] = msgs.map((m) => ({
            role: m.role,
            content: parseMessageContent(m),
          }));
          // If the last message is from the user, the assistant response
          // was lost (e.g. stream disconnected while on another page).
          // Show the messages and add a note so the user can resend.
          const lastMsg = restored[restored.length - 1];
          if (lastMsg && lastMsg.role === "user") {
            restored.push({
              role: "assistant",
              content:
                "It looks like my previous response didn\u2019t come through. " +
                "Could you resend your last message?",
            });
          }
          setMessages(restored);
        }
        setRestoringChat(false);
      });
    } else {
      setRestoringChat(false);
    }
  }, []);

  // Save conversation_id whenever it changes
  useEffect(() => {
    if (conversationId) {
      saveConversationId(conversationId);
    }
  }, [conversationId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText, thinkingTool]);

  useEffect(() => {
    if (!restoringChat) {
      inputRef.current?.focus();
    }
  }, [restoringChat]);

  // Close history dropdown on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (historyRef.current && !historyRef.current.contains(e.target as Node)) {
        setShowHistory(false);
      }
    }
    if (showHistory) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [showHistory]);

  const handleToggleHistory = useCallback(async () => {
    if (!showHistory) {
      const convos = await loadRecentConversations();
      setConversations(convos);
    }
    setShowHistory((prev) => !prev);
  }, [showHistory]);

  const handleLoadConversation = useCallback(async (id: string) => {
    setShowHistory(false);
    const msgs = await loadConversation(id);
    if (msgs && msgs.length > 0) {
      setConversationId(id);
      saveConversationId(id);
      setMessages(
        msgs.map((m) => ({
          role: m.role,
          content: parseMessageContent(m),
        }))
      );
      setStreamingText("");
      setThinkingTool(null);
      setPendingAction(null);
    }
  }, []);

  const handleActionClick = useCallback(
    async (action: ActionEvent) => {
      setLoading(true);
      setPendingAction(null);
      try {
        if (action.type === "submit_training_job") {
          await createTrainingJob(action.config as never);
        } else if (action.type === "submit_sdg_job") {
          await createSDGJob(action.config as never);
        }
        setMessages((prev) => [
          ...prev,
          { role: "user", content: `Yes, ${action.label.toLowerCase()}` },
        ]);
        let accumulated = "";
        await streamChatMessage(
          `The user confirmed: "${action.label}". The job has been submitted with config: ${JSON.stringify(action.config)}. Let them know it's been submitted and how to check status.`,
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
                content: "Job submitted successfully! Use the Jobs page to monitor progress.",
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
            content: "Failed to submit the job. Please try again.",
          },
        ]);
        setLoading(false);
      }
    },
    [conversationId]
  );

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);
    setStreamingText("");
    setThinkingTool(null);
    setPendingAction(null);

    try {
      let accumulated = "";
      let lastAction: ActionEvent | null = null;

      await streamChatMessage(
        text,
        conversationId,
        (delta) => {
          setThinkingTool(null);
          accumulated += delta;
          setStreamingText(accumulated);
        },
        (fullText) => {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: fullText, action: lastAction },
          ]);
          setStreamingText("");
          setThinkingTool(null);
          setLoading(false);
          if (lastAction) {
            setPendingAction(lastAction);
          }
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
          setThinkingTool(null);
          setLoading(false);
        },
        {
          onThinking: (data) => {
            setThinkingTool(data.tool);
          },
          onToolResult: () => {
            setThinkingTool(null);
          },
          onAction: (data) => {
            lastAction = data;
          },
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
      setThinkingTool(null);
      setLoading(false);
    }
  }, [input, loading, conversationId]);

  const handleNewChat = useCallback(() => {
    setMessages([]);
    setConversationId(undefined);
    setStreamingText("");
    setThinkingTool(null);
    setPendingAction(null);
    clearConversationId();
  }, []);

  const isCenter = mode === "center";

  // Derive title from first user message
  const conversationTitle = messages.find((m) => m.role === "user")?.content.slice(0, 40);

  if (restoringChat) {
    return (
      <div className={cn("flex flex-col items-center justify-center", isCenter ? "h-full" : "h-full")}>
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex flex-col",
        isCenter ? "h-full w-full max-w-3xl mx-auto" : "h-full w-full"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-2 min-w-0">
          <h2
            className={cn(
              "font-semibold text-foreground truncate",
              isCenter ? "text-lg" : "text-sm"
            )}
          >
            {conversationTitle || "Amortized Assistant"}
          </h2>
        </div>
        <div className="flex items-center gap-1 relative" ref={historyRef}>
          <button
            onClick={handleToggleHistory}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground px-2 py-1 rounded-md hover:bg-muted transition-colors"
            title="Chat history"
          >
            <History className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">History</span>
          </button>
          <button
            onClick={handleNewChat}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground px-2 py-1 rounded-md hover:bg-muted transition-colors"
            title="New chat"
          >
            <MessageSquarePlus className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">New</span>
          </button>

          {/* History dropdown */}
          {showHistory && (
            <div className="absolute right-0 top-full mt-1 z-50 w-72 rounded-md border border-border bg-popover shadow-lg">
              <div className="px-3 py-2 border-b border-border">
                <span className="text-xs font-medium text-muted-foreground">Recent conversations</span>
              </div>
              <div className="max-h-64 overflow-y-auto">
                {conversations.length === 0 ? (
                  <div className="px-3 py-4 text-xs text-muted-foreground text-center">
                    No conversations yet
                  </div>
                ) : (
                  conversations.map((conv) => (
                    <button
                      key={conv.id}
                      onClick={() => handleLoadConversation(conv.id)}
                      className={cn(
                        "w-full text-left px-3 py-2 text-sm hover:bg-muted transition-colors flex items-center justify-between gap-2",
                        conv.id === conversationId && "bg-muted"
                      )}
                    >
                      <span className="truncate text-foreground">
                        {conv.title || "Untitled"}
                      </span>
                      <span className="text-xs text-muted-foreground whitespace-nowrap">
                        {formatTime(conv.updated_at)}
                      </span>
                    </button>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
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
            <p
              className={cn(
                "mb-2 font-medium",
                isCenter ? "text-lg" : "text-sm"
              )}
            >
              {WELCOME_MESSAGE}
            </p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i}>
            <div
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
            {/* Action button rendered after the assistant message */}
            {msg.role === "assistant" && msg.action && pendingAction === msg.action && (
              <div className="flex justify-start mt-2">
                <div className="bg-muted rounded-lg p-3 text-sm">
                  <div className="font-medium mb-1">{msg.action.label}</div>
                  <pre className="text-xs text-muted-foreground overflow-x-auto mb-3">
                    {JSON.stringify(msg.action.config, null, 2)}
                  </pre>
                  <button
                    onClick={() => handleActionClick(msg.action!)}
                    disabled={loading}
                    className={cn(
                      "flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium",
                      "bg-primary text-primary-foreground",
                      "hover:bg-primary/90 transition-colors",
                      "disabled:opacity-50 disabled:cursor-not-allowed"
                    )}
                  >
                    <Play className="h-4 w-4" />
                    Confirm & Submit
                  </button>
                </div>
              </div>
            )}
          </div>
        ))}

        {/* Thinking indicator */}
        {thinkingTool && (
          <div className="flex justify-start">
            <div className="bg-muted rounded-lg px-3 py-2 flex items-center gap-2 text-sm text-muted-foreground">
              <Wrench className="h-4 w-4 animate-pulse" />
              {TOOL_LABELS[thinkingTool] || thinkingTool}...
            </div>
          </div>
        )}

        {/* Streaming text */}
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

        {/* Loading spinner (when no streaming text and no thinking indicator) */}
        {loading && !streamingText && !thinkingTool && (
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
