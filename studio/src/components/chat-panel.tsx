"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Send, Loader2, Wrench, Play } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  streamChatMessage,
  ActionEvent,
  createTrainingJob,
  createSDGJob,
} from "@/lib/api";

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

export function ChatPanel({ mode = "panel" }: ChatPanelProps) {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [streamingText, setStreamingText] = useState("");
  const [thinkingTool, setThinkingTool] = useState<string | null>(null);
  const [pendingAction, setPendingAction] = useState<ActionEvent | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText, thinkingTool]);

  useEffect(() => {
    inputRef.current?.focus();
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
        // Send a confirmation message through the chat
        setMessages((prev) => [
          ...prev,
          { role: "user", content: `Yes, ${action.label.toLowerCase()}` },
        ]);
        // Let the agent know the user confirmed
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
  }, []);

  const isCenter = mode === "center";

  return (
    <div
      className={cn(
        "flex flex-col",
        isCenter ? "h-full w-full max-w-3xl mx-auto" : "h-full w-full"
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
                  {msg.action.label}
                </button>
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
