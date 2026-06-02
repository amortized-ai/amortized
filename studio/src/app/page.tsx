import { ChatPanel } from "@/components/chat-panel";

export default function DashboardPage() {
  return (
    <div className="h-[calc(100vh-4rem)]">
      <ChatPanel mode="center" />
    </div>
  );
}
