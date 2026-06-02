"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { LayoutDashboard, Briefcase, Workflow, Settings, MessageCircle, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { ChatPanel } from "@/components/chat-panel";
import "./globals.css";

const navItems = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/jobs", label: "Jobs", icon: Briefcase },
  { href: "/flows", label: "Flows", icon: Workflow },
  { href: "/settings", label: "Settings", icon: Settings },
];

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const [chatOpen, setChatOpen] = useState(false);
  const isDashboard = pathname === "/";

  return (
    <html lang="en" className="dark">
      <body className="flex min-h-screen">
        <aside className="hidden md:flex w-56 border-r border-border bg-sidebar flex-col">
          <div className="text-lg font-semibold text-foreground p-4 pb-0 mb-6">
            Amortized
          </div>
          <nav className="flex flex-col gap-1 px-3">
            {navItems.map((item) => {
              const isActive =
                item.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors",
                    isActive
                      ? "bg-muted text-foreground font-medium"
                      : "text-sidebar-foreground hover:bg-muted hover:text-foreground"
                  )}
                >
                  <item.icon className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </aside>
        <div className="flex-1 flex flex-col min-h-screen">
          <header className="h-14 border-b border-border bg-sidebar flex items-center px-6">
            <span className="text-lg font-semibold text-foreground md:hidden">Amortized</span>
            <nav className="ml-8 flex gap-4 md:hidden">
              {navItems.map((item) => {
                const isActive =
                  item.href === "/"
                    ? pathname === "/"
                    : pathname.startsWith(item.href);
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "text-sm transition-colors",
                      isActive
                        ? "text-foreground font-medium"
                        : "text-sidebar-foreground hover:text-foreground"
                    )}
                  >
                    {item.label}
                  </Link>
                );
              })}
            </nav>
            {!isDashboard && (
              <button
                onClick={() => setChatOpen(!chatOpen)}
                className={cn(
                  "ml-auto flex items-center gap-2 px-3 py-1.5 rounded-md text-sm",
                  "text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                )}
                aria-label={chatOpen ? "Close chat" : "Open chat"}
              >
                {chatOpen ? (
                  <X className="h-4 w-4" />
                ) : (
                  <MessageCircle className="h-4 w-4" />
                )}
                <span className="hidden sm:inline">
                  {chatOpen ? "Close" : "Assistant"}
                </span>
              </button>
            )}
          </header>
          <div className="flex-1 flex overflow-hidden">
            <main className={cn("flex-1 p-8 overflow-y-auto", !isDashboard && chatOpen && "mr-0")}>
              {children}
            </main>
            {!isDashboard && chatOpen && (
              <aside className="w-96 max-w-full border-l border-border bg-background flex flex-col">
                <ChatPanel mode="panel" />
              </aside>
            )}
          </div>
        </div>
      </body>
    </html>
  );
}
