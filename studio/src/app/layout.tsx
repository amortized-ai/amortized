"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, Briefcase, Workflow, Settings } from "lucide-react";
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
          <header className="h-14 border-b border-border bg-sidebar flex items-center px-6 md:hidden">
            <span className="text-lg font-semibold text-foreground">Amortized</span>
            <nav className="ml-8 flex gap-4">
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
          </header>
          <main className="flex-1 p-8">{children}</main>
        </div>
        <ChatPanel />
      </body>
    </html>
  );
}
