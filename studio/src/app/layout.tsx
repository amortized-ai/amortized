import { useEffect } from "react"
import { Link, Outlet, useLocation } from "react-router"
import {
  Home,
  MessageSquare,
  Briefcase,
  Database,
  FileText,
  Box,
  BookOpen,
  Settings,
  Sun,
  Moon,
  Bot,
  PanelRightClose,
} from "lucide-react"
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarInset,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable"
import { Separator } from "@/components/ui/separator"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { Breadcrumbs } from "@/components/breadcrumbs"
import { CreateMenu } from "@/components/create-menu"
import { ActiveJobsBadge } from "@/components/active-jobs-badge"
import { useHealth } from "@/features/settings/api/use-settings"
import { useUIStore } from "@/stores/ui-store"
import { useChatStore } from "@/stores/chat-store"
import { Button } from "@/components/ui/button"
import { CommandPalette } from "@/components/command-palette"
import { TutorialOverlay } from "@/components/tutorial"
import { ChatSidebar } from "@/features/chat/components/chat-sidebar"

import type { LucideIcon } from "lucide-react"

type NavItem = { label: string; path: string; icon: LucideIcon }

const NAV_ITEMS: NavItem[] = [
  { label: "Overview", path: "/overview", icon: Home },
  { label: "Chat", path: "/chat", icon: MessageSquare },
  { label: "Jobs", path: "/jobs", icon: Briefcase },
  { label: "Datasets", path: "/datasets", icon: Database },
  { label: "Documents", path: "/documents", icon: FileText },
  { label: "Models", path: "/models", icon: Box },
  { label: "Recipes", path: "/recipes", icon: BookOpen },
  { label: "Settings", path: "/settings", icon: Settings },
]

function ConnectionDot() {
  const { data: health, isError: healthError, isLoading } = useHealth({ refetchInterval: 30000 })

  const isConnected = !healthError && !isLoading && health?.status === "ok"

  const color = isConnected
    ? "bg-rh-green"
    : isLoading
      ? "bg-rh-yellow"
      : "bg-rh-red"

  const label = isConnected
    ? "Connected"
    : isLoading
      ? "Checking..."
      : "Disconnected"

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className={`inline-block h-2 w-2 rounded-full ${color}`}
          data-testid="connection-dot"
          aria-label={label}
        />
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function ThemeToggle() {
  const { theme, setTheme } = useUIStore()

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-muted-foreground hover:text-foreground"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          data-testid="theme-toggle"
        >
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          <span className="sr-only">Toggle theme</span>
        </Button>
      </TooltipTrigger>
      <TooltipContent>{theme === "dark" ? "Light mode" : "Dark mode"}</TooltipContent>
    </Tooltip>
  )
}

function TourButton() {
  const startTutorial = useUIStore((s) => s.startTutorial)
  const tutorialActive = useUIStore((s) => s.tutorialActive)

  if (tutorialActive) return null

  return (
    <Button
      variant="ghost"
      size="sm"
      className="h-7 text-xs text-muted-foreground hover:text-foreground transition-colors"
      onClick={startTutorial}
      data-testid="tour-button"
    >
      Take a Tour
    </Button>
  )
}

function ChatToggle() {
  const panelOpen = useChatStore((s) => s.panelOpen)
  const togglePanel = useChatStore((s) => s.togglePanel)
  const location = useLocation()

  if (location.pathname.startsWith("/chat")) return null

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant={panelOpen ? "secondary" : "ghost"}
          size="icon"
          className="h-8 w-8 text-muted-foreground hover:text-foreground"
          onClick={togglePanel}
          data-testid="chat-sidebar-toggle"
          aria-label={panelOpen ? "Close chat" : "Open chat"}
        >
          {panelOpen ? (
            <PanelRightClose className="h-4 w-4" />
          ) : (
            <MessageSquare className="h-4 w-4" />
          )}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{panelOpen ? "Close chat" : "Chat with Morty"}</TooltipContent>
    </Tooltip>
  )
}

export function AppLayout() {
  const location = useLocation()
  const theme = useUIStore((s) => s.theme)
  const panelOpen = useChatStore((s) => s.panelOpen)
  const isChatPage = location.pathname.startsWith("/chat")
  const showChatSidebar = panelOpen && !isChatPage

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark")
    document.documentElement.style.colorScheme = theme === "dark" ? "dark" : "light"
  }, [theme])

  return (
    <SidebarProvider>
      <Sidebar collapsible="icon">
        <SidebarHeader>
          <Link to="/" className="flex items-center gap-2 px-2 py-1 rounded-lg transition-colors duration-200 hover:bg-muted">
            <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-rh-red">
              <Bot className="h-3.5 w-3.5 text-white" />
            </div>
            <span className="text-sm font-semibold font-display group-data-[collapsible=icon]:hidden">
              Amortized Studio
            </span>
          </Link>
        </SidebarHeader>
        <SidebarContent>
          <SidebarGroup data-tutorial="sidebar-nav">
            <SidebarGroupLabel>Navigation</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {NAV_ITEMS.map((entry) => (
                  <SidebarMenuItem key={entry.path}>
                    <SidebarMenuButton
                      asChild
                      isActive={location.pathname.startsWith(entry.path)}
                    >
                      <Link to={entry.path}>
                        <entry.icon />
                        <span>{entry.label}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
      </Sidebar>
      <SidebarInset>
        <header className="relative z-20 flex h-12 items-center gap-2 border-b bg-background px-4">
          <SidebarTrigger data-testid="sidebar-trigger" />
          <Separator orientation="vertical" className="mx-2 h-4" />
          <Breadcrumbs />
          <div className="ml-auto flex items-center gap-3">
            <TourButton />
            <ActiveJobsBadge />
            <CreateMenu />
            <ChatToggle />
            <ConnectionDot />
            <ThemeToggle />
            <Link
              to="/settings"
              className="text-muted-foreground hover:text-foreground"
            >
              <Settings className="h-4 w-4" />
              <span className="sr-only">Settings</span>
            </Link>
          </div>
        </header>
        <ResizablePanelGroup orientation="horizontal" className="flex-1 min-h-0">
          <ResizablePanel defaultSize={100} minSize={40}>
            <main className="h-full min-w-0 overflow-auto p-4">
              <Outlet />
            </main>
          </ResizablePanel>
          {showChatSidebar && (
            <>
              <ResizableHandle withHandle />
              <ResizablePanel
                defaultSize={30}
                minSize={20}
                maxSize={50}
                id="chat-sidebar"
              >
                <ChatSidebar />
              </ResizablePanel>
            </>
          )}
        </ResizablePanelGroup>
      </SidebarInset>
      <CommandPalette />
      <TutorialOverlay />
    </SidebarProvider>
  )
}
