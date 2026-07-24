import { useEffect, useState } from "react"
import { useNavigate } from "react-router"
import {
  CommandDialog,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
} from "@/components/ui/command"
import {
  Home,
  MessageSquare,
  Briefcase,
  Database,
  FileText,
  Box,
  BookOpen,
  Settings,
} from "lucide-react"

const PAGES = [
  { label: "Overview", path: "/overview", icon: Home },
  { label: "Chat", path: "/chat", icon: MessageSquare },
  { label: "Jobs", path: "/jobs", icon: Briefcase },
  { label: "Datasets", path: "/datasets", icon: Database },
  { label: "Documents", path: "/documents", icon: FileText },
  { label: "Models", path: "/models", icon: Box },
  { label: "Recipes", path: "/recipes", icon: BookOpen },
  { label: "Settings", path: "/settings", icon: Settings },
]

export function CommandPalette() {
  const [open, setOpen] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault()
        setOpen((prev) => !prev)
      }
    }
    document.addEventListener("keydown", onKeyDown)
    return () => document.removeEventListener("keydown", onKeyDown)
  }, [])

  function handleSelect(path: string) {
    setOpen(false)
    navigate(path)
  }

  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <CommandInput placeholder="Type a command or search..." />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>
        <CommandGroup heading="Pages">
          {PAGES.map((page) => (
            <CommandItem
              key={page.path}
              value={page.label}
              onSelect={() => handleSelect(page.path)}
            >
              <page.icon className="h-4 w-4 text-muted-foreground" />
              {page.label}
            </CommandItem>
          ))}
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  )
}
