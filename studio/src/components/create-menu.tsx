import { useNavigate } from "react-router"
import { Plus, MessageSquare, BookOpen } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu"

const CREATE_OPTIONS = [
  { label: "New Chat", icon: MessageSquare, path: "/chat" },
  { label: "Create Recipe", icon: BookOpen, path: "/recipes" },
] as const

export function CreateMenu() {
  const navigate = useNavigate()

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button size="sm">
          <Plus className="h-4 w-4" />
          Create
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {CREATE_OPTIONS.map((option) => (
          <DropdownMenuItem
            key={option.path}
            onClick={() => navigate(option.path)}
          >
            <option.icon className="h-4 w-4" />
            {option.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
