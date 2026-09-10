import { BrowserRouter } from "react-router"
import { MutationCache, QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { toast } from "sonner"
import { Toaster } from "sonner"
import { TooltipProvider } from "@/components/ui/tooltip"

const queryClient = new QueryClient({
  mutationCache: new MutationCache({
    onError: (error) => {
      toast.error(error.message)
    },
  }),
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 2,
    },
  },
})

// Client-side routing base, derived from the Vite build base (import.meta.env.
// BASE_URL). "/" for the standalone/kind deployment; a subpath (e.g.
// "/amortized-studio-embed/") when served behind the RHOAI dashboard proxy.
const routerBasename = import.meta.env.BASE_URL.replace(/\/+$/, "") || "/"

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <BrowserRouter basename={routerBasename}>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          {children}
          <Toaster position="bottom-right" richColors />
        </TooltipProvider>
      </QueryClientProvider>
    </BrowserRouter>
  )
}
