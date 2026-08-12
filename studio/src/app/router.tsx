import { lazy, Suspense } from "react"
import { TableSkeleton } from "@/components/table-skeleton"
import { Navigate, Route, Routes } from "react-router"
import { ErrorBoundary } from "react-error-boundary"
import { AppLayout } from "@/app/layout"
import { RouteErrorFallback } from "@/components/route-error-fallback"
import ChatPage from "@/features/chat/page"
import JobsPage from "@/features/jobs/page"
import DatasetsPage from "@/features/datasets/page"
import { DatasetDetailPage } from "@/features/datasets/components/dataset-detail"
import ModelsPage from "@/features/models/page"
import DocumentsPage from "@/features/documents/page"
import SettingsPage from "@/features/settings/page"

const OverviewPage = lazy(() => import("@/features/overview/page"))
const RecipesPage = lazy(() => import("@/features/recipes/page"))

export function AppRouter() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Navigate to="/overview" replace />} />
        <Route
          path="overview"
          element={
            <ErrorBoundary FallbackComponent={RouteErrorFallback}>
              <Suspense fallback={<div className="p-4"><TableSkeleton columns={4} /></div>}>
                <OverviewPage />
              </Suspense>
            </ErrorBoundary>
          }
        />
        <Route path="chat" element={<ErrorBoundary FallbackComponent={RouteErrorFallback}><ChatPage /></ErrorBoundary>} />
        <Route path="jobs" element={<ErrorBoundary FallbackComponent={RouteErrorFallback}><JobsPage /></ErrorBoundary>} />
        <Route path="datasets" element={<ErrorBoundary FallbackComponent={RouteErrorFallback}><DatasetsPage /></ErrorBoundary>} />
        <Route path="datasets/:id" element={<ErrorBoundary FallbackComponent={RouteErrorFallback}><DatasetDetailPage /></ErrorBoundary>} />
        <Route path="documents" element={<ErrorBoundary FallbackComponent={RouteErrorFallback}><DocumentsPage /></ErrorBoundary>} />
        <Route path="models" element={<ErrorBoundary FallbackComponent={RouteErrorFallback}><ModelsPage /></ErrorBoundary>} />
        <Route
          path="recipes"
          element={
            <ErrorBoundary FallbackComponent={RouteErrorFallback}>
              <Suspense fallback={<div className="p-4"><TableSkeleton columns={4} /></div>}>
                <RecipesPage />
              </Suspense>
            </ErrorBoundary>
          }
        />
        <Route path="settings" element={<ErrorBoundary FallbackComponent={RouteErrorFallback}><SettingsPage /></ErrorBoundary>} />
      </Route>
    </Routes>
  )
}
