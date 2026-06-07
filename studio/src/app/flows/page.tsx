"use client";

import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { listFlows, type SDGFlow } from "@/lib/api";

export default function FlowsPage() {
  const [flows, setFlows] = useState<SDGFlow[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedFlow, setSelectedFlow] = useState<SDGFlow | null>(null);

  useEffect(() => {
    listFlows()
      .then(setFlows)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">SDG Flows</h1>

      {loading ? (
        <p className="text-muted-foreground">Loading flows...</p>
      ) : flows.length === 0 ? (
        <p className="text-muted-foreground">
          No flows available. Make sure the runtime is running and asynth is configured.
        </p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {flows.map((flow) => (
            <Card
              key={flow.name}
              className="cursor-pointer hover:border-accent/50 transition-colors"
              onClick={() => setSelectedFlow(flow)}
            >
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">{flow.name}</CardTitle>
                  {flow.supports_multi_turn && (
                    <Badge variant="secondary">Multi-turn</Badge>
                  )}
                </div>
                <CardDescription>{flow.description}</CardDescription>
              </CardHeader>
            </Card>
          ))}
        </div>
      )}

      <Dialog open={!!selectedFlow} onOpenChange={() => setSelectedFlow(null)}>
        {selectedFlow && (
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{selectedFlow.name}</DialogTitle>
              <DialogDescription>{selectedFlow.description}</DialogDescription>
            </DialogHeader>
            <div className="space-y-3 text-sm">
              <div>
                <span className="text-muted-foreground">Multi-turn: </span>
                <Badge variant="secondary">{selectedFlow.supports_multi_turn ? "Yes" : "No"}</Badge>
              </div>
            </div>
          </DialogContent>
        )}
      </Dialog>
    </div>
  );
}
