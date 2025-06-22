"use client";

import { useSymbolStore } from "@/lib/store";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { OverviewChart } from "@/components/overview-chart";
import { SentimentTab } from "@/components/sentiment-tab";
import { VolatilityTab } from "@/components/volatility-tab";
import { AIAnalystTab } from "@/components/ai-analyst-tab";
import { AlertsTab } from "@/components/alerts-tab";

export default function Dashboard() {
  const { currentSymbol, setCurrentSymbol } = useSymbolStore();

  const handleSymbolChange = (newSymbol: string) => {
    setCurrentSymbol(newSymbol);
  };

  return (
    <Tabs defaultValue="ai-analyst" className="space-y-4">
      <div className="flex items-center">
        <TabsList>
          <TabsTrigger value="ai-analyst">AI Analyst</TabsTrigger>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="sentiment">Sentiment</TabsTrigger>
          <TabsTrigger value="volatility">Volatility</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>
      </div>

      <TabsContent value="ai-analyst">
        <AIAnalystTab symbol={currentSymbol} />
      </TabsContent>

      <TabsContent value="overview">
        <Card>
          <CardHeader>
            <CardTitle>Price Overview</CardTitle>
          </CardHeader>
          <CardContent className="pl-2">
            <OverviewChart symbol={currentSymbol} />
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="sentiment">
        <SentimentTab symbol={currentSymbol} />
      </TabsContent>

      <TabsContent value="volatility">
        <VolatilityTab symbol={currentSymbol} />
      </TabsContent>

      <TabsContent value="alerts">
        <AlertsTab symbol={currentSymbol} onSymbolChange={handleSymbolChange} />
      </TabsContent>
    </Tabs>
  );
} 