"use client";

import { useState } from "react";
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
import { SymbolSwitcher } from "@/components/symbol-switcher";
import { OverviewChart } from "@/components/overview-chart";
import { SentimentTab } from "@/components/sentiment-tab";
import { VolatilityTab } from "@/components/volatility-tab";
import { AIAnalystTab } from "@/components/ai-analyst-tab";
import { AlertsTab } from "@/components/alerts-tab";

export default function Dashboard() {
  const [symbol, setSymbol] = useState("AAPL");

  const handleSymbolChange = (newSymbol: string) => {
    setSymbol(newSymbol);
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
        <div className="ml-auto flex items-center gap-2">
          <SymbolSwitcher symbol={symbol} setSymbol={handleSymbolChange} />
        </div>
      </div>

      <TabsContent value="ai-analyst">
        <AIAnalystTab symbol={symbol} />
      </TabsContent>

      <TabsContent value="overview">
        <Card>
          <CardHeader>
            <CardTitle>Price Overview</CardTitle>
          </CardHeader>
          <CardContent className="pl-2">
            <OverviewChart symbol={symbol} />
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="sentiment">
        <SentimentTab symbol={symbol} />
      </TabsContent>

      <TabsContent value="volatility">
        <VolatilityTab symbol={symbol} />
      </TabsContent>

      <TabsContent value="alerts">
        <AlertsTab symbol={symbol} onSymbolChange={handleSymbolChange} />
      </TabsContent>
    </Tabs>
  );
} 