"use client";

import { useCallback, useEffect, useState } from "react";
import { useSymbolStore } from "@/lib/store";
import api from "@/lib/api";
import { toast } from "sonner";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { OverviewChart } from "@/components/overview-chart";
import { SentimentTab } from "@/components/sentiment-tab";
import { VolatilityTab } from "@/components/volatility-tab";
import { AIAnalystTab } from "@/components/ai-analyst-tab";

// Define interfaces for the data to be fetched
interface SentimentData {
  news_sentiment_details: any;
  reddit_sentiment_details: any;
  news_sentiment_score: number;
  reddit_sentiment_score: number;
}

interface VolatilityData {
  predicted_volatility: number;
  prediction_range: {
    low: number;
    high: number;
  };
  volatility_regime: string;
  trend: string;
  confidence_score: number;
}

interface ChartData {
  date: string;
  price: number;
}

interface AIAnalystData {
  symbol: string;
  generated_at: string;
  overall_outlook: string;
  summary: string;
  insights: any[];
}

export default function DashboardPage() {
  const { currentSymbol } = useSymbolStore();

  // States for data
  const [sentimentData, setSentimentData] = useState<SentimentData | null>(null);
  const [volatilityData, setVolatilityData] = useState<VolatilityData | null>(null);
  const [historicalPrices, setHistoricalPrices] = useState<ChartData[]>([]);
  const [historicalVolatility, setHistoricalVolatility] = useState([]);
  const [aiAnalystData, setAiAnalystData] = useState<AIAnalystData | null>(null);

  // Granular loading states
  const [isSentimentLoading, setIsSentimentLoading] = useState(true);
  const [isVolatilityLoading, setIsVolatilityLoading] = useState(true);
  const [isChartLoading, setIsChartLoading] = useState(true);
  const [isAiAnalystLoading, setIsAiAnalystLoading] = useState(true);

  const fetchData = useCallback(async () => {
    if (!currentSymbol) return;

    console.log(`Starting to fetch data for symbol: ${currentSymbol}`);

    // Reset states
    setIsSentimentLoading(true);
    setIsVolatilityLoading(true);
    setIsChartLoading(true);
    setIsAiAnalystLoading(true);
    
    toast.info(`Fetching data for ${currentSymbol}...`);

    try {
      const sentimentPromise = api.get(`/sentiment/${currentSymbol}`);
      const volatilityPromise = api.post('/volatility/predict', { symbol: currentSymbol });
      const pricesPromise = api.get(`/visualization/historical-prices/${currentSymbol}`);
      const historicalVolatilityPromise = api.get(`/volatility/historical/${currentSymbol}`);
      const aiAnalystPromise = api.get(`/insights/${currentSymbol}`);

      console.log('Making API calls...');

      const [
        sentimentResult,
        volatilityResult,
        pricesResult,
        historicalVolatilityResult,
        aiAnalystResult,
      ] = await Promise.allSettled([
        sentimentPromise,
        volatilityPromise,
        pricesPromise,
        historicalVolatilityPromise,
        aiAnalystPromise,
      ]);

      console.log('API calls completed:', {
        sentiment: sentimentResult.status,
        volatility: volatilityResult.status,
        prices: pricesResult.status,
        historicalVolatility: historicalVolatilityResult.status,
        aiAnalyst: aiAnalystResult.status,
      });

      // Handle sentiment data
      if (sentimentResult.status === 'fulfilled') {
        console.log('Sentiment data received:', sentimentResult.value.data);
        setSentimentData(sentimentResult.value.data);
      } else {
        console.error('Sentiment API failed:', sentimentResult.reason);
        toast.error(`Failed to fetch sentiment data for ${currentSymbol}`);
        setSentimentData(null);
      }
      setIsSentimentLoading(false);

      // Handle volatility data
      if (volatilityResult.status === 'fulfilled') {
        console.log('Volatility data received:', volatilityResult.value.data);
        setVolatilityData(volatilityResult.value.data);
      } else {
        console.error('Volatility API failed:', volatilityResult.reason);
        toast.error(`Failed to fetch volatility data for ${currentSymbol}`);
        setVolatilityData(null);
      }
      setIsVolatilityLoading(false);

      // Handle historical prices
      if (pricesResult.status === 'fulfilled') {
        console.log('Price data received:', pricesResult.value.data);
        setHistoricalPrices(pricesResult.value.data);
      } else {
        console.error('Prices API failed:', pricesResult.reason);
        toast.error(`Failed to fetch price chart data for ${currentSymbol}`);
        setHistoricalPrices([]);
      }
      setIsChartLoading(false);

      // Handle historical volatility for the tab
      if (historicalVolatilityResult.status === 'fulfilled') {
        console.log('Historical volatility data received:', historicalVolatilityResult.value.data);
        setHistoricalVolatility(historicalVolatilityResult.value.data);
      } else {
        console.error('Historical volatility API failed:', historicalVolatilityResult.reason);
        toast.error(`Failed to fetch historical volatility for ${currentSymbol}`);
        setHistoricalVolatility([]);
      }

      // Handle AI Analyst data
      if (aiAnalystResult.status === 'fulfilled') {
        console.log('AI Analyst data received:', aiAnalystResult.value.data);
        setAiAnalystData(aiAnalystResult.value.data);
      } else {
        console.error('AI Analyst API failed:', aiAnalystResult.reason);
        toast.error(`Failed to fetch AI analysis for ${currentSymbol}`);
        setAiAnalystData(null);
      }
      setIsAiAnalystLoading(false);

    } catch (error) {
      console.error('Error in fetchData:', error);
      toast.error('An error occurred while fetching data');
    }

  }, [currentSymbol]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return (
    <div className="grid gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Price Overview for {currentSymbol}</CardTitle>
            <CardDescription>
              Historical price data for the selected symbol.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <OverviewChart data={historicalPrices} isLoading={isChartLoading} />
          </CardContent>
        </Card>
        
        <Tabs defaultValue="ai_analyst">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="ai_analyst">AI Analyst</TabsTrigger>
            <TabsTrigger value="sentiment">Sentiment</TabsTrigger>
            <TabsTrigger value="volatility">Volatility</TabsTrigger>
          </TabsList>
          <TabsContent value="ai_analyst">
            <AIAnalystTab data={aiAnalystData} isLoading={isAiAnalystLoading} />
          </TabsContent>
          <TabsContent value="sentiment">
            <SentimentTab sentimentData={sentimentData} isLoading={isSentimentLoading} />
          </TabsContent>
          <TabsContent value="volatility">
            <VolatilityTab 
              volatilityData={volatilityData} 
              historicalData={historicalVolatility} 
              isLoading={isVolatilityLoading} />
          </TabsContent>
        </Tabs>
    </div>
  );
} 