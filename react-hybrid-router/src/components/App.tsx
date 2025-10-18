import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  ConversationExchange, 
  SessionStats, 
  RouterStrategy, 
  ConversationInsights, 
  SystemStatus,
  PerformanceMetrics,
  ConversationMetadata 
} from '../types';
import ChatInterface from './ChatInterface';
import Analytics from './Analytics';
import { HybridRouterAPI } from '../services/api';
import './App.css';

const App: React.FC = () => {
  const [conversationHistory, setConversationHistory] = useState<ConversationExchange[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<RouterStrategy>('hybrid');
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean>(false);
  const [performanceHistory, setPerformanceHistory] = useState<PerformanceMetrics[]>([]);
  const [sessionStats, setSessionStats] = useState<SessionStats>({
    start_time: new Date(),
    total_exchanges: 0,
    model_switches: 0,
    last_model: null,
    context_preserved: 0,
    fallback_used: 0
  });

  const api = useMemo(() => new HybridRouterAPI(), []);

  // Check system status with useCallback to avoid dependency issues
  const checkSystemStatus = useCallback(async () => {
    try {
      const status = await api.getSystemStatus();
      setSystemStatus(status);
      setApiConnected(status.systemHealth !== 'error');
    } catch (error) {
      console.warn('Failed to get system status:', error);
      setApiConnected(false);
      setSystemStatus({
        availableRouters: { hybrid: false, rule_based: false, bert: false, phi: false },
        systemHealth: 'error'
      });
    }
  }, [api]);

  useEffect(() => {
    // Check system status on component mount and periodically
    checkSystemStatus();
    const interval = setInterval(checkSystemStatus, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, [checkSystemStatus]);

  const availableStrategies: Record<RouterStrategy, boolean> = {
    hybrid: systemStatus?.availableRouters?.hybrid ?? true,
    rule_based: systemStatus?.availableRouters?.hybrid_foundry_apim ?? systemStatus?.availableRouters?.rule_based ?? true,
    bert: systemStatus?.availableRouters?.bert ?? false,
    phi: systemStatus?.availableRouters?.phi ?? false
  };

  const getConnectionStatusMessage = () => {
    if (!systemStatus) return 'Checking connection...';
    
    switch (systemStatus.systemHealth) {
      case 'healthy':
        return '🟢 All systems operational';
      case 'degraded':
        return '🟡 Some services unavailable';
      case 'error':
        return '🔴 API connection failed - using mock responses';
      default:
        return 'Unknown status';
    }
  };

  const getRouterStatusText = () => {
    if (!systemStatus) return 'Checking...';
    
    const availableRouters = Object.entries(systemStatus.availableRouters)
      .filter(([_, available]) => available)
      .map(([name]) => name);
    
    if (availableRouters.length === 0) {
      return 'No routers available';
    }
    
    return `Active: ${availableRouters.join(', ')}`;
  };

  const addExchange = (userMsg: string, aiResponse: string, source: string, responseTime: number, metadata?: ConversationMetadata) => {
    const newExchange: ConversationExchange = {
      timestamp: new Date(),
      user_message: userMsg,
      ai_response: aiResponse,
      source,
      response_time: responseTime,
      exchange_number: conversationHistory.length + 1,
      metadata,
      model_switched: sessionStats.last_model !== null && sessionStats.last_model !== source
    };

    // Track if this was a fallback/mock response
    const isFallback = source === 'mock' || metadata?.error === true;

    // Add to conversation history (keep last 15 exchanges)
    setConversationHistory(prev => [...prev, newExchange].slice(-15));
    
    // Update session stats
    setSessionStats(prev => ({
      ...prev,
      total_exchanges: prev.total_exchanges + 1,
      model_switches: newExchange.model_switched ? prev.model_switches + 1 : prev.model_switches,
      last_model: source,
      fallback_used: isFallback ? prev.fallback_used + 1 : prev.fallback_used,
      context_preserved: metadata?.context_used ? prev.context_preserved + 1 : prev.context_preserved
    }));

    // Add to performance history
    const performanceMetric: PerformanceMetrics = {
      timestamp: new Date(),
      response_time: responseTime,
      source,
      context_used: metadata?.context_used || false,
      strategy: selectedStrategy,
      success: !metadata?.error
    };

    setPerformanceHistory(prev => [...prev, performanceMetric].slice(-50)); // Keep last 50
  };

  const clearContext = async () => {
    setConversationHistory([]);
    setPerformanceHistory([]);
    setSessionStats({
      start_time: new Date(),
      total_exchanges: 0,
      model_switches: 0,
      last_model: null,
      context_preserved: 0,
      fallback_used: 0
    });
    
    // Clear API context (both frontend and backend)
    try {
      await api.clearContext();
      console.log('✅ Session context cleared successfully');
    } catch (error) {
      console.warn('⚠️ Failed to clear backend context:', error);
    }
  };

  const getConversationInsights = (): ConversationInsights | null => {
    if (conversationHistory.length === 0) return null;

    const responseTimes = conversationHistory.map(ex => ex.response_time);
    const sources = conversationHistory.map(ex => ex.source);
    const sourceDistribution: Record<string, number> = {};
    
    sources.forEach(source => {
      sourceDistribution[source] = (sourceDistribution[source] || 0) + 1;
    });

    const contextUsedCount = conversationHistory.filter(ex => ex.metadata?.context_used).length;

    return {
      total_exchanges: conversationHistory.length,
      avg_response_time: responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length,
      fastest_response: Math.min(...responseTimes),
      slowest_response: Math.max(...responseTimes),
      source_distribution: sourceDistribution,
      model_switches: sessionStats.model_switches,
      context_usage_rate: (contextUsedCount / conversationHistory.length) * 100,
      session_duration: (new Date().getTime() - sessionStats.start_time.getTime()) / (1000 * 60)
    };
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🤖 Hybrid AI Router - Multi-Turn Demo</h1>
        <p>Advanced context-aware routing with multiple strategies</p>
        <div className="connection-status">
          <span className="status-indicator">{getConnectionStatusMessage()}</span>
          <div className="router-status">
            <small>{getRouterStatusText()}</small>
          </div>
        </div>
      </header>

      <div className="app-layout">
        <main className="chat-section">
          <ChatInterface
            conversationHistory={conversationHistory}
            selectedStrategy={selectedStrategy}
            sessionStats={sessionStats}
            onAddExchange={addExchange}
            onClearContext={clearContext}
            apiConnected={apiConnected}
            api={api}
          />
        </main>

        <aside className="analytics-section">
          <Analytics
            selectedStrategy={selectedStrategy}
            availableStrategies={availableStrategies}
            onStrategyChange={setSelectedStrategy}
            conversationInsights={getConversationInsights()}
            conversationHistory={conversationHistory}
            systemStatus={systemStatus}
            performanceHistory={performanceHistory}
          />
        </aside>
      </div>
    </div>
  );
};

export default App;