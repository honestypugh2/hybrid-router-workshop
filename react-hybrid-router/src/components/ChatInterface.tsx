import React, { useState } from 'react';
import { ConversationExchange, RouterStrategy, SessionStats, ConversationMetadata } from '../types';
import { HybridRouterAPI } from '../services/api';

interface ChatInterfaceProps {
  conversationHistory: ConversationExchange[];
  selectedStrategy: RouterStrategy;
  sessionStats: SessionStats;
  onAddExchange: (userMsg: string, aiResponse: string, source: string, responseTime: number, metadata?: ConversationMetadata) => void;
  onClearContext: () => void;
  apiConnected: boolean;
  api: HybridRouterAPI;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  conversationHistory,
  selectedStrategy,
  sessionStats,
  onAddExchange,
  onClearContext,
  apiConnected,
  api
}) => {
  const [userInput, setUserInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  const sourceIcons = {
    local: '🟢 LOCAL',
    apim: '🔵 APIM', 
    foundry: '🟣 FOUNDRY',
    hybrid: '🤖 HYBRID',
    azure: '🟠 AZURE',
    cloud: '☁️ CLOUD',
    error: '🔴 ERROR',
    mock: '⚪ MOCK'
  };

  const generateMockResponse = (query: string): [string, string] => {
    const responses = [
      [`Mock response for "${query.substring(0, 30)}..." - This would be processed by the ${selectedStrategy} router.`, 'mock'],
      [`Simulated AI response using ${selectedStrategy} strategy for: ${query.substring(0, 40)}...`, 'azure'],
      [`Demo response: Processing "${query}" through hybrid routing system.`, 'local']
    ];
    return responses[Math.floor(Math.random() * responses.length)] as [string, string];
  };

  const handleSendMessage = async () => {
    if (!userInput.trim() || isProcessing) return;

    setIsProcessing(true);
    const startTime = Date.now();
    
    try {
      // Use the actual API with context enabled
      const result = await api.routeQuery(userInput, selectedStrategy, true);
      
      const metadata: ConversationMetadata = {
        // Include the routing metadata first, then override specific fields
        ...(result.metadata || {}),
        strategy: selectedStrategy,
        context_used: result.metadata?.context_used || conversationHistory.length > 0,
        routing_info: result.metadata,
        api_connected: apiConnected,
        context_length: conversationHistory.length,
        query_length: userInput.length,
      };

      // Debug: Log the metadata structure to console
      console.log('🔍 Metadata received:', {
        resultMetadata: result.metadata,
        finalMetadata: metadata,
        source: result.source
      });

      onAddExchange(userInput, result.response, result.source, result.responseTime, metadata);
      setUserInput('');
    } catch (error) {
      console.error('Error processing message:', error);
      
      // Fallback to mock response on error
      const [response, source] = generateMockResponse(userInput);
      const responseTime = (Date.now() - startTime) / 1000;
      
      const metadata: ConversationMetadata = {
        strategy: selectedStrategy,
        context_used: conversationHistory.length > 0,
        error: true,
        error_message: error instanceof Error ? error.message : 'Unknown error',
        api_connected: false,
        context_length: conversationHistory.length,
        query_length: userInput.length
      };

      onAddExchange(userInput, response, source, responseTime, metadata);
      setUserInput('');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleExampleQuery = (query: string) => {
    setUserInput(query);
  };

  const getRouterUsed = (exchange: ConversationExchange): string => {
    // Check both routing_info and direct metadata for router information
    const routingInfo = exchange.metadata?.routing_info;
    const metadata = exchange.metadata;
    
    if (routingInfo?.analysis?.router_used) {
      return routingInfo.analysis.router_used.replace('_', ' ').toUpperCase();
    }
    if ((metadata as any)?.analysis?.router_used) {
      return (metadata as any).analysis.router_used.replace('_', ' ').toUpperCase();
    }
    if (routingInfo?.strategy) {
      return routingInfo.strategy.replace('_', ' ').toUpperCase();
    }
    if (metadata?.strategy) {
      return metadata.strategy.replace('_', ' ').toUpperCase();
    }
    return 'UNKNOWN';
  };

  return (
    <div className="chat-interface">
      <h2>💬 Intelligent Multi-Turn Chat</h2>
      
      {conversationHistory.length > 0 && (
        <div className="context-status">
          <p>🧠 Context: {conversationHistory.length} exchanges in memory | Strategy: {selectedStrategy.replace('_', ' ').toUpperCase()} | Last model: {sessionStats.last_model?.toUpperCase() || 'NONE'}</p>
        </div>
      )}

      <div className="input-section">
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          placeholder="Ask a question, follow up, or start a new topic..."
          disabled={isProcessing}
          className="chat-input"
        />
        <div className="button-row">
          <button 
            onClick={handleSendMessage} 
            disabled={!userInput.trim() || isProcessing}
            className="send-button"
          >
            {isProcessing ? '⏳ Processing...' : '💬 Send Message'}
          </button>
          <button onClick={onClearContext} className="clear-button">
            🧹 Clear Context
          </button>
        </div>
      </div>

      <div className="example-queries">
        <h3>🚀 Quick Starters</h3>
        <div className="example-buttons">
          <button onClick={() => handleExampleQuery('Hello! How can you help me today?')}>👋 Greeting</button>
          <button onClick={() => handleExampleQuery('What are the best practices for enterprise AI deployment?')}>💼 Enterprise</button>
          <button onClick={() => handleExampleQuery('Can you analyze the implications of quantum computing?')}>🔍 Complex Analysis</button>
          <button onClick={() => handleExampleQuery('Can you elaborate on that point?')}>🔄 Follow-up</button>
          <button onClick={() => handleExampleQuery('What was my previous question about?')}>🧠 Context Test</button>
        </div>
      </div>

      <div className="conversation-display">
        <h3>📜 Conversation Flow</h3>
        {conversationHistory.length === 0 ? (
          <p>👋 Start a conversation! The system will remember context across exchanges and intelligently switch between models.</p>
        ) : (
          <div className="exchanges">
            {conversationHistory.slice(-8).reverse().map((exchange, index) => (
              <div key={exchange.exchange_number} className="exchange">
                <div className="exchange-meta">
                  <span>Exchange #{exchange.exchange_number}</span>
                  {exchange.model_switched && <span>🔄 Model switched</span>}
                  <span>⏱️ {exchange.response_time.toFixed(3)}s</span>
                </div>
                <div className="user-message">
                  <strong>🧑 You:</strong> {exchange.user_message}
                </div>
                <div className="ai-response">
                  <strong>🤖 AI [{sourceIcons[exchange.source as keyof typeof sourceIcons] || exchange.source.toUpperCase()}]:</strong> {exchange.ai_response}
                </div>
                
                {/* Enhanced routing details */}
                <div className="routing-details">
                  <div className="routing-summary">
                    <span className="router-used">Router: {getRouterUsed(exchange)}</span>
                    {/* Confidence from multiple possible sources */}
                    {(
                      (exchange.metadata?.routing_info as any)?.confidence || 
                      (exchange.metadata as any)?.confidence ||
                      (exchange.metadata as any)?.ml_confidence
                    ) && (
                      <span className="confidence">Confidence: {(
                        ((exchange.metadata?.routing_info as any)?.confidence || 
                        (exchange.metadata as any)?.confidence ||
                        (exchange.metadata as any)?.ml_confidence || 0) * 100
                      ).toFixed(1)}%</span>
                    )}
                    {/* Reason from multiple possible sources */}
                    {(() => {
                      const reason = 
                        (exchange.metadata?.routing_info as any)?.reason || 
                        (exchange.metadata as any)?.routing_decision ||
                        (exchange.metadata as any)?.reason ||
                        (exchange.metadata?.routing_info as any)?.analysis?.reason ||
                        (exchange.metadata as any)?.analysis?.reason;
                      
                      return reason && (
                        <span className="routing-reason">Reason: {reason}</span>
                      );
                    })()}
                  </div>
                  
                  {/* Expandable routing details */}
                  {(exchange.metadata?.routing_info || exchange.metadata) && (
                    <details className="routing-details-expandable">
                      <summary>🔍 Routing Details</summary>
                      <div className="routing-info-grid">
                        <div>
                          <strong>Strategy:</strong> {(exchange.metadata?.routing_info as any)?.strategy || exchange.metadata?.strategy || 'N/A'}
                        </div>
                        <div>
                          <strong>Target:</strong> {(exchange.metadata?.routing_info as any)?.target || exchange.source}
                        </div>
                        <div>
                          <strong>Source:</strong> {(exchange.metadata?.routing_info as any)?.source || exchange.source}
                        </div>
                        <div>
                          <strong>Reason:</strong> {(() => {
                            const reason = 
                              (exchange.metadata?.routing_info as any)?.reason || 
                              (exchange.metadata as any)?.routing_decision ||
                              (exchange.metadata as any)?.reason ||
                              (exchange.metadata?.routing_info as any)?.analysis?.reason ||
                              (exchange.metadata as any)?.analysis?.reason ||
                              (exchange.metadata as any)?.routing_reason;
                            return reason || 'Not specified';
                          })()}
                        </div>
                        <div>
                          <strong>Context Used:</strong> {(exchange.metadata?.context_used || (exchange.metadata as any)?.context_used) ? '✅' : '❌'}
                        </div>
                        <div>
                          <strong>Exchange #:</strong> {exchange.exchange_number}
                        </div>
                        <div>
                          <strong>Response Time:</strong> {exchange.response_time.toFixed(3)}s
                        </div>
                        <div className="analysis-info">
                          <strong>Analysis:</strong>
                          <ul>
                            {/* Check for greeting detection in multiple places */}
                            {(
                              (exchange.metadata?.routing_info as any)?.analysis?.is_greeting || 
                              (exchange.metadata as any)?.analysis?.is_greeting ||
                              (exchange.metadata as any)?.is_greeting ||
                              exchange.user_message.toLowerCase().includes('hello') ||
                              exchange.user_message.toLowerCase().includes('hi')
                            ) && <li>✅ Greeting detected</li>}
                            
                            {/* Check for calculation detection */}
                            {(
                              (exchange.metadata?.routing_info as any)?.analysis?.is_calculation || 
                              (exchange.metadata as any)?.analysis?.is_calculation ||
                              (exchange.metadata as any)?.is_calculation ||
                              /\d+\s*[+\-*/]\s*\d+/.test(exchange.user_message)
                            ) && <li>✅ Calculation detected</li>}
                            
                            {/* ML Confidence from various sources */}
                            {(
                              (exchange.metadata?.routing_info as any)?.analysis?.ml_confidence || 
                              (exchange.metadata as any)?.analysis?.ml_confidence ||
                              (exchange.metadata as any)?.ml_confidence ||
                              (exchange.metadata as any)?.confidence
                            ) && (
                              <li>ML Confidence: {(
                                ((exchange.metadata?.routing_info as any)?.analysis?.ml_confidence || 
                                (exchange.metadata as any)?.analysis?.ml_confidence ||
                                (exchange.metadata as any)?.ml_confidence ||
                                (exchange.metadata as any)?.confidence || 0) * 100
                              ).toFixed(1)}%</li>
                            )}
                            
                            {/* Model switching indicator */}
                            {(exchange.model_switched || (exchange.metadata as any)?.model_switched) && <li>🔄 Model switched this exchange</li>}
                            
                            {/* Error indicator */}
                            {(exchange.metadata as any)?.error && <li>❌ Error occurred</li>}
                            
                            {/* Routing decision info */}
                            {(exchange.metadata as any)?.routing_decision && <li>🎯 Decision: {(exchange.metadata as any).routing_decision}</li>}
                            
                            {/* Router reason from metadata */}
                            {(() => {
                              const reason = 
                                (exchange.metadata?.routing_info as any)?.reason || 
                                (exchange.metadata as any)?.reason ||
                                (exchange.metadata as any)?.routing_reason ||
                                (exchange.metadata?.routing_info as any)?.analysis?.reason ||
                                (exchange.metadata as any)?.analysis?.reason;
                              
                              return reason && <li>💭 Reason: {reason}</li>;
                            })()}
                            
                            {/* Strategy used */}
                            <li>📊 Strategy: {exchange.metadata?.strategy || selectedStrategy}</li>
                            
                            {/* Source information */}
                            <li>🎯 Source: {exchange.source}</li>
                            
                            {/* Context information */}
                            <li>🧠 Context: {exchange.metadata?.context_used ? 'Used' : 'Not used'} ({conversationHistory.length} exchanges)</li>
                            
                            {/* Query complexity */}
                            <li>📏 Query length: {exchange.user_message.length} characters</li>
                            
                            {/* API connection status */}
                            {exchange.metadata?.api_connected !== undefined && (
                              <li>🔗 API: {exchange.metadata.api_connected ? 'Connected' : 'Disconnected'}</li>
                            )}
                          </ul>
                        </div>
                      </div>
                    </details>
                  )}
                </div>
                
                {exchange.metadata?.context_used && (
                  <div className="context-indicator">🧠 Used conversation context</div>
                )}
                
                {exchange.metadata?.error && (
                  <div className="error-indicator">⚠️ Error: {exchange.metadata.error_message}</div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatInterface;