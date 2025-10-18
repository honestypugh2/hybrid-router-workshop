// Enhanced API service for hybrid router backend (mirroring streamlit_multiturn_demo.py)
import { RouterStrategy, SystemStatus, RoutingMetadata } from '../types';

export class HybridRouterAPI {
  private baseUrl: string;
  private enableMockFallback: boolean;
  private sessionId: string;

  constructor(baseUrl?: string) {
    // Use environment variable or fallback to default
    this.baseUrl = baseUrl || process.env.REACT_APP_API_BASE_URL || 'http://localhost:8080';
    this.enableMockFallback = process.env.REACT_APP_ENABLE_MOCK_FALLBACK === 'true';
    this.sessionId = this.getOrCreateSessionId();
    
    if (process.env.REACT_APP_DEBUG_MODE === 'true') {
      console.log('🔧 HybridRouterAPI initialized:', {
        baseUrl: this.baseUrl,
        enableMockFallback: this.enableMockFallback,
        sessionId: this.sessionId
      });
    }
  }

  private getOrCreateSessionId(): string {
    // Generate or retrieve session ID for context tracking
    let sessionId = sessionStorage.getItem('hybrid-router-session-id');
    if (!sessionId) {
      sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      sessionStorage.setItem('hybrid-router-session-id', sessionId);
    }
    return sessionId;
  }

  async routeQuery(query: string, strategy: RouterStrategy, contextEnabled: boolean = true): Promise<{
    response: string;
    source: string;
    responseTime: number;
    metadata: RoutingMetadata;
  }> {
    const startTime = Date.now();
    
    if (process.env.REACT_APP_DEBUG_MODE === 'true') {
      console.log('🚀 Routing query:', { 
        query: query.substring(0, 50), 
        strategy, 
        contextEnabled,
        sessionId: this.sessionId 
      });
    }
    
    try {
      // Determine endpoint based on strategy
      let endpoint = '/route';
      if (strategy === 'bert') {
        endpoint = '/route/bert';
      } else if (strategy === 'phi') {
        endpoint = '/route/phi';
      }
      
      const requestBody = {
        query,
        strategy,
        session_id: contextEnabled ? this.sessionId : undefined
      };
      
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status} - ${response.statusText}`);
      }

      const data = await response.json();
      const responseTime = (Date.now() - startTime) / 1000;

      if (process.env.REACT_APP_DEBUG_MODE === 'true') {
        console.log('✅ Query successful:', {
          source: data.source,
          responseTime: data.responseTime || responseTime,
          metadata: data.metadata
        });
      }

      return {
        response: data.response,
        source: data.source,
        responseTime: data.responseTime || responseTime,
        metadata: data.metadata || this.createFallbackMetadata(strategy, data.source, responseTime)
      };
    } catch (error) {
      console.warn('⚠️ API request failed:', error);
      
      // Only use mock response if explicitly enabled
      if (this.enableMockFallback) {
        console.log('🔄 Falling back to mock response');
        return this.getMockResponse(query, strategy, startTime);
      } else {
        // Re-throw the error to let the UI handle it properly
        throw new Error(`Failed to connect to hybrid router API: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }
  }

  private createFallbackMetadata(strategy: RouterStrategy, source: string, responseTime: number): RoutingMetadata {
    return {
      strategy,
      target: source,
      confidence: 0.5,
      reason: 'API response without detailed metadata',
      source,
      response_time: responseTime,
      success: true,
      timestamp: new Date().toISOString(),
      context_used: false
    };
  }

  private getMockResponse(query: string, strategy: RouterStrategy, startTime: number) {
    const responseTime = (Date.now() - startTime) / 1000;
    
    console.log('🎭 Generating mock response (API unavailable)');
    
    // Generate context-aware mock response based on query
    const queryLower = query.toLowerCase();
    let mockSource = 'mock';
    let mockResponse = '';
    
    if (queryLower.includes('hello') || queryLower.includes('hi')) {
      mockSource = 'local';
      mockResponse = `Hello! This is a mock response from the ${strategy} router. The actual API is unavailable.`;
    } else if (queryLower.includes('enterprise') || queryLower.includes('business')) {
      mockSource = 'apim';
      mockResponse = `Enterprise query detected: This would be processed through APIM with ${strategy} routing.`;
    } else if (queryLower.includes('analyze') || queryLower.includes('complex')) {
      mockSource = 'foundry';
      mockResponse = `Complex analysis requested: Azure AI Foundry would handle this with ${strategy} strategy.`;
    } else {
      mockSource = 'azure';
      mockResponse = `General query processed with ${strategy} routing strategy. This is a mock response.`;
    }
    
    const metadata: RoutingMetadata = {
      strategy,
      target: mockSource,
      confidence: 0.5,
      reason: 'API unavailable - using fallback mock response',
      source: mockSource,
      response_time: responseTime,
      success: true,
      timestamp: new Date().toISOString(),
      context_used: false,
      analysis: {
        router_used: strategy,
        is_greeting: queryLower.includes('hello') || queryLower.includes('hi'),
        is_calculation: false,
        ml_confidence: 0.5
      }
    };
    
    return {
      response: `⚠️ **Mock Response** (API Unavailable)\n\n${mockResponse}\n\nQuery: "${query.substring(0, 100)}${query.length > 100 ? '...' : ''}"\n\nTo use actual AI responses, please start the backend API server.`,
      source: mockSource,
      responseTime,
      metadata
    };
  }

  async getSystemStatus(): Promise<SystemStatus> {
    try {
      if (process.env.REACT_APP_DEBUG_MODE === 'true') {
        console.log('🔍 Checking system status...');
      }
      
      const response = await fetch(`${this.baseUrl}/status`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Status check failed: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (process.env.REACT_APP_DEBUG_MODE === 'true') {
        console.log('✅ System status:', data);
      }
      
      return {
        availableRouters: data.availableRouters || {},
        systemHealth: data.systemHealth || 'error',
        hybrid_router_targets: data.hybrid_router_targets,
        capabilities: data.capabilities
      };
    } catch (error) {
      console.warn('⚠️ System status check failed:', error);
      
      return {
        availableRouters: {
          hybrid: false,
          rule_based: false,
          bert: false,
          phi: false,
          local: false,
          cloud: false
        },
        systemHealth: 'error'
      };
    }
  }

  async getCapabilities(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/capabilities`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Capabilities check failed: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.warn('⚠️ Capabilities check failed:', error);
      return {
        available_targets: {},
        hybrid_routing: false,
        bert_routing: false,
        phi_routing: false,
        context_management: false,
        multi_turn_support: false
      };
    }
  }

  // Clear session context
  async clearContext(): Promise<void> {
    try {
      // Clear context on the backend
      await fetch(`${this.baseUrl}/session/${this.sessionId}/context`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    } catch (error) {
      console.warn('⚠️ Failed to clear server-side context:', error);
    }
    
    // Always clear local session ID and create new one
    sessionStorage.removeItem('hybrid-router-session-id');
    this.sessionId = this.getOrCreateSessionId();
  }

  // Get session context from backend
  async getSessionContext(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/session/${this.sessionId}/context`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        if (response.status === 404) {
          return { session_id: this.sessionId, chat_history: [], total_exchanges: 0 };
        }
        throw new Error(`Session context fetch failed: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.warn('⚠️ Failed to get session context:', error);
      return { session_id: this.sessionId, chat_history: [], total_exchanges: 0 };
    }
  }

  // Get session insights from backend
  async getSessionInsights(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/session/${this.sessionId}/insights`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        if (response.status === 404) {
          return { message: "No conversation data available" };
        }
        throw new Error(`Session insights fetch failed: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.warn('⚠️ Failed to get session insights:', error);
      return { message: "Failed to retrieve insights" };
    }
  }

  // Get current session ID
  getSessionId(): string {
    return this.sessionId;
  }
}