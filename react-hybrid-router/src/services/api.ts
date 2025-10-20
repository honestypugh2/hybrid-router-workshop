// Enhanced API service for hybrid router backend with dual backend support
import { RouterStrategy, SystemStatus, RoutingMetadata } from '../types';

interface BackendConfig {
  url: string;
  endpointPattern: 'basic' | 'enhanced'; // basic = /route, enhanced = /api/*
  port: number;
  name: string;
}

export class HybridRouterAPI {
  private backends: BackendConfig[];
  private enableMockFallback: boolean;
  private sessionId: string;
  private activeBackend: BackendConfig | null = null;

  constructor() {
    // Configure multiple backends to try
    this.backends = [
      {
        url: 'http://localhost:8000',
        endpointPattern: 'enhanced',
        port: 8000,
        name: 'Enhanced Backend (root level backend_api.py)'
      },
      {
        url: 'http://localhost:8080',
        endpointPattern: 'basic',
        port: 8080,
        name: 'Basic Backend (react directory backend_api.py)'
      }
    ];
    
    this.enableMockFallback = process.env.REACT_APP_ENABLE_MOCK_FALLBACK === 'true';
    this.sessionId = this.getOrCreateSessionId();
    
    console.log('🔧 HybridRouterAPI initialized with dual backend support:', {
      backends: this.backends,
      enableMockFallback: this.enableMockFallback,
      sessionId: this.sessionId
    });
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

  private async testBackendConnection(backend: BackendConfig): Promise<boolean> {
    try {
      const healthEndpoint = backend.endpointPattern === 'enhanced' ? '/api/health' : '/health';
      const response = await fetch(`${backend.url}${healthEndpoint}`, {
        method: 'GET',
        signal: AbortSignal.timeout(3000) // 3 second timeout
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  private async findActiveBackend(): Promise<BackendConfig | null> {
    if (this.activeBackend) {
      // Test if current active backend is still working
      if (await this.testBackendConnection(this.activeBackend)) {
        return this.activeBackend;
      }
    }

    // Test all backends to find a working one
    for (const backend of this.backends) {
      console.log(`🔍 Testing backend: ${backend.name} at ${backend.url}`);
      if (await this.testBackendConnection(backend)) {
        console.log(`✅ Found working backend: ${backend.name}`);
        this.activeBackend = backend;
        return backend;
      }
    }

    console.warn('❌ No working backends found');
    return null;
  }

  async routeQuery(query: string, strategy: RouterStrategy, contextEnabled: boolean = true): Promise<{
    response: string;
    source: string;
    responseTime: number;
    metadata: RoutingMetadata;
  }> {
    const startTime = Date.now();
    
    console.log('🚀 Routing query:', { 
      query: query.substring(0, 50), 
      strategy, 
      contextEnabled,
      sessionId: this.sessionId 
    });
    
    const backend = await this.findActiveBackend();
    if (!backend) {
      if (this.enableMockFallback) {
        return this.getMockResponse(query, strategy, Date.now() - startTime);
      }
      throw new Error('No backend servers are available');
    }

    try {
      // Configure request based on backend type
      let endpoint: string;
      let requestBody: any;

      if (backend.endpointPattern === 'enhanced') {
        // Enhanced backend (/api/* endpoints)
        endpoint = '/api/query';
        requestBody = {
          query,
          strategy,
          session_id: contextEnabled ? this.sessionId : undefined,
          context_enabled: contextEnabled
        };
      } else {
        // Basic backend (/route endpoints)
        endpoint = '/route';
        if (strategy === 'bert') {
          endpoint = '/route/bert';
        } else if (strategy === 'phi') {
          endpoint = '/route/phi';
        }
        requestBody = {
          query,
          strategy,
          session_id: contextEnabled ? this.sessionId : undefined
        };
      }
      
      const response = await fetch(`${backend.url}${endpoint}`, {
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

      console.log('✅ Query successful:', {
        backend: backend.name,
        source: data.source,
        responseTime: data.responseTime || responseTime,
        metadata: data.metadata
      });

      return {
        response: data.response,
        source: data.source,
        responseTime: data.responseTime || responseTime,
        metadata: {
          ...data.metadata,
          backend_used: backend.name,
          backend_pattern: backend.endpointPattern,
          backend_url: backend.url
        }
      };
    } catch (error) {
      console.warn(`⚠️ API request failed for ${backend.name}:`, error);
      
      // Mark this backend as failed and try to find another
      this.activeBackend = null;
      const fallbackBackend = await this.findActiveBackend();
      
      if (fallbackBackend && fallbackBackend !== backend) {
        console.log(`🔄 Retrying with fallback backend: ${fallbackBackend.name}`);
        return this.routeQuery(query, strategy, contextEnabled);
      }
      
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
    const backend = await this.findActiveBackend();
    
    if (!backend) {
      return {
        availableRouters: { hybrid: false, rule_based: false, bert: false, phi: false },
        systemHealth: 'error'
      };
    }

    try {
      const endpoint = backend.endpointPattern === 'enhanced' ? '/api/system-status' : '/status';
      const response = await fetch(`${backend.url}${endpoint}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('✅ System status retrieved:', {
        backend: backend.name,
        systemHealth: data.systemHealth,
        availableRouters: data.availableRouters
      });

      return {
        availableRouters: data.availableRouters || { hybrid: false, rule_based: false, bert: false, phi: false },
        systemHealth: data.systemHealth || 'unknown',
        hybrid_router_targets: data.hybrid_router_targets,
        capabilities: data.capabilities
      };
    } catch (error) {
      console.warn(`⚠️ Failed to get system status from ${backend.name}:`, error);
      
      // Try fallback backend
      this.activeBackend = null;
      const fallbackBackend = await this.findActiveBackend();
      
      if (fallbackBackend && fallbackBackend !== backend) {
        return this.getSystemStatus();
      }
      
      return {
        availableRouters: { hybrid: false, rule_based: false, bert: false, phi: false },
        systemHealth: 'error'
      };
    }
  }

  async getCapabilities(): Promise<any> {
    const backend = await this.findActiveBackend();
    
    if (!backend) {
      return {
        available_targets: {},
        hybrid_routing: false,
        bert_routing: false,
        phi_routing: false,
        context_management: false,
        multi_turn_support: false
      };
    }

    try {
      // Enhanced backend has capabilities endpoint, basic doesn't
      if (backend.endpointPattern === 'enhanced') {
        const response = await fetch(`${backend.url}/api/capabilities`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });

        if (response.ok) {
          return await response.json();
        }
      }
      
      // Fallback for basic backend or failed enhanced request
      return {
        available_targets: {
          local: true,
          apim: true,
          foundry: true,
          azure: true
        },
        hybrid_routing: true,
        bert_routing: backend.endpointPattern === 'basic',
        phi_routing: backend.endpointPattern === 'basic',
        context_management: true,
        multi_turn_support: true
      };
    } catch (error) {
      console.warn(`⚠️ Failed to get capabilities from ${backend.name}:`, error);
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
    const backend = await this.findActiveBackend();
    
    if (backend) {
      try {
        const endpoint = backend.endpointPattern === 'enhanced' 
          ? `/api/clear-context/${this.sessionId}`
          : `/session/${this.sessionId}/context`;
          
        await fetch(`${backend.url}${endpoint}`, {
          method: 'DELETE',
          headers: {
            'Content-Type': 'application/json',
          },
        });
      } catch (error) {
        console.warn('⚠️ Failed to clear server-side context:', error);
      }
    }
    
    // Always clear local session ID and create new one
    sessionStorage.removeItem('hybrid-router-session-id');
    this.sessionId = this.getOrCreateSessionId();
  }

  // Get session context from backend
  async getSessionContext(): Promise<any> {
    const backend = await this.findActiveBackend();
    
    if (!backend) {
      return { session_id: this.sessionId, chat_history: [], total_exchanges: 0 };
    }

    try {
      const endpoint = backend.endpointPattern === 'enhanced'
        ? `/api/conversation-history/${this.sessionId}`
        : `/session/${this.sessionId}/context`;
        
      const response = await fetch(`${backend.url}${endpoint}`, {
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
    const backend = await this.findActiveBackend();
    
    if (!backend) {
      return { message: "No backend available" };
    }

    try {
      const endpoint = backend.endpointPattern === 'enhanced'
        ? `/api/conversation-insights/${this.sessionId}`
        : `/session/${this.sessionId}/insights`;
        
      const response = await fetch(`${backend.url}${endpoint}`, {
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