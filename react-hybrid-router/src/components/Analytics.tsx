import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { RouterStrategy, ConversationInsights, ConversationExchange, SystemStatus, PerformanceMetrics } from '../types';
import './Analytics.css';

interface AnalyticsProps {
  selectedStrategy: RouterStrategy;
  availableStrategies: Record<RouterStrategy, boolean>;
  onStrategyChange: (strategy: RouterStrategy) => void;
  conversationInsights: ConversationInsights | null;
  conversationHistory: ConversationExchange[];
  systemStatus: SystemStatus | null;
  performanceHistory: PerformanceMetrics[];
}

const Analytics: React.FC<AnalyticsProps> = ({
  selectedStrategy,
  availableStrategies,
  onStrategyChange,
  conversationInsights,
  conversationHistory,
  systemStatus,
  performanceHistory
}) => {
  const strategyDescriptions = {
    hybrid: '🎭 Full hybrid router with 3-tier routing (Local → APIM → Foundry)',
    rule_based: '📋 Pattern-based routing using query characteristics',
    bert: '🧠 BERT ML model for intelligent query classification',
    phi: '🔬 PHI small language model for query routing'
  };

  const getSystemHealthColor = () => {
    if (!systemStatus) return '#999';
    switch (systemStatus.systemHealth) {
      case 'healthy': return '#4CAF50';
      case 'degraded': return '#FF9800';
      case 'error': return '#f44336';
      default: return '#999';
    }
  };

  const getPerformanceTrend = () => {
    if (performanceHistory.length < 3) return null;
    const recent = performanceHistory.slice(-5);
    const avgTime = recent.reduce((sum, p) => sum + p.response_time, 0) / recent.length;
    return avgTime;
  };

  return (
    <div className="analytics">
      <h2>📊 Configuration & Analytics</h2>
      
      <div className="strategy-selector">
        <h3>🎛️ Routing Strategy</h3>
        <label htmlFor="strategy-select" className="strategy-label">
          Select routing strategy:
        </label>
        <select 
          id="strategy-select"
          value={selectedStrategy} 
          onChange={(e) => onStrategyChange(e.target.value as RouterStrategy)}
        >
          {Object.entries(availableStrategies)
            .filter(([, available]) => available)
            .map(([strategy]) => (
              <option key={strategy} value={strategy}>
                {strategy.replace('_', ' ').toUpperCase()}
              </option>
            ))
          }
        </select>
        <p className="strategy-description">
          {strategyDescriptions[selectedStrategy]}
        </p>
      </div>

      <div className="system-status">
        <h3>🔧 System Status</h3>
        <div
          className={`status-indicator ${systemStatus ? `status-${systemStatus.systemHealth}` : 'status-unknown'}`}
        >
          {systemStatus ? systemStatus.systemHealth.toUpperCase() : 'CHECKING...'}
        </div>
        <div className="available-routers">
          <h4>Available Routers:</h4>
          {Object.entries(availableStrategies).map(([strategy, available]) => (
            <div key={strategy} className="system-status-row">
              {available ? '✅' : '❌'} {strategy.replace('_', ' ').toUpperCase()}
            </div>
          ))}
        </div>
        
        {systemStatus?.hybrid_router_targets && (
          <div className="hybrid-targets">
            <h4>Hybrid Router Targets:</h4>
            {Object.entries(systemStatus.hybrid_router_targets).map(([target, available]) => (
              <div key={target} className="system-status-row">
                {available ? '✅' : '❌'} {target.replace('_', ' ').toUpperCase()}
              </div>
            ))}
          </div>
        )}
      </div>

      {conversationInsights && (
        <>
          <div className="conversation-metrics">
            <h3>📈 Conversation Metrics</h3>
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-value">{conversationInsights.total_exchanges}</div>
                <div className="metric-label">Exchanges</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{conversationInsights.model_switches}</div>
                <div className="metric-label">Model Switches</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{conversationInsights.avg_response_time.toFixed(3)}s</div>
                <div className="metric-label">Avg Response</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{conversationInsights.context_usage_rate.toFixed(1)}%</div>
                <div className="metric-label">Context Usage</div>
              </div>
            </div>
            
            <div className="additional-metrics">
              <div className="metric-row">
                <span>Session Duration:</span>
                <span>{conversationInsights.session_duration.toFixed(1)} min</span>
              </div>
              <div className="metric-row">
                <span>Fastest Response:</span>
                <span>{conversationInsights.fastest_response.toFixed(3)}s</span>
              </div>
              <div className="metric-row">
                <span>Slowest Response:</span>
                <span>{conversationInsights.slowest_response.toFixed(3)}s</span>
              </div>
            </div>
          </div>

          <div className="source-distribution">
            <h3>🎯 Model Usage Distribution</h3>
            {conversationInsights.source_distribution && (
              <>
                {/* Pie Chart */}
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={Object.entries(conversationInsights.source_distribution).map(([source, count]) => ({
                          name: source.toUpperCase(),
                          value: count,
                          percentage: ((count / conversationInsights.total_exchanges) * 100).toFixed(1)
                        }))}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percentage }) => `${name} (${percentage}%)`}
                        outerRadius={60}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {Object.entries(conversationInsights.source_distribution).map((entry, index) => {
                          const colors = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];
                          return <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />;
                        })}
                      </Pie>
                      <Tooltip 
                        formatter={(value: number, name: string) => [
                          `${value} exchanges (${((value / conversationInsights.total_exchanges) * 100).toFixed(1)}%)`,
                          name
                        ]}
                      />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                
                {/* Table View */}
                <div className="distribution-table">
                  {Object.entries(conversationInsights.source_distribution)
                    .sort(([,a], [,b]) => b - a)
                    .map(([source, count]) => (
                      <div key={source} className="source-distribution-row">
                        <span className="source-name">{source.toUpperCase()}</span>
                        <span className="source-count">{count}</span>
                        <span className="source-percentage">
                          ({((count / conversationInsights.total_exchanges) * 100).toFixed(1)}%)
                        </span>
                      </div>
                    ))
                  }
                </div>
              </>
            )}
          </div>

          {performanceHistory.length > 3 && (
            <div className="performance-trend">
              <h3>⚡ Performance Trend</h3>
              <div className="trend-summary">
                <div className="trend-metric">
                  <span>Recent Avg:</span>
                  <span>{getPerformanceTrend()?.toFixed(3)}s</span>
                </div>
                <div className="trend-metric">
                  <span>Success Rate:</span>
                  <span>
                    {((performanceHistory.filter(p => p.success).length / performanceHistory.length) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      <div className="conversation-starters">
        <h3>🚀 Example Queries by Complexity</h3>
        <div className="conversation-starters-grid">
          <div className="complexity-category">
            <h4>Simple (Local):</h4>
            <ul>
              <li>"Hello there!"</li>
              <li>"What is 25 + 17?"</li>
              <li>"What time is it?"</li>
            </ul>
          </div>
          <div className="complexity-category">
            <h4>Moderate:</h4>
            <ul>
              <li>"Explain machine learning"</li>
              <li>"Compare Python and Java"</li>
              <li>"What is cloud computing?"</li>
            </ul>
          </div>
          <div className="complexity-category">
            <h4>Complex (Cloud):</h4>
            <ul>
              <li>"Analyze hybrid AI architecture"</li>
              <li>"Write a business case for AI adoption"</li>
              <li>"Design a recommendation system"</li>
            </ul>
          </div>
        </div>
      </div>

      {conversationHistory.length > 0 && (
        <div className="session-summary">
          <h3>📋 Session Summary</h3>
          <div className="summary-stats">
            <div>Started: {conversationHistory[0]?.timestamp.toLocaleTimeString()}</div>
            <div>Last Activity: {conversationHistory[conversationHistory.length - 1]?.timestamp.toLocaleTimeString()}</div>
            <div>Strategy: {selectedStrategy.replace('_', ' ').toUpperCase()}</div>
            <div>Context Enabled: ✅</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Analytics;