import streamlit as st
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add project root directory for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Try to import hybrid router and context manager
try:
    from modules.hybrid_router import HybridFoundryAPIMRouter, HybridRouterConfig, create_hybrid_router_from_env
    from modules.context_manager import ConversationContextManager
    router_available = True
except ImportError:
    router_available = False

# Try to import additional routing modules
try:
    from modules.bert_router import BertQueryRouter, BertRouterConfig
    from modules.phi_router import PhiQueryRouter, PhiRouterConfig
    additional_routers_available = True
except ImportError:
    additional_routers_available = False

# Disable warnings that might clutter the Streamlit interface
import warnings
warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(
    page_title="Hybrid AI Router - Multi-Turn Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitMultiTurnDemo:
    """Advanced Streamlit demo with multi-turn conversation and context management."""
    
    def __init__(self):
        self.router = None
        
        # Use ConversationContextManager instead of custom ConversationContext
        self.context_manager = ConversationContextManager(
            session_id=f"streamlit_session_{int(time.time())}", 
            max_history=15
        )
        
        self.routing_stats = {
            "local": 0, "apim": 0, "foundry": 0, "azure": 0, "mock": 0, "error": 0
        }
        self.performance_history = []
        self.available_routers = {'hybrid': True, 'rule_based': False, 'bert': False, 'phi': False}
        self.selected_strategy = 'hybrid'  # Default to hybrid router
        self.init_router()
        self.init_additional_routers()
    
    def init_router(self):
        """Initialize the hybrid router with error handling."""
        if router_available:
            try:
                # Suppress stdout/stderr during router initialization
                import io
                import contextlib
                
                f = io.StringIO()
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    # Create router without session_id since ConversationContextManager integration was removed
                    self.router = create_hybrid_router_from_env()
                
                if self.router:
                    print(f"✅ Hybrid router initialized for session: {self.context_manager.session_id}")
                    self.available_routers['hybrid'] = True
                    return True
                else:
                    print("⚠️ Router creation failed")
                    self.available_routers['hybrid'] = False
                    return False
                    
            except Exception as e:
                # Don't use st.error here as it might not be available during init
                print(f"Hybrid router initialization failed: {e}")
                self.available_routers['hybrid'] = False
                return False
        
        self.available_routers['hybrid'] = False
        return False
    
    def init_additional_routers(self):
        """Initialize additional routing strategies."""
        if not additional_routers_available:
            return
        
        # Rule-based routing is now handled through hybrid router
        self.available_routers['rule_based'] = True
        
        # Initialize BERT router
        try:
            bert_config = BertRouterConfig(
                model_path="./notebooks/mobilbert_query_router_trained",
                confidence_threshold=0.7
            )
            self.bert_router = BertQueryRouter(bert_config)
            self.available_routers['bert'] = True
        except Exception as e:
            print(f"BERT router initialization failed: {e}")
            self.bert_router = None
            self.available_routers['bert'] = False
        
        # Initialize PHI router
        try:
            phi_config = PhiRouterConfig(model_path="./notebooks/phi_router_model")
            self.phi_router = PhiQueryRouter(phi_config)
            self.available_routers['phi'] = True
        except Exception as e:
            print(f"PHI router initialization failed: {e}")
            self.phi_router = None
            self.available_routers['phi'] = False
    
    def route_with_selected_strategy(self, query: str) -> Tuple[str, str, float, Dict]:
        """Route query using selected strategy and generate response."""
        start_time = time.time()
        
        if not self.router:
            # Generate mock response without router
            response_time = time.time() - start_time
            mock_response, source = self._generate_contextual_mock_response(query)
            
            metadata = {
                'strategy': 'mock',
                'target': source,
                'confidence': 0.5,
                'reason': 'Router not available',
                'source': source,
                'response_time': response_time,
                'success': True,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return mock_response, source, response_time, metadata
        
        try:
            # Get routing decision using enhanced strategy matching
            target, reason, confidence = self._route_with_strategy(query, self.selected_strategy)
            
            # Get analysis information for router used details
            analysis = {}
            if self.router and self.selected_strategy in ['hybrid', 'rule_based']:
                try:
                    analysis = self.router.analyze_query_for_hybrid_routing(query)
                except Exception:
                    analysis = {'router_used': self.selected_strategy}
            else:
                analysis = {'router_used': self.selected_strategy}
            
            # Generate response using the router
            response = self.router.route(query, show_reasoning=False)
            
            # Parse response to extract source
            actual_source = target
            if response.startswith("["):
                end_bracket = response.find("]")
                if end_bracket != -1:
                    actual_source = response[1:end_bracket].lower()
                    response = response[end_bracket+1:].strip()
            
            response_time = time.time() - start_time
            
            metadata = {
                'strategy': self.selected_strategy,
                'target': target,
                'confidence': confidence,
                'reason': reason,
                'source': actual_source,
                'response_time': response_time,
                'success': True,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'analysis': analysis
            }
            
            return response, actual_source, response_time, metadata
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Error processing request: {str(e)}"
            
            metadata = {
                'strategy': 'error',
                'target': 'error',
                'confidence': 0.0,
                'reason': f'Routing error: {str(e)}',
                'source': 'error',
                'response_time': response_time,
                'success': False,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return error_msg, "error", response_time, metadata
    
    def _route_with_strategy(self, query: str, strategy: str) -> Tuple[str, str, float]:
        """Route query using selected strategy (matches app_hybrid_enhanced.py)."""
        try:
            if strategy == 'hybrid':
                target, reason, priority = self.router.route_hybrid_query(query)
                analysis = self.router.analyze_query_for_hybrid_routing(query)
                confidence = analysis.get('ml_confidence', 0.75)
            elif strategy == 'rule_based':
                analysis = self.router.analyze_query_for_hybrid_routing(query)
                if analysis['is_greeting'] or analysis['is_calculation']:
                    target, reason = 'local', 'Simple query - rule based'
                else:
                    target, reason = 'cloud', 'Complex query - rule based'
                confidence = 0.85
            elif strategy == 'bert':
                if hasattr(self.router, 'bert_router') and self.router.bert_router:
                    target, reason, metadata = self.router.bert_router.route_query(query)
                    confidence = metadata.get('confidence', 0.75)
                else:
                    target, reason, confidence = 'cloud', 'BERT router not available', 0.5
            elif strategy == 'phi':
                if hasattr(self.router, 'phi_router') and self.router.phi_router:
                    target, reason, metadata = self.router.phi_router.route_query(query)
                    confidence = metadata.get('confidence', 0.75)
                else:
                    target, reason, confidence = 'cloud', 'PHI router not available', 0.5
            else:
                target, reason, confidence = 'cloud', 'Unknown strategy', 0.5
                
            return target, reason, confidence
            
        except Exception as e:
            return 'cloud', f'Routing error: {str(e)[:50]}', 0.5
    
    def process_conversation_turn(self, user_input: str) -> Tuple[str, str, float, Dict]:
        """Process a conversation turn with context preservation and routing strategy."""        
        # Generate context-aware query using ConversationContextManager
        recent_messages = self.context_manager.get_messages_for_model('both', include_system=False)
        context_query = self.context_manager.get_conversation_context(recent_messages[-3:]) if recent_messages else user_input
        
        if context_query != "No previous context available.":
            full_query = f"{context_query}\n\nCurrent question: {user_input}"
        else:
            full_query = user_input
        
        # Route using selected strategy
        response, source, response_time, routing_metadata = self.route_with_selected_strategy(full_query)
        
        # Additional metadata
        metadata = {
            'context_length': len(self.context_manager.chat_history),
            'query_length': len(user_input),
            'context_used': len(self.context_manager.chat_history) > 0,
            'routing_info': routing_metadata
        }
        
        # Add exchange to context manager using correct parameter name
        self.context_manager.add_exchange(
            user_message=user_input,
            ai_response=response,  # Note: this becomes 'response' field in the exchange dict
            source=source,
            response_time=response_time,
            metadata=metadata
        )
        
        return response, source, response_time, metadata
    
    def _generate_contextual_mock_response(self, query: str) -> Tuple[str, str]:
        """Generate context-aware mock responses."""
        import random
        
        query_lower = query.lower()
        
        # Check for context-dependent queries
        is_followup = any(word in query_lower for word in ["what about", "and", "also", "furthermore", "additionally", "can you", "elaborate", "explain more"])
        has_context = len(self.context_manager.chat_history) > 0
        
        if is_followup and has_context:
            last_exchange = self.context_manager.chat_history[-1]
            last_source = last_exchange['source']
            
            responses = [
                f"Following up on our previous discussion about '{last_exchange['user_message'][:30]}...', let me elaborate: {query[:50]}",
                f"Based on our conversation context, I can expand on that topic: {query[:50]}",
                f"Continuing from my previous {last_source} response, here's more detail: {query[:50]}"
            ]
            return random.choice(responses), last_source
        
        # Standard routing logic with context awareness
        if any(word in query_lower for word in ["hello", "hi", "start", "begin"]):
            if has_context:
                responses = ["Welcome back! I remember our previous conversation.", "Hello again! Ready to continue our discussion?"]
            else:
                responses = ["Hello! I'm ready to start our conversation.", "Hi there! Let's begin our chat."]
            source = "local"
        elif any(word in query_lower for word in ["enterprise", "business", "production", "scalable"]):
            responses = [
                "Enterprise analysis: This complex business query would be routed through APIM for optimal processing.",
                "Business context detected: APIM router would handle this enterprise-level request."
            ]
            source = "apim"
        elif any(word in query_lower for word in ["analyze", "complex", "detailed", "comprehensive", "reasoning"]):
            responses = [
                "Complex analysis requested: Foundry Agent would provide detailed reasoning and comprehensive insights.",
                "Advanced query detected: Azure AI Foundry would handle this sophisticated analysis."
            ]
            source = "foundry"
        else:
            responses = [
                "General query: Azure OpenAI would process this standard request with cloud capabilities.",
                "Standard processing: Cloud model would handle this general inquiry."
            ]
            source = "azure"
        
        return random.choice(responses), source
    
    def update_performance_metrics(self, response_time: float, source: str, context_used: bool):
        """Update performance tracking."""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'response_time': response_time,
            'source': source,
            'context_used': context_used
        })
        
        # Keep last 50 entries
        if len(self.performance_history) > 50:
            self.performance_history.pop(0)
    
    def get_conversation_insights(self) -> Dict:
        """Generate conversation insights and analytics."""
        if not self.context_manager.chat_history:
            return {}
        
        # Get insights from the ConversationContextManager
        context_summary = self.context_manager.get_session_summary()
        
        # Calculate additional metrics from chat_history
        total_time = sum(ex['response_time'] for ex in self.context_manager.chat_history)
        avg_response_time = total_time / len(self.context_manager.chat_history)
        
        # Source distribution
        sources = [ex['source'] for ex in self.context_manager.chat_history]
        source_counts = {source: sources.count(source) for source in set(sources)}
        
        # Context usage analysis
        context_used_count = sum(1 for ex in self.context_manager.chat_history if ex.get('metadata', {}).get('context_used', False))
        
        return {
            'total_exchanges': len(self.context_manager.chat_history),
            'avg_response_time': avg_response_time,
            'fastest_response': min(ex['response_time'] for ex in self.context_manager.chat_history),
            'slowest_response': max(ex['response_time'] for ex in self.context_manager.chat_history),
            'source_distribution': source_counts,
            'model_switches': context_summary['routing_stats'].get('model_switches', 0),
            'context_usage_rate': (context_used_count / len(self.context_manager.chat_history)) * 100 if self.context_manager.chat_history else 0,
            'session_duration': context_summary['session_info'].get('duration', '0:00:00'),
            'session_summary': context_summary
        }

def main():
    """Main Streamlit application."""
    
    try:
        # Initialize session state
        if "multi_turn_demo" not in st.session_state:
            st.session_state.multi_turn_demo = StreamlitMultiTurnDemo()
        if "selected_router" not in st.session_state:
            st.session_state.selected_router = "hybrid"
        
        demo = st.session_state.multi_turn_demo
        
        # Sync selected strategy with session state
        if hasattr(st.session_state, 'selected_router'):
            demo.selected_strategy = st.session_state.selected_router
            
    except Exception as e:
        st.error(f"Initialization error: {e}")
        st.info("Please refresh the page to retry initialization.")
        return
    
    # Header
    st.title("🤖 Hybrid AI Router - Multi-Turn Conversation Demo")
    st.markdown("**Advanced context-aware routing with multiple strategies**")
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Conversation interface
        st.header("💬 Intelligent Multi-Turn Chat")
        
        # Context status
        if demo.context_manager.chat_history:
            # Get the latest exchange for router info
            latest_exchange = demo.context_manager.chat_history[-1]
            router_used = "Unknown"
            
            # Extract router used from the latest exchange metadata
            if 'metadata' in latest_exchange and 'routing_info' in latest_exchange['metadata']:
                routing_info = latest_exchange['metadata']['routing_info']
                if 'analysis' in routing_info and 'router_used' in routing_info['analysis']:
                    router_used = routing_info['analysis']['router_used'].replace('_', ' ').title()
                else:
                    router_used = routing_info.get('strategy', 'Unknown').title()
            
            st.info(f"🧠 Context: {len(demo.context_manager.chat_history)} exchanges in memory | Strategy: {demo.selected_strategy.title()} | Router Used: {router_used} | Session: {demo.context_manager.session_id}")
        
        # Chat input
        # Handle example query selection by using value parameter
        input_value = ""
        if hasattr(st.session_state, 'selected_example') and st.session_state.selected_example:
            input_value = st.session_state.selected_example
            # Clear the selected example immediately to prevent repeated population
            st.session_state.selected_example = ""
        
        user_input = st.text_input(
            "Continue the conversation:",
            value=input_value,
            placeholder="Ask a question, follow up, or start a new topic...",
            key="conversation_input"
        )
        
        # Send button
        col_send, col_clear = st.columns([1, 1])
        
        with col_send:
            if st.button("💬 Send Message", disabled=not user_input):
                with st.spinner("Processing with context awareness..."):
                    response, source, response_time, metadata = demo.process_conversation_turn(user_input)
                    
                    # Update stats
                    if source in demo.routing_stats:
                        demo.routing_stats[source] += 1
                    
                    # Update performance tracking
                    demo.update_performance_metrics(response_time, source, metadata.get('context_used', False))
                
                # Clear input after sending by triggering a rerun which will reset the input
                st.rerun()
        
        with col_clear:
            if st.button("🧹 Clear Context"):
                demo.context_manager.clear_conversation()
                demo.routing_stats = {k: 0 for k in demo.routing_stats}
                demo.performance_history.clear()
                st.rerun()
        
        # Conversation display
        st.subheader("📜 Conversation Flow")
        
        if demo.context_manager.chat_history:
            for exchange in reversed(demo.context_manager.chat_history[-8:]):  # Show last 8 exchanges
                with st.container():
                    # Exchange header
                    col_meta1, col_meta2, col_meta3 = st.columns([1, 1, 1])
                    with col_meta1:
                        st.caption(f"Exchange #{exchange['exchange_number']}")
                    with col_meta2:
                        if exchange.get('model_switched', False):
                            st.caption("🔄 Model switched")
                    with col_meta3:
                        st.caption(f"⏱️ {exchange['response_time']:.3f}s")
                    
                    # User message
                    st.write(f"**🧑 You:** {exchange['user_message']}")
                    
                    # AI response with enhanced source display
                    source_icons = {
                        "local": "🟢 LOCAL",
                        "apim": "🔵 APIM", 
                        "foundry": "🟣 FOUNDRY",
                        "azure": "🟠 AZURE",
                        "error": "🔴 ERROR",
                        "mock": "⚪ MOCK"
                    }
                    
                    source_display = source_icons.get(exchange['source'], f"❓ {exchange['source'].upper()}")
                    st.write(f"**🤖 AI [{source_display}]:** {exchange['response']}")
                    
                    # Routing details expandable section
                    if 'metadata' in exchange and 'routing_info' in exchange['metadata']:
                        with st.expander("🔍 Routing Details"):
                            routing_info = exchange['metadata']['routing_info']
                            detail_col1, detail_col2 = st.columns(2)
                            with detail_col1:
                                st.metric("Strategy", routing_info.get('strategy', 'Unknown').title())
                                st.metric("Target", routing_info.get('target', 'Unknown').title())
                                st.metric("Confidence", f"{routing_info.get('confidence', 0):.2f}")
                            with detail_col2:
                                st.metric("Source", routing_info.get('source', 'Unknown').title())
                                st.metric("Response Time", f"{routing_info.get('response_time', 0):.2f}s")
                                st.metric("Success", "✅" if routing_info.get('success', True) else "❌")
                            st.write("**Reason:**", routing_info.get('reason', 'No reason provided'))
                    
                    # Context indicator
                    if exchange.get('metadata', {}).get('context_used', False):
                        st.caption("🧠 Used conversation context")
                    
                    st.divider()
        else:
            st.info("👋 Start a conversation! The system will remember context across exchanges and intelligently switch between models.")
    
    with col2:
        # Analytics sidebar
        
        st.header("📊 Configuration & Analytics")
        
        # Routing Strategy Selection
        with st.expander("🎛️ Routing Strategy", expanded=True):
            available_strategies = []
            for strategy, available in demo.available_routers.items():
                if available:
                    available_strategies.append(strategy)
            
            if available_strategies:
                new_strategy = st.selectbox(
                    "Select Routing Strategy:",
                    available_strategies,
                    index=available_strategies.index(demo.selected_strategy) if demo.selected_strategy in available_strategies else 0
                )
                
                if new_strategy != demo.selected_strategy:
                    demo.selected_strategy = new_strategy
                    st.session_state.selected_router = new_strategy
                    st.rerun()
                
                # Show strategy description
                strategy_descriptions = {
                    'hybrid': '🎭 Full hybrid router with 3-tier routing (Local → APIM → Foundry)',
                    'rule_based': '📋 Pattern-based routing using query characteristics',
                    'bert': '🧠 BERT ML model for intelligent query classification',
                    'phi': '🔬 PHI small language model for query routing'
                }
                
                st.caption(strategy_descriptions.get(demo.selected_strategy, 'Custom routing strategy'))
            else:
                st.warning("No routing strategies available")
        
        # System status
        with st.expander("🔧 System Status", expanded=True):
            st.subheader("Available Routers")
            for strategy, available in demo.available_routers.items():
                status = "✅" if available else "❌"
                st.write(f"{status} {strategy.replace('_', ' ').title()}")
            
            if demo.router and hasattr(demo.router, 'get_system_capabilities'):
                try:
                    capabilities = demo.router.get_system_capabilities()
                    targets = capabilities.get("available_targets", {})
                    st.subheader("Hybrid Router Targets")
                    for target, available in targets.items():
                        status = "✅" if available else "❌"
                        st.write(f"{status} {target.replace('_', ' ').title()}")
                except:
                    st.write("Basic routing available")
            
        # Conversation insights
        if demo.context_manager.chat_history:
            insights = demo.get_conversation_insights()
            
            # Key metrics
            st.subheader("📈 Conversation Metrics")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Exchanges", insights['total_exchanges'])
                st.metric("Model Switches", insights['model_switches'])
            with col_m2:
                st.metric("Avg Response", f"{insights['avg_response_time']:.3f}s")
                st.metric("Context Usage", f"{insights['context_usage_rate']:.1f}%")
            
            # Source distribution chart
            st.subheader("🎯 Model Usage")
            if insights['source_distribution']:
                fig = px.pie(
                    values=list(insights['source_distribution'].values()),
                    names=list(insights['source_distribution'].keys()),
                    title="Routing Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance trend
            if len(demo.performance_history) > 3:
                st.subheader("⚡ Performance Trend")
                times = [p['response_time'] for p in demo.performance_history]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=times,
                    mode='lines+markers',
                    name='Response Time',
                    line=dict(color='blue')
                ))
                fig.update_layout(
                    title="Response Time History",
                    xaxis_title="Request #",
                    yaxis_title="Time (s)",
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Quick conversation starters
        st.subheader("🚀 Conversation Starters")
        if st.button("👋 Greeting"):
            st.session_state.selected_example = "Hello! How can you help me today?"
            st.rerun()
        if st.button("💼 Enterprise Query"):
            st.session_state.selected_example = "What are the best practices for enterprise AI deployment?"
            st.rerun()
        if st.button("🔍 Complex Analysis"):
            st.session_state.selected_example = "Can you analyze the implications of quantum computing on cybersecurity?"
            st.rerun()
        if st.button("🔄 Follow-up"):
            st.session_state.selected_example = "Can you elaborate on that point?"
            st.rerun()
        if st.button("🎯 Context Test"):
            st.session_state.selected_example = "What was my previous question about?"
            st.rerun()

    # Example queries section (outside column layout)
    st.markdown("---")
    st.subheader("💡 Example Queries by Complexity")
    col_ex1, col_ex2, col_ex3 = st.columns(3)

    examples = {
        "Simple (Local)": [
            "Hello there!",
            "What is 25 + 17?",
            "What time is it?"
        ],
        "Moderate": [
            "Explain machine learning",
            "Compare Python and Java",
            "What is cloud computing?"
        ],
        "Complex (Cloud)": [
            "Analyze hybrid AI architecture benefits",
            "Write a business case for AI adoption",
            "Design a recommendation system"
        ]
    }
    
    for i, (category, queries) in enumerate(examples.items()):
        with [col_ex1, col_ex2, col_ex3][i]:
            st.markdown(f"**{category}**")
            for query in queries:
                if st.button(f"💬 {query[:20]}...", key=f"example_{i}_{query[:10]}"):
                    st.session_state.selected_example = query
                    st.rerun()
    
    # Footer with session info
    st.markdown("---")
    if demo.context_manager.chat_history:
        session_summary = demo.context_manager.get_session_summary()
        session_info = session_summary['session_info']
        st.caption(f"🕒 Session: {session_info['session_id']} | Duration: {session_info['duration']} | 💬 {len(demo.context_manager.chat_history)} exchanges | 🧠 Context-aware routing active | 🎛️ Strategy: {demo.selected_strategy.title()}")
    
    st.markdown("🤖 **Hybrid AI Router Demo** - Multi-strategy intelligent routing with context preservation")

if __name__ == "__main__":
    main()