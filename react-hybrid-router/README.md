# Hybrid AI Router React Demo

A comprehensive React.js + TypeScript application that provides a sophisticated multi-turn conversation interface with advanced AI routing capabilities. This app mirrors the functionality of `streamlit_multiturn_demo.py` while offering enhanced user experience and production-ready features.

## 🌟 Features

### Core Functionality

- **Multi-turn Conversations**: Context-aware chat with session management and conversation history
- **Multiple Routing Strategies**: Switch between hybrid, rule-based, BERT, and PHI routing
- **Real-time Performance Tracking**: Response times, model switches, and comprehensive analytics
- **Context Management**: Intelligent context preservation across conversations
- **System Status Monitoring**: Live router availability and health checking

### Advanced Features

- **Expandable Routing Details**: Click exchanges to see detailed routing analysis
- **Performance Analytics**: Comprehensive metrics, trends, and model usage distribution
- **Session Insights**: Duration tracking, context usage, and efficiency metrics
- **Example Query Suggestions**: Categorized by complexity for comprehensive testing
- **Responsive Design**: Mobile-friendly interface with modern CSS Grid layout
- **TypeScript**: Full type safety and IntelliSense support
- **Mock API Fallback**: Graceful degradation when backend is unavailable

## 🚀 Quick Start

### Option 1: Use the Startup Script (Recommended)

```bash
# Run the enhanced demo startup script
./start_enhanced_demo.bat
```

This script will:

- Check prerequisites (Python, Node.js)
- Set up virtual environment
- Install dependencies
- Start both backend and frontend servers

### Option 2: Manual Startup

#### Backend (FastAPI)

```bash
# Install Python dependencies
pip install fastapi uvicorn pydantic

# Start the backend API
python backend_api.py
```

#### Frontend (React)

```bash
# Navigate to React app directory
cd react-hybrid-router

# Install dependencies (first time only)
npm install

# Start development server
npm start
```

### Option 3: Development Setup

```bash
# Build for production
npm run build

# Install React dependencies for development
npm install react @types/react
```

## 🔗 Access Points

- **React Frontend**: <http://localhost:3000>
- **FastAPI Backend**: <http://localhost:8000>
- **API Documentation**: <http://localhost:8000/docs>
- **API Health Check**: <http://localhost:8000/api/health>

## 📱 User Interface

### Main Components

#### Chat Interface

- **Message Input**: Submit queries with selected routing strategy
- **Conversation History**: Expandable exchanges with detailed metadata
- **Routing Analysis**: Click any exchange to see:
  - Model selection rationale
  - Context usage details
  - Performance metrics
  - Confidence scores

#### Analytics Panel

- **Strategy Selector**: Choose between routing methods
- **System Status**: Live router availability indicators
- **Performance Metrics**: Real-time statistics and trends
- **Usage Distribution**: Visual model utilization breakdown
- **Example Queries**: Categorized test cases for different scenarios

### Interactive Features

#### Expandable Exchange Details

Each conversation exchange can be expanded to show:

- **Routing Decision**: Why this model was chosen
- **Context Information**: What context was used
- **Performance Data**: Response time and confidence scores
- **Model Switching**: When and why models changed

#### Real-time Analytics

- **Live Metrics**: Updated with each exchange
- **Performance Trends**: Historical response times
- **Context Usage Tracking**: How often context helps improve responses
- **Model Efficiency**: Success rates by strategy

## 🎯 Testing Scenarios

### Simple Queries (Local Routing)

```
"Hello there!"
"What is 25 + 17?"
"What time is it?"
```

### Moderate Complexity

```
"Explain machine learning"
"Compare Python and Java"
"What is cloud computing?"
```

### Complex Queries (Cloud Routing)

```
"Analyze hybrid AI architecture patterns"
"Write a business case for AI adoption"
"Design a recommendation system architecture"
```

## 🔧 Architecture

### Project Structure

```
react-hybrid-router/
├── src/
│   ├── components/
│   │   ├── App.tsx              # Main application orchestrator
│   │   ├── ChatInterface.tsx    # Enhanced chat with expandable details
│   │   ├── Analytics.tsx        # Comprehensive metrics and configuration
│   │   └── App.css             # Application styles
│   ├── services/
│   │   └── api.ts              # Enhanced API service with session management
│   ├── types/
│   │   └── index.ts            # Comprehensive TypeScript interfaces
│   └── index.tsx               # Application entry point
├── backend_api.py              # FastAPI backend with full feature parity
├── package.json                # Dependencies and scripts
└── README.md                   # This file
```

### Frontend (React + TypeScript)

- **App.tsx**: Main container with state management and layout orchestration
- **ChatInterface.tsx**: Enhanced chat interface with conversation history and expandable details
- **Analytics.tsx**: Comprehensive metrics dashboard and configuration panel
- **api.ts**: Structured API service layer with session management and fallback handling

### Backend (FastAPI + Python)

- **Enhanced API**: Full feature parity with Streamlit version
- **Session Management**: Context-aware conversation handling
- **Multiple Routers**: Integration with hybrid, BERT, PHI, and rule-based routing
- **Real-time Status**: System health and availability monitoring

### Router Integration

- **Hybrid Router**: 3-tier routing (Local → APIM → Foundry)
- **BERT Router**: ML-based query classification
- **PHI Router**: Small language model routing
- **Rule-based**: Pattern-based routing logic

## 📊 Analytics & Insights

### Conversation Metrics

- Total exchanges and model switches
- Average, fastest, and slowest response times
- Context usage rate and session duration
- Model distribution and efficiency analysis

### System Monitoring

- Router availability status with real-time updates
- Health check indicators and performance trends
- Hybrid router target status monitoring
- Error handling and fallback system status

## 🔄 API Integration

### Core Endpoints

- `POST /api/query`: Route queries with context management
- `GET /api/system-status`: Get comprehensive router availability
- `GET /api/conversation-insights/{session_id}`: Get detailed session analytics
- `DELETE /api/clear-context/{session_id}`: Clear session context
- `GET /api/health`: System health check

### Request/Response Models

- Comprehensive Pydantic models for type safety
- Enhanced metadata in all responses
- Session management with UUIDs
- Robust error handling with fallback responses

## 🎨 Key Differences from Streamlit Version

### Technical Improvements

- **React Hooks**: Advanced state management with useState, useEffect
- **Component Architecture**: Modular, reusable, and testable components
- **TypeScript Types**: Strongly typed interfaces and comprehensive props
- **Modern Styling**: CSS Grid layout with responsive design
- **Enhanced API Layer**: Structured service architecture with error boundaries
- **Event Handling**: React synthetic events with optimal performance

### User Experience Enhancements

- **Interactive Elements**: Expandable details and hover effects
- **Real-time Updates**: Live metrics and status indicators
- **Mobile Responsiveness**: Works seamlessly across devices
- **Performance Optimization**: Efficient rendering and state management

## 🚨 Error Handling & Reliability

- **Graceful Degradation**: Automatic fallback to mock responses
- **Connection Monitoring**: Real-time API connection status
- **Router Availability**: Live status updates with retry logic
- **User Feedback**: Clear error messages and helpful status indicators
- **Memory Management**: Efficient history retention and cleanup

## 📈 Performance Features

- **Optimized Builds**: Production-ready compilation with code splitting
- **Efficient State Management**: Minimal re-renders and optimized updates
- **Context Optimization**: Smart context preservation and cleanup
- **Lazy Loading**: Dynamic imports for improved initial load times

## 🤝 Integration & Compatibility

This React application seamlessly integrates with:

- **Existing Router Modules**: Hybrid, BERT, PHI, and rule-based routers
- **Context Manager**: Advanced conversation context handling
- **Configuration System**: Centralized configuration management
- **Telemetry System**: Comprehensive performance and usage tracking
- **Backend APIs**: Compatible with existing Python infrastructure

## 🛠️ Development

### Prerequisites

- Node.js 16+ and npm/yarn
- Python 3.8+ (for backend)
- Modern web browser with JavaScript enabled

### Development Workflow

```bash
# Install dependencies
npm install

# Start development with hot reload
npm start

# Run type checking
npm run type-check

# Build for production
npm run build

# Run tests (when available)
npm test
```

### Configuration Notes

The lint errors in minimal setups are resolved by:

1. Installing React dependencies: `npm install react @types/react`
2. Setting up proper TypeScript configuration
3. Adding JSX runtime configuration
4. Installing development dependencies

---

**This enhanced React demo provides a production-ready, feature-rich alternative to the Streamlit version, suitable for integration into enterprise applications while maintaining full compatibility with the existing Python router infrastructure.**
