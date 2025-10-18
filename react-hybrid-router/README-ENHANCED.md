# Enhanced Hybrid AI Router React Demo

This is an enhanced React application that mirrors the functionality of `streamlit_multiturn_demo.py`, providing a sophisticated multi-turn conversation interface with advanced AI routing capabilities.

## 🌟 Features

### Core Functionality
- **Multi-turn Conversations**: Context-aware conversations with session management
- **Multiple Routing Strategies**: Hybrid, Rule-based, BERT, and PHI routing
- **Real-time Performance Tracking**: Response times, model switches, and analytics
- **Context Management**: Intelligent context preservation across conversations
- **System Status Monitoring**: Live router availability and health checking

### Advanced Features
- **Expandable Routing Details**: Click exchanges to see detailed routing analysis
- **Performance Analytics**: Comprehensive metrics and trends
- **Model Usage Distribution**: Visual breakdown of which models were used
- **Session Insights**: Duration, context usage, and efficiency metrics
- **Example Query Suggestions**: Categorized by complexity for testing

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

## 🔗 Access Points

- **React Frontend**: http://localhost:3000
- **FastAPI Backend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/api/health

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
- **System Status**: Live router availability
- **Performance Metrics**: Real-time statistics
- **Usage Distribution**: Model utilization breakdown
- **Example Queries**: Categorized test cases

### Interactive Features

#### Expandable Exchange Details
Each conversation exchange can be expanded to show:
- **Routing Decision**: Why this model was chosen
- **Context Information**: What context was used
- **Performance Data**: Response time and confidence
- **Model Switching**: When and why models changed

#### Real-time Analytics
- **Live Metrics**: Updated with each exchange
- **Performance Trends**: Historical response times
- **Context Usage Tracking**: How often context helps
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

### Frontend (React + TypeScript)
- **Components**:
  - `App.tsx`: Main orchestrator with state management
  - `ChatInterface.tsx`: Enhanced chat with expandable details
  - `Analytics.tsx`: Comprehensive metrics and configuration
- **Services**:
  - `api.ts`: Enhanced API service with session management
  - `types.ts`: Comprehensive TypeScript interfaces

### Backend (FastAPI + Python)
- **Enhanced API**: `backend_api.py` with full feature parity
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
- Model distribution and efficiency

### System Monitoring
- Router availability status
- Health check indicators
- Hybrid router target status
- Performance trend analysis

## 🛠️ Development

### Project Structure
```
react-hybrid-router/
├── src/
│   ├── components/
│   │   ├── App.tsx              # Main application
│   │   ├── ChatInterface.tsx    # Enhanced chat interface
│   │   └── Analytics.tsx        # Analytics and configuration
│   ├── services/
│   │   └── api.ts              # Enhanced API service
│   ├── types/
│   │   └── index.ts            # TypeScript interfaces
│   └── styles/                 # CSS files
└── backend_api.py              # FastAPI backend
```

### Key Features Implemented

#### Context Management
- Session-based conversation history
- Intelligent context extraction
- Context usage tracking and analytics

#### Multi-Strategy Routing
- Dynamic strategy selection
- Real-time availability checking
- Fallback handling with mock responses

#### Enhanced User Experience
- Expandable exchange details
- Real-time performance metrics
- Interactive analytics dashboard
- Example query categorization

## 🔄 API Endpoints

### Core Routes
- `POST /api/query`: Route queries with context
- `GET /api/system-status`: Get router availability
- `GET /api/conversation-insights/{session_id}`: Get session analytics
- `DELETE /api/clear-context/{session_id}`: Clear session context
- `GET /api/health`: Health check

### Request/Response Models
- Comprehensive Pydantic models for type safety
- Enhanced metadata in all responses
- Session management with UUIDs
- Error handling with fallback responses

## 🎨 Styling

The app uses CSS modules with:
- **Responsive Design**: Works on desktop and mobile
- **Modern UI**: Clean, professional interface
- **Interactive Elements**: Hover effects and animations
- **Status Indicators**: Color-coded system health
- **Expandable Sections**: Collapsible details

## 🚨 Error Handling

- **Graceful Degradation**: Falls back to mock responses
- **Connection Monitoring**: Shows API connection status
- **Router Availability**: Real-time status updates
- **User Feedback**: Clear error messages and status indicators

## 📈 Performance

- **Optimized Builds**: Production-ready compilation
- **Efficient State Management**: Minimal re-renders
- **Context Optimization**: Smart context preservation
- **Memory Management**: Limited history retention

## 🤝 Integration

This React app is designed to work seamlessly with:
- **Existing Router Modules**: Hybrid, BERT, PHI routers
- **Context Manager**: Conversation context handling
- **Configuration System**: Centralized config management
- **Telemetry System**: Performance and usage tracking

---

**Note**: This enhanced React demo provides the same core functionality as the Streamlit version but with a more sophisticated, production-ready interface suitable for integration into larger applications.