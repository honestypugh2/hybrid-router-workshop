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

## � Prerequisites

### System Requirements

- **Node.js 16.0+** (Download from [nodejs.org](https://nodejs.org/))
- **Python 3.8+** (for backend API)
- **npm** or **yarn** (comes with Node.js)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Backend Dependencies

The backend requires the hybrid router modules from the parent project. Install from the main workshop directory:

```bash
# From the main hybrid-llm-router-workshop directory
pip install -r requirements.txt
```

Key backend dependencies:

- `fastapi>=0.104.0` - Web framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `pydantic>=2.4.0` - Data validation
- `python-dotenv>=1.0.0` - Environment management
- `azure-ai-inference>=1.0.0b1` - Azure AI services
- Router modules (hybrid, BERT, PHI) from parent directory

## 🏗️ Dual Backend Architecture

This React app intelligently supports **two backend API patterns**:

### 🚀 Enhanced Backend (Port 8000)
- **Location**: Root level `backend_api.py`
- **API Pattern**: `/api/*` endpoints (e.g., `/api/query`, `/api/system-status`)
- **Features**: Advanced analytics, conversation insights, enhanced context management
- **Version**: 2.0.0 (Enhanced)

### ⚡ Basic Backend (Port 8080)
- **Location**: `react-hybrid-router/backend_api.py`
- **API Pattern**: `/route` endpoints (e.g., `/route`, `/route/bert`)
- **Features**: Core routing functionality, basic analytics
- **Version**: 1.0.0 (Basic)

### � Automatic Backend Detection

The React app automatically:
1. **Tests both backends** on startup (ports 8000 and 8080)
2. **Prioritizes Enhanced backend** (8000) when available
3. **Falls back to Basic backend** (8080) if Enhanced is unavailable
4. **Switches backends** during runtime if active backend fails
5. **Adapts API calls** to match the detected backend pattern

## �🚀 Quick Start

### Option 1: Use the Startup Scripts (Recommended)

### Option 1: Use the Startup Scripts (Recommended)

**All scripts are now located in the `react-hybrid-router` directory for easier access.**

**Windows Scripts:**

```bash
# Navigate to the React directory first
cd react-hybrid-router

# Enhanced demo with dual backend support (Recommended)
./start_enhanced_demo.bat

# Basic demo with original backend
./start_demo.bat  

# React-specific demo with Python startup
./start_react_demo.bat

# Or using npm scripts
npm run demo-enhanced    # Enhanced demo
npm run demo            # Basic demo
npm run demo-react      # React demo
npm run demo-python     # Python startup script
```

**Linux/macOS:**

```bash
# Navigate to the React directory first
cd react-hybrid-router

# Setup and start the application
./start.sh
```

**Script Features:**

- **start_enhanced_demo.bat**: Starts Enhanced backend (port 8000) + React frontend
- **start_demo.bat**: Starts Basic backend (port 8080) + React frontend  
- **start_react_demo.bat**: Launches Python startup script for React demo
- **start_react_demo.py**: Python-based startup with enhanced error handling

### Option 2: Manual Setup

#### Frontend (React)

```bash
# Navigate to React app directory
cd react-hybrid-router

# Install dependencies (first time only)
npm install

# Start development server
npm start
```

#### Backend (FastAPI) - Optional

```bash
# From the main workshop directory, activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Navigate to React app directory and start backend
cd react-hybrid-router
python backend_api.py
```

**Note:** The frontend can run in mock mode without the backend for demonstration purposes.

### Option 3: Production Build

```bash
# Build for production
npm run build

# Serve production build
npm run serve
```

## 🔗 Access Points

- **React Frontend**: <http://localhost:3000> (development server)
- **Production Build**: Served via `npm run serve` after `npm run build`
- **Backend API** (when running): <http://localhost:8000>
- **API Status** (when backend active): <http://localhost:8000/status>

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
├── public/                      # Static assets
│   └── index.html              # HTML template
├── src/
│   ├── components/
│   │   ├── App.tsx             # Main application orchestrator
│   │   ├── App.css             # Application styles
│   │   ├── ChatInterface.tsx   # Enhanced chat interface
│   │   ├── Analytics.tsx       # Metrics and configuration panel
│   │   └── Analytics.css       # Analytics component styles
│   ├── services/
│   │   └── api.ts              # API service with session management
│   ├── types.ts                # TypeScript type definitions
│   ├── index.tsx               # Application entry point
│   └── index.css               # Global styles
├── build/                       # Production build output (generated)
├── node_modules/               # npm dependencies (generated)
├── backend_api.py              # FastAPI backend (optional)
├── package.json                # npm configuration and dependencies
├── package-lock.json           # npm lock file (generated)
├── tsconfig.json              # TypeScript configuration
├── setup.bat                   # Windows setup script
├── setup.sh                   # Linux/macOS setup script
├── start.bat                   # Windows startup script
├── start.sh                   # Linux/macOS startup script
├── .env                       # Environment variables (optional)
└── README.md                  # This file
```

### Frontend Dependencies (React + TypeScript)

**Production Dependencies:**

- `react` & `react-dom` - React framework
- `typescript` - TypeScript support
- `recharts` - Charts and visualization library
- `axios` - HTTP client for API calls
- `web-vitals` - Performance monitoring

**Development Dependencies:**

- `react-scripts` - Create React App toolchain
- `@types/react` & `@types/react-dom` - TypeScript definitions
- `@testing-library/*` - Testing utilities
- `eslint` & TypeScript ESLint plugins - Code linting
- `serve` - Static file server for production builds

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

### Frontend-Only Mode (Default)

The React app can run independently with mock responses for demonstration purposes. No backend setup required.

### With Backend API (Optional)

When `backend_api.py` is running, the app connects to live router services.

### Core API Endpoints (when backend is active)

- `POST /route` - Route queries with hybrid strategy
- `POST /route/bert` - Route queries with BERT strategy  
- `POST /route/phi` - Route queries with PHI strategy
- `GET /status` - Get system status and router availability
- `GET /capabilities` - Get available router capabilities

### Request/Response Models

- JSON request/response format
- Session-based context management
- Comprehensive metadata in responses
- Error handling with fallback modes

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

## 🛠️ Development & Deployment

### Prerequisites

- Node.js 16+ and npm/yarn
- Python 3.8+ (for optional backend)
- Modern web browser with JavaScript enabled

### Local Development

1. **Setup the project:**

   ```bash
   npm install
   ```

2. **Start development server:**

   ```bash
   npm start
   ```

3. **Optional - Start backend API:**

   ```bash
   # From main workshop directory
   pip install -r requirements.txt
   cd react-hybrid-router
   python backend_api.py
   ```

### Production Deployment

1. **Build the application:**

   ```bash
   npm run build
   ```

2. **Deploy the `build/` directory** to your web server or hosting platform

3. **Hosting Options:**
   - Static hosting: Netlify, Vercel, GitHub Pages
   - Cloud platforms: Azure Static Web Apps, AWS S3 + CloudFront
   - Traditional web servers: Apache, Nginx

### Docker Deployment (Optional)

The application can be containerized for consistent deployments across environments.

### Development Workflow

```bash
# Install dependencies
npm install

# Start development server with hot reload
npm start

# Type checking
npm run type-check

# Build for production
npm run build

# Serve production build locally
npm run serve

# Run tests (when available)
npm test
```

### Available npm Scripts

- `npm start` - Start development server (localhost:3000)
- `npm run build` - Create production build
- `npm run test` - Run tests
- `npm run eject` - Eject from Create React App (irreversible)
- `npm run serve` - Serve production build locally
- `npm run type-check` - Run TypeScript type checking

### Environment Configuration

The app can be configured using environment variables in `.env` file:

```bash
# API Configuration
REACT_APP_API_BASE_URL=http://localhost:8000
REACT_APP_ENABLE_MOCK_FALLBACK=true
REACT_APP_DEBUG_MODE=false
```

---

**This React application provides a modern, interactive frontend for the Hybrid AI Router system. It can operate independently with mock responses or connect to the full router backend for live AI interactions. Perfect for demonstrations, development, and production deployments.**
