## React.js + TypeScript Hybrid AI Router

This is a React.js + TypeScript conversion of the Streamlit multi-turn demo application for the Hybrid AI Router system.

### Project Structure

```
react-hybrid-router/
├── src/
│   ├── components/
│   │   ├── App.tsx          # Main application component
│   │   ├── ChatInterface.tsx # Chat interface with conversation history
│   │   ├── Analytics.tsx     # Analytics sidebar component
│   │   └── App.css          # Application styles
│   ├── services/
│   │   └── api.ts           # API service layer
│   ├── types.ts             # TypeScript type definitions
│   └── index.tsx            # Application entry point
├── package.json             # Dependencies and scripts
└── README.md               # This file
```

### Features

- **Multi-turn Conversations**: Context-aware chat with conversation history
- **Routing Strategies**: Switch between hybrid, rule-based, BERT, and PHI routing
- **Real-time Analytics**: Performance metrics, source distribution, and insights
- **Responsive Design**: Mobile-friendly interface with CSS Grid layout
- **TypeScript**: Full type safety and IntelliSense support
- **Mock API**: Fallback mock responses when backend is unavailable

### Key Components

1. **App.tsx**: Main container managing state and layout
2. **ChatInterface.tsx**: Handles user input, message display, and example queries
3. **Analytics.tsx**: Shows routing configuration, metrics, and system status
4. **api.ts**: Service layer for backend communication with fallbacks

### Setup Instructions

1. **Install Dependencies**:

   ```bash
   cd react-hybrid-router
   npm install
   ```

2. **Start Development Server**:

   ```bash
   npm start
   ```

3. **Build for Production**:

   ```bash
   npm run build
   ```

### Key Differences from Streamlit Version

- **React Hooks**: Uses useState, useEffect for state management
- **Component Architecture**: Modular, reusable components
- **TypeScript Types**: Strongly typed interfaces and props
- **CSS Modules**: Scoped styling with CSS classes
- **API Layer**: Structured service for backend communication
- **Event Handling**: React synthetic events instead of Streamlit widgets

### Integration Notes

- Backend API expected at `http://localhost:8000`
- Graceful fallback to mock responses when API unavailable
- Compatible with existing Python hybrid router backend
- Maintains same conversation context and routing logic

### Development Notes

The lint errors shown are expected in a minimal setup and would be resolved by:

1. Installing React dependencies: `npm install react @types/react`
2. Setting up proper TypeScript configuration
3. Adding JSX runtime configuration
4. Installing development dependencies

This provides a complete React.js + TypeScript equivalent of the Streamlit application with modern web development patterns and full type safety.
