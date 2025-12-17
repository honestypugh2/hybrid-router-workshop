# Hybrid LLM Model Router Workshop

A hands-on workshop for building hybrid AI chatbots that intelligently route between on-device and cloud models

> **⚠️ DISCLAIMER**: This workshop and all associated code are provided **for development and educational purposes only**. This is not production-ready code and should not be deployed in production environments without proper security review, testing, and hardening. Use at your own risk.

## 🎯 Overview

This workshop guides you through building a **hybrid AI chatbot** that combines the best of both worlds:

- **On-device models** (Azure Foundry Local) for fast, private responses
- **Cloud models** (Azure AI Foundry) for complex, sophisticated analysis
- **Intelligent routing** that automatically selects the best model for each query
- **Seamless context sharing** across model transitions

## 🖼️ Frontend Preview

![Chat Interface](images/react_hybrid_app_main.png)
*Interactive React chat interface with model routing transparency*

## 🏗️ Architecture

```
User Query → Router Logic → [Local Model] or [Cloud Model] → Unified Response
                ↓                ↓              ↓
            Analytics     Fast Response   Smart Analysis

Where:
• Local Model = Azure Foundry Local
• Cloud Model = Azure AI Foundry Agent Service (Agent Framework) | Azure OpenAI Direct | APIM
```

The system uses a **central Router Logic** powered by ML models (BERT/Phi SLM) that analyzes each query and intelligently routes to either the Local Model (Azure Foundry Local) for fast responses, or Cloud Model options including **Azure AI Foundry Agent Service using Agent Framework** (with ephemeral and persistent agent support), Azure OpenAI Direct, or APIM for smart analysis. The new **Agent Framework integration** provides async/await patterns, thread-based conversation persistence, and enhanced context management. Each path generates appropriate analytics while maintaining unified response handling and context continuity across model transitions.

## 🚀 Key Features

- **⚡ Low-Latency Local Responses**: Simple queries answered instantly on-device via Azure Foundry Local
- **🧠 Intelligent Cloud Escalation**: Complex tasks automatically routed to Azure AI Foundry Agent Service using Agent Framework
- **🤖 Agent Framework Integration**: Modern async/await patterns with ephemeral and persistent agents, thread-based conversation persistence
- **🔄 Seamless Context Sharing**: Unified conversation across model switches with native agent threads
- **🎯 ML-Powered Routing**: BERT and Phi SLM-based query classification for intelligent routing decisions
- **👁️ Full Transparency**: Clear indication of processing location with expandable routing details
- **📊 Comprehensive Observability**: Telemetry, performance monitoring, and real-time analytics
- **🎨 Interactive Frontend**: Modern React + TypeScript chat interface with dual backend support and automatic failover

## 📋 Prerequisites

- Python 3.10+
- Node.js 16.0+ and npm (for React frontend)
- Azure subscription with AI Foundry access
- **Azure AI Foundry Agent Service** with Agent Framework support
- Azure Foundry Local installation
- Azure OpenAI
- Azure API Management (optional)
- Application Insights
- Log Analytics Workspace (optional)
- Basic knowledge of Python, Azure services, and React/TypeScript
- Deployed models: gpt-4-1, gpt-4o, gpt-4o-mini, model-router
- Python packages: `azure-ai-projects`, `azure-ai-inference`, `agent-framework`
- Roles: [Azure AI Foundry project roles](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/rbac-azure-ai-foundry#azure-ai-foundry-project-roles)

### Assumption

This workshop uses a Windows machine for on-device.

> **⚠️ IMPORTANT**: Do not proceed to the Setup section until **ALL** prerequisites above are complete. Ensure you have:
>
> - ✅ Azure subscription with AI Foundry access configured
> - ✅ Azure Foundry Local installed and running
> - ✅ All required Azure services (OpenAI, APIM, Application Insights, etc.) deployed
> - ✅ All models (gpt-4-1, gpt-4o, gpt-4o-mini, model-router) deployed and accessible
> - ✅ Appropriate Azure roles assigned
> - ✅ Network connectivity verified between all services
>
> Missing prerequisites will cause setup and lab failures. Use the infrastructure deployment guide (`infra/README.md`) to deploy required Azure resources first.
>
> **NOTE: Infra APIM bicep script currently deploys API's backend using the incorrect format. Please follow Post-Deploymment instructions below.**

## 🛠️ Setup

1. **Clone the workshop materials**:

   ```bash
   git clone <repository-url>
   cd hybrid-router-workshop
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:

   ```bash
   cp .env.example .env
   # Edit .env with your Azure credentials and endpoints
   ```

5. **Login into Azure**:

   ```bash
   az login
   ```

6. **Add environment to Jupyter kernelspec to use in Notebook**:

   Activate the virtual environment if not done so:

      ```bash
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate
      ```

      ```bash
      python -m ipykernel install --user --name .venv
      ```

      When you open up the lab Notebooks, select Kernel to be the created virtual environment.

## 🔧 Post-Deployment Configuration

After deploying infrastructure and models, complete these configuration steps:

### Step 1: Create Azure AI Foundry Agent

1. Navigate to Azure AI Foundry portal
2. Go to your deployed AI Foundry project
3. Select **Agents** → **Create new agent**
4. Configure agent with deployed models (gpt-4o, gpt-4o-mini, etc.)
5. Note the agent endpoint URL for API Management

### Step 2: Configure API Management

1. Open Azure API Management service
2. Go to **APIs** → **Add API** → **Blank API**
3. Create APIs for:
   - Azure AI Foundry models (`/ai-foundry/*`)
   - Azure OpenAI models (`/openai/*`)
4. Set backend services to your deployed model endpoints
5. Configure authentication and rate limiting policies

### Step 3: Test Integration

1. Test API Management endpoints
2. Verify agent responses
3. Confirm routing functionality

## 🚀 Quick Demo Scripts

> **⚠️ PREREQUISITES REQUIRED**: Before running any demo scripts, ensure you have:
>
> - ✅ **Installed Python dependencies** via `pip install -r requirements.txt`
> - ✅ **Installed Node.js dependencies** via `cd react-hybrid-router && npm install`
> - ✅ **Completed infrastructure deployment** (see `infra/README.md`)
> - ✅ **Verified model deployments** and API endpoints are accessible
> - ✅ **Configured environment variables** in `.env` file with your Azure credentials
> - ✅ **Azure AI Foundry Agent Service** configured with Agent Framework support
> - ✅ **Finished Lab 1** (Environment Setup) - Required for basic connectivity
> - ✅ **Finished Lab 2** (Local Model Testing) - Required for local routing
> - ✅ **Finished Lab 3** (Azure Model Testing + Agent Framework) - Required for cloud routing
> - ✅ **Finished Remaining Labs** from Lab Notebooks Section Below
>
> **Running demos without completing these prerequisites will result in connection errors, authentication failures, or incomplete functionality.**

**📁 All demo scripts are located in the `react-hybrid-router` directory.**

### Available Demos

```bash
# Navigate to React directory first
cd react-hybrid-router

# Enhanced Demo (Recommended) - Uses port 8000 with advanced features
.\start_enhanced_demo.bat       # Windows
./start_enhanced_demo.sh        # Linux/macOS

# Basic Demo - Uses port 8080 with original backend 
.\start_demo.bat                # Windows
./start_demo.sh                 # Linux/macOS

# React Demo with Python startup
.\start_react_demo.bat          # Windows
./start_react_demo.sh           # Linux/macOS
python start_react_demo.py

# Or use npm scripts
npm run demo-enhanced    # Enhanced demo (port 8000)
npm run demo            # Basic demo (port 8080)
npm run demo-react      # React demo
npm run demo-python     # Python startup
```

**From project root, use the helper scripts:**

```bash
# Windows - Helper script that navigates to react-hybrid-router
.\start_demos.bat

# Linux/macOS - Helper script that navigates to react-hybrid-router
./start_demos.sh
```

### 🌐 React App Features

The React + TypeScript demo provides:

- **Dual Backend Support**: Automatically detects Enhanced (8000) or Basic (8080) backend
- **Intelligent Failover**: Switches backends if active backend becomes unavailable
- **Expandable Routing Details**: Click any message to view detailed routing analysis
- **Performance Analytics**: Real-time metrics, trends, and model usage distribution
- **Multiple Router Types**: Switch between Hybrid, Rule-based, BERT, and PHI routing
- **Session Management**: Multi-turn conversations with context preservation
- **Example Queries**: Pre-configured test queries categorized by complexity
- **System Status**: Live monitoring of router availability and health
- **Mock API Fallback**: Graceful degradation when backends are unavailable
- **Responsive Design**: Mobile-friendly interface with modern CSS Grid layout

> **📝 Note**: All demo scripts are located in the `react-hybrid-router` directory for better organization. The React app provides a modern, responsive interface with TypeScript support and advanced features like expandable routing details and performance analytics.

![React App Chat Interface](images/react_hybrid_app_main.png)

![React App Chat Routing Details](images/react_hybrid_app_routingdetails.png)

## �📚 Workshop Structure

### 🔧 Core Modules (`modules/`)

- **`router.py`**: Rule-based intelligent query routing logic
- **`bert_router.py`**: BERT-based query classification router with ML-powered routing
- **`phi_router.py`**: Phi SLM-based query routing with fine-tuning capabilities
- **`hybrid_router.py`**: Multi-tier hybrid routing orchestration (APIM-based)
- **`hybrid_router_agent_framework.py`**: **NEW** - Agent Framework integration with async/await patterns, ephemeral and persistent agents
- **`hybrid_agent_context.py`**: **NEW** - Unified conversation management with native Agent Framework thread support and analytics
- **`azure_ai_manager.py`**: Azure AI Foundry integration and management
- **`context_manager.py`**: Conversation history and session management (legacy)
- **`telemetry.py`**: Observability, performance tracking, and comprehensive analytics
- **`config.py`**: Configuration management and environment settings
- **`helper.py`**: Utility functions and common operations
- **`utils.py`**: Additional utility functions for routing and processing
- **`onnx_converter.py`**: ONNX model conversion utilities for BERT/MobileBERT

### 📓 Lab Notebooks (`notebooks/`)

#### **Lab 1: Environment Setup** (`lab1_environment_setup.ipynb`)

- Configure Azure Foundry Local and Cloud environments
- Test connectivity and model availability
- Verify authentication and basic functionality

#### **Lab 2: Local Model Testing** (`lab2_local_model_testing.ipynb`)

- Explore on-device model capabilities
- Test response times and quality
- Understand local model limitations and strengths

#### **Lab 3: Azure Model Testing** (`lab3_azure_model_testing.ipynb`)

- Configure Azure OpenAI direct integration
- Test cloud model capabilities and API patterns
- Compare performance with local models

#### **Lab 3 (Foundry Agents): Azure AI Foundry Agents Testing** (`lab3_azure_ai_foundry_agents_testing.ipynb`)

- Work with Azure AI Foundry Agent Service (legacy patterns)
- Implement basic agent-based conversation patterns
- Explore Azure AI Foundry agent capabilities

#### **Lab 3 (Agent Framework): Agent Framework + Foundry Testing** (`lab3_agent_framework_foundry_testing.ipynb`) ⭐ **NEW**

- **Modern Agent Framework integration** with Azure AI Foundry
- Implement async/await patterns and thread-based persistence
- Work with ephemeral and persistent agents
- Explore enhanced conversation context management

#### **Lab 4: Model Routing Logic** (`lab4_model_routing.ipynb`)

- Implement intelligent query analysis
- Build routing decision engine
- Test routing accuracy with various query types

#### **Lab 4 (Extended): BERT Query Router** (`lab4_bert_query_router.ipynb`)

- Deep dive into BERT-based query classification
- Fine-tune models for specific routing scenarios
- Evaluate routing accuracy and performance

#### **Lab 4 (SLM): Phi Small Language Model Routing** (`lab4_phi_slm_routing.ipynb`)

- Implement routing using Phi small language models
- Fine-tune Phi models for routing decisions
- Compare SLM vs traditional ML approaches

#### **Lab 4 (Foundry Agents): Foundry Agent Routing** (`lab4_foundry_agent_routing.ipynb`)

- Route between different Azure Foundry agents
- Implement multi-agent coordination patterns
- Build agent-specific routing logic

#### **Lab 4 (Agent Framework): Agent Framework Routing** (`lab4_agent_framework_routing.ipynb`) ⭐ **NEW**

- **Implement Agent Framework-based routing** with async patterns
- Build intelligent routing using ephemeral and persistent agents
- Integrate ML-powered routing decisions (BERT/Phi) with Agent Framework
- Advanced thread-based conversation management

#### **Lab 4 (Integration): API Management Router** (`lab4_apim_model_router.ipynb`)

- Integrate with Azure API Management
- Implement enterprise-grade routing patterns
- Add rate limiting, authentication, and backend routing

#### **Lab 5: Hybrid Orchestration** (`lab5_hybrid_orchestration.ipynb`)

- Combine local and cloud models seamlessly (APIM-based)
- Implement conversation context sharing
- Build multi-turn conversation management with analytics

#### **Lab 5 (Agent Framework): Agent Framework Orchestration** (`lab5_agent_framework_orchestration.ipynb`) ⭐ **NEW**

- **Modern hybrid orchestration with Agent Framework**
- Combine Azure Foundry Local with Agent Framework cloud routing
- Implement native thread-based persistence with routing analytics
- ML-powered routing decisions with seamless context sharing

#### **Lab 6: Observability & Telemetry** (`lab6_observability_telemetry.ipynb`)

- Add comprehensive logging and monitoring
- Implement performance metrics collection
- Create analytics dashboard for insights

#### **Lab 6 Alternative: Telemetry with HybridFoundryAPIMRouter** (`lab6_alt_telemetry.ipynb`)

- Implement comprehensive telemetry collection
- Track performance across three routing tiers
- Monitor ML-powered routing decisions

#### **Lab 7: Frontend Chat Interface** (`lab7_frontend_chat_interface.ipynb`)

- Build React + TypeScript chat UI with comprehensive features
- Implement real-time multi-turn conversation interface
- **Dual backend support** with automatic detection and failover (ports 8000/8080)
- Add visual indicators for model routing with **expandable routing details**
- Display performance analytics, session insights, and model usage distribution
- Implement example query suggestions categorized by complexity
- Explore modern frontend architecture patterns with responsive design
- Mock API fallback for graceful degradation

#### **Lab 7 (Advanced): Advanced Routing Agents** (`lab7_advanced_routing_agents.ipynb`)

- Implementing sophisticated multi-agent routing systems
- Building complex decision trees for agent selection
- Creating advanced conversation orchestration patterns
- Exploring advanced Agent Framework capabilities

## 🎯 Success Criteria

By completing this workshop, you will have built a system that demonstrates:

✅ **Low-Latency Local Responses**: Simple queries answered in <0.5s on-device  
✅ **Seamless Cloud Escalation**: Complex tasks automatically routed without user friction  
✅ **Context Continuity**: Unified conversation memory across model switches  
✅ **Transparency & Control**: Clear indication of processing location  
✅ **Performance Optimization**: Measurable speed and efficiency gains  
✅ **Stakeholder-Ready Demo**: Interactive interface for business validation  

### 📱 User Interface Examples

![Local Response Example](images/react_hybrid_app_routingdetails.png)
*Fast local model response with performance indicators*

## 🏗️ Architecture Highlights

**Architecture highlights (Router Logic Design)**:

- **Router Logic**: Central decision engine with ML-powered routing (BERT, Phi SLM) and rule-based fallback
- **Local Model**: Azure Foundry Local for fast responses with local analytics
- **Cloud Model Options**:
  - **Azure AI Foundry Agent Service** (Agent Framework with ephemeral/persistent agents, async patterns) ⭐ **RECOMMENDED**
  - Azure AI Foundry Agents (legacy agent-based processing)
  - Azure OpenAI Direct (direct API access)
  - Azure APIM (API Management with backend routing)
- **Agent Framework Features**:
  - Native thread-based conversation persistence
  - Async/await patterns for better performance
  - Ephemeral agents for quick interactions
  - Persistent agents for long-running conversations
  - Enhanced context management with conversation analytics
- **Frontend**: React + TypeScript with dual backend support (Enhanced/Basic) and automatic failover
- **Supporting Services**: Cosmos DB, Blob Storage, Key Vault, Log Analytics, Application Insights
- **Unified Response**: Single response path regardless of processing location
- **Context Continuity**: Maintained across all processing paths with native agent threads

## 📊 Expected Outcomes

### Performance Metrics

- **Local queries**: 70-80% of simple queries, <0.5s response time
- **Cloud queries**: 20-30% of complex queries, 1-3s response time
- **Context preservation**: 100% accuracy across model transitions
- **User satisfaction**: Seamless experience with transparency

### Technical Achievements

- **Hybrid Architecture**: Production-ready routing system
- **Scalable Design**: Modular components for easy enhancement
- **Comprehensive Monitoring**: Full observability stack
- **Business Value**: Clear ROI demonstration through metrics

## 🛟 Troubleshooting

### Common Issues

**Connection Problems**:

- Verify Azure credentials in `.env`
- Check Azure Foundry Local service status
- Confirm network connectivity

**Model Loading Issues**:

- Ensure sufficient local resources
- Verify model deployment in Azure
- Check API quotas and limits

**Routing Problems**:

- Review query classification logic
- Adjust complexity thresholds
- Test with various query types

## 🔗 Additional Resources

### Azure AI & Agent Framework

- [Azure AI Foundry Documentation](https://docs.microsoft.com/azure/ai-foundry)
- [Azure AI Agent Service Documentation](https://learn.microsoft.com/azure/ai-services/agents/overview)
- [Agent Framework for Python](https://github.com/microsoft/agent-framework)
- [Azure AI Projects SDK](https://learn.microsoft.com/python/api/overview/azure/ai-projects-readme)
- [Azure AI Inference SDK](https://learn.microsoft.com/python/api/overview/azure/ai-inference-readme)
- [Azure Foundry Local Setup Guide](https://github.com/guygregory/FoundryLocal)

### Models & ML

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [MobileBERT Reference](https://huggingface.co/docs/transformers/model_doc/mobilebert#mobilebert)
- [Phi CookBook Reference](https://github.com/microsoft/PhiCookBook)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)

### Frontend & Development

- [React Documentation](https://react.dev)
- [TypeScript Documentation](https://www.typescriptlang.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)

## 🤝 Contributing

This workshop is designed for educational purposes. Feedback and improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## 📄 License

This workshop is provided under the MIT License. See `LICENSE` file for details.

## 🆘 Support

For workshop-specific questions:

- Create an issue in this repository
- Contact the workshop facilitators
- Join the discussion in our community forum

---

## Ready to build the future of hybrid AI? Let's get started with Lab 1! 🚀
