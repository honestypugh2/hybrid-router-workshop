# Production Scaling Guide: Hybrid LLM Router

**Target:** Scale to 100s simultaneous users  
**Last Updated:** December 5, 2025

## 🏗️ Architecture

### Container Orchestration

**Deploy to Azure Container Apps (ACA)**

- Built-in autoscaling (KEDA)
- Zero-downtime deployments
- Cost-effective for variable workloads

```bicep
scale: {
  minReplicas: 2
  maxReplicas: 20
  rules: [
    { name: 'http-rule', http: { metadata: { concurrentRequests: '50' }}}
    { name: 'cpu-rule', custom: { type: 'cpu', metadata: { value: '70' }}}
  ]
}
```

### Load Balancing

- **Azure API Management (APIM)** - Already integrated ✅
  - Rate limiting, caching, circuit breakers
- **Azure Front Door** for global distribution, DDoS protection

## 💾 State Management

### Replace In-Memory Storage

**Current Issue:** `session_contexts = {}` in `api_server.py` loses state on restart

**Solution:**

- **Redis Cache** - Hot session data (3600s TTL)
- **Cosmos DB** - Persistent conversation history
- **Azure Table Storage** - Cost-effective metadata

```python
# modules/session_manager.py
class DistributedSessionManager:
    async def get_context(self, session_id: str):
        # Try Redis first (fast)
        cached = await self.redis.get(f"session:{session_id}")
        if cached: return ConversationContextManager.from_json(cached)
        
        # Fallback to Cosmos DB
        context = await self.cosmos.read_item(session_id, partition_key=session_id)
        await self.redis.setex(f"session:{session_id}", 3600, context.to_json())
        return context
```

## ⚡ Performance

### Async Throughout

```python
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    result = await router.route_async(request.query)
    
    # Parallel operations
    await asyncio.gather(
        telemetry_manager.log_event_async(...),
        context_manager.add_interaction_async(...)
    )
```

### Connection Pooling

```python
async_client = AsyncOpenAI(
    max_retries=3,
    timeout=30.0,
    http_client=httpx.AsyncClient(
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
    )
)
```

### Response Caching

```python
async def get_cached_response(query: str, model: str) -> Optional[str]:
    cache_key = f"response:{hashlib.md5(f'{query}:{model}'.encode()).hexdigest()}"
    return await redis_client.get(cache_key)  # 1hr TTL
```

## 🔐 Security

### Authentication & Rate Limiting

```python
from fastapi.security import HTTPBearer
from fastapi_limiter.depends import RateLimiter

@app.post("/query")
@limiter.limit("10/minute")  # Per user
async def query_endpoint(request: QueryRequest, user = Depends(verify_token)):
    pass
```

### Circuit Breakers

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
async def call_azure_openai(query: str):
    return await azure_client.chat.completions.create(...)
```

## 📊 Observability

### Enhanced Telemetry

```python
# Instrument FastAPI + OpenAI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

FastAPIInstrumentor.instrument_app(app)
OpenAIInstrumentor().instrument()

# Custom metrics
from prometheus_client import Counter, Histogram
request_counter = Counter('router_requests_total', 'Total', ['model', 'status'])
response_time = Histogram('router_response_seconds', 'Response time', ['model'])
```

### Health Checks

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/ready")
async def readiness_check():
    checks = {
        "router": hybrid_router is not None,
        "redis": await check_redis_connection(),
        "cosmos": await check_cosmos_connection()
    }
    if all(checks.values()): return {"status": "ready", "checks": checks}
    raise HTTPException(status_code=503, detail=checks)
```

## 🚀 Infrastructure

### Add to main.bicep

```bicep
// Redis Cache
module redisCache 'modules/redis-cache.bicep' = {
  name: 'redisCache'
  params: {
    name: '${workloadName}-${environmentName}-redis'
    sku: 'Standard'  // Premium for production clustering
    capacity: 1
    enableNonSslPort: false
  }
}

// Cosmos DB
module cosmosDb 'modules/cosmos-db.bicep' = {
  name: 'cosmosDb'
  params: {
    name: '${workloadName}-${environmentName}-cosmos'
    consistencyLevel: 'Session'
    enableAutomaticFailover: true
    enableMultipleWriteLocations: true
  }
}

// Container Registry
module containerRegistry 'modules/container-registry.bicep' = {
  name: 'containerRegistry'
  params: {
    name: replace('${workloadName}${environmentName}acr', '-', '')
    sku: 'Standard'
    adminUserEnabled: false
  }
}
```

## 📦 Production Dependencies

Add to `requirements.txt`:

```txt
# Production server
gunicorn>=21.2.0

# Connection pooling
httpx>=0.25.0

# Distributed caching
redis>=5.0.0
azure-cosmos>=4.5.0

# Rate limiting
fastapi-limiter>=0.1.6
slowapi>=0.1.9

# Monitoring
prometheus-client>=0.19.0
opencensus-ext-azure>=1.1.13

# Reliability
circuitbreaker>=1.4.0
tenacity>=8.2.3
```

## 🧪 Load Testing

```python
# tests/load_test.py
import asyncio, aiohttp, time
from statistics import mean, median

async def load_test(concurrent_users=100, requests_per_user=10):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for user in range(concurrent_users):
            for req in range(requests_per_user):
                tasks.append(send_request(session, f"Query {user}-{req}"))
        results = await asyncio.gather(*tasks)
    
    times = [r[0] for r in results]
    print(f"Mean: {mean(times):.3f}s | Median: {median(times):.3f}s | Max: {max(times):.3f}s")
```

## ✅ Production Checklist

**Critical Path:**

- [ ] Replace in-memory state with Redis/Cosmos DB
- [ ] Add authentication & per-user rate limiting
- [ ] Enable response caching for frequent queries
- [ ] Implement circuit breakers for external services
- [ ] Configure auto-scaling (min 2, max 20 replicas)

**Observability:**

- [ ] Add health checks (/health, /ready, /metrics)
- [ ] Enable Application Insights with custom metrics
- [ ] Set up alerting (error rate, latency, quota)
- [ ] Implement request/response logging (PII filtering)

**Deployment:**

- [ ] Set up blue-green deployments
- [ ] Configure APIM policies (rate limit, cache, retry)
- [ ] Load test with 100+ concurrent users
- [ ] Multi-region failover (if needed)

**Cost Optimization:**

- [ ] Use Azure Reserved Instances for baseline
- [ ] Enable auto-pause for dev/test environments
- [ ] Set up cost alerts at 80% and 90% of budget
- [ ] Right-size containers based on metrics

## 💰 Cost Management

1. **Reserved Instances** for predictable baseline load
2. **Aggressive caching** to reduce model API calls
3. **Auto-pause** Cosmos DB in dev/test
4. **Cost alerts** at budget thresholds
5. **Right-sizing** based on actual usage

## 🔄 Graceful Shutdown

```python
@app.on_event("shutdown")
async def shutdown_event():
    # Persist all active sessions
    for session_id, context in session_contexts.items():
        await save_session_to_cosmos(session_id, context)
    
    # Close connections
    await redis_client.close()
    await cosmos_client.close()
```

## 📈 Scaling Metrics

**Monitor:**

- Concurrent sessions (Gauge)
- Response time p50/p95/p99 (Histogram)
- Requests per second (Counter)
- Cache hit rate (%)
- Error rate per endpoint (%)
- Model API call costs ($)

**Scale triggers:**

- CPU > 70%
- Concurrent requests > 50 per instance
- Response time p95 > 3s
- Error rate > 5%

## 🚦 Deployment Strategy

1. **Validate** with `azd provision --preview`
2. **Canary**: 10% traffic to new version
3. **Monitor**: 15 minutes for errors/latency
4. **Promote**: 100% traffic if healthy
5. **Rollback**: Instant if issues detected

---

**Next Steps:**  
Start with session management → caching → auto-scaling for immediate impact.
