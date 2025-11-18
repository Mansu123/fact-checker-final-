
#!/usr/bin/env python3
"""
Quick OpenSearch Connection Test (No Authentication)
Run this to verify OpenSearch is working
"""

from opensearchpy import OpenSearch

# Connect to OpenSearch WITHOUT authentication (security disabled)
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    http_compress=True,
    use_ssl=False,  # No SSL
    timeout=10
)

# Test connection
try:
    if client.ping():
        print("✅ OpenSearch is running!")
        
        # Get cluster info
        info = client.info()
        print(f"✅ Version: {info['version']['number']}")
        
        # Get cluster health
        health = client.cluster.health()
        print(f"✅ Status: {health['status']}")
        print(f"✅ Nodes: {health['number_of_nodes']}")
        
        print("\n🎉 OpenSearch setup is working perfectly!")
    else:
        print("❌ OpenSearch is not responding")
        print("Run: docker-compose up -d opensearch")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("  1. Check if container is running: docker ps | grep opensearch")
    print("  2. Test from container: docker exec opensearch curl http://localhost:9200")
    print("  3. Check logs: docker logs opensearch | tail -30")