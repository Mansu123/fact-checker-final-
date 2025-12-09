
#!/usr/bin/env python3
"""
OpenSearch Connection Test and Setup Verification
"""

import sys
from opensearchpy import OpenSearch

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header():
    print(f"{BLUE}{'='*60}")
    print("OpenSearch Connection Test")
    print(f"{'='*60}{RESET}\n")


def test_connection():
    """Test OpenSearch connection"""
    print(f"{YELLOW}Testing OpenSearch connection...{RESET}\n")
    
    try:
        client = OpenSearch(
            hosts=[{'host': 'localhost', 'port': 9200}],
            http_compress=True,
            http_auth=('admin', 'admin12345678'),
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False
        )
        
        # Test ping
        if client.ping():
            print(f"{GREEN}✓ Successfully connected to OpenSearch{RESET}")
        else:
            print(f"{RED}✗ OpenSearch is not responding{RESET}")
            return False
        
        # Get cluster info
        info = client.info()
        print(f"\n{BLUE}Cluster Information:{RESET}")
        print(f"  Name: {info['cluster_name']}")
        print(f"  Version: {info['version']['number']}")
        
        # Get cluster health
        health = client.cluster.health()
        status = health['status']
        status_color = GREEN if status == 'green' else YELLOW if status == 'yellow' else RED
        print(f"  Status: {status_color}{status}{RESET}")
        print(f"  Nodes: {health['number_of_nodes']}")
        print(f"  Data Nodes: {health['number_of_data_nodes']}")
        
        # List indices
        indices = client.cat.indices(format='json')
        if indices:
            print(f"\n{BLUE}Existing Indices:{RESET}")
            for idx in indices:
                print(f"  - {idx['index']} (docs: {idx['docs.count']}, size: {idx['store.size']})")
        else:
            print(f"\n{YELLOW}No indices found yet{RESET}")
        
        # Test kNN plugin
        print(f"\n{YELLOW}Checking kNN plugin...{RESET}")
        try:
            cat_plugins = client.cat.plugins(format='json')
            knn_found = any('knn' in plugin.get('component', '').lower() for plugin in cat_plugins)
            if knn_found:
                print(f"{GREEN}✓ kNN plugin is installed{RESET}")
            else:
                print(f"{YELLOW}⚠ kNN plugin not found - may need to install{RESET}")
        except:
            print(f"{YELLOW}⚠ Could not verify kNN plugin{RESET}")
        
        client.close()
        
        print(f"\n{GREEN}{'='*60}")
        print("✓ OpenSearch is ready for use!")
        print(f"{'='*60}{RESET}\n")
        
        return True
        
    except Exception as e:
        print(f"{RED}✗ Connection failed: {e}{RESET}")
        print(f"\n{YELLOW}Troubleshooting:{RESET}")
        print("1. Make sure OpenSearch is running:")
        print("   docker-compose up -d opensearch")
        print("2. Wait for OpenSearch to start (may take 1-2 minutes)")
        print("3. Check if port 9200 is accessible:")
        print("   curl -k -u admin:admin12345678 https://localhost:9200")
        return False


def show_next_steps():
    """Show next steps after successful connection"""
    print(f"{BLUE}Next Steps:{RESET}\n")
    print(f"1. {GREEN}Preprocess your data:{RESET}")
    print(f"   python data_preprocessing.py\n")
    print(f"2. {GREEN}Start the API server:{RESET}")
    print(f"   python backend.py\n")
    print(f"3. {GREEN}Test the API:{RESET}")
    print(f"   curl http://localhost:8000/health\n")


def main():
    print_header()
    
    if test_connection():
        show_next_steps()
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Cancelled by user{RESET}")
        sys.exit(0)