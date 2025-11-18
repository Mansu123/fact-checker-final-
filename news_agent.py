import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict
import time
from vector_db import get_vector_db, EmbeddingService
from config import settings

class NewsCollectionAgent:
    """
    Daily news collection agent for TRUSTED sources only:
    - Prothom Alo
    - The Daily Star
    - BBC Bangla
    - Bangladesh Pratidin
    - NCTB Books / Govt PDFs
    """
    
    def __init__(self):
        self.vector_db = get_vector_db()
        self.embedding_service = EmbeddingService()
        self.collection_name = "news_articles"
        self.sources = self.load_trusted_sources()
    
    def load_trusted_sources(self) -> List[Dict]:
        """Load TRUSTED news sources configuration ONLY"""
        return [
            {
                "name": "Prothom Alo",
                "rss": "https://www.prothomalo.com/feed/",
                "language": "bn",
                "type": "rss",
                "trusted": True
            },
            {
                "name": "The Daily Star",
                "rss": "https://www.thedailystar.net/rss.xml",
                "language": "en",
                "type": "rss",
                "trusted": True
            },
            {
                "name": "BBC Bangla",
                "rss": "https://feeds.bbci.co.uk/bengali/rss.xml",
                "language": "bn",
                "type": "rss",
                "trusted": True
            },
            {
                "name": "Bangladesh Pratidin",
                "url": "https://www.bd-pratidin.com/",
                "language": "bn",
                "type": "scrape",
                "trusted": True
            }
            # Add NCTB Books / Govt PDFs here when available
        ]
    
    def collect_from_rss(self, source: Dict) -> List[Dict]:
        """Collect news from RSS feed"""
        articles = []
        
        try:
            print(f"  Fetching from {source['name']}...")
            feed = feedparser.parse(source['rss'])
            
            for entry in feed.entries[:20]:  # Get latest 20 articles
                article = {
                    "title": entry.get('title', ''),
                    "content": entry.get('summary', ''),
                    "link": entry.get('link', ''),
                    "published": entry.get('published', str(datetime.now())),
                    "source": source['name'],
                    "language": source['language'],
                    "trusted": source.get('trusted', True),
                    "collected_at": str(datetime.now())
                }
                articles.append(article)
            
            print(f"  ✓ Collected {len(articles)} articles from {source['name']}")
        except Exception as e:
            print(f"  ✗ Error collecting from {source['name']}: {e}")
        
        return articles
    
    def collect_from_scrape(self, source: Dict) -> List[Dict]:
        """Scrape news from website"""
        articles = []
        
        try:
            print(f"  Fetching from {source['name']}...")
            response = requests.get(source['url'], timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links (customize selectors for each site)
            article_links = soup.find_all('a', class_=['article', 'news-item', 'post'])[:20]
            
            for link in article_links:
                article = {
                    "title": link.get_text().strip(),
                    "link": link.get('href', ''),
                    "source": source['name'],
                    "language": source['language'],
                    "trusted": source.get('trusted', True),
                    "collected_at": str(datetime.now())
                }
                articles.append(article)
            
            print(f"  ✓ Scraped {len(articles)} articles from {source['name']}")
        except Exception as e:
            print(f"  ✗ Error scraping {source['name']}: {e}")
        
        return articles
    
    def collect_daily_news(self) -> List[Dict]:
        """Collect news from all TRUSTED sources"""
        all_articles = []
        
        print("\n" + "="*70)
        print("📰 Starting TRUSTED news collection...")
        print(f"⏰ Time: {datetime.now()}")
        print("="*70)
        print("\nTrusted Sources:")
        for source in self.sources:
            print(f"  • {source['name']}")
        print("="*70 + "\n")
        
        for source in self.sources:
            if source['type'] == 'rss':
                articles = self.collect_from_rss(source)
            else:
                articles = self.collect_from_scrape(source)
            
            all_articles.extend(articles)
            time.sleep(2)  # Be respectful to servers
        
        print(f"\n{'='*70}")
        print(f"✓ Total collected: {len(all_articles)} articles from TRUSTED sources")
        print(f"{'='*70}\n")
        return all_articles
    
    def store_news_in_vectordb(self, articles: List[Dict]):
        """Store collected news in vector database"""
        if not articles:
            print("No articles to store")
            return
        
        print("📊 Storing articles in vector database...")
        
        # Create collection if it doesn't exist
        try:
            # Check if collection exists, if not create it
            self.vector_db.create_collection(self.collection_name, 768)
        except:
            print("  Collection already exists, continuing...")
        
        # Prepare documents and embeddings
        documents = []
        texts = []
        
        for article in articles:
            # Combine title and content for better search
            full_text = f"{article['title']} {article.get('content', '')}"
            texts.append(full_text)
            
            documents.append({
                "id": hash(article['link']),
                "title": article['title'],
                "content": article.get('content', ''),
                "link": article['link'],
                "source": article['source'],
                "language": article['language'],
                "trusted": article.get('trusted', True),
                "published": article.get('published', ''),
                "collected_at": article['collected_at']
            })
        
        # Generate embeddings
        print("  Generating embeddings for articles...")
        embeddings = self.embedding_service.embed_documents(texts)
        
        # Store in vector database
        self.vector_db.add_documents(self.collection_name, documents, embeddings)
        
        print(f"  ✓ Stored {len(documents)} articles in vector database\n")
    
    def save_collection_log(self, articles: List[Dict]):
        """Save collection log for tracking"""
        log_file = f"news_logs/collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        os.makedirs("news_logs", exist_ok=True)
        
        log_data = {
            "date": str(datetime.now()),
            "total_articles": len(articles),
            "sources": list(set(a['source'] for a in articles)),
            "trusted_only": True,
            "articles": articles
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved collection log: {log_file}\n")
    
    def run_daily_collection(self):
        """Main function to run daily news collection"""
        print("="*70)
        print("🤖 TRUSTED News Collection Agent")
        print("="*70)
        print("Sources: Prothom Alo, The Daily Star, BBC Bangla, Bangladesh Pratidin")
        print("="*70 + "\n")
        
        # Collect news
        articles = self.collect_daily_news()
        
        if not articles:
            print("⚠ No articles collected today")
            return
        
        # Store in vector database
        self.store_news_in_vectordb(articles)
        
        # Save log
        self.save_collection_log(articles)
        
        print("="*70)
        print("✓ Daily collection complete!")
        print(f"  Total articles: {len(articles)}")
        print(f"  All from TRUSTED sources")
        print("="*70 + "\n")
        
        return len(articles)

def run_agent():
    """Run the news collection agent"""
    agent = NewsCollectionAgent()
    agent.run_daily_collection()

if __name__ == "__main__":
    run_agent()