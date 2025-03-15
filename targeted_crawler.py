"""
Targeted Crawler - A tool for analyzing specific URLs based on site map reports
Uses Gemini for content analysis and relevance scoring
"""

import os
import sys
import json
import asyncio
import nest_asyncio
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

# Apply nest_asyncio to allow async operations within Jupyter-like environments
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class PageAnalysis(BaseModel):
    """Model for page analysis results"""
    url: str = Field(..., description="The URL of the page")
    title: str = Field(None, description="Page title")
    content_summary: str = Field(None, description="Summary of the page content")
    relevance_score: float = Field(0.0, description="Relevance score to the search prompt")
    relevance_reason: str = Field(None, description="Explanation of relevance")
    source_relevance: float = Field(0.0, description="Original relevance score from site map")
    source_reason: str = Field(None, description="Original relevance reason from site map")
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())

class TargetedCrawler:
    def __init__(self, site_map_file: str, search_prompt: str, max_pages: int = 10):
        self.site_map_file = site_map_file
        self.search_prompt = search_prompt
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.results: List[PageAnalysis] = []
        
        # Create necessary directories
        os.makedirs('analysis_results', exist_ok=True)
        
        # Initialize Gemini model
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Configure session with common headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        
        print(f"Initialized TargetedCrawler")
        print(f"Site map file: {site_map_file}")
        print(f"Search prompt: {search_prompt}")
        print(f"Max pages to analyze: {max_pages}")

    def load_site_map(self) -> List[Dict[str, Any]]:
        """Load and parse the site map file"""
        try:
            with open(self.site_map_file, 'r', encoding='utf-8') as f:
                site_map = json.load(f)
            
            # Convert to list of entries and sort by relevance score
            entries = []
            for url, data in site_map.items():
                entries.append({
                    'url': url,
                    'relevance_score': data.get('relevance_score', 0.0),
                    'relevance_reason': data.get('relevance_reason', ''),
                    'content_summary': data.get('content_summary', '')
                })
            
            # Sort by relevance score
            entries.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            print(f"Loaded {len(entries)} entries from site map")
            return entries
            
        except Exception as e:
            print(f"Error loading site map: {e}")
            return []

    def is_product_page(self, url: str) -> bool:
        """Check if the URL is likely to be a product page"""
        if not url:
            return False
            
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        # Exclude support, contact, and other non-product pages
        exclude_patterns = [
            '/support/', '/contact/', '/help/', '/faq/', '/service/',
            '/warranty/', '/repair/', '/service-center/', '/instant-ink/',
            '/drivers/', '/software/', '/downloads/', '/community/'
        ]
        
        if any(pattern in path for pattern in exclude_patterns):
            return False
        
        # HP-specific product page indicators
        product_indicators = [
            '/product/', '/item/', '/detail/', '/specs/', '/laptop/', '/desktop/', '/computer/',
            '/victus/', '/omen/', '/pavilion/', '/envy/', '/elitebook/', '/probook/',
            '/model/', '/series/', '/configuration/', '/tech/', '/feature/',
            '/shop/', '/store/'
        ]
        
        # Must contain at least one product indicator
        has_product_indicator = any(indicator in path for indicator in product_indicators)
        
        # Additional checks for HP store
        if 'hp.com/ca-en/shop' in url:
            # Must be a product detail page, not a category or search page
            return has_product_indicator and not any(x in path for x in ['search.aspx', 'category', 'list'])
        
        return has_product_indicator

    async def extract_product_links(self, url: str) -> List[str]:
        """Extract product links from a search results page"""
        try:
            # Fetch the page content
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            html_content = response.text
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for product links in common containers
            product_links = []
            
            # Common product container classes/IDs
            product_containers = [
                'product-item', 'product-card', 'product-tile', 'product-grid-item',
                'search-result-item', 'product-list-item', 'item-product'
            ]
            
            for container in product_containers:
                for element in soup.find_all(class_=re.compile(container, re.I)):
                    # Look for links within the container
                    for a_tag in element.find_all('a', href=True):
                        link = a_tag['href']
                        if self.is_product_page(link):
                            normalized_url = self.normalize_url(link, url)
                            if normalized_url and normalized_url not in self.visited_urls:
                                product_links.append(normalized_url)
            
            # If no products found in containers, try direct link search
            if not product_links:
                for a_tag in soup.find_all('a', href=True):
                    link = a_tag['href']
                    if self.is_product_page(link):
                        normalized_url = self.normalize_url(link, url)
                        if normalized_url and normalized_url not in self.visited_urls:
                            product_links.append(normalized_url)
            
            # Remove duplicates
            return list(set(product_links))
            
        except Exception as e:
            print(f"Error extracting product links from {url}: {e}")
            return []

    async def evaluate_content_relevance(self, url: str, content: str) -> tuple[float, str]:
        """Evaluate how relevant the content is to the search prompt"""
        try:
            prompt = f"""
            Analyze this content and evaluate its relevance to the search prompt: "{self.search_prompt}"
            
            Consider:
            1. How directly the content addresses the search prompt
            2. The depth and quality of information
            3. Whether it contains specific details or just general information
            4. Whether it's a primary source or just mentions the topic
            
            Format your response as a JSON object with:
            - relevance_score: number between 0 and 1
            - relevance_reason: string explaining the score
            
            URL being analyzed: {url}
            """
            
            # Generate response from Gemini
            response = self.gemini_model.generate_content([
                {"text": prompt},
                {"text": content[:100000]}  # Limit content size
            ])
            
            # Extract JSON from response
            json_text = self._extract_json_from_response(response.text)
            evaluation = json.loads(json_text)
            
            return (
                evaluation.get('relevance_score', 0.0),
                evaluation.get('relevance_reason', 'No relevance explanation provided')
            )
            
        except Exception as e:
            print(f"Error evaluating content relevance for {url}: {e}")
            return 0.0, "Error evaluating relevance"

    async def extract_content(self, url: str) -> tuple[str, str]:
        """Extract content from a page"""
        try:
            # Fetch the page content
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            html_content = response.text
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else "No title"
            
            # Extract main content (focus on article, main, or content areas)
            content = ""
            
            # Special handling for search pages
            if 'search.aspx' in url:
                # Look for search results container
                search_results = soup.find_all(['div', 'section'], class_=re.compile('search-results|product-grid|product-list|search-container', re.I))
                if search_results:
                    content = search_results[0].get_text(separator=' ', strip=True)
                else:
                    # If no specific container, look for product items
                    product_items = soup.find_all(['div', 'article'], class_=re.compile('product-item|product-card|search-result-item', re.I))
                    if product_items:
                        content = "\n".join(item.get_text(separator=' ', strip=True) for item in product_items)
            
            # For product pages, first try to find product specifications
            elif self.is_product_page(url):
                # First try to find product specifications
                spec_sections = soup.find_all(['div', 'section'], class_=re.compile('specs|specifications|technical-details|product-info', re.I))
                if spec_sections:
                    content = spec_sections[0].get_text(separator=' ', strip=True)
                
                # If no specs found, try to find product details
                if not content:
                    product_details = soup.find_all(['div', 'section'], class_=re.compile('product-details|product-description|product-overview', re.I))
                    if product_details:
                        content = product_details[0].get_text(separator=' ', strip=True)
            
            # If still no content, try other content areas
            if not content:
                # Try to find main content areas
                main_content = soup.find(['article', 'main', 'div'], class_=re.compile('content|main-content|article-content|product-content', re.I))
                if main_content:
                    content = main_content.get_text(separator=' ', strip=True)
            
            # If still no content, get all text but clean it up
            if not content:
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "header", "footer"]):
                    script.decompose()
                
                # Get remaining text
                content = soup.get_text(separator=' ', strip=True)
            
            # Clean up the content
            content = re.sub(r'\s+', ' ', content)  # Replace multiple spaces with single space
            content = content.strip()
            
            # Ensure we have some content
            if not content:
                content = "No content found on page"
            
            # Add URL to content for context
            content = f"URL: {url}\n\nContent:\n{content}"
            
            return title, content
            
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return "No title", f"Error extracting content from {url}"

    def _extract_json_from_response(self, text):
        """Extract JSON from Gemini response"""
        json_match = re.search(r'{.*}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        return "{}"

    async def analyze_pages(self):
        """Analyze the most relevant pages from the site map"""
        print(f"Starting analysis for: {self.search_prompt}")
        
        # Load site map entries
        entries = self.load_site_map()
        if not entries:
            print("No entries found in site map")
            return []
        
        # Get top entries based on max_pages
        top_entries = entries[:self.max_pages]
        
        # Progress bar
        pbar = tqdm(total=self.max_pages, desc="Analyzing pages")
        
        for entry in top_entries:
            url = entry['url']
            if url in self.visited_urls:
                continue
                
            print(f"\nProcessing: {url}")
            self.visited_urls.add(url)
            
            # If this is a search page, extract product links
            if 'search.aspx' in url:
                product_links = await self.extract_product_links(url)
                print(f"Found {len(product_links)} product links")
                
                # Process each product link
                for product_url in product_links:
                    if product_url in self.visited_urls:
                        continue
                        
                    print(f"\nAnalyzing product: {product_url}")
                    self.visited_urls.add(product_url)
                    
                    # Extract content
                    title, content = await self.extract_content(product_url)
                    
                    # Evaluate relevance
                    relevance_score, relevance_reason = await self.evaluate_content_relevance(product_url, content)
                    
                    # Create page analysis
                    analysis = PageAnalysis(
                        url=product_url,
                        title=title,
                        content_summary=content[:500] + "..." if len(content) > 500 else content,
                        relevance_score=relevance_score,
                        relevance_reason=relevance_reason,
                        source_relevance=entry['relevance_score'],
                        source_reason=entry['relevance_reason']
                    )
                    
                    # Add to results if relevant enough
                    if relevance_score >= 0.3:  # Minimum relevance threshold
                        self.results.append(analysis)
                        print(f"Found relevant content (score: {relevance_score:.2f})")
                    
                    pbar.update(1)
            else:
                # Process non-search pages as before
                title, content = await self.extract_content(url)
                relevance_score, relevance_reason = await self.evaluate_content_relevance(url, content)
                
                analysis = PageAnalysis(
                    url=url,
                    title=title,
                    content_summary=content[:500] + "..." if len(content) > 500 else content,
                    relevance_score=relevance_score,
                    relevance_reason=relevance_reason,
                    source_relevance=entry['relevance_score'],
                    source_reason=entry['relevance_reason']
                )
                
                if relevance_score >= 0.3:
                    self.results.append(analysis)
                    print(f"Found relevant content (score: {relevance_score:.2f})")
                
                pbar.update(1)
        
        pbar.close()
        print(f"\nAnalysis completed. Processed {len(self.visited_urls)} pages")
        print(f"Found {len(self.results)} relevant pages")
        
        return self.results

    def save_results(self):
        """Save the analysis results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sort results by relevance score
        sorted_results = sorted(self.results, key=lambda x: x.relevance_score, reverse=True)
        
        # Save as JSON
        json_filename = f"analysis_results/results_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump([r.model_dump() for r in sorted_results], f, indent=2)
        
        # Convert to DataFrame and save as CSV
        results_data = []
        for result in sorted_results:
            result_data = {
                'URL': result.url,
                'Title': result.title,
                'Relevance Score': result.relevance_score,
                'Relevance Reason': result.relevance_reason,
                'Source Relevance': result.source_relevance,
                'Source Reason': result.source_reason,
                'Content Summary': result.content_summary,
                'Last Updated': result.last_updated
            }
            results_data.append(result_data)
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(results_data)
        csv_filename = f"analysis_results/results_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        
        print(f"Results saved to:")
        print(f"- JSON: {json_filename}")
        print(f"- CSV: {csv_filename}")
        
        return json_filename, csv_filename

    def normalize_url(self, url: str, parent_url: str = None) -> str:
        """Normalize the URL to handle relative paths and ensure proper format"""
        if not url or url == '/' or url == '#' or url.startswith('javascript:'):
            return None
            
        # Remove fragments
        if '#' in url:
            url = url.split('#')[0]
            
        # Handle protocol-relative URLs (//example.com)
        if url.startswith('//'):
            parsed_parent = urlparse(parent_url or self.site_map_file)
            url = f"{parsed_parent.scheme}:{url}"
            
        # Handle mailto: and tel: links
        if url.startswith(('mailto:', 'tel:', 'sms:')):
            return None
            
        # Fix common issues with URLs
        if '//' in url and not url.startswith('http'):
            url = re.sub(r'([^:])//+', r'\1/', url)
            
        # Handle query parameters
        if '?' in url:
            base_url = url.split('?')[0]
            query = url.split('?')[1]
            
            # Remove tracking parameters
            tracking_params = ['utm_', 'fbclid', 'gclid', 'msclkid', 'zanpid', 'dclid']
            
            if query:
                query_parts = query.split('&')
                filtered_parts = [p for p in query_parts if not any(t in p.lower() for t in tracking_params)]
                
                if filtered_parts:
                    url = f"{base_url}?{'&'.join(filtered_parts)}"
                else:
                    url = base_url
            
        # Use parent URL for relative URLs if provided
        base_for_join = parent_url if parent_url else self.site_map_file
        
        # Handle relative URLs
        if url.startswith('/'):
            parsed_base = urlparse(self.site_map_file)
            base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
            return f"{base_domain}{url}"
        
        # Handle URLs without scheme
        if not url.startswith('http'):
            return urljoin(base_for_join, url)
        
        return url

async def main():
    if len(sys.argv) < 4:
        print("Usage: python targeted_crawler.py <site_map_file> \"<search_prompt>\" <max_pages>")
        sys.exit(1)
    
    site_map_file = sys.argv[1]
    search_prompt = sys.argv[2]
    max_pages = int(sys.argv[3])
    
    crawler = TargetedCrawler(site_map_file, search_prompt, max_pages)
    
    # Analyze pages
    results = await crawler.analyze_pages()
    
    # Save results
    json_file, csv_file = crawler.save_results()
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"- Pages processed: {len(crawler.visited_urls)}")
    print(f"- Relevant pages found: {len(results)}")
    print(f"- Results saved to: {json_file}")
    print(f"- CSV export: {csv_file}")
    
    # Print top results
    if results:
        print("\nTop Results:")
        for result in sorted(results, key=lambda x: x.relevance_score, reverse=True)[:5]:
            print(f"\nTitle: {result.title}")
            print(f"URL: {result.url}")
            print(f"Relevance Score: {result.relevance_score:.2f}")
            print(f"Relevance Reason: {result.relevance_reason}")
            print(f"Source Relevance: {result.source_relevance:.2f}")
            print(f"Source Reason: {result.source_reason}")
            print("-" * 80)

if __name__ == "__main__":
    asyncio.run(main()) 