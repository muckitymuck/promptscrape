"""
Gemini Site Mapper - A tool for crawling websites and creating relevance-based site maps
Uses Gemini for content extraction and analysis
"""

import os
import sys
import json
import asyncio
import nest_asyncio
import re
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

# Apply nest_asyncio to allow async operations within Jupyter-like environments
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class LinkNode(BaseModel):
    """Model for a link node in the site map"""
    url: str = Field(..., description="The URL of the page")
    title: str = Field(None, description="The title of the page")
    relevance_score: float = Field(0.0, description="Relevance score from 0.0 to 1.0")
    relevance_reason: str = Field(None, description="Reason for the relevance score")
    content_summary: str = Field(None, description="Brief summary of the page content")
    outgoing_links: List[str] = Field(default_factory=list, description="List of outgoing links from this page")
    depth: int = Field(0, description="Depth level from the base URL")
    visited: bool = Field(False, description="Whether this link has been visited")
    parent_url: str = Field(None, description="The parent URL that led to this URL")

class GeminiSiteMapper:
    def __init__(self, base_url: str, search_prompt: str, max_depth: int = 3, max_pages: int = 50):
        self.base_url = base_url
        self.search_prompt = search_prompt
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls = set()
        self.site_map = {}  # Dictionary of LinkNode objects
        self.domain = urlparse(base_url).netloc
        
        # Create necessary directories
        os.makedirs('site_maps', exist_ok=True)
        
        # Initialize Gemini model
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        
        print(f"Initialized GeminiSiteMapper for {base_url}")
        print(f"Search prompt: {search_prompt}")
        print(f"Maximum depth: {max_depth}")
        print(f"Maximum pages: {max_pages}")
        
        # Configure session with common headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })

    def normalize_url(self, url: str, parent_url: str = None) -> str:
        """Normalize the URL to handle relative paths and ensure proper format"""
        if not url or url == '/' or url == '#' or url.startswith('javascript:'):
            return None
            
        # Remove fragments
        if '#' in url:
            url = url.split('#')[0]
            
        # Handle protocol-relative URLs (//example.com)
        if url.startswith('//'):
            parsed_parent = urlparse(parent_url or self.base_url)
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
        base_for_join = parent_url if parent_url else self.base_url
        
        # Handle relative URLs
        if url.startswith('/'):
            parsed_base = urlparse(self.base_url)
            base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
            return f"{base_domain}{url}"
        
        # Handle URLs without scheme
        if not url.startswith('http'):
            return urljoin(base_for_join, url)
        
        return url

    def is_valid_url(self, url: str) -> bool:
        """Check if the URL is valid and should be crawled"""
        if not url:
            return False
            
        parsed_url = urlparse(url)
        
        # Check if the URL is from the same domain or a subdomain
        base_domain = self.domain.split('.')[-2:] if len(self.domain.split('.')) > 1 else [self.domain]
        url_domain = parsed_url.netloc.split('.')[-2:] if len(parsed_url.netloc.split('.')) > 1 else [parsed_url.netloc]
        
        if '.'.join(base_domain) != '.'.join(url_domain):
            return False
        
        # Skip URLs with certain extensions
        skip_extensions = [
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.ico',  # Images
            '.css', '.js', '.json', '.xml',  # Web resources
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',  # Documents
            '.zip', '.tar', '.gz', '.rar',  # Archives
            '.woff', '.woff2', '.ttf', '.eot',  # Fonts
            '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm'  # Media
        ]
        
        if any(parsed_url.path.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        # Check for product or specification pages - these should always be allowed
        is_product_page = any(indicator in parsed_url.path.lower() for indicator in [
            '/product/', '/item/', '/detail/', '/specs/', '/laptop/', '/desktop/', '/computer/',
            '/victus/', '/omen/', '/pavilion/', '/envy/', '/elitebook/', '/probook/',  # HP specific
            '/xps/', '/inspiron/', '/alienware/', '/latitude/', '/precision/',  # Dell specific
            '/thinkpad/', '/ideapad/', '/yoga/', '/legion/',  # Lenovo specific
            '/rog/', '/tuf/', '/zenbook/', '/vivobook/',  # Asus specific
            '/predator/', '/nitro/', '/swift/', '/aspire/',  # Acer specific
            '/surface/',  # Microsoft specific
            '/model/', '/series/', '/configuration/', '/tech/', '/feature/'
        ])
        
        if is_product_page:
            return True
            
        # Skip common non-content URLs unless they're search pages
        skip_patterns = [
            '/cdn-cgi/', '/wp-admin/', '/wp-includes/',  # CMS/CDN paths
            '/login', '/logout', '/signin', '/signout', '/register',  # Auth paths
            '/cart', '/checkout', '/basket', '/shopping-cart',  # E-commerce paths
            '/account', '/profile', '/user/',  # User account paths
            '/feed/', '/rss/', '/atom/',  # Feed paths
            '/print/', '/email/', '/share/',  # Utility paths
            '/comment', '/trackback', '/pingback'  # Comment paths
        ]
        
        # Don't skip search pages on e-commerce sites
        is_search_page = 'search' in parsed_url.path.lower() or 'q=' in parsed_url.query.lower()
        is_ecommerce = any(domain in self.domain.lower() for domain in [
            'hp.com', 'dell.com', 'lenovo.com', 'asus.com', 'acer.com', 'microsoft.com',
            'amazon.com', 'bestbuy.com', 'newegg.com', 'walmart.com', 'target.com'
        ])
        
        if any(pattern in parsed_url.path.lower() for pattern in skip_patterns):
            return is_search_page and is_ecommerce
        
        return True

    async def extract_content(self, url: str) -> tuple[str, str]:
        """Extract content and links from page for parsing"""
        try:
            # Fetch the page content with additional headers for HP website
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0'
            }
            
            response = self.session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Print response status and headers for debugging
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            html_content = response.text
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else "No title"
            
            # Extract main content (focus on article, main, or content areas)
            content = ""
            
            # Special handling for search pages
            if 'search.aspx' in url:
                # First extract all product links
                product_links = []
                
                # Look for product containers with more specific HP classes
                product_containers = soup.find_all(['div', 'article'], class_=re.compile(
                    'product-item|product-card|search-result-item|product-tile|product-grid-item|'
                    'product-list-item|item-product|product-result|search-result|product-container',
                    re.I
                ))
                
                print(f"Found {len(product_containers)} product containers")
                
                for container in product_containers:
                    # Extract product title and link
                    title_elem = container.find(['h2', 'h3', 'h4', 'a'], class_=re.compile(
                        'product-title|item-title|result-title|product-name|product-heading|'
                        'product-link|item-link|result-link',
                        re.I
                    ))
                    
                    if title_elem:
                        link = title_elem.get('href')
                        if link:
                            normalized_url = self.normalize_url(link, url)
                            if normalized_url:
                                product_links.append(normalized_url)
                                print(f"Found product link: {normalized_url}")
                    
                    # Extract product details
                    details = container.find(['div', 'p'], class_=re.compile(
                        'product-details|item-details|result-details|product-description|'
                        'product-specs|product-features|item-description|result-description',
                        re.I
                    ))
                    if details:
                        content += details.get_text(separator=' ', strip=True) + "\n"
                
                # If no products found in containers, try direct link search
                if not product_links:
                    print("No products found in containers, trying direct link search")
                    for a_tag in soup.find_all('a', href=True):
                        link = a_tag['href']
                        if self.is_product_page(link):
                            normalized_url = self.normalize_url(link, url)
                            if normalized_url:
                                product_links.append(normalized_url)
                                print(f"Found product link (direct search): {normalized_url}")
                
                # Update outgoing links in site map
                if url in self.site_map:
                    self.site_map[url].outgoing_links = list(set(product_links))
                    print(f"Updated outgoing links: {len(self.site_map[url].outgoing_links)} links")
                
                # Add product links to content for context
                if product_links:
                    content += "\nProduct Links Found:\n" + "\n".join(product_links)
            
            # For product pages, first try to find product specifications
            elif self.is_product_page(url):
                # First try to find product specifications
                spec_sections = soup.find_all(['div', 'section'], class_=re.compile(
                    'specs|specifications|technical-details|product-info|product-specs|'
                    'technical-specs|product-features|product-details',
                    re.I
                ))
                if spec_sections:
                    content = spec_sections[0].get_text(separator=' ', strip=True)
                
                # If no specs found, try to find product details
                if not content:
                    product_details = soup.find_all(['div', 'section'], class_=re.compile(
                        'product-details|product-description|product-overview|product-summary|'
                        'product-info|product-content',
                        re.I
                    ))
                    if product_details:
                        content = product_details[0].get_text(separator=' ', strip=True)
            
            # If still no content, try other content areas
            if not content:
                # Try to find main content areas
                main_content = soup.find(['article', 'main', 'div'], class_=re.compile(
                    'content|main-content|article-content|product-content|page-content|'
                    'main-wrapper|content-wrapper',
                    re.I
                ))
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
            
        except requests.exceptions.RequestException as e:
            print(f"Request error for {url}: {str(e)}")
            return "No title", f"Request error: {str(e)}"
        except Exception as e:
            print(f"Error extracting content from {url}: {str(e)}")
            return "No title", f"Error extracting content: {str(e)}"

    async def evaluate_content_relevance(self, url: str, content: str) -> Tuple[float, str, str]:
        """Evaluate the relevance of the content to the search prompt"""
        try:
            # Check if this is a search results page
            is_search_page = 'search' in url.lower() or 'q=' in url.lower()
            is_ecommerce = any(domain in self.domain.lower() for domain in [
                'hp.com', 'dell.com', 'lenovo.com', 'asus.com', 'acer.com', 'microsoft.com'
            ])
            
            prompt = f"""
            Evaluate the relevance of this web page content to the search prompt:
            SEARCH PROMPT: "{self.search_prompt}"
            
            {f'''IMPORTANT: This is a search results page on an e-commerce site. 
            Consider it relevant if it shows results related to the products we're looking for 
            or contains links to product specification pages.''' if is_search_page and is_ecommerce else ''}
            
            Provide:
            1. A relevance score from 0.0 to 1.0:
               - 1.0: Contains exactly what we're looking for
               - 0.7-0.9: Contains most of what we're looking for
               - 0.4-0.6: Contains some useful information
               - 0.1-0.3: Has minimal useful information
               - 0.0: Not relevant at all
            
            2. A brief explanation for the score
            3. A concise content summary
            
            Format as JSON with:
            - relevance_score: float
            - relevance_reason: string
            - content_summary: string
            
            URL: {url}
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
                float(evaluation.get('relevance_score', 0.0)),
                evaluation.get('relevance_reason', "No reason provided"),
                evaluation.get('content_summary', "No summary provided")
            )
            
        except Exception as e:
            print(f"Error evaluating content relevance for {url}: {e}")
            return 0.0, f"Error: {str(e)}", "No summary available due to error"

    def _extract_json_from_response(self, text):
        """Extract JSON from Gemini response"""
        json_match = re.search(r'{.*}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        return "{}"

    async def crawl(self):
        """Crawl the website and build a site map with relevance scores"""
        print(f"Starting crawl of {self.base_url} with prompt: {self.search_prompt}")
        
        # Initialize with the base URL
        base_node = LinkNode(url=self.base_url, depth=0)
        self.site_map[self.base_url] = base_node
        
        # Queue of URLs to visit (url, depth)
        queue = [(self.base_url, 0)]
        queued_urls = {self.base_url}
        
        # Progress bar
        pbar = tqdm(total=self.max_pages, desc="Crawling pages")
        
        while queue and len(self.visited_urls) < self.max_pages:
            # Sort queue to prioritize product and search pages
            if len(self.visited_urls) > 5:
                queue.sort(key=lambda x: (
                    -1 if self._is_product_or_search_page(x[0]) else 0,
                    -self._get_parent_relevance(x[0]),
                    x[1]
                ))
            
            current_url, depth = queue.pop(0)
            queued_urls.remove(current_url)
            
            if current_url in self.visited_urls or depth > self.max_depth:
                continue
            
            print(f"\nCrawling: {current_url} (Depth: {depth})")
            self.visited_urls.add(current_url)
            
            # Mark as visited in the site map
            if current_url in self.site_map:
                self.site_map[current_url].visited = True
            
            # Extract content and links
            title, content = await self.extract_content(current_url)
            
            # Evaluate content relevance
            relevance_score, relevance_reason, content_summary = await self.evaluate_content_relevance(current_url, content)
            
            # Update the site map
            if current_url in self.site_map:
                node = self.site_map[current_url]
                node.title = title
                node.relevance_score = relevance_score
                node.relevance_reason = relevance_reason
                node.content_summary = content_summary
            
            print(f"Relevance score: {relevance_score:.2f} - {relevance_reason}")
            
            # Add new links to the queue
            for link in self.site_map[current_url].outgoing_links:
                if link not in self.site_map and link not in queued_urls:
                    self.site_map[link] = LinkNode(
                        url=link,
                        depth=depth + 1,
                        parent_url=current_url
                    )
                    
                    if depth + 1 <= self.max_depth:
                        queue.append((link, depth + 1))
                        queued_urls.add(link)
            
            pbar.update(1)
            pbar.set_postfix({"Queue": len(queue), "Mapped": len(self.site_map)})
        
        pbar.close()
        print(f"Crawl completed. Visited {len(self.visited_urls)} pages out of {len(self.site_map)} discovered URLs.")
        
        return self.site_map

    def _is_product_or_search_page(self, url: str) -> bool:
        """Check if a URL is likely a product page or search results page"""
        if not url:
            return False
            
        parsed_url = urlparse(url)
        
        # Check for product page indicators
        product_indicators = [
            '/product/', '/item/', '/detail/', '/specs/', '/laptop/', '/desktop/', '/computer/',
            '/victus/', '/omen/', '/pavilion/', '/envy/', '/elitebook/', '/probook/',  # HP specific
            '/xps/', '/inspiron/', '/alienware/', '/latitude/', '/precision/',  # Dell specific
            '/thinkpad/', '/ideapad/', '/yoga/', '/legion/',  # Lenovo specific
            '/rog/', '/tuf/', '/zenbook/', '/vivobook/',  # Asus specific
            '/predator/', '/nitro/', '/swift/', '/aspire/',  # Acer specific
            '/surface/',  # Microsoft specific
            '/model/', '/series/', '/configuration/', '/tech/', '/feature/'
        ]
        
        # Check for search page indicators
        is_search_page = 'search' in parsed_url.path.lower() or 'q=' in parsed_url.query.lower()
        
        return any(indicator in parsed_url.path.lower() for indicator in product_indicators) or is_search_page

    def _get_parent_relevance(self, url: str) -> float:
        """Get the relevance score of the parent URL"""
        if url not in self.site_map:
            return 0.0
        parent_url = self.site_map[url].parent_url
        if not parent_url or parent_url not in self.site_map:
            return 0.0
        return self.site_map[parent_url].relevance_score

    def generate_site_map_report(self):
        """Generate a comprehensive report of the site map"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"site_maps/site_map_{timestamp}.json"
        
        # Convert site map to serializable format
        serializable_map = {}
        for url, node in self.site_map.items():
            serializable_map[url] = node.model_dump()
        
        # Save to JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_map, f, indent=2)
        
        print(f"Site map saved to: {filename}")
        
        # Generate markdown report
        md_filename = f"site_maps/site_map_report_{timestamp}.md"
        
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write(f"# Site Map Report\n\n")
            f.write(f"**Base URL:** {self.base_url}\n\n")
            f.write(f"**Search Prompt:** {self.search_prompt}\n\n")
            f.write(f"**Crawl Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Pages Visited:** {len(self.visited_urls)}\n\n")
            
            f.write("## Most Relevant Pages\n\n")
            
            # Sort by relevance score
            relevant_pages = sorted(
                [node for node in self.site_map.values() if node.visited],
                key=lambda x: x.relevance_score,
                reverse=True
            )
            
            f.write("| URL | Title | Relevance | Summary |\n")
            f.write("|-----|-------|-----------|----------|\n")
            
            for page in relevant_pages[:20]:
                title = page.title or "No title"
                score = f"{page.relevance_score:.2f}"
                summary = page.content_summary or "No summary"
                if len(summary) > 100:
                    summary = summary[:97] + "..."
                f.write(f"| [{title}]({page.url}) | {title} | {score} | {summary} |\n")
            
            f.write("\n\n## Complete Site Map\n\n")
            
            for depth in range(self.max_depth + 1):
                depth_pages = [node for node in self.site_map.values() if node.depth == depth]
                
                if depth_pages:
                    f.write(f"\n### Depth {depth}\n\n")
                    
                    for page in depth_pages:
                        visited_mark = "✓" if page.visited else "✗"
                        title = page.title or "Not visited"
                        score = f"{page.relevance_score:.2f}" if page.visited else "N/A"
                        
                        f.write(f"- {visited_mark} [{title}]({page.url}) (Relevance: {score})\n")
                        if page.relevance_reason and page.visited:
                            f.write(f"  - *{page.relevance_reason}*\n")
        
        print(f"Site map report saved to: {md_filename}")
        return filename, md_filename

    def visualize_site_map(self):
        """Create a visual representation of the site map"""
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for url, node in self.site_map.items():
            if node.visited:
                G.add_node(url, relevance=node.relevance_score, title=node.title or "No title")
        
        # Add edges
        for url, node in self.site_map.items():
            if node.visited and node.outgoing_links:
                for link in node.outgoing_links:
                    if link in self.site_map and self.site_map[link].visited:
                        G.add_edge(url, link)
        
        if len(G.nodes()) < 1:
            print("Not enough visited pages to create a visualization.")
            return None, None
        
        # Create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Full site map
        plt.figure(figsize=(15, 12))
        pos = nx.spring_layout(G, seed=42, k=0.15)
        
        # Draw nodes colored by relevance
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_color=[G.nodes[node]['relevance'] for node in G.nodes()],
            cmap=plt.cm.YlOrRd,
            node_size=300,
            alpha=0.8
        )
        
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True)
        
        # Add labels for high-relevance nodes
        high_relevance_nodes = {
            node: G.nodes[node]['title'] 
            for node in G.nodes() 
            if G.nodes[node]['relevance'] > 0.6
        }
        nx.draw_networkx_labels(G, pos, labels=high_relevance_nodes, font_size=8)
        
        plt.colorbar(nodes, label='Relevance Score')
        plt.title(f"Site Map for {self.base_url}\nColored by relevance to: {self.search_prompt}")
        
        viz_filename = f"site_maps/site_map_visualization_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(viz_filename, dpi=300)
        plt.close()
        
        # High-relevance visualization
        high_relevance_nodes_list = [
            node for node in G.nodes() 
            if G.nodes[node]['relevance'] > 0.4
        ]
        
        if not high_relevance_nodes_list:
            print("No high-relevance nodes found.")
            return viz_filename, None
        
        plt.figure(figsize=(12, 10))
        H = G.subgraph(high_relevance_nodes_list)
        pos_h = nx.spring_layout(H, seed=42, k=0.2)
        
        h_nodes = nx.draw_networkx_nodes(
            H, pos_h,
            node_color=[H.nodes[node]['relevance'] for node in H.nodes()],
            cmap=plt.cm.YlOrRd,
            node_size=400,
            alpha=0.9
        )
        
        nx.draw_networkx_edges(H, pos_h, alpha=0.4, arrows=True)
        
        labels = {node: H.nodes[node]['title'] for node in H.nodes()}
        nx.draw_networkx_labels(H, pos_h, labels=labels, font_size=8)
        
        plt.colorbar(h_nodes, label='Relevance Score')
        plt.title(f"High Relevance Pages for: {self.search_prompt}")
        
        high_rel_viz_filename = f"site_maps/high_relevance_map_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(high_rel_viz_filename, dpi=300)
        plt.close()
        
        return viz_filename, high_rel_viz_filename

    def is_product_page(self, url: str) -> bool:
        """Check if a URL is likely a product page"""
        if not url:
            return False
            
        parsed_url = urlparse(url)
        
        # Check for product page indicators
        product_indicators = [
            '/product/', '/item/', '/detail/', '/specs/', '/laptop/', '/desktop/', '/computer/',
            '/victus/', '/omen/', '/pavilion/', '/envy/', '/elitebook/', '/probook/',  # HP specific
            '/xps/', '/inspiron/', '/alienware/', '/latitude/', '/precision/',  # Dell specific
            '/thinkpad/', '/ideapad/', '/yoga/', '/legion/',  # Lenovo specific
            '/rog/', '/tuf/', '/zenbook/', '/vivobook/',  # Asus specific
            '/predator/', '/nitro/', '/swift/', '/aspire/',  # Acer specific
            '/surface/',  # Microsoft specific
            '/model/', '/series/', '/configuration/', '/tech/', '/feature/'
        ]
        
        # Check for excluded patterns (non-product pages)
        excluded_patterns = [
            '/support/', '/contact/', '/help/', '/about/', '/careers/', '/news/',
            '/privacy/', '/terms/', '/legal/', '/accessibility/', '/sitemap/',
            '/login/', '/register/', '/account/', '/cart/', '/checkout/'
        ]
        
        # First check if it's an excluded page
        if any(pattern in parsed_url.path.lower() for pattern in excluded_patterns):
            return False
            
        # Then check if it's a product page
        return any(indicator in parsed_url.path.lower() for indicator in product_indicators)

async def main():
    if len(sys.argv) < 3:
        print("Usage: python gemini_site_mapper.py <base_url> <search_prompt> [max_depth] [max_pages]")
        sys.exit(1)
    
    base_url = sys.argv[1]
    search_prompt = sys.argv[2]
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    max_pages = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    
    mapper = GeminiSiteMapper(base_url, search_prompt, max_depth, max_pages)
    
    # Crawl the site
    await mapper.crawl()
    
    # Generate reports
    json_file, md_file = mapper.generate_site_map_report()
    
    # Visualize the site map
    viz_file, high_rel_viz = mapper.visualize_site_map()
    
    print(f"\nProcess completed successfully!")
    print(f"Site map data: {json_file}")
    print(f"Site map report: {md_file}")
    
    if viz_file:
        print(f"Site map visualization: {viz_file}")
    if high_rel_viz:
        print(f"High relevance visualization: {high_rel_viz}")
    
    # Print statistics
    total_urls = len(mapper.site_map)
    visited_urls = len(mapper.visited_urls)
    relevant_urls = sum(1 for node in mapper.site_map.values() 
                      if node.visited and node.relevance_score >= 0.5)
    highly_relevant_urls = sum(1 for node in mapper.site_map.values() 
                             if node.visited and node.relevance_score >= 0.7)
    
    print(f"\nStatistics:")
    print(f"- Total URLs discovered: {total_urls}")
    print(f"- URLs visited: {visited_urls} ({visited_urls/total_urls*100:.1f}% of discovered)")
    print(f"- Relevant URLs (score ≥ 0.5): {relevant_urls} ({relevant_urls/visited_urls*100:.1f}% of visited)")
    print(f"- Highly relevant URLs (score ≥ 0.7): {highly_relevant_urls} ({highly_relevant_urls/visited_urls*100:.1f}% of visited)")
    
    print("\nTop 5 most relevant URLs:")
    top_urls = sorted(
        [node for node in mapper.site_map.values() if node.visited],
        key=lambda x: x.relevance_score,
        reverse=True
    )[:5]
    
    for i, node in enumerate(top_urls, 1):
        print(f"{i}. [{node.relevance_score:.2f}] {node.title or 'No title'}: {node.url}")
        print(f"   {node.relevance_reason}")

if __name__ == "__main__":
    asyncio.run(main())
