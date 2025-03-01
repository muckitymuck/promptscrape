"""
Site Mapper - A tool for crawling websites and creating relevance-based site maps
This script crawls websites, evaluates page relevance using AI, and generates 
visual and textual reports of the site structure.
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

# Import functions from jina.py
from jina import get_jina_response

# Apply nest_asyncio to allow async operations within Jupyter-like environments
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

"""
Data model for representing pages in the site map.
Each node contains metadata about a URL including its relevance to the search query,
content summary, and relationships to other pages.
"""

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

"""
Main crawler class that handles site mapping operations.
Manages the crawling process, content evaluation, and report generation.
"""

class SiteMapper:
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
        os.makedirs('jina_responses', exist_ok=True)
        
        # Initialize Gemini model
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        print(f"Initialized SiteMapper for {base_url}")
        print(f"Search prompt: {search_prompt}")
        print(f"Maximum depth: {max_depth}")
        print(f"Maximum pages: {max_pages}")
    
    """
    Standardizes URLs to ensure consistent format and handling.
    - Resolves relative URLs to absolute
    - Handles special cases for specific domains
    - Cleans tracking parameters
    - Fixes common URL formatting issues
    """
    def normalize_url(self, url: str, parent_url: str = None) -> str:
        """Normalize the URL to handle relative paths and ensure proper format"""
        # Handle empty URLs
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
        # 1. Replace double slashes in path (not in protocol)
        if '//' in url and not url.startswith('http'):
            url = re.sub(r'([^:])//+', r'\1/', url)
            
        # 2. Handle URLs with known issues for specific sites
        if 'rog.asus.com' in (parent_url or self.base_url):
            # For ROG website, ensure proper group URLs
            # This is a general pattern matching approach rather than hardcoding
            if '/laptops/' in url and not any(pattern in url for pattern in ['-group', '/group/']):
                # Look for product category URLs that should use the -group suffix
                for category in ['laptops', 'motherboards', 'graphics-cards', 'monitors', 'phones', 'desktops']:
                    if f'/{category}/' in url:
                        url = url.replace(f'/{category}/', f'/{category}-group/')
                        break
            
        # Handle query parameters
        if '?' in url:
            base_url = url.split('?')[0]
            query = url.split('?')[1]
            
            # Remove tracking parameters
            tracking_params = ['utm_', 'fbclid', 'gclid', 'msclkid', 'zanpid', 'dclid']
            
            # If we have query parameters
            if query:
                query_parts = query.split('&')
                filtered_parts = [p for p in query_parts if not any(t in p.lower() for t in tracking_params)]
                
                # Rebuild the URL with filtered query parameters
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
    
    """
    Determines if a URL should be crawled based on:
    - Domain matching (including subdomains)
    - File extensions (skips media/resource files)
    - URL patterns (skips login, admin, etc.)
    - Query parameters
    """
    def is_valid_url(self, url: str) -> bool:
        """Check if the URL is valid and should be crawled"""
        if not url:
            return False
            
        # Parse the URL
        parsed_url = urlparse(url)
        
        # Check if the URL is from the same domain or a subdomain
        base_domain = self.domain.split('.')[-2:] if len(self.domain.split('.')) > 1 else [self.domain]
        url_domain = parsed_url.netloc.split('.')[-2:] if len(parsed_url.netloc.split('.')) > 1 else [parsed_url.netloc]
        
        # Allow same domain or subdomains of the same organization
        if '.'.join(base_domain) != '.'.join(url_domain):
            # Special case for known related domains
            if ('asus.com' in self.domain and 'asus.com' in parsed_url.netloc) or \
               ('rog.asus.com' in self.domain and 'asus.com' in parsed_url.netloc):
                # Allow related domains
                pass
            else:
                return False
        
        # Skip URLs with certain extensions (media files, stylesheets, scripts, etc.)
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
            
        # Skip common non-content URLs
        skip_patterns = [
            '/cdn-cgi/', '/wp-admin/', '/wp-includes/',  # CMS/CDN paths
            '/login', '/logout', '/signin', '/signout', '/register',  # Auth paths
            '/cart', '/checkout', '/basket', '/shopping-cart',  # E-commerce paths
            '/account', '/profile', '/user/',  # User account paths
            '/search', '/tag/', '/category/',  # Search/taxonomy paths that often duplicate content
            '/feed/', '/rss/', '/atom/',  # Feed paths
            '/print/', '/email/', '/share/',  # Utility paths
            '/comment', '/trackback', '/pingback'  # Comment paths
        ]
        
        # Check if the URL path contains any of the skip patterns
        if any(pattern in parsed_url.path.lower() for pattern in skip_patterns):
            return False
        
        # For query parameters, be selective but not overly restrictive
        if parsed_url.query:
            # Skip tracking parameters but keep important ones
            skip_params = ['utm_', 'fbclid', 'gclid', 'msclkid', 'zanpid', 'dclid']
            
            # If the query only contains tracking parameters, skip it
            query_parts = parsed_url.query.split('&')
            if all(any(skip in param.lower() for skip in skip_params) for param in query_parts):
                return False
        
        return True
    
    """
    Extracts and processes links from a webpage using Jina Reader API.
    - Gets page content via Jina
    - Parses HTML for links
    - Normalizes and validates found URLs
    - Handles special cases for specific domains
    - Saves responses for debugging
    """
    async def extract_links(self, url: str) -> List[str]:
        """Extract links from a page using Jina Reader API"""
        try:
            # Get content using Jina Reader
            content = get_jina_response(url)
            
            # Check if content is empty or very short
            if not content or len(content) < 50:
                print(f"Warning: Received very short or empty content from {url}")
                return [], "Empty or very short content received"
            
            # Save the response
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_url = re.sub(r'[^\w]', '_', url)[-30:]
            filename = f"jina_responses/jina_response_{clean_url}_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Parse the content
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else "No title"
            
            # Extract all links
            links = []
            all_links_found = []
            
            # Debug: Print all links found in the page
            print(f"Extracting links from {url}")
            print(f"Page title: {title}")
            
            # Extract all links from the page
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                all_links_found.append(href)
                
                # Normalize the URL
                normalized_url = self.normalize_url(href, url)
                
                # Check if it's a valid URL to crawl
                if normalized_url and self.is_valid_url(normalized_url) and normalized_url not in links:
                    # For ROG website, prioritize product category links
                    if 'rog.asus.com' in url and any(category in normalized_url for category in 
                                                    ['-group', '/group/', '/series/', '/models/', '/specs/']):
                        # Add to the beginning of the list to prioritize
                        links.insert(0, normalized_url)
                    else:
                        links.append(normalized_url)
            
            # Debug information
            if links:
                print(f"Found {len(links)} valid links")
                # Print the first 5 links for debugging
                for i, link in enumerate(links[:5]):
                    print(f"  {i+1}. {link}")
            else:
                print(f"Warning: No valid links found on page {url}")
                print("Sample of raw links found:")
                for i, link in enumerate(all_links_found[:10]):  # Print first 10 raw links
                    print(f"  {i+1}. {link}")
                
                # Try to extract links from markdown content
                markdown_links = re.findall(r'\[.*?\]\((.*?)\)', content)
                if markdown_links:
                    print(f"Found {len(markdown_links)} links in markdown format")
                    for href in markdown_links:
                        normalized_url = self.normalize_url(href, url)
                        if normalized_url and self.is_valid_url(normalized_url) and normalized_url not in links:
                            links.append(normalized_url)
                    
                    if links:
                        print(f"Added {len(links)} valid links from markdown content")
            
            # Update the site map with the title and outgoing links
            if url in self.site_map:
                self.site_map[url].title = title
                self.site_map[url].outgoing_links = links
            
            return links, content
            
        except Exception as e:
            print(f"Error extracting links from {url}: {e}")
            return [], f"Error: {str(e)}"
    
    """
    Uses Gemini AI to evaluate how relevant a page's content is to the search prompt.
    Returns:
    - Relevance score (0.0-1.0)
    - Explanation of the score
    - Content summary
    """
    async def evaluate_content_relevance(self, url: str, content: str) -> Tuple[float, str, str]:
        """
        Evaluate the relevance of the content to the search prompt
        Returns (relevance_score, reason, content_summary)
        """
        prompt = f"""
        Evaluate the relevance of the following web page content to this search prompt:
        
        SEARCH PROMPT: "{self.search_prompt}"
        
        For the content, provide:
        1. A relevance score from 0.0 to 1.0, where:
           - 1.0: Highly relevant, contains exactly what we're looking for
           - 0.7-0.9: Very relevant, contains most of what we're looking for
           - 0.4-0.6: Somewhat relevant, contains some useful information
           - 0.1-0.3: Slightly relevant, has minimal useful information
           - 0.0: Not relevant at all
        
        2. A brief explanation for the score (1-2 sentences)
        
        3. A concise summary of the page content (3-5 sentences)
        
        Format your response as a valid JSON object with these fields:
        - relevance_score: float
        - relevance_reason: string
        - content_summary: string
        
        URL being evaluated: {url}
        """
        
        try:
            # Generate response from Gemini
            response = self.gemini_model.generate_content(prompt + "\n\nContent:\n" + content[:100000])  # Limit content size
            
            # Extract JSON from response
            json_text = self._extract_json_from_response(response.text)
            
            # Parse the JSON
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
        # Look for JSON object
        json_match = re.search(r'{.*}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return "{}"  # Return empty object if no JSON found
    
    """
    Main crawling logic that:
    - Implements breadth-first search with relevance prioritization
    - Manages crawl depth and page limits
    - Updates site map data structure
    - Tracks progress with tqdm
    """
    async def crawl(self):
        """Crawl the website and build a site map with relevance scores"""
        print(f"Starting crawl of {self.base_url} with prompt: {self.search_prompt}")
        
        # Initialize with the base URL
        base_node = LinkNode(url=self.base_url, depth=0)
        self.site_map[self.base_url] = base_node
        
        # Queue of URLs to visit (url, depth)
        queue = [(self.base_url, 0)]
        queued_urls = {self.base_url}  # Track URLs already in queue to avoid duplicates
        
        # Progress bar
        pbar = tqdm(total=self.max_pages, desc="Crawling pages")
        
        while queue and len(self.visited_urls) < self.max_pages:
            # Sort queue to prioritize breadth-first but also higher relevance paths
            # This helps ensure we map the entire site structure more effectively
            if len(self.visited_urls) > 5:  # After visiting a few pages, we can start prioritizing
                # Sort by a combination of depth (primary) and parent relevance (secondary)
                queue.sort(key=lambda x: (x[1], -self._get_parent_relevance(x[0])))
            
            current_url, depth = queue.pop(0)
            queued_urls.remove(current_url)  # Remove from tracking set
            
            if current_url in self.visited_urls or depth > self.max_depth:
                continue
            
            print(f"\nCrawling: {current_url} (Depth: {depth})")
            self.visited_urls.add(current_url)
            
            # Mark as visited in the site map
            if current_url in self.site_map:
                self.site_map[current_url].visited = True
            
            # Extract links and content
            links, content = await self.extract_links(current_url)
            
            # Evaluate content relevance
            relevance_score, relevance_reason, content_summary = await self.evaluate_content_relevance(current_url, content)
            
            # Update the site map
            if current_url in self.site_map:
                self.site_map[current_url].relevance_score = relevance_score
                self.site_map[current_url].relevance_reason = relevance_reason
                self.site_map[current_url].content_summary = content_summary
            
            print(f"Relevance score: {relevance_score:.2f} - {relevance_reason}")
            
            # Add new links to the queue and site map
            for link in links:
                if link not in self.site_map and link not in queued_urls:
                    # Add to site map
                    self.site_map[link] = LinkNode(
                        url=link,
                        depth=depth + 1,
                        parent_url=current_url
                    )
                    
                    # Add to queue if within depth limit
                    if depth + 1 <= self.max_depth:
                        queue.append((link, depth + 1))
                        queued_urls.add(link)  # Track in our queued set
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({"Queue": len(queue), "Mapped": len(self.site_map)})
        
        pbar.close()
        print(f"Crawl completed. Visited {len(self.visited_urls)} pages out of {len(self.site_map)} discovered URLs.")
        
        return self.site_map
    
    def _get_parent_relevance(self, url: str) -> float:
        """Get the relevance score of the parent URL to help prioritize crawling"""
        if url not in self.site_map:
            return 0.0
            
        parent_url = self.site_map[url].parent_url
        if not parent_url or parent_url not in self.site_map:
            return 0.0
            
        return self.site_map[parent_url].relevance_score
    
    """
    Creates detailed reports of the crawl results:
    - JSON data export
    - Markdown report with:
      - Crawl statistics
      - Most relevant pages
      - Complete site structure
    """
    def generate_site_map_report(self):
        """Generate a comprehensive report of the site map with relevance scores"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"site_maps/site_map_{timestamp}.json"
        
        # Convert site map to serializable format
        serializable_map = {}
        for url, node in self.site_map.items():
            serializable_map[url] = node.model_dump()  # Use model_dump() instead of deprecated dict()
        
        # Save to JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_map, f, indent=2)
        
        print(f"Site map saved to: {filename}")
        
        # Generate a markdown report
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
            
            for page in relevant_pages[:20]:  # Top 20 most relevant
                title = page.title or "No title"
                score = f"{page.relevance_score:.2f}"
                summary = page.content_summary or "No summary"
                # Truncate long summaries
                if len(summary) > 100:
                    summary = summary[:97] + "..."
                
                f.write(f"| [{title}]({page.url}) | {title} | {score} | {summary} |\n")
            
            f.write("\n\n## Complete Site Map\n\n")
            
            # Group by depth
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
    
    """
    Creates two network visualizations:
    1. Complete site map showing all visited pages
    2. High-relevance view showing only the most relevant pages
    Uses NetworkX for graph creation and Matplotlib for visualization
    """
    def visualize_site_map(self):
        """Create a visual representation of the site map"""
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for url, node in self.site_map.items():
            if node.visited:
                # Use relevance score for color intensity
                G.add_node(url, relevance=node.relevance_score, title=node.title or "No title")
        
        # Add edges
        for url, node in self.site_map.items():
            if node.visited and node.outgoing_links:
                for link in node.outgoing_links:
                    if link in self.site_map and self.site_map[link].visited:
                        G.add_edge(url, link)
        
        # Check if we have enough nodes to create a visualization
        if len(G.nodes()) < 1:
            print("Not enough visited pages to create a visualization.")
            return None, None
        
        # Create the visualization
        plt.figure(figsize=(15, 12))
        
        # Position nodes using force-directed layout
        # Use k parameter to spread nodes more (higher values = more spread)
        pos = nx.spring_layout(G, seed=42, k=0.15)
        
        # Get relevance scores for coloring
        relevance_scores = [G.nodes[node]['relevance'] for node in G.nodes()]
        
        # Draw nodes with color based on relevance
        nodes = nx.draw_networkx_nodes(
            G, pos, 
            node_color=relevance_scores, 
            cmap=plt.cm.YlOrRd, 
            node_size=300,
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True)
        
        # Add labels for high-relevance nodes only (to avoid clutter)
        high_relevance_nodes = {node: G.nodes[node]['title'] for node in G.nodes() 
                               if G.nodes[node]['relevance'] > 0.6}
        nx.draw_networkx_labels(G, pos, labels=high_relevance_nodes, font_size=8, font_color='black')
        
        # Add a color bar
        plt.colorbar(nodes, label='Relevance Score')
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_filename = f"site_maps/site_map_visualization_{timestamp}.png"
        plt.title(f"Site Map for {self.base_url}\nColored by relevance to: {self.search_prompt}")
        plt.tight_layout()
        plt.savefig(viz_filename, dpi=300)
        plt.close()  # Close the figure to free memory
        
        # Create a second visualization showing only high-relevance nodes
        high_relevance_nodes_list = [node for node in G.nodes() if G.nodes[node]['relevance'] > 0.4]
        
        # Check if we have enough high-relevance nodes
        if len(high_relevance_nodes_list) < 1:
            print("No high-relevance nodes found. Skipping high-relevance visualization.")
            return viz_filename, None
        
        plt.figure(figsize=(12, 10))
        
        # Create subgraph with only high-relevance nodes
        H = G.subgraph(high_relevance_nodes_list)
        
        # Position nodes
        pos_h = nx.spring_layout(H, seed=42, k=0.2)
        
        # Get relevance scores
        h_relevance_scores = [H.nodes[node]['relevance'] for node in H.nodes()]
        
        # Draw nodes
        h_nodes = nx.draw_networkx_nodes(
            H, pos_h, 
            node_color=h_relevance_scores, 
            cmap=plt.cm.YlOrRd, 
            node_size=400,
            alpha=0.9
        )
        
        # Draw edges
        nx.draw_networkx_edges(H, pos_h, alpha=0.4, arrows=True)
        
        # Add labels for all nodes in this view
        labels = {node: H.nodes[node]['title'] for node in H.nodes()}
        nx.draw_networkx_labels(H, pos_h, labels=labels, font_size=8, font_color='black')
        
        # Create a colorbar explicitly specifying the axis
        # Get the current axis and create a colorbar
        ax = plt.gca()
        plt.colorbar(h_nodes, ax=ax, label='Relevance Score')
        
        # Save the high-relevance visualization
        high_rel_viz_filename = f"site_maps/high_relevance_map_{timestamp}.png"
        plt.title(f"High Relevance Pages for: {self.search_prompt}")
        plt.tight_layout()
        plt.savefig(high_rel_viz_filename, dpi=300)
        plt.close()  # Close the figure to free memory
        
        print(f"Site map visualization saved to: {viz_filename}")
        if high_rel_viz_filename:
            print(f"High relevance map saved to: {high_rel_viz_filename}")
        
        return viz_filename, high_rel_viz_filename

"""
Command-line interface for the site mapper.
Usage: python site_mapper.py <base_url> <search_prompt> [max_depth] [max_pages]
Handles special cases for specific domains and outputs crawl statistics.
"""
async def main():
    if len(sys.argv) < 3:
        print("Usage: python site_mapper.py <base_url> <search_prompt> [max_depth] [max_pages]")
        sys.exit(1)
    
    base_url = sys.argv[1]
    
    # Special handling for ASUS ROG URLs
    if 'rog.asus.com' in base_url:
        # Fix common issues with ASUS ROG URLs
        if '/laptops/' in base_url and '/laptops-group/' not in base_url:
            corrected_url = base_url.replace('/laptops/', '/laptops-group/')
            print(f"WARNING: Detected ASUS ROG laptops URL. The /laptops/ endpoint often returns errors.")
            print(f"CORRECTED: Changed base URL from {base_url} to {corrected_url}")
            base_url = corrected_url
    
    search_prompt = sys.argv[2]
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    max_pages = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    
    mapper = SiteMapper(base_url, search_prompt, max_depth, max_pages)
    
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
    else:
        print("No site map visualization was created (not enough data).")
        
    if high_rel_viz:
        print(f"High relevance visualization: {high_rel_viz}")
    else:
        print("No high-relevance visualization was created (no high-relevance nodes found).")
    
    # Print statistics
    total_urls = len(mapper.site_map)
    visited_urls = len(mapper.visited_urls)
    relevant_urls = sum(1 for node in mapper.site_map.values() 
                      if node.visited and node.relevance_score >= 0.5)
    highly_relevant_urls = sum(1 for node in mapper.site_map.values() 
                             if node.visited and node.relevance_score >= 0.7)
    
    print(f"\nStatistics:")
    print(f"- Total URLs discovered: {total_urls}")
    
    # Protect against division by zero
    discovered_percent = (visited_urls/total_urls*100) if total_urls > 0 else 0
    print(f"- URLs visited: {visited_urls} ({discovered_percent:.1f}% of discovered)")
    
    # Protect against division by zero
    relevant_percent = (relevant_urls/visited_urls*100) if visited_urls > 0 else 0
    highly_relevant_percent = (highly_relevant_urls/visited_urls*100) if visited_urls > 0 else 0
    
    print(f"- Relevant URLs (score ≥ 0.5): {relevant_urls} ({relevant_percent:.1f}% of visited)")
    print(f"- Highly relevant URLs (score ≥ 0.7): {highly_relevant_urls} ({highly_relevant_percent:.1f}% of visited)")
    
    # Print top 5 most relevant URLs
    print("\nTop 5 most relevant URLs:")
    top_urls = sorted([node for node in mapper.site_map.values() if node.visited], 
                     key=lambda x: x.relevance_score, reverse=True)[:5]
                     
    for i, node in enumerate(top_urls, 1):
        print(f"{i}. [{node.relevance_score:.2f}] {node.title or 'No title'}: {node.url}")
        print(f"   {node.relevance_reason}")
        
    print("\nYou can find the full report in the generated Markdown file.")

if __name__ == "__main__":
    asyncio.run(main()) 