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
from bs4 import BeautifulSoup
from tqdm import tqdm

# Import functions from jina.py
from jina import get_jina_response

# Import SiteMapper from site_mapper.py
from site_mapper import SiteMapper

# Apply nest_asyncio to allow async operations
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class HardwareSpecification(BaseModel):
    """Model for hardware technical specifications"""
    model_number: str = Field(..., description="Model/Product number")
    brand: str = Field(None, description="Product brand")
    series: str = Field(None, description="Product series")
    specifications: dict = Field(default_factory=dict, description="Technical specifications key-value pairs")
    summary_specs: List[str] = Field(default_factory=list, description="List of summarized specifications in bullet point format")
    source_url: str = Field(None, description="Source URL where the specifications were found")
    relevance_score: float = Field(0.0, description="Relevance score from site mapping")

class SmartSpecsCrawler:
    def __init__(self, base_url: str, search_prompt: str, max_depth: int = 3, max_pages: int = 50):
        self.base_url = base_url
        self.search_prompt = search_prompt
        self.max_depth = max_depth
        self.max_pages = max_pages
        
        # Initialize the site mapper
        self.site_mapper = SiteMapper(base_url, search_prompt, max_depth, max_pages)
        
        # Initialize Gemini model for content processing
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Create necessary directories
        os.makedirs('specs_output', exist_ok=True)
        
        print(f"Initialized SmartSpecsCrawler for {base_url}")
        print(f"Search prompt: {search_prompt}")
        
    async def extract_specs_from_content(self, url: str, content: str, relevance_score: float) -> List[HardwareSpecification]:
        """Extract specifications from page content using Gemini"""
        prompt = f"""
        Extract technical specifications for hardware products from the provided content.
        Focus on finding complete and accurate specifications.
        
        For each model number found, provide:
        1. The complete model number
        2. Brand name (if found)
        3. Product series (if found)
        4. All technical specifications organized by category
        5. A summary list of key specifications
        
        Important specifications to capture:
        - CPU/Processor
        - GPU/Graphics
        - Display/Screen
        - Memory/RAM
        - Storage
        - Operating System
        - Ports/Connectivity
        - Battery (for portable devices)
        - Dimensions/Weight
        
        Format the output as a valid JSON array where each item contains:
        - model_number: string (required)
        - brand: string (if found)
        - series: string (if found)
        - specifications: dictionary of specification categories
        - summary_specs: array of key specifications as bullet points
        
        URL being processed: {url}
        """
        
        try:
            # Generate response from Gemini
            response = self.gemini_model.generate_content(prompt + "\n\nContent:\n" + content[:100000])
            
            # Extract JSON from response
            json_text = self._extract_json_from_response(response.text)
            
            # Parse the JSON
            specs_data = json.loads(json_text)
            if not isinstance(specs_data, list):
                specs_data = [specs_data]
            
            # Convert to HardwareSpecification objects and add metadata
            specs_list = []
            for spec in specs_data:
                spec['source_url'] = url
                spec['relevance_score'] = relevance_score
                specs_list.append(HardwareSpecification(**spec))
            
            return specs_list
            
        except Exception as e:
            print(f"Error extracting specifications from {url}: {e}")
            return []
    
    def _extract_json_from_response(self, text):
        """Extract JSON from Gemini response"""
        # Look for JSON array
        json_match = re.search(r'\[\s*{.*}\s*\]', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # Try looking for JSON object
        json_match = re.search(r'{.*}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return "[]"  # Return empty array if no JSON found
    
    async def crawl_and_extract(self):
        """Crawl the site and extract specifications from relevant pages"""
        print("Phase 1: Mapping the site structure...")
        
        # First, use the site mapper to identify relevant pages
        site_map = await self.site_mapper.crawl()
        
        # Get pages sorted by relevance score
        relevant_pages = sorted(
            [node for node in site_map.values() if node.visited and node.relevance_score >= 0.4],
            key=lambda x: x.relevance_score,
            reverse=True
        )
        
        print(f"\nFound {len(relevant_pages)} relevant pages to process")
        
        # Process the most relevant pages
        all_specs = []
        processed_urls = set()
        
        print("\nPhase 2: Extracting specifications from relevant pages...")
        for page in tqdm(relevant_pages, desc="Processing pages"):
            if page.url in processed_urls:
                continue
                
            try:
                # Get content using Jina Reader
                content = get_jina_response(page.url)
                
                # Extract specifications
                specs = await self.extract_specs_from_content(
                    page.url, 
                    content, 
                    page.relevance_score
                )
                
                if specs:
                    print(f"\nFound {len(specs)} product specifications on {page.url}")
                    all_specs.extend(specs)
                
                processed_urls.add(page.url)
                
                # Also process outgoing links if they seem relevant
                for link in page.outgoing_links:
                    if link not in processed_urls and any(term in link.lower() for term in 
                        ['spec', 'technical', 'detail', 'product', 'model']):
                        try:
                            content = get_jina_response(link)
                            specs = await self.extract_specs_from_content(
                                link,
                                content,
                                page.relevance_score * 0.9  # Slightly lower relevance for linked pages
                            )
                            if specs:
                                print(f"Found {len(specs)} product specifications on linked page {link}")
                                all_specs.extend(specs)
                            processed_urls.add(link)
                        except Exception as e:
                            print(f"Error processing linked page {link}: {e}")
                
            except Exception as e:
                print(f"Error processing {page.url}: {e}")
        
        # Consolidate specifications
        consolidated_specs = self._consolidate_specs(all_specs)
        
        # Save the results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"specs_output/specifications_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(
                [spec.model_dump() for spec in consolidated_specs],
                f,
                indent=2
            )
        
        print(f"\nProcess completed successfully!")
        print(f"Found specifications for {len(consolidated_specs)} unique models")
        print(f"Results saved to: {output_file}")
        
        return consolidated_specs
    
    def _consolidate_specs(self, specs_list: List[HardwareSpecification]) -> List[HardwareSpecification]:
        """Consolidate specifications by model number, merging data from multiple sources"""
        model_dict = {}
        
        for spec in specs_list:
            model = spec.model_number
            if not model:
                continue
            
            if model not in model_dict:
                model_dict[model] = spec
            else:
                # Keep the version from the more relevant source
                if spec.relevance_score > model_dict[model].relevance_score:
                    model_dict[model] = spec
                elif spec.relevance_score == model_dict[model].relevance_score:
                    # If same relevance, merge specifications
                    existing_spec = model_dict[model]
                    
                    # Merge specifications dictionaries
                    for category, value in spec.specifications.items():
                        if category not in existing_spec.specifications:
                            existing_spec.specifications[category] = value
                    
                    # Add any new summary specs
                    existing_spec.summary_specs.extend(
                        spec for spec in spec.summary_specs 
                        if spec not in existing_spec.summary_specs
                    )
        
        return list(model_dict.values())

async def main():
    if len(sys.argv) < 3:
        print("Usage: python smart_specs_crawler.py <base_url> <search_prompt> [max_depth] [max_pages]")
        sys.exit(1)
    
    base_url = sys.argv[1]
    search_prompt = sys.argv[2]
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    max_pages = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    
    crawler = SmartSpecsCrawler(base_url, search_prompt, max_depth, max_pages)
    await crawler.crawl_and_extract()

if __name__ == "__main__":
    asyncio.run(main()) 