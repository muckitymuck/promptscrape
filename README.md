# Hardware Specifications Crawler

This project combines AutoGen, Gemini, and Jina Reader to extract and organize technical specifications for hardware products.

## Features

- Uses AutoGen's WebSurfer agent to crawl websites for technical specifications
- **Fully agentic LLM-powered link evaluation** that dynamically assesses relevance to the search context
- **Extracts summarized specifications from product cards** on listing pages
- Enhances content extraction using Jina Reader API
- Processes and organizes data using Google's Gemini AI
- Outputs structured JSON files with technical specifications organized by model number
- Combines data from multiple pages for more comprehensive results
- Adapts to different product types and websites without hardcoded rules

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

## Usage

### Jina Reader

To use Jina Reader directly:

```bash
python jina.py <url_to_extract>
```

Example:
```bash
python jina.py https://rog.asus.com/laptops/rog-zephyrus/rog-zephyrus-g16-2025-gu605
```

### AutoGen Specifications Crawler

To crawl a website, extract specifications, and organize them by model number:

```bash
python jina_autogen_specs.py <base_url> [search_context]
```

Example:
```bash
# Using automatic search context extraction from URL
python jina_autogen_specs.py https://rog.asus.com/laptops/rog-zephyrus/rog-zephyrus-g16-2025-gu605

# With explicit search context
python jina_autogen_specs.py https://rog.asus.com/laptops/rog-zephyrus/rog-zephyrus-g16-2025-gu605 "ROG Zephyrus G16 gaming laptop specifications"

# For product listing pages with multiple models
python jina_autogen_specs.py https://rog.asus.com/laptops/rog-zephyrus-series/?items=20393
```

### Site Mapper

To create a comprehensive site map with relevance scores based on a search prompt:

```bash
python site_mapper.py <base_url> <search_prompt> [max_depth] [max_pages]
```

Example:
```bash
# Basic usage
python site_mapper.py https://rog.asus.com/laptops "gaming laptops with RTX 4090"

# With custom depth and page limit
python site_mapper.py https://rog.asus.com/laptops "gaming laptops with RTX 4090" 4 100
```

The site mapper will:
1. Start at the provided base URL
2. Crawl the website up to the specified depth (default: 3)
3. Visit up to the specified number of pages (default: 50)
4. Evaluate each page's relevance to the search prompt using Gemini
5. Generate a comprehensive site map with relevance scores
6. Create a visual representation of the site structure
7. Produce detailed reports in JSON and Markdown formats

#### Site Mapper Output

The script generates three main outputs:
1. **JSON Data File**: Complete site map data with all details
2. **Markdown Report**: Human-readable report with relevance rankings and site structure
3. **Visualization**: Network graph showing the site structure with nodes colored by relevance

These outputs are saved in the `site_maps/` directory with timestamps in the filenames.

### Smart Specs Crawler

The Smart Specs Crawler builds on the Site Mapper's capabilities to intelligently extract hardware specifications from the most relevant pages:

```bash
python smart_specs_crawler.py <base_url> <search_prompt> [max_depth] [max_pages]
```

Example:
```bash
# Basic usage
python smart_specs_crawler.py https://rog.asus.com/laptops "ROG gaming laptops specifications"

# With custom depth and page limit
python smart_specs_crawler.py https://rog.asus.com/laptops "ROG gaming laptops specifications" 4 100
```

#### How It Works

The Smart Specs Crawler operates in two phases:

1. **Site Mapping Phase**:
   - Uses the most recent output from the Site Mapper to identify relevant pages
   - Automatically prioritizes pages with higher relevance scores (≥ 0.4)
   - Creates a structured map of the website focused on your search context

2. **Specification Extraction Phase**:
   - Processes pages in order of relevance (highest first)
   - Uses Jina Reader to extract clean content from each page
   - Applies Gemini AI to identify and extract hardware specifications
   - Intelligently follows relevant links containing terms like "spec", "technical", "detail", etc.
   - Consolidates specifications from multiple sources, prioritizing data from more relevant pages
   - Merges complementary information when the same model appears on multiple pages

#### Integration with Site Mapper

The Smart Specs Crawler directly integrates with the Site Mapper in the following ways:

1. **Direct Class Integration**: 
   - The crawler imports and instantiates the `SiteMapper` class from `site_mapper.py`
   - It passes the same parameters (`base_url`, `search_prompt`, `max_depth`, `max_pages`) to ensure consistency

2. **Real-time Site Mapping**:
   - Rather than reading from saved files, the crawler calls `await self.site_mapper.crawl()` to generate a fresh site map
   - This ensures the most up-to-date information is used for specification extraction

3. **Relevance-Based Filtering**:
   - After site mapping completes, the crawler filters pages using: 
     ```python
     relevant_pages = sorted(
         [node for node in site_map.values() if node.visited and node.relevance_score >= 0.4],
         key=lambda x: x.relevance_score,
         reverse=True
     )
     ```
   - This selects only pages with relevance scores ≥ 0.4 and sorts them from highest to lowest relevance

4. **Relevance Score Inheritance**:
   - When processing linked pages not in the original site map, the crawler assigns them a slightly reduced relevance score:
     ```python
     specs = await self.extract_specs_from_content(
         link,
         content,
         page.relevance_score * 0.9  # Slightly lower relevance for linked pages
     )
     ```

5. **Metadata Preservation**:
   - The crawler preserves the relevance scores and source URLs in the final output
   - This allows users to trace back to the original pages and understand the confidence level of each specification

#### Smart Specs Crawler Output

The script generates a comprehensive JSON file containing structured hardware specifications:

```json
[
  {
    "model_number": "GU605CX-XS98-CA",
    "brand": "ASUS",
    "series": "ROG Zephyrus",
    "specifications": {
      "Operating System": "Windows 11 Pro",
      "Graphics": "NVIDIA® GeForce RTX™ 5090 Laptop GPU",
      "Processor": "Intel® Core™ Ultra 9 Processor 285H",
      "Display": "16\" 2.5K (2560 x 1600, WQXGA) 16:10 240Hz OLED ROG Nebula Display",
      "Storage": "2TB M.2 NVMe™ PCIe® 4.0 SSD storage"
    },
    "summary_specs": [
      "Windows 11 Pro",
      "NVIDIA® GeForce RTX™ 5090 Laptop GPU",
      "Intel® Core™ Ultra 9 Processor 285H",
      "16\" 2.5K (2560 x 1600, WQXGA) 16:10 240Hz OLED ROG Nebula Display",
      "2TB M.2 NVMe™ PCIe® 4.0 SSD storage"
    ],
    "source_url": "https://rog.asus.com/laptops/rog-zephyrus/rog-zephyrus-g16-2025-gu605",
    "relevance_score": 0.95
  }
]
```

The output is saved in the `specs_output/` directory with a timestamp in the filename.

#### Key Features

- **Intelligent Page Selection**: Automatically prioritizes the most relevant pages based on site mapping
- **Comprehensive Specification Extraction**: Captures detailed hardware specifications across multiple categories
- **Model-Based Organization**: Organizes specifications by model number for easy reference
- **Data Consolidation**: Merges information from multiple sources to create comprehensive specifications
- **Source Tracking**: Maintains references to source URLs for verification
- **Relevance Scoring**: Includes relevance scores to indicate confidence in the extracted data

## Output

All outputs will be timestamped and saved in the appropriate directories:
1. Log files containing raw crawler responses from each visited URL (in `logs/`)
2. JSON files with organized technical specifications by model number (in `output/`)
3. Original Jina Reader responses for each visited URL (in `jina_responses/`)
4. Site maps with relevance scores and visualizations (in `site_maps/`)
5. Extracted hardware specifications organized by model (in `specs_output/`)

## Structure

- `jina.py` - Simple script to extract content using Jina Reader API
- `jina_autogen_specs.py` - Main script that combines AutoGen, Gemini, and Jina with agentic link evaluation
- `site_mapper.py` - Creates comprehensive site maps with relevance scoring
- `smart_specs_crawler.py` - Extracts hardware specifications using site mapper data
- `logs/` - Directory containing logged responses
- `output/` - Directory containing JSON output files
- `jina_responses/` - Directory containing Jina Reader API responses
- `site_maps/` - Directory containing site mapping data and visualizations
- `specs_output/` - Directory containing extracted hardware specifications 