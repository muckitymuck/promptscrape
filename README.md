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

## Output

The script will generate a JSON file with specifications organized by model number. Each model includes:

```json
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
  ]
}
```

All outputs will be timestamped and saved in the appropriate directories:
1. Log files containing raw crawler responses from each visited URL (in `logs/`)
2. JSON files with organized technical specifications by model number (in `output/`)
3. Original Jina Reader responses for each visited URL (in `jina_responses/`)

## Structure

- `jina.py` - Simple script to extract content using Jina Reader API
- `jina_autogen_specs.py` - Main script that combines AutoGen, Gemini, and Jina with agentic link evaluation
- `logs/` - Directory containing logged responses
- `output/` - Directory containing JSON output files
- `jina_responses/` - Directory containing Jina Reader API responses 