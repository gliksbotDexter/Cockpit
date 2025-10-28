from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import requests
from urllib.parse import urljoin, urlparse
import time
import base64
from dataclasses import dataclass

# Optional imports with fallbacks
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    
    def to_markdown(self) -> str:
        return f"- **{self.title}**\n  {self.snippet}\n  URL: {self.url}"

class RobustWebSearchTool:
    name = "web_search"
    description = """Search and browse the web with multiple fallback strategies.
    Features:
    - HTML-based search (no API keys)
    - Direct URL access
    - Content extraction with fallbacks
    - Basic OCR capability (if available)
    - Resilient to various web challenges
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        self.session = requests.Session()
        # Rotate user agents to avoid blocking
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        self.session.headers.update({'User-Agent': self.user_agents[0]})
        
        # Configuration
        self.timeout = cfg.get('timeout', 15)
        self.max_retries = cfg.get('max_retries', 3)
        
    def _make_request(self, url: str, method: str = 'GET', **kwargs) -> Optional[requests.Response]:
        """Make HTTP request with retries and rotation"""
        kwargs.setdefault('timeout', self.timeout)
        
        for attempt in range(self.max_retries):
            try:
                # Rotate user agent
                ua = self.user_agents[attempt % len(self.user_agents)]
                self.session.headers.update({'User-Agent': ua})
                
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def search_duckduckgo(self, query: str, max_results: int = 5) -> str:
        """Search DuckDuckGo with fallback strategies"""
        try:
            # Try HTML search first
            results = self._search_ddg_html(query, max_results)
            if results:
                return results
                
            # Fallback to API-like approach
            return self._search_ddg_api_fallback(query, max_results)
            
        except Exception as e:
            return f"Search failed: {str(e)}. Try visiting URLs directly."
    
    def _search_ddg_html(self, query: str, max_results: int) -> str:
        """HTML-based DuckDuckGo search"""
        if not BS4_AVAILABLE:
            return ""
            
        try:
            search_url = "https://html.duckduckgo.com/html/"
            data = {'q': query, 'kl': 'us-en'}
            
            response = self._make_request('POST', search_url, data=data)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            result_divs = soup.find_all('div', class_='result')[:max_results]
            
            for div in result_divs:
                try:
                    title_link = div.find('a', class_='result__a')
                    if title_link:
                        title = title_link.get_text(strip=True)
                        url = title_link.get('href', '')
                        
                        snippet_div = div.find('a', class_='result__snippet')
                        snippet = snippet_div.get_text(strip=True) if snippet_div else ''
                        
                        if title and url:
                            results.append(SearchResult(title=title, url=url, snippet=snippet))
                except:
                    continue
            
            if results:
                output = ["Search Results:"]
                for i, result in enumerate(results, 1):
                    output.append(f"{i}. {result.to_markdown()}")
                return "\n".join(output)
                
        except Exception:
            pass
        return ""
    
    def _search_ddg_api_fallback(self, query: str, max_results: int) -> str:
        """API-style fallback for DuckDuckGo"""
        try:
            # DuckDuckGo Instant Answer API
            api_url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = self._make_request('GET', api_url, params=params)
            data = response.json()
            
            results = []
            
            # Add abstract/result
            if data.get('Abstract'):
                results.append(SearchResult(
                    title=data.get('Heading', 'Result'),
                    url=data.get('AbstractURL', ''),
                    snippet=data.get('Abstract', '')
                ))
            
            # Add related topics
            for topic in data.get('RelatedTopics', [])[:max_results-1]:
                if 'FirstURL' in topic:
                    results.append(SearchResult(
                        title=topic.get('Text', '').split(' - ')[0],
                        url=topic.get('FirstURL', ''),
                        snippet=''
                    ))
            
            if results:
                output = ["Search Results:"]
                for i, result in enumerate(results, 1):
                    output.append(f"{i}. {result.to_markdown()}")
                return "\n".join(output)
                
        except Exception:
            pass
        return "No results found with available methods."
    
    def fetch_and_analyze_url(self, url: str, extract_images: bool = False) -> str:
        """Fetch URL with multiple analysis strategies"""
        try:
            response = self._make_request('GET', url)
            
            content_type = response.headers.get('content-type', '').lower()
            
            # Handle different content types
            if 'text/html' in content_type:
                return self._analyze_html_content(response, url, extract_images)
            elif 'image' in content_type and OCR_AVAILABLE:
                return self._analyze_image_content(response, url)
            elif 'application/pdf' in content_type:
                return self._analyze_pdf_content(response, url)
            else:
                return self._analyze_generic_content(response, url)
                
        except Exception as e:
            return f"Failed to analyze {url}: {str(e)}"
    
    def _analyze_html_content(self, response: requests.Response, url: str, extract_images: bool) -> str:
        """Analyze HTML content with multiple strategies"""
        try:
            if BS4_AVAILABLE:
                return self._analyze_with_beautifulsoup(response, url, extract_images)
            else:
                return self._analyze_with_regex(response, url)
        except Exception as e:
            return self._analyze_raw_content(response, url)
    
    def _analyze_with_beautifulsoup(self, response: requests.Response, url: str, extract_images: bool) -> str:
        """Use BeautifulSoup for detailed analysis"""
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else "No title"
        
        # Extract main content
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        # Try to find main content area
        main_content = ""
        content_selectors = [
            'main', 'article', '.content', '#content', 
            '.main-content', '[role="main"]'
        ]
        
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                main_content = element.get_text(strip=True)
                break
        
        if not main_content:
            main_content = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in main_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True)[:10]:
            text = link.get_text(strip=True)
            href = urljoin(url, link['href'])
            if text:
                links.append(f"  - {text}: {href}")
        
        # Extract images if requested
        images = []
        if extract_images:
            for img in soup.find_all('img')[:5]:
                alt = img.get('alt', 'No alt text')
                src = img.get('src', '')
                if src:
                    full_src = urljoin(url, src)
                    images.append(f"  - {alt}: {full_src}")
        
        output = [
            f"📄 Page Analysis: {url}",
            f"Title: {title}",
            f"Content-Type: {response.headers.get('content-type', 'unknown')}",
            "",
            "📝 Content Preview:",
            clean_text[:1000] + ("..." if len(clean_text) > 1000 else ""),
            ""
        ]
        
        if links:
            output.extend(["🔗 Important Links:", *links[:10], ""])
        
        if images and extract_images:
            output.extend(["🖼️ Images:", *images, ""])
        
        return "\n".join(output)
    
    def _analyze_with_regex(self, response: requests.Response, url: str) -> str:
        """Fallback regex-based analysis"""
        import re
        
        text = response.text
        
        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', text, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else "No title"
        
        # Remove HTML tags for basic content
        clean_text = re.sub(r'<[^>]+>', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return f"""📄 Page Analysis: {url}
Title: {title}
Content-Type: {response.headers.get('content-type', 'unknown')}

📝 Content Preview:
{clean_text[:800]}{'...' if len(clean_text) > 800 else ''}"""
    
    def _analyze_raw_content(self, response: requests.Response, url: str) -> str:
        """Raw content analysis fallback"""
        preview = response.text[:800]
        if len(response.text) > 800:
            preview += "..."
            
        return f"""📄 Raw Content Analysis: {url}
Content-Type: {response.headers.get('content-type', 'unknown')}
Content-Length: {len(response.text)} bytes

Preview:
{preview}"""
    
    def _analyze_image_content(self, response: requests.Response, url: str) -> str:
        """Analyze image content with OCR if available"""
        if not OCR_AVAILABLE:
            return f"🖼️ Image detected: {url}\n(Content-Type: {response.headers.get('content-type')})\nOCR not available - install Pillow and pytesseract for text extraction."
        
        try:
            from io import BytesIO
            image = Image.open(BytesIO(response.content))
            
            # Extract text with OCR
            text = pytesseract.image_to_string(image)
            
            # Get image info
            width, height = image.size
            mode = image.mode
            
            return f"""🖼️ Image Analysis: {url}
Dimensions: {width}x{height}
Mode: {mode}
Content-Type: {response.headers.get('content-type')}

🔍 OCR Text Extracted:
{text[:500]}{'...' if len(text) > 500 else ''}"""
            
        except Exception as e:
            return f"🖼️ Image detected: {url}\nOCR failed: {str(e)}"
    
    def _analyze_pdf_content(self, response: requests.Response, url: str) -> str:
        """Analyze PDF content"""
        try:
            # Check if PyPDF2 or similar is available
            import io
            has_pypdf = False
            try:
                import PyPDF2
                has_pypdf = True
            except ImportError:
                pass
            
            if has_pypdf:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(response.content))
                num_pages = len(pdf_reader.pages)
                
                # Extract text from first few pages
                text = ""
                for i in range(min(3, num_pages)):
                    text += pdf_reader.pages[i].extract_text() + "\n"
                
                return f"""📄 PDF Document: {url}
Pages: {num_pages}
Content-Type: {response.headers.get('content-type')}

📝 Text Preview (first 3 pages):
{text[:1000]}{'...' if len(text) > 1000 else ''}"""
            else:
                return f"📄 PDF Document: {url}\n(Content-Type: {response.headers.get('content-type')})\nPyPDF2 not installed for text extraction."
                
        except Exception as e:
            return f"📄 PDF Document: {url}\nAnalysis failed: {str(e)}"
    
    def _analyze_generic_content(self, response: requests.Response, url: str) -> str:
        """Generic content analysis"""
        content_preview = response.text[:500]
        if len(response.text) > 500:
            content_preview += "..."
            
        return f"""📄 File: {url}
Content-Type: {response.headers.get('content-type', 'unknown')}
Content-Length: {len(response.content)} bytes

Preview:
{content_preview}"""
    
    def run(self, query: str = None, url: str = None, 
            max_results: int = 5, extract_images: bool = False) -> str:
        """
        Main tool entry point
        
        Args:
            query: Search query
            url: URL to analyze directly
            max_results: Number of search results
            extract_images: Whether to extract image information
        """
        if query:
            return self.search_duckduckgo(query, max_results)
        elif url:
            return self.fetch_and_analyze_url(url, extract_images)
        else:
            return "Please provide either a search query or a URL to analyze."

# Enhanced version with basic "mouse/keyboard" simulation concepts
class AdvancedWebTool(RobustWebSearchTool):
    name = "advanced_web_browse"
    description = """Advanced web browsing with simulated interaction capabilities.
    Can simulate basic navigation patterns and handle complex web scenarios."""
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.navigation_history = []
        self.current_context = {}
    
    def simulate_click_sequence(self, urls: List[str], delay: float = 1.0) -> str:
        """Simulate clicking through a sequence of URLs"""
        results = []
        
        for i, url in enumerate(urls):
            if i > 0:
                time.sleep(delay)  # Simulate human-like delays
            
            result = self.fetch_and_analyze_url(url)
            results.append(f"Step {i+1}: {url}\n{result}")
            self.navigation_history.append(url)
        
        return "\n---\n".join(results)
    
    def extract_structured_data(self, url: str, data_types: List[str] = None) -> str:
        """Extract specific types of structured data"""
        if data_types is None:
            data_types = ['tables', 'lists', 'forms', 'metadata']
        
        try:
            response = self._make_request('GET', url)
            if not BS4_AVAILABLE:
                return "BeautifulSoup required for structured data extraction"
            
            soup = BeautifulSoup(response.text, 'html.parser')
            extracted_data = []
            
            if 'tables' in data_types:
                tables = soup.find_all('table')
                extracted_data.append(f"Tables found: {len(tables)}")
                for i, table in enumerate(tables[:3]):
                    rows = table.find_all('tr')
                    extracted_data.append(f"Table {i+1}: {len(rows)} rows")
            
            if 'lists' in data_types:
                lists = soup.find_all(['ul', 'ol'])
                extracted_data.append(f"Lists found: {len(lists)}")
            
            if 'forms' in data_types:
                forms = soup.find_all('form')
                extracted_data.append(f"Forms found: {len(forms)}")
                for i, form in enumerate(forms):
                    inputs = form.find_all(['input', 'textarea', 'select'])
                    extracted_data.append(f"Form {i+1}: {len(inputs)} input fields")
            
            if 'metadata' in data_types:
                meta_tags = soup.find_all('meta')
                extracted_data.append(f"Meta tags found: {len(meta_tags)}")
            
            return f"Structured Data from {url}:\n" + "\n".join(extracted_data)
            
        except Exception as e:
            return f"Structured data extraction failed: {str(e)}"
    
    def run(self, query: str = None, url: str = None, action: str = "browse",
            urls_sequence: List[str] = None, extract_data_types: List[str] = None) -> str:
        """Enhanced run method with advanced capabilities"""
        if action == "sequence" and urls_sequence:
            return self.simulate_click_sequence(urls_sequence)
        elif action == "extract_structured" and url:
            return self.extract_structured_data(url, extract_data_types)
        else:
            return super().run(query=query, url=url)

# Installation helper
def get_installation_instructions() -> str:
    """Get installation instructions for optional dependencies"""
    return """
Optional Dependencies for Full Functionality:

pip install beautifulsoup4      # For better HTML parsing
pip install Pillow              # For image processing
pip install pytesseract         # For OCR capabilities
pip install PyPDF2             # For PDF text extraction

For OCR, also install Tesseract OCR engine:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- macOS: brew install tesseract
- Linux: sudo apt-get install tesseract-ocr

Note: The tool works without these dependencies but with reduced functionality.
"""
