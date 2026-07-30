import urllib.request
import urllib.parse
import re
import html
from html.parser import HTMLParser
from typing import List, Dict, Tuple, Any, Optional

class PageParser(HTMLParser):
    """HTML parser to extract text and label interactive components (links, forms, inputs, buttons)."""
    
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.text_parts = []
        self.elements = []  # list of {"type": "link/input/button", "text": str, "url": str, "attrs": dict, "form": dict}
        
        # Form tracking
        self.current_link = None
        self.current_button = None
        self.current_form = None
        self.forms = []
        self.in_script_style = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        tag_lower = tag.lower()
        
        if tag_lower in ("script", "style", "noscript", "head", "iframe", "svg"):
            self.in_script_style = True
            return
            
        if self.in_script_style:
            return
            
        if tag_lower == "form":
            action = attrs_dict.get("action", "")
            method = attrs_dict.get("method", "get").lower()
            self.current_form = {
                "action": urllib.parse.urljoin(self.base_url, action),
                "method": method,
                "inputs": []
            }
            self.forms.append(self.current_form)
            
        elif tag_lower == "a":
            href = attrs_dict.get("href", "")
            if href and not href.startswith("javascript:") and not href.startswith("mailto:") and not href.startswith("tel:"):
                abs_url = urllib.parse.urljoin(self.base_url, href)
                self.current_link = {
                    "type": "link",
                    "text": "",
                    "url": abs_url,
                    "attrs": attrs_dict
                }
                
        elif tag_lower == "input":
            input_type = attrs_dict.get("type", "text").lower()
            name = attrs_dict.get("name", "")
            value = attrs_dict.get("value", "")
            
            if input_type in ("text", "search", "url", "email", "password", "number", "hidden") and name:
                elem = {
                    "type": "input",
                    "text": attrs_dict.get("placeholder", "") or name,
                    "name": name,
                    "value": value,
                    "input_type": input_type,
                    "attrs": attrs_dict,
                    "form": self.current_form
                }
                self.elements.append(elem)
                if self.current_form:
                    self.current_form["inputs"].append(elem)
                # Render inline placeholder
                self.text_parts.append(f" [Input #{len(self.elements)}: {elem['text']} ({name})] ")
                
            elif input_type in ("submit", "button"):
                elem = {
                    "type": "button",
                    "text": attrs_dict.get("value", "") or "Submit",
                    "name": name,
                    "attrs": attrs_dict,
                    "form": self.current_form
                }
                self.elements.append(elem)
                if self.current_form:
                    self.current_form["inputs"].append(elem)
                self.text_parts.append(f" [Button #{len(self.elements)}: {elem['text']}] ")
                
        elif tag_lower == "button":
            self.current_button = {
                "type": "button",
                "text": "",
                "name": attrs_dict.get("name", ""),
                "attrs": attrs_dict,
                "form": self.current_form
            }

    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        if tag_lower in ("script", "style", "noscript", "head", "iframe", "svg"):
            self.in_script_style = False
            return
            
        if self.in_script_style:
            return
            
        if tag_lower == "form":
            self.current_form = None
        elif tag_lower == "a" and self.current_link:
            text = self.current_link["text"].strip()
            if text:
                self.elements.append(self.current_link)
                self.text_parts.append(f" [Link #{len(self.elements)}: {text}] ")
            self.current_link = None
        elif tag_lower == "button" and self.current_button:
            text = self.current_button["text"].strip() or "Button"
            self.elements.append(self.current_button)
            self.text_parts.append(f" [Button #{len(self.elements)}: {text}] ")
            if self.current_form:
                self.current_form["inputs"].append(self.current_button)
            self.current_button = None

    def handle_data(self, data):
        if self.in_script_style:
            return
        
        if self.current_link:
            self.current_link["text"] += data
        elif self.current_button:
            self.current_button["text"] += data
        else:
            cleaned = re.sub(r'\s+', ' ', data)
            if cleaned and cleaned != ' ':
                self.text_parts.append(cleaned)


class BrowserServer:
    """Natively emulates a web session, cookie tracking, and form submissions."""
    
    def __init__(self, guard, session_name: str):
        self.guard = guard
        self.session_name = session_name
        self.current_url = None
        self.current_page_text = "No page loaded. Call web_navigate first."
        self.elements: List[Dict[str, Any]] = []
        self.history: List[str] = []
        self.inputs_state: Dict[int, str] = {}
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'
        
    def _fetch_url(self, url: str, data: bytes = None) -> Tuple[str, str]:
        req = urllib.request.Request(url, data=data, headers={'User-Agent': self.user_agent})
        with urllib.request.urlopen(req, timeout=10) as response:
            html_bytes = response.read()
            charset = response.headers.get_content_charset() or 'utf-8'
            try:
                html_str = html_bytes.decode(charset, errors='replace')
            except:
                html_str = html_bytes.decode('utf-8', errors='replace')
            return html_str, response.geturl()

    def web_navigate(self, url: str) -> str:
        try:
            if not (url.startswith("http://") or url.startswith("https://")):
                url = "https://" + url
                
            # Safety checks via WorkspaceGuard
            self.guard.validate_path(".") # Ensure we reside inside environment
            
            html_str, final_url = self._fetch_url(url)
            self.current_url = final_url
            self.history.append(final_url)
            
            # Parse components
            parser = PageParser(final_url)
            parser.feed(html_str)
            
            self.elements = parser.elements
            self.inputs_state = {}
            
            # Format visual layout
            body_text = "".join(parser.text_parts)
            body_text = re.sub(r'\n+', '\n', body_text)
            body_text = re.sub(r' +', ' ', body_text)
            
            elems_desc = []
            for i, elem in enumerate(self.elements):
                idx = i + 1
                if elem["type"] == "link":
                    elems_desc.append(f"[{idx}] Link: '{elem['text']}' -> {elem['url']}")
                elif elem["type"] == "input":
                    val_str = f" (Value: '{self.inputs_state.get(idx, elem.get('value', ''))}')"
                    elems_desc.append(f"[{idx}] Input field: '{elem['text']}' (name='{elem['name']}'){val_str}")
                elif elem["type"] == "button":
                    elems_desc.append(f"[{idx}] Button: '{elem['text']}'")
                    
            out = f"--- Loaded Page: {final_url} ---\n\n"
            out += body_text[:2000]
            if len(body_text) > 2000:
                out += "\n... [truncated]"
            out += "\n\n--- Interactive Elements on Page ---\n"
            out += "\n".join(elems_desc) if elems_desc else "None"
            
            self.current_page_text = out
            return out
        except Exception as e:
            return f"Error loading '{url}': {str(e)}"

    def web_click(self, index: int) -> str:
        idx_zero = index - 1
        if idx_zero < 0 or idx_zero >= len(self.elements):
            return f"Error: Element index #{index} out of range (1-{len(self.elements)})."
            
        elem = self.elements[idx_zero]
        if elem["type"] == "link":
            return self.web_navigate(elem["url"])
            
        elif elem["type"] == "button":
            form = elem.get("form")
            if not form:
                return "Error: Button not associated with any form."
                
            form_data = {}
            for input_elem in form.get("inputs", []):
                if input_elem["type"] == "input":
                    for i, e in enumerate(self.elements):
                        if e is input_elem:
                            val = self.inputs_state.get(i + 1, input_elem.get("value", ""))
                            form_data[input_elem["name"]] = val
                            break
                            
            action = form["action"]
            method = form["method"]
            
            try:
                if method == "post":
                    encoded_data = urllib.parse.urlencode(form_data).encode('utf-8')
                    html_str, final_url = self._fetch_url(action, data=encoded_data)
                else:
                    sep = "&" if "?" in action else "?"
                    query_url = action + sep + urllib.parse.urlencode(form_data)
                    html_str, final_url = self._fetch_url(query_url)
                    
                self.current_url = final_url
                self.history.append(final_url)
                
                parser = PageParser(final_url)
                parser.feed(html_str)
                self.elements = parser.elements
                self.inputs_state = {}
                
                body_text = "".join(parser.text_parts)
                body_text = re.sub(r'\n+', '\n', body_text)
                body_text = re.sub(r' +', ' ', body_text)
                
                elems_desc = []
                for i, el in enumerate(self.elements):
                    idx = i + 1
                    if el["type"] == "link":
                        elems_desc.append(f"[{idx}] Link: '{el['text']}' -> {el['url']}")
                    elif el["type"] == "input":
                        val_str = f" (Value: '{self.inputs_state.get(idx, el.get('value', ''))}')"
                        elems_desc.append(f"[{idx}] Input field: '{el['text']}' (name='{el['name']}'){val_str}")
                    elif el["type"] == "button":
                        elems_desc.append(f"[{idx}] Button: '{el['text']}'")
                        
                out = f"--- Form Submitted. Loaded Page: {final_url} ---\n\n"
                out += body_text[:2000]
                if len(body_text) > 2000:
                    out += "\n... [truncated]"
                out += "\n\n--- Interactive Elements on Page ---\n"
                out += "\n".join(elems_desc) if elems_desc else "None"
                
                self.current_page_text = out
                return out
            except Exception as e:
                return f"Error submitting form to '{action}': {str(e)}"
        else:
            return f"Error: Element index #{index} is of type '{elem['type']}' and cannot be clicked."

    def web_type(self, index: int, text: str) -> str:
        idx_zero = index - 1
        if idx_zero < 0 or idx_zero >= len(self.elements):
            return f"Error: Element index #{index} out of range (1-{len(self.elements)})."
            
        elem = self.elements[idx_zero]
        if elem["type"] != "input":
            return f"Error: Element index #{index} is of type '{elem['type']}' and cannot be typed into."
            
        self.inputs_state[index] = text
        return f"Success: Entered '{text}' into input field #{index} (name='{elem['name']}')."

    def web_back(self) -> str:
        if len(self.history) < 2:
            return "Error: No history to go back to."
            
        self.history.pop()
        prev_url = self.history.pop()
        return self.web_navigate(prev_url)
